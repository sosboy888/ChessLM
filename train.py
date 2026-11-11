"""Train a character-level chess model on prepared game records.

    python train.py --size tiny  --max_iters 2000    # sanity run
    python train.py --size small                      # main run (~24M params)

Each training example is one whole game (padded). Padding positions are excluded from the loss.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from chess_tokenizer import PAD, decode, encode
from model import GPT, GPTConfig

SIZES = {
    "tiny":  dict(n_layer=6, n_head=6, n_kv_head=6, dim=384, hidden_dim=1024),
    "small": dict(n_layer=8, n_head=8, n_kv_head=4, dim=512, hidden_dim=1408),
    "base":  dict(n_layer=12, n_head=10, n_kv_head=5, dim=640, hidden_dim=1728),   # ~55M
}

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=SIZES, default="small")
ap.add_argument("--data_dir", default="data")
ap.add_argument("--out_dir", default="out")
ap.add_argument("--batch_size", type=int, default=32)      # games per micro-step
ap.add_argument("--grad_accum", type=int, default=1)
ap.add_argument("--max_iters", type=int, default=20_000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--min_lr_frac", type=float, default=0.1)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--weight_decay", type=float, default=0.1)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--eval_batches", type=int, default=50)
ap.add_argument("--resume", action="store_true", help="continue from out/<size>_last.pt")
args = ap.parse_args()

torch.manual_seed(1337)
np.random.seed(1337)
device = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs(args.out_dir, exist_ok=True)

meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
L = meta["row_len"]


def load(name):
    return np.memmap(os.path.join(args.data_dir, f"{name}.bin"), dtype=np.uint8, mode="r").reshape(-1, L)


train_data, val_data = load("train"), load("val")
cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=meta["block_size"], **SIZES[args.size])
model = GPT(cfg).to(device)
print(f"model: {model.num_params() / 1e6:.1f}M params | {len(train_data):,} train games, {len(val_data):,} val games")


def get_batch(data):
    ix = np.sort(np.random.randint(0, len(data), args.batch_size))
    rows = torch.from_numpy(data[ix].astype(np.int64)).to(device)
    x, y = rows[:, :-1], rows[:, 1:].clone()
    y[y == PAD] = -1          # don't learn to predict padding
    return x, y


decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95), fused=True,
)


def lr_at(it):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    progress = (it - args.warmup) / max(1, args.max_iters - args.warmup)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(args.eval_batches):
            x, y = get_batch(data)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def sample(prompt=";1.e4 e5 2.Nf3", n=160):
    model.eval()
    ids = torch.tensor([encode(prompt)], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(ids, n, temperature=0.7, top_k=10)
    model.train()
    return decode(out[0].tolist())


def save(path):
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "args": vars(args)}, path)


def save_last(it):
    # full training state, so an interrupted run can continue with --resume
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "best_val": best_val,
                "cfg": cfg.__dict__, "args": vars(args)}, os.path.join(args.out_dir, f"{args.size}_last.pt"))


best_val, start_it, t0 = float("inf"), 0, time.time()
if args.resume:
    ck = torch.load(os.path.join(args.out_dir, f"{args.size}_last.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    start_it, best_val = ck["it"], ck["best_val"]
    print(f"resumed from iter {start_it} (best val {best_val:.4f})")
model.train()
for it in range(start_it, args.max_iters + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)

    if it % args.eval_every == 0:
        losses = estimate_loss()
        print(f"\n[eval] iter {it}: train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"| peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        print("[sample]", sample(), "\n")
        if losses["val"] < best_val:
            best_val = losses["val"]
            save(os.path.join(args.out_dir, f"{args.size}_best.pt"))
        save_last(it)
    if it == args.max_iters:
        break

    for _ in range(args.grad_accum):
        x, y = get_batch(train_data)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        (loss / args.grad_accum).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    opt.step()
    opt.zero_grad(set_to_none=True)

    if it % 20 == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {it:6d} | loss {loss.item():.4f} | lr {lr_at(it):.2e} | gnorm {norm:.2f} "
              f"| {dt * 1000 / (20 if it else 1):.0f} ms/iter", flush=True)

save(os.path.join(args.out_dir, f"{args.size}_final.pt"))
print("done")
