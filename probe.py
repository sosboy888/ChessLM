"""Board-state probing: does the model represent the chess board internally?

The model only ever sees game text like ";1.e4 e5 2.Nf3". If a simple readout of its hidden activations can recover
which piece sits on each of the 64 squares, then the model has built something like a board inside itself.

Method
  * Take held-out games (never seen in training). Before every move, record the residual-stream activation at the
    character just ahead of it (the "." after a move number, or the space after White's move), at every layer
    (layer 0 = token embedding, layer k = after block k). That is the point where the model has to know the position
    in order to choose its next move.
  * Label each position with the true board: for each of the 64 squares, one of 13 classes (empty, 6 white pieces,
    6 black pieces).
  * Fit one probe per layer (activation -> 64 x 13 logits) on one set of games and test it on different games.
    Two probe types: "linear" (a single linear layer) and "mlp" (one hidden layer of 512 units).
  * Control: repeat everything on a randomly initialised copy of the same network. A random network still carries
    simple statistics of the text it has read (which characters appeared so far), so this tells us how much of the
    board a probe can recover without any learned chess knowledge.
  * Baseline: for each square, always predict its most common class.

    python probe.py                          # 1500 train games, 300 test games
    python probe.py --n_train 60 --n_test 20 --epochs 2     # quick check
"""
import argparse
import json
import math
import os
import random
import re
import time

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_tokenizer import decode
from model import GPT
from play import Engine, find_ckpt

HERE = os.path.dirname(os.path.abspath(__file__))
N_CLASSES = 13   # 0 empty | 1-6 white P N B R Q K | 7-12 black p n b r q k
KINDS = ("linear", "mlp")


def board_labels(board):
    out = np.zeros(64, dtype=np.int8)                  # square 0 = a1, 7 = h1, 8 = a2, ..., 63 = h8
    for sq, piece in board.piece_map().items():
        out[sq] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
    return out


@torch.no_grad()
def extract(model, rows, device, batch=16):
    """Residual-stream activations just before every move, for every layer."""
    n_layers = len(model.blocks) + 1
    cap = {}
    hooks = [model.tok_emb.register_forward_hook(lambda m, i, o: cap.__setitem__(0, o))]
    for l, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l + 1, o)))

    feats, labels, plies, games = [], [], [], []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        positions = []
        for b, row in enumerate(chunk):
            text = decode(row)
            starts, sans, off = [], [], 0
            for tok in text.split(" "):                       # tokens look like ";1.e4", "e5", "2.Nf3"
                m = re.match(r"^;?(?:\d+\.)?", tok)
                starts.append(off + m.end())                  # index of the move's first character
                sans.append(tok[m.end():])
                off += len(tok) + 1
            board, pos = chess.Board(), []
            board.push_san(sans[0])
            for p in range(1, len(sans)):
                # probe the character just before move p+1 starts: the "." after a move number, or the space
                # after White's move. The model must already know the position here to choose its next move.
                pos.append(starts[p] - 1)
                labels.append(board_labels(board))            # board after p plies
                plies.append(p)
                games.append(start + b)
                board.push_san(sans[p])
            positions.append(pos)
        x = torch.from_numpy(chunk[:, :-1].astype(np.int64)).to(device)
        model(x)
        for b, pos in enumerate(positions):
            if pos:
                idx = torch.tensor(pos, device=device)
                feats.append(torch.stack([cap[l][b, idx] for l in range(n_layers)], dim=1).half().cpu())
    for h in hooks:
        h.remove()
    return (torch.cat(feats), torch.from_numpy(np.stack(labels)), np.array(plies), np.array(games))


def fit_probe(kind, Xtr, Ytr, Xte, epochs, device, seed=0, bs=1024):
    """Activation -> 64 squares x 13 classes. Returns predicted classes for the test set and train accuracy."""
    torch.manual_seed(seed)
    Xtr, Xte = Xtr.to(device).float(), Xte.to(device).float()
    mean, std = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-5
    Xtr, Xte = (Xtr - mean) / std, (Xte - mean) / std
    Ytr = Ytr.to(device).long()
    d_in, d_out = Xtr.size(1), 64 * N_CLASSES
    net = nn.Linear(d_in, d_out) if kind == "linear" else \
        nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(), nn.Linear(512, d_out))
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3 if kind == "linear" else 1e-3, weight_decay=1e-2)
    n = len(Xtr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * math.ceil(n / bs))
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(net(Xtr[idx]).view(-1, N_CLASSES), Ytr[idx].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        train_acc = (net(Xtr[:20000]).view(-1, 64, N_CLASSES).argmax(-1) == Ytr[:20000].view(-1, 64)).float().mean().item()
        return net(Xte).view(-1, 64, N_CLASSES).argmax(-1).cpu(), train_acc


def metrics(pred, Y):
    correct = pred == Y.long()
    occ = Y != 0
    per_sq_occ = [(correct[:, s] & occ[:, s]).sum().item() / max(occ[:, s].sum().item(), 1) for s in range(64)]
    return {"acc_all": correct.float().mean().item(),
            "acc_occ": (correct & occ).sum().item() / occ.sum().item(),
            "exact": correct.all(1).float().mean().item(),
            "per_square_occ": per_sq_occ}


def run_model(name, model, rows_tr, rows_te, args, device):
    t0 = time.time()
    Ftr, Ytr, _, _ = extract(model, rows_tr, device)
    Fte, Yte, pte, gte = extract(model, rows_te, device)
    print(f"[{name}] extracted {len(Ftr):,} train / {len(Fte):,} test positions in {time.time() - t0:.0f}s", flush=True)
    result, best_preds = {}, {}
    for kind in KINDS:
        layers, preds = [], []
        for l in range(Ftr.size(1)):
            pred, train_acc = fit_probe(kind, Ftr[:, l], Ytr, Fte[:, l], args.epochs, device, seed=args.seed)
            m = metrics(pred, Yte)
            m["layer"], m["train_acc_all"] = l, train_acc
            layers.append(m)
            preds.append(pred)
            print(f"[{name}/{kind}] layer {l:2d}: all squares {100 * m['acc_all']:.1f}% | occupied "
                  f"{100 * m['acc_occ']:.1f}% | whole board {100 * m['exact']:.1f}% | (train, all: {100 * train_acc:.1f}%)",
                  flush=True)
        best = max(range(len(layers)), key=lambda i: layers[i]["acc_occ"])
        by_ply = []                                   # accuracy by how far into the game, at the best layer
        for lo in range(0, 80, 10):
            sel = torch.from_numpy((pte > lo) & (pte <= lo + 10))
            if sel.sum() >= 50:
                mm = metrics(preds[best][sel], Yte[sel])
                by_ply.append({"plies": f"{lo + 1}-{lo + 10}", "n": int(sel.sum()),
                               "acc_all": mm["acc_all"], "acc_occ": mm["acc_occ"]})
        result[kind] = {"layers": layers, "best_layer": best, "by_ply": by_ply}
        best_preds[kind] = preds[best]
    return result, (Yte, pte, gte, best_preds, Ytr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data_big"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    eng = Engine(args.ckpt or find_ckpt(), device)
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    allrows = np.fromfile(os.path.join(args.data_dir, "val.bin"), dtype=np.uint8).reshape(-1, meta["row_len"])
    pick = random.Random(args.seed).sample(range(len(allrows)), args.n_train + args.n_test)
    rows_tr, rows_te = allrows[pick[:args.n_train]], allrows[pick[args.n_train:]]     # different games
    print(f"model {eng.name} | {args.n_train} train games, {args.n_test} test games (all held out from LM training)")

    trained, (Yte, pte, gte, best_pred, Ytr) = run_model("trained", eng.model, rows_tr, rows_te, args, device)
    torch.manual_seed(123)
    random_model = GPT(eng.cfg).to(device).eval()
    control, (_, _, _, rand_pred, _) = run_model("random-init", random_model, rows_tr, rows_te, args, device)

    # majority-class baseline: for each square, always guess its most common class in the training positions
    majority = torch.mode(Ytr.long(), dim=0).values
    base = metrics(majority.unsqueeze(0).expand(len(Yte), -1), Yte)

    # one example position for the figure (around move 15, i.e. ply 30), decoded by the linear probes
    # representative, not cherry-picked: among positions at plies 25-35, take the median case of
    # (squares the trained probe gets right) minus (squares the untrained control gets right)
    cand = np.where((pte >= 25) & (pte <= 35))[0]
    diff = ((best_pred["linear"][cand] == Yte[cand].long()).sum(1)
            - (rand_pred["linear"][cand] == Yte[cand].long()).sum(1)).numpy()
    ex = int(cand[np.argsort(diff, kind="stable")[len(cand) // 2]])
    sans = decode(rows_te[gte[ex]]).lstrip(";").split()[:int(pte[ex])]
    example = {"ply": int(pte[ex]), "moves": " ".join(sans), "true": Yte[ex].tolist(),
               "trained_pred": best_pred["linear"][ex].tolist(), "random_pred": rand_pred["linear"][ex].tolist(),
               "trained_layer": trained["linear"]["best_layer"], "random_layer": control["linear"]["best_layer"]}

    out = {"model": eng.name, "n_train_games": args.n_train, "n_test_games": args.n_test,
           "n_test_positions": len(Yte), "epochs": args.epochs, "seed": args.seed,
           "trained": trained, "random_init": control, "majority": base, "example": example}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"probe_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(out, open(path, "w"), indent=2)

    print("\nSummary on held-out games (occupied squares / all squares):")
    for kind in KINDS:
        t, c = trained[kind], control[kind]
        bt, bc = t["layers"][t["best_layer"]], max(c["layers"], key=lambda m: m["acc_occ"])
        print(f"  {kind:6s} trained (layer {t['best_layer']:2d}): {100 * bt['acc_occ']:.1f}% / {100 * bt['acc_all']:.1f}%   "
              f"random-init (layer {bc['layer']:2d}): {100 * bc['acc_occ']:.1f}% / {100 * bc['acc_all']:.1f}%")
    print(f"  majority baseline: {100 * base['acc_occ']:.1f}% / {100 * base['acc_all']:.1f}%")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
