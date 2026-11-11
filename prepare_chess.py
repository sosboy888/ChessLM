"""Turn a Lichess PGN dump (.pgn.zst) into fixed-length character-level game records.

Each game becomes one row of (block_size + 1) uint8 ids:  ";1.e4 e5 2.Nf3 ..." then zero-padding.
Games are never split or concatenated, so the model always sees a game from its first move.

    python prepare_chess.py data/raw/lichess_db_standard_rated_2013-01.pgn.zst --min_elo 0
    python prepare_chess.py <dump> --min_elo 1800 --max_games 2000000
"""
import argparse
import io
import json
import os
import re

import numpy as np
import zstandard

from chess_tokenizer import PAD, VOCAB_SIZE, encode

HEADER = re.compile(r'\[(\w+) "(.*)"\]')
COMMENT = re.compile(r"\{[^}]*\}")        # {[%clk 0:03:00]} style annotations
NAG = re.compile(r"\$\d+")
MOVE_NUM = re.compile(r"\d+\.(?:\.\.)?")  # "12." or "12..."
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def iter_games(path):
    """Yield (headers, movetext_line) for every game in a PGN file (plain or .zst)."""
    def lines():
        if path.endswith(".zst"):
            with open(path, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as rd:
                yield from io.TextIOWrapper(rd, encoding="utf-8", errors="replace")
        else:
            with open(path, encoding="utf-8", errors="replace") as fh:
                yield from fh

    headers = {}
    for line in lines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            m = HEADER.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
        else:  # the movetext is a single line in Lichess dumps
            yield headers, line
            headers = {}


def all_games(paths):
    for p in paths:
        print(f"\n== {p}", flush=True)
        yield from iter_games(p)


def clean_moves(movetext):
    """Strip comments/NAGs/results and renumber compactly -> list of SAN moves."""
    t = COMMENT.sub(" ", movetext)
    t = NAG.sub(" ", t)
    t = MOVE_NUM.sub(" ", t)
    t = t.replace("?", "").replace("!", "")
    return [tok for tok in t.split() if tok not in RESULTS]


def format_game(moves):
    parts = [f"{i // 2 + 1}.{m}" if i % 2 == 0 else m for i, m in enumerate(moves)]
    return ";" + " ".join(parts)


def keep_game(h, args):
    try:
        if min(int(h["WhiteElo"]), int(h["BlackElo"])) < args.min_elo:
            return False
    except (KeyError, ValueError):
        return False
    if h.get("Termination") != "Normal":
        return False
    if not args.keep_bullet and "Bullet" in h.get("Event", ""):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="+", help="one or more Lichess .pgn.zst (or plain .pgn) files")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--block_size", type=int, default=1024, help="max characters per game")
    ap.add_argument("--min_elo", type=int, default=1800, help="both players must be at least this rated")
    ap.add_argument("--min_plies", type=int, default=10)
    ap.add_argument("--max_games", type=int, default=0, help="stop after keeping this many (0 = all)")
    ap.add_argument("--val_every", type=int, default=200, help="every Nth kept game goes to validation")
    ap.add_argument("--keep_bullet", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    L = args.block_size + 1
    files = {n: open(os.path.join(args.out_dir, f"{n}.bin"), "wb") for n in ("train", "val")}
    buf = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    seen = kept = too_long = too_short = 0

    def flush(name):
        if buf[name]:
            files[name].write(np.stack(buf[name]).tobytes())
            buf[name].clear()

    for headers, movetext in all_games(args.dump):
        seen += 1
        if seen % 100_000 == 0:
            print(f"\rread {seen:,} games | kept {kept:,} | too long {too_long:,}", end="", flush=True)
        if not keep_game(headers, args):
            continue
        moves = clean_moves(movetext)
        if len(moves) < args.min_plies:
            too_short += 1
            continue
        text = format_game(moves)
        if len(text) > args.block_size:   # biases the data toward shorter games; see README
            too_long += 1
            continue
        row = np.full(L, PAD, dtype=np.uint8)
        ids = encode(text)
        row[:len(ids)] = ids
        name = "val" if kept % args.val_every == 0 else "train"
        buf[name].append(row)
        counts[name] += 1
        kept += 1
        if len(buf[name]) >= 10_000:
            flush(name)
        if args.max_games and kept >= args.max_games:
            break

    for n in files:
        flush(n)
        files[n].close()
    meta = dict(block_size=args.block_size, row_len=L, vocab_size=VOCAB_SIZE,
                n_train=counts["train"], n_val=counts["val"])
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nread {seen:,} | kept {kept:,} (train {counts['train']:,}, val {counts['val']:,}) "
          f"| too long {too_long:,} | too short {too_short:,}")


if __name__ == "__main__":
    main()
