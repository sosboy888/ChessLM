"""Sanity check: replay prepared games with python-chess. Every game should be 100% legal.

    python verify_data.py            # checks the first 2000 validation games
"""
import argparse
import json
import os
import re

import chess
import numpy as np

from chess_tokenizer import decode

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="data")
ap.add_argument("--split", default="val")
ap.add_argument("--n", type=int, default=2000)
args = ap.parse_args()

meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
rows = np.fromfile(os.path.join(args.data_dir, f"{args.split}.bin"), dtype=np.uint8).reshape(-1, meta["row_len"])
print(f"{args.split}: {len(rows):,} games, {meta['row_len'] - 1} chars max per game")

ok = 0
lengths = []
for i, row in enumerate(rows[:args.n]):
    text = decode(row)
    lengths.append(len(text))
    board = chess.Board()
    try:
        for tok in text.lstrip(";").split():
            board.push_san(re.sub(r"^\d+\.", "", tok))
        ok += 1
    except Exception as e:
        if i < 5 or ok == i:
            print(f"game {i} failed at {tok!r}: {e}\n  {text[:120]}")
n = min(args.n, len(rows))
print(f"legal replays: {ok}/{n}  | mean length {np.mean(lengths):.0f} chars")
print("example:", decode(rows[0])[:200])
