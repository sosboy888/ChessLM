"""Evaluate the chess model.

    python evaluate.py legal                                        # legality + human-move agreement
    python evaluate.py stockfish --opponents skill:0 skill:2 skill:4 --games 20

Each command prints a README-ready markdown table and writes results/<name>_<timestamp>.json (and .md).
The stockfish command also writes a PGN with every game.

legal      Positions are sampled from held-out Lichess games (never seen in training). For each one the model is
           given the real game so far and asked for the next move:
             * legal first try   - is the move it writes legal? (free sampling at --temperature)
             * top-1/3/5         - how often is the move the human actually played among the model's top-k
                                   legal moves (ranked by likelihood)? Random guessing is shown for comparison.
stockfish  Plays the model against Stockfish. Each opening (the first --opening_plies plies of a real held-out
           game) is played twice, once with each colour, to cancel out the opening's bias.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from chess_tokenizer import decode
from play import Engine, build_prompt, choose_move, find_ckpt, fits_context

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKETS = [(0, 10, "opening (plies 1-10)"), (10, 30, "early middlegame (11-30)"),
           (30, 60, "middlegame (31-60)"), (60, 10 ** 6, "late game (61+)")]


# ---------------------------------------------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - w) / d, (c + w) / d


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / max(n, 1):.1f}% ({100 * lo:.1f}-{100 * hi:.1f})"


def elo_from_score(s):
    s = min(max(s, 0.005), 0.995)
    return -400 * math.log10(1 / s - 1)


def load_rows(data_dir):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    return np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint8).reshape(-1, meta["row_len"])


def row_to_sans(row):
    return [re.sub(r"^\d+\.", "", t) for t in decode(row).lstrip(";").split()]


def save(args, name, payload, markdown):
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.out_dir, f"{name}_{stamp}")
    with open(base + ".json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(markdown + "\n")
    return base


# ---------------------------------------------------------------------------------------------------- legal
def cmd_legal(args, eng):
    rows = load_rows(args.data_dir)
    rng = random.Random(args.seed)
    S = {b[2]: dict(n=0, legal=0, top1=0, top3=0, top5=0, rand1=0.0) for b in BUCKETS}
    t0, done = time.time(), 0
    while done < args.n:
        sans = row_to_sans(rows[rng.randrange(len(rows))])
        if len(sans) < 2:
            continue
        k = rng.randrange(len(sans))                     # predict ply k+1 from the real moves before it
        prefix = sans[:k]
        if not fits_context(eng, prefix):                # a few very long games would overflow the context
            continue
        board = chess.Board()
        for s in prefix:
            board.push_san(s)
        human = board.parse_san(sans[k])
        prompt = build_prompt(prefix)

        raw = eng.sample(prompt, args.temperature)
        try:
            board.parse_san(raw)
            legal_ok = True
        except ValueError:
            legal_ok = False

        legal = list(board.legal_moves)
        scores = eng.score(prompt, [board.san(m) for m in legal])
        order = sorted(range(len(legal)), key=lambda j: -scores[j])
        rank = order.index(legal.index(human))

        b = next(x[2] for x in BUCKETS if x[0] <= k < x[1])
        for key in (b, "ALL"):
            d = S.setdefault(key, dict(n=0, legal=0, top1=0, top3=0, top5=0, rand1=0.0))
            d["n"] += 1
            d["legal"] += legal_ok
            d["top1"] += rank < 1
            d["top3"] += rank < 3
            d["top5"] += rank < 5
            d["rand1"] += 1 / len(legal)
        done += 1
        if done % 250 == 0:
            print(f"  {done}/{args.n} positions ({(time.time() - t0) / done * 1000:.0f} ms each)", flush=True)

    lines = [f"Held-out Lichess positions, n={args.n}, free sampling at temperature {args.temperature}. "
             "95% Wilson intervals in brackets.", "",
             "| Game phase | Positions | Legal first try | Top-1 = human move | Top-3 | Top-5 | Random top-1 |",
             "|---|---|---|---|---|---|---|"]
    for name in [b[2] for b in BUCKETS] + ["ALL"]:
        d = S[name]
        if d["n"] == 0:
            continue
        lines.append(f"| {'**all**' if name == 'ALL' else name} | {d['n']} | {fmt(d['legal'], d['n'])} | "
                     f"{fmt(d['top1'], d['n'])} | {100 * d['top3'] / d['n']:.1f}% | {100 * d['top5'] / d['n']:.1f}% | "
                     f"{100 * d['rand1'] / d['n']:.1f}% |")
    md = "\n".join(lines)
    print("\n" + md)
    base = save(args, "legal", {"model": eng.name, "n": args.n, "temperature": args.temperature,
                                "seed": args.seed, "stats": S}, md)
    print(f"\nsaved {base}.json / .md")


# ---------------------------------------------------------------------------------------------------- stockfish
def find_stockfish(path):
    if path:
        return path
    hits = glob.glob(os.path.join(HERE, "tools", "**", "stockfish*.exe"), recursive=True)
    return hits[0] if hits else shutil.which("stockfish")


def parse_opponent(spec):
    if spec == "random":
        return {"name": "Random mover", "options": None, "elo": None}
    kind, val = spec.split(":")
    val = int(val)
    if kind == "skill":
        return {"name": f"Stockfish skill {val}", "options": {"Skill Level": val}, "elo": None}
    if kind == "elo":
        return {"name": f"Stockfish ~{val} Elo", "options": {"UCI_LimitStrength": True, "UCI_Elo": val}, "elo": val}
    raise SystemExit(f"bad opponent '{spec}': use random, skill:N (0-20) or elo:N (e.g. elo:1500)")


def adjudicate(sf, board, threshold=300):
    if sf is None:                                        # random-mover opponent: no engine to judge the position
        return "1/2-1/2"
    cp = sf.analyse(board, chess.engine.Limit(depth=14))["score"].white().score(mate_score=10000)
    return "1-0" if cp >= threshold else "0-1" if cp <= -threshold else "1/2-1/2"


def play_game(eng, sf, limit, model_white, opening, args):
    board, san = chess.Board(), []
    for s in opening:
        mv = board.parse_san(s)
        san.append(board.san(mv))
        board.push(mv)
    moves = illegal_first = illegal_samples = fallbacks = 0
    result = term = None
    adjudicated = False
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome:
            result, term = outcome.result(), outcome.termination.name.lower().replace("_", " ")
            break
        if (board.turn == chess.WHITE) == model_white:
            if not fits_context(eng, san):
                result, term, adjudicated = adjudicate(sf, board), "adjudicated (model context full)", True
                break
            move, info = choose_move(eng, board, san, args.policy, args.temperature, want_candidates=False)
            moves += 1
            illegal_first += bool(info["attempts"])
            illegal_samples += len(info["attempts"])
            fallbacks += info["fallback"]
            if move is None:                              # strict policy: an illegal move forfeits the game
                result, term = ("0-1" if model_white else "1-0"), "forfeit (illegal move)"
                break
        else:
            move = random.choice(list(board.legal_moves)) if sf is None else sf.play(board, limit).move
        san.append(board.san(move))
        board.push(move)
    won = (result == "1-0") == model_white if result in ("1-0", "0-1") else None
    return dict(result=result, outcome="draw" if won is None else "win" if won else "loss", termination=term,
                plies=len(san), model_moves=moves, illegal_first=illegal_first, illegal_samples=illegal_samples,
                fallbacks=int(fallbacks), adjudicated=adjudicated, board=board, model_white=model_white)


def cmd_stockfish(args, eng):
    path = find_stockfish(args.stockfish)
    if not path:
        raise SystemExit("Stockfish not found. Pass --stockfish PATH or put it under tools/.")
    rows = load_rows(args.data_dir)
    rng = random.Random(args.seed)
    limit = chess.engine.Limit(time=args.movetime)
    pairs = max(1, args.games // 2)
    openings = []
    while len(openings) < pairs:                          # real openings from held-out games
        sans = row_to_sans(rows[rng.randrange(len(rows))])
        if len(sans) > args.opening_plies + 2:
            openings.append(sans[:args.opening_plies])

    pgn_path = os.path.join(args.out_dir, f"stockfish_{time.strftime('%Y%m%d_%H%M%S')}.pgn")
    os.makedirs(args.out_dir, exist_ok=True)
    summary, rows_md = [], []
    base_path = os.path.join(args.out_dir, f"stockfish_summary_{time.strftime('%Y%m%d_%H%M%S')}")

    def write_summary():
        """Write the table so far. Called after every opponent, so a crash never loses finished results."""
        header = [f"Model `{eng.name}` vs Stockfish, {args.movetime}s per move, policy `{args.policy}`, "
                  f"temperature {args.temperature}, {2 * pairs} games per opponent (each opening played with both "
                  "colours). Elo difference is the model's, relative to the opponent, from the score (95% interval); "
                  "'Est. Elo' is only given for `elo:` opponents and is on Stockfish's own scale, not Lichess or FIDE.",
                  "", "| Opponent | Games | W / D / L | Score | Elo diff | Est. Elo | Legal first try | Adjudicated |",
                  "|---|---|---|---|---|---|---|---|"]
        md = "\n".join(header + rows_md)
        with open(base_path + ".json", "w") as f:
            json.dump({"model": eng.name, "stockfish": path, "movetime": args.movetime, "policy": args.policy,
                       "temperature": args.temperature, "opponents": summary}, f, indent=2)
        with open(base_path + ".md", "w", encoding="utf-8") as f:
            f.write(md + "\n")
        return md

    sf = chess.engine.SimpleEngine.popen_uci(path)
    try:
        for spec in args.opponents:
            opp = parse_opponent(spec)
            if opp["options"] is not None:
                try:
                    sf.configure({"Threads": 1, "Hash": 16, **opp["options"]})
                except chess.engine.EngineError as e:
                    raise SystemExit(f"Stockfish rejected {spec}: {e}")
            res = []
            for i, opening in enumerate(openings):
                for model_white in (True, False):
                    g = play_game(eng, None if opp["options"] is None else sf, limit, model_white, opening, args)
                    res.append(g)
                    print(f"[{spec}] game {len(res)}/{2 * pairs} model as {'White' if model_white else 'Black'}: "
                          f"{g['outcome']} ({g['termination']}, {g['plies']} plies, "
                          f"{g['illegal_first']}/{g['model_moves']} illegal first guesses)", flush=True)
                    pg = chess.pgn.Game.from_board(g["board"])
                    pg.headers.update({"Event": "chessLM evaluation", "Round": str(len(res)),
                                       "White": eng.name if model_white else opp["name"],
                                       "Black": opp["name"] if model_white else eng.name,
                                       "Result": g["result"], "Termination": g["termination"],
                                       "Opening": " ".join(opening)})
                    with open(pgn_path, "a", encoding="utf-8") as f:
                        print(pg, file=f, end="\n\n")

            n = len(res)
            w, d, l = (sum(g["outcome"] == k for g in res) for k in ("win", "draw", "loss"))
            score = (w + 0.5 * d) / n
            lo, hi = wilson(w + 0.5 * d, n)
            mm = sum(g["model_moves"] for g in res)
            fl = sum(g["illegal_first"] for g in res)
            elo = f"{elo_from_score(score):+.0f} ({elo_from_score(lo):+.0f} to {elo_from_score(hi):+.0f})"
            est = f"{opp['elo'] + elo_from_score(score):.0f}" if opp["elo"] else "n/a"
            rows_md.append(f"| {opp['name']} | {n} | {w} / {d} / {l} | {100 * score:.0f}% | {elo} | {est} | "
                           f"{100 * (mm - fl) / max(mm, 1):.1f}% | {sum(g['adjudicated'] for g in res)} |")
            summary.append(dict(opponent=opp["name"], spec=spec, games=n, wins=w, draws=d, losses=l, score=score,
                                legal_first_try=(mm - fl) / max(mm, 1), model_moves=mm, illegal_first=fl,
                                fallbacks=sum(g["fallbacks"] for g in res),
                                adjudicated=sum(g["adjudicated"] for g in res),
                                mean_plies=sum(g["plies"] for g in res) / n))
            write_summary()
    finally:
        sf.quit()

    print("\n" + write_summary())
    print(f"\nsaved {base_path}.json / .md and {pgn_path}")


# ---------------------------------------------------------------------------------------------------- main
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", default=None)
    common.add_argument("--data_dir", default=os.path.join(HERE, "data_big"))
    common.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    common.add_argument("--temperature", type=float, default=0.5)
    common.add_argument("--seed", type=int, default=0)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("legal", parents=[common], help="legality and human-move agreement on held-out positions")
    p.add_argument("--n", type=int, default=2000)
    p = sub.add_parser("stockfish", parents=[common], help="play against Stockfish")
    p.add_argument("--stockfish", default=None, help="path to the stockfish executable (auto-detected in tools/)")
    p.add_argument("--opponents", nargs="+", default=["skill:0", "skill:2", "skill:4"],
                   help="random, skill:N (Skill Level 0-20) or elo:N (UCI_Elo, minimum about 1320)")
    p.add_argument("--games", type=int, default=20, help="games per opponent (rounded down to an even number)")
    p.add_argument("--movetime", type=float, default=0.1, help="seconds per Stockfish move")
    p.add_argument("--opening_plies", type=int, default=4, help="plies copied from a real game (use an even number)")
    p.add_argument("--policy", choices=["free", "strict"], default="free",
                   help="free: retry illegal moves then fall back to the best legal move; strict: an illegal move loses")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    eng = Engine(args.ckpt or find_ckpt(), "cuda" if torch.cuda.is_available() else "cpu")
    print(f"model {eng.name} ({eng.params_m:.1f}M) on {eng.device}")
    {"legal": cmd_legal, "stockfish": cmd_stockfish}[args.cmd](args, eng)


if __name__ == "__main__":
    main()
