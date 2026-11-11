"""Play against the trained chess model in your browser.

    python play.py                           # picks the newest checkpoint in out/
    python play.py --ckpt out/base_best.pt   # or choose one
    python play.py --port 8001 --no_browser

The model only knows game text like ";1.e4 e5 2.Nf3". To get its move we feed it the game so far and either

  free   : sample characters until a space; if the move is illegal, resample (up to 5 times), then fall back to
           the highest-scoring legal move. Illegal attempts are counted, which shows how well it learned the rules.
  legal  : score every legal move by its likelihood under the model and pick one (temperature controls randomness).

The server only listens on 127.0.0.1.
"""
import argparse
import json
import math
import os
import random
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess
import torch

from chess_tokenizer import decode, encode
from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_ORDER = ["base_final.pt", "base_best.pt", "small_final.pt", "small_best.pt"]
MAX_RETRIES = 5
MAX_MOVE_CHARS = 10


def find_ckpt():
    for name in CKPT_ORDER:
        p = os.path.join(HERE, "out", name)
        if os.path.exists(p):
            return p
    raise SystemExit("No checkpoint found in out/. Train first, or pass --ckpt PATH.")


def build_prompt(san_moves):
    """Game so far in the training format. Ends where the model must write its move."""
    parts = [f"{i // 2 + 1}.{m}" if i % 2 == 0 else m for i, m in enumerate(san_moves)]
    text = ";" + " ".join(parts)
    n = len(san_moves)
    if n % 2 == 0:   # white to move: the next thing is a move number, e.g. "...e5 2."
        text += (" " if parts else "") + f"{n // 2 + 1}."
    else:            # black to move: next is black's move after a space
        text += " "
    return text


class Engine:
    def __init__(self, ckpt_path, device):
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        self.cfg = GPTConfig(**ck["cfg"])
        self.model = GPT(self.cfg).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.device = device
        self.name = os.path.basename(ckpt_path)
        self.params_m = self.model.num_params() / 1e6

    def amp(self):
        return torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda")

    @torch.no_grad()
    def sample(self, prompt, temperature, top_k=10):
        """Write the next move as text: sample characters, cut at the first space."""
        ids = torch.tensor([encode(prompt)], device=self.device)
        with self.amp():
            out = self.model.generate(ids, MAX_MOVE_CHARS, temperature=temperature, top_k=top_k)
        return decode(out[0, ids.size(1):].tolist()).split(" ")[0]

    @torch.no_grad()
    def score(self, prompt, moves):
        """log P(move + ' ' | prompt) for each candidate move string, summed over characters."""
        p = encode(prompt)
        seqs = [p + encode(m + " ") for m in moves]
        L = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), L, dtype=torch.long)       # right-padded with PAD (causal attention ignores it)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
        x = x.to(self.device)
        with self.amp():
            logits, _ = self.model(x[:, :-1])
        logp = torch.log_softmax(logits.float(), dim=-1)
        scores = []
        for i, s in enumerate(seqs):
            pos = torch.arange(len(s) - 1, device=self.device)
            lp = logp[i, pos, x[i, 1:len(s)]]                 # lp[j] = log P(token j+1 | tokens <= j)
            scores.append(lp[len(p) - 1:].sum().item())       # only the characters of the move itself
        return scores


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    z = sum(e)
    return [v / z for v in e]


def fits_context(engine, san):
    """True if the game so far plus one more move still fits in the model's context window."""
    return len(build_prompt(san)) + MAX_MOVE_CHARS + 2 <= engine.cfg.block_size


def choose_move(engine, board, san, mode, temperature, want_candidates=True):
    """Pick the model's move for the side to move.

    mode: 'free'   - sample the move as text, retry if illegal (up to MAX_RETRIES), then fall back to the
                     highest-scoring legal move
          'strict' - one sample; if it is illegal, move is None (the caller decides what that means)
          'legal'  - sample from the model's likelihood over the legal moves
    Returns (move, info); info has candidates (top 5 legal), raw (last sampled text), attempts (illegal
    samples) and fallback (True if the legal-move fallback was used).
    """
    prompt = build_prompt(san)
    legal = list(board.legal_moves)
    sans = [board.san(m) for m in legal]
    cache = {}

    def ranked():      # score every legal move once, only if needed
        if not cache:
            cache["scores"] = engine.score(prompt, sans)
            cache["probs"] = softmax(cache["scores"])
            cache["order"] = sorted(range(len(legal)), key=lambda i: -cache["probs"][i])
        return cache

    info = {"candidates": [], "raw": None, "attempts": [], "fallback": False}
    move = None
    if mode in ("free", "strict"):
        for _ in range(MAX_RETRIES if mode == "free" else 1):
            info["raw"] = engine.sample(prompt, temperature)
            try:
                move = board.parse_san(info["raw"])
                break
            except ValueError:
                info["attempts"].append(info["raw"] or "(nothing)")
        if move is None and mode == "free":
            move, info["fallback"] = legal[ranked()["order"][0]], True
    else:
        r = ranked()
        if temperature <= 0.05:
            idx = r["order"][0]
        else:
            m = max(r["scores"])
            w = [math.exp((s - m) / temperature) for s in r["scores"]]
            idx = random.choices(range(len(legal)), weights=w)[0]
        move = legal[idx]

    if want_candidates:
        r = ranked()
        info["candidates"] = [{"san": sans[i], "p": r["probs"][i]} for i in r["order"][:5]]
    return move, info


class Game:
    def __init__(self, engine):
        self.engine = engine
        self.mode, self.temperature = "free", 0.5
        self.new("w")

    def new(self, color):
        self.board = chess.Board()
        self.san = []
        self.user = chess.WHITE if color == "w" else chess.BLACK
        self.last = {}
        self.stats = {"turns": 0, "first_try": 0, "illegal": 0, "fallbacks": 0}
        self.context_full = False
        self.model_move_if_needed()

    # ---- moves -----------------------------------------------------------------------------------------
    def push(self, move):
        self.san.append(self.board.san(move))
        self.board.push(move)

    def user_move(self, uci):
        if self.board.is_game_over() or self.board.turn != self.user or self.context_full:
            raise ValueError("It is not your turn.")
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {uci}")
        self.push(move)
        self.last = {}
        self.model_move_if_needed()

    def undo(self):
        popped = 0
        while self.board.move_stack:
            self.board.pop()
            self.san.pop()
            popped += 1
            if self.board.turn == self.user:
                break
        self.last, self.context_full = {}, False
        self.model_move_if_needed()   # e.g. user plays black and we undid back to the start

    def model_move_if_needed(self):
        if self.board.is_game_over() or self.board.turn == self.user:
            return
        if not fits_context(self.engine, self.san):
            self.context_full = True
            self.last = {"note": f"The game is longer than the model's {self.engine.cfg.block_size}-character "
                                 "context, so it cannot move any more."}
            return

        move, info = choose_move(self.engine, self.board, self.san, self.mode, self.temperature)
        attempts, note = info["attempts"], ""
        if self.mode == "free":
            self.stats["turns"] += 1
            self.stats["illegal"] += len(attempts)
            if info["fallback"]:
                self.stats["fallbacks"] += 1
                note = (f"All {MAX_RETRIES} samples were illegal ({', '.join(attempts)}). "
                        "Played its highest-scoring legal move instead.")
            elif attempts:
                note = f"Illegal attempts before a legal move: {', '.join(attempts)}."
            else:
                self.stats["first_try"] += 1

        played = self.board.san(move)
        self.push(move)
        self.last = {"played": played, "candidates": info["candidates"], "raw": info["raw"], "note": note}

    # ---- view ------------------------------------------------------------------------------------------
    def state(self):
        b = self.board
        legal, promos = {}, set()
        my_turn = b.turn == self.user and not b.is_game_over() and not self.context_full
        if my_turn:
            for m in b.legal_moves:
                f, t = chess.square_name(m.from_square), chess.square_name(m.to_square)
                legal.setdefault(f, set()).add(t)
                if m.promotion:
                    promos.add(f + t)
        outcome = b.outcome()
        if outcome:
            if outcome.winner is None:
                text = f"Draw ({outcome.termination.name.replace('_', ' ').lower()})"
            else:
                who = "White" if outcome.winner == chess.WHITE else "Black"
                text = f"Checkmate. {who} wins" if outcome.termination == chess.Termination.CHECKMATE else f"{who} wins"
        else:
            text = ""
        last = b.move_stack[-1] if b.move_stack else None
        return {
            "fen": b.fen(),
            "user": "w" if self.user == chess.WHITE else "b",
            "san": self.san,
            "legal": {k: sorted(v) for k, v in legal.items()},
            "promos": sorted(promos),
            "last_move": [chess.square_name(last.from_square), chess.square_name(last.to_square)] if last else None,
            "check": chess.square_name(b.king(b.turn)) if b.is_check() else None,
            "over": bool(outcome) or self.context_full,
            "status": text,
            "my_turn": my_turn,
            "last": self.last,
            "stats": self.stats,
            "settings": {"mode": self.mode, "temperature": self.temperature},
            "model": {"name": self.engine.name, "params_m": round(self.engine.params_m, 1),
                      "context": self.engine.cfg.block_size, "device": self.engine.device},
            "prompt_chars": len(build_prompt(self.san)),
        }


LOCK = threading.Lock()
GAME = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "play.html"), "rb") as f:   # re-read so edits show on refresh
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/api/state":
            with LOCK:
                self._json(GAME.state())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            with LOCK:
                if self.path == "/api/new":
                    GAME.new("b" if body.get("color") == "b" else "w")
                elif self.path == "/api/move":
                    GAME.user_move(str(body.get("from", "")) + str(body.get("to", "")) + str(body.get("promotion", "")))
                elif self.path == "/api/undo":
                    GAME.undo()
                elif self.path == "/api/settings":
                    if body.get("mode") in ("free", "legal"):
                        GAME.mode = body["mode"]
                    if "temperature" in body:
                        GAME.temperature = min(2.0, max(0.0, float(body["temperature"])))
                else:
                    return self._json({"error": "not found"}, 404)
                self._json(GAME.state())
        except (ValueError, KeyError) as e:
            self._json({"error": str(e)}, 400)


def main():
    global GAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no_browser", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or find_ckpt()
    engine = Engine(ckpt, device)
    GAME = Game(engine)
    print(f"loaded {engine.name} ({engine.params_m:.1f}M params) on {device}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"open {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
