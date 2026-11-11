"""Make the README figures from the training log and the results/ folder.

    python make_figures.py          # writes figures/*.png; skips any figure whose data is not there yet

Colours follow a validated categorical palette (blue and orange pass the colour-blind and contrast checks); text is
always in ink colours, never in a series colour. Every figure has a legend or direct labels, hairline grids, and no
dual axes.
"""
import glob
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, DRAW = "#2a78d6", "#eb6834", "#c3c2b7"
CRITICAL = "#d03b3b"
SEQ = LinearSegmentedColormap.from_list("blue_seq", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
})


def latest(pattern):
    hits = sorted(glob.glob(os.path.join(HERE, "results", pattern)))
    return hits[-1] if hits else None


def load(pattern):
    p = latest(pattern)
    return json.load(open(p)) if p else None


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - w) / d, (c + w) / d


def titles(fig, claim, sub, y=0.965):
    fig.text(0.015, y, claim, fontsize=15, fontweight="semibold", color=INK, ha="left", va="top")
    fig.text(0.015, y - 0.075, sub, fontsize=10.5, color=INK2, ha="left", va="top")


def dot(ax, x, y, color, ms=9, **kw):
    ax.plot(x, y, "o", ms=ms, mfc=color, mec=SURFACE, mew=2, zorder=4, **kw)   # 2px surface ring


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------------------------------------ training loss
def fig_loss():
    log = os.path.join(HERE, "train_base.log")
    if not os.path.exists(log):
        return print("skip loss curve (no train_base.log)")
    raw = open(log, "rb").read()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
    pts = [(int(a), float(b), float(c)) for a, b, c in
           re.findall(r"\[eval\] iter (\d+): train ([\d.]+) \| val ([\d.]+)", text)]
    if not pts:
        return print("skip loss curve (no eval lines in log)")
    it, tr, va = (np.array(v) for v in zip(*pts))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.76)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.plot(it, va, color=ORANGE, lw=2, solid_capstyle="round", label="Validation (held-out games)")
    ax.plot(it, tr, color=BLUE, lw=2, solid_capstyle="round", label="Training")
    dot(ax, it[-1], tr[-1], BLUE)
    dot(ax, it[-1], va[-1], ORANGE)
    ax.set_yscale("log")
    ax.set_yticks([0.3, 0.5, 1, 2, 4])
    ax.set_yticklabels(["0.3", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlim(-800, it[-1] + 1500)
    ax.set_xticks(range(0, int(it[-1]) + 1, 10000))
    ax.set_xticklabels([f"{v // 1000}k" if v else "0" for v in range(0, int(it[-1]) + 1, 10000)])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (nats per character, log scale)")
    ax.annotate(f"final: train {tr[-1]:.3f}, val {va[-1]:.3f}", xy=(it[-1], va[-1]), xytext=(it[-1] - 1500, 1.1),
                ha="right", va="center", color=INK2, fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=1))
    ax.legend(loc="upper right", frameon=False, labelcolor=INK2)
    titles(fig, "Training loss falls steadily and validation tracks it",
           "54.6M-parameter model, 40,000 steps on 2.57M Lichess games. No sign of overfitting.")
    save(fig, "loss_curve.png")


# ------------------------------------------------------------------------------------------------ legality / human match
PHASES = ["opening (plies 1-10)", "early middlegame (11-30)", "middlegame (31-60)", "late game (61+)"]


def fig_phases():
    d = load("legal_*.json")
    if not d:
        return print("skip phases (no legal_*.json)")
    S = d["stats"]
    names = PHASES + ["ALL"]
    labels = ["Opening\n(plies 1-10)", "Early middlegame\n(11-30)", "Middlegame\n(31-60)", "Late game\n(61+)", "All positions"]
    ys = [4.5, 3.5, 2.5, 1.5, 0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.17, top=0.72, wspace=0.08)
    for ax, key, xlim, ticks, xl in (
            (axes[0], "legal", (84, 101), [85, 90, 95, 100], "Move is legal on the first try (%)"),
            (axes[1], "top1", (0, 75), [0, 20, 40, 60], "Top move = the move a human played (%)")):
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        for nme, y in zip(names, ys):
            s = S[nme]
            k, n = s[key], s["n"]
            lo, hi = wilson(k, n)
            ax.errorbar(100 * k / n, y, xerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]], fmt="none",
                        ecolor=MUTED, elinewidth=1.4, capsize=3, zorder=3)
            dot(ax, 100 * k / n, y, BLUE)
            if key == "top1":
                ax.plot(100 * s["rand1"] / n, y, "o", ms=8, mfc=SURFACE, mec=MUTED, mew=1.6, zorder=4)
            if nme == "ALL":
                ax.text(100 * hi + (xlim[1] - xlim[0]) * 0.015, y, f"{100 * k / n:.1f}%", ha="left", va="center",
                        fontsize=11, fontweight="semibold", color=INK)
        ax.axhline(0.75, color=GRID, lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, 5.1)
        ax.set_xticks(ticks)
        ax.set_xlabel(xl)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels)
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=BLUE, mec=SURFACE, mew=2, label="chessLM (bars: 95% interval)"),
               Line2D([], [], marker="o", ls="", ms=8, mfc=SURFACE, mec=MUTED, mew=1.6, label="Random guess among legal moves")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.15, 0.80), ncol=2, frameon=False, labelcolor=INK2)
    titles(fig, "It follows the rules and often picks the move a human would",
           f"{d['n']:,} positions from held-out Lichess games. Legality dips later in the game; "
           "human-move agreement is about 9x random guessing overall.")
    save(fig, "legal_and_human_match.png")


# ------------------------------------------------------------------------------------------------ Stockfish
def fig_stockfish():
    d = load("stockfish_summary_*.json")
    if not d or not d.get("opponents"):
        return print("skip stockfish (no stockfish_summary_*.json)")
    rows = d["opponents"]
    fig, ax = plt.subplots(figsize=(10, 1.9 + 0.8 * len(rows)))
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.17, top=1 - (1.9 / (1.9 + 0.8 * len(rows))) * 0.62)
    ax.set_axisbelow(True)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        left = 0.0
        for key, color, tc in (("wins", BLUE, "#ffffff"), ("draws", DRAW, INK), ("losses", ORANGE, "#ffffff")):
            w = 100 * r[key] / r["games"]
            ax.barh(y, w, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2)
            if w >= 6:       # a label only where the number fits with padding
                ax.text(left + w / 2, y, str(r[key]), ha="center", va="center", color=tc, fontsize=11, fontweight="semibold")
            left += w
        ax.text(102, y, f"score {100 * r['score']:.0f}%", ha="left", va="center", color=INK2, fontsize=10.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["opponent"] for r in rows][::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of games (numbers inside the bars are game counts)")
    ax.spines["left"].set_visible(False)
    handles = [Patch(color=BLUE, label="chessLM wins"), Patch(color=DRAW, label="Draw"), Patch(color=ORANGE, label="chessLM loses")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 0.80), ncol=3, frameon=False, labelcolor=INK2)
    g = rows[0]["games"]
    titles(fig, "Clearly better than random, clearly weaker than Stockfish",
           f"{g} games per opponent, {d['movetime']}s per Stockfish move, each opening played with both colours. "
           "Stockfish cannot be set weaker than ~1320 Elo.")
    save(fig, "vs_stockfish.png")


# ------------------------------------------------------------------------------------------------ probing
def fig_probe_layers():
    d = load("probe_*.json")
    if not d:
        return print("skip probe figures (no probe_*.json)")
    base = 100 * d["majority"]["acc_occ"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.1), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.64, wspace=0.07)
    for ax, kind, ttl in ((axes[0], "linear", "Linear probe"), (axes[1], "mlp", "Small non-linear (MLP) probe")):
        t = [100 * m["acc_occ"] for m in d["trained"][kind]["layers"]]
        r = [100 * m["acc_occ"] for m in d["random_init"][kind]["layers"]]
        x = np.arange(len(t))
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.axhline(base, color=MUTED, lw=1.4)
        ax.plot(x, r, color=ORANGE, lw=2, solid_capstyle="round")
        ax.plot(x, t, color=BLUE, lw=2, solid_capstyle="round")
        b = int(np.argmax(t))
        dot(ax, b, t[b], BLUE)
        ax.annotate(f"{t[b]:.1f}%  (layer {b})", xy=(b, t[b]), xytext=(b + 0.3, t[b] + 9), ha="right", color=INK,
                    fontsize=10.5, fontweight="semibold")
        ax.set_title(ttl, loc="left", fontsize=11, color=INK2, pad=8)
        ax.set_xlim(-0.3, len(t) - 0.7)
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xlabel("Layer (0 = embedding, 12 = last block)")
    axes[0].set_ylabel("Occupied squares decoded correctly (%)")
    handles = [Line2D([], [], color=BLUE, lw=2, label="Trained model"),
               Line2D([], [], color=ORANGE, lw=2, label="Untrained network (control)"),
               Line2D([], [], color=MUTED, lw=1.4, label="Always guess each square's most common piece")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.805), ncol=3, frameon=False, labelcolor=INK2)
    titles(fig, "The model carries some board information, but not a clean board",
           f"Probes decode the piece on each of 64 squares from hidden activations. {d['n_test_positions']:,} positions "
           f"from {d['n_test_games']} held-out games.")
    save(fig, "probe_by_layer.png")


def fig_probe_heatmap():
    d = load("probe_*.json")
    if not d:
        return
    best = d["trained"]["linear"]["best_layer"]
    maps = [("Trained model, layer %d" % best, d["trained"]["linear"]["layers"][best]["per_square_occ"]),
            ("Untrained network, layer %d" % d["random_init"]["linear"]["best_layer"],
             d["random_init"]["linear"]["layers"][d["random_init"]["linear"]["best_layer"]]["per_square_occ"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.5))
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.17, top=0.78, wspace=0.12)
    im = None
    for ax, (ttl, vals) in zip(axes, maps):
        A = np.zeros((8, 8))
        for s, v in enumerate(vals):
            A[7 - s // 8, s % 8] = v
        im = ax.imshow(A, cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(range(8))
        ax.set_xticklabels(list("abcdefgh"))
        ax.set_yticks(range(8))
        ax.set_yticklabels([str(8 - i) for i in range(8)])
        ax.tick_params(length=0)
        ax.set_title(ttl, loc="left", fontsize=11, color=INK2, pad=8)
        for k in range(9):          # 2px surface gaps between cells
            ax.axhline(k - 0.5, color=SURFACE, lw=1.5)
            ax.axvline(k - 0.5, color=SURFACE, lw=1.5)
    cax = fig.add_axes([0.28, 0.08, 0.44, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelcolor=INK2)
    cb.set_label("Share of times a piece on this square was identified correctly", color=INK2, fontsize=10)
    titles(fig, "The trained model's advantage is in the middle of the board",
           "Linear probe accuracy per square, only when it holds a piece (White at the bottom).")
    save(fig, "probe_by_square.png")


def fig_probe_by_ply():
    d = load("probe_*.json")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.64, wspace=0.07)
    for ax, kind, ttl in ((axes[0], "linear", "Linear probe"), (axes[1], "mlp", "Small non-linear (MLP) probe")):
        t, r = d["trained"][kind]["by_ply"], d["random_init"][kind]["by_ply"]
        x = np.arange(len(t))
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.plot(x, [100 * v["acc_occ"] for v in r], color=ORANGE, lw=2, solid_capstyle="round")
        ax.plot(x, [100 * v["acc_occ"] for v in t], color=BLUE, lw=2, solid_capstyle="round")
        ax.set_title(ttl, loc="left", fontsize=11, color=INK2, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([v["plies"] for v in t])
        ax.set_ylim(0, 100)
        ax.set_xlabel("How far into the game (plies played)")
    axes[0].set_ylabel("Occupied squares decoded correctly (%)")
    handles = [Line2D([], [], color=BLUE, lw=2, label="Trained model (best layer)"),
               Line2D([], [], color=ORANGE, lw=2, label="Untrained network (control)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.805), ncol=2, frameon=False, labelcolor=INK2)
    titles(fig, "Board readout gets harder as the game goes on",
           "Same probes as before, split by how many moves had been played.")
    save(fig, "probe_by_move_number.png")


GLYPH = {1: "♟", 2: "♞", 3: "♝", 4: "♜", 5: "♛", 6: "♚"}


def draw_board(ax, labels, truth=None):
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    wrong = 0
    for s in range(64):
        f, r = s % 8, s // 8
        ax.add_patch(Rectangle((f, r), 1, 1, color="#f0efec" if (f + r) % 2 else "#c3c2b7", lw=0))
        c = labels[s]
        if c:
            white = c <= 6
            ax.text(f + 0.5, r + 0.47, GLYPH[c if white else c - 6], ha="center", va="center", fontsize=21,
                    family="DejaVu Sans", color="#ffffff" if white else "#151515",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="#151515" if white else "#fcfcfb")])
        if truth is not None and labels[s] != truth[s]:
            wrong += 1
            ax.add_patch(Rectangle((f + 0.06, r + 0.06), 0.88, 0.88, fill=False, ec=CRITICAL, lw=2.4))
    return wrong


def fig_probe_example():
    d = load("probe_*.json")
    if not d:
        return
    ex = d["example"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.9))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.72, wspace=0.06)
    draw_board(axes[0], ex["true"])
    w1 = draw_board(axes[1], ex["trained_pred"], ex["true"])
    w2 = draw_board(axes[2], ex["random_pred"], ex["true"])
    axes[0].set_title("True position", loc="left", fontsize=11, color=INK2, pad=8)
    axes[1].set_title(f"Trained model, layer {ex['trained_layer']}: {64 - w1}/64 squares right", loc="left",
                      fontsize=11, color=INK2, pad=8)
    axes[2].set_title(f"Untrained network, layer {ex['random_layer']}: {64 - w2}/64 right", loc="left",
                      fontsize=11, color=INK2, pad=8)
    titles(fig, "One position, as seen by the probes",
           f"After {ex['ply']} plies of a held-out game: the median case around move 15, not the best or worst. "
           "Red outlines mark wrong squares.")
    save(fig, "probe_example.png")


if __name__ == "__main__":
    fig_loss()
    fig_phases()
    fig_stockfish()
    fig_probe_layers()
    fig_probe_heatmap()
    fig_probe_by_ply()
    fig_probe_example()
