# v2 plan: does more compute sharpen the board representation?

**Question.** v1 plays legal, human-looking chess but is weak and carries only a partial board representation
(small probe: 76.9% of occupied squares, 15.1% of whole boards right). Does training on more data for more steps
improve that, and the play with it? This is a small scaling study, so the comparison has to be clean.

**Default design (decide the open questions below before starting).** Same 54.6M model and same pipeline. Change
only the compute: about 2x the data (four more Lichess months, 2016-05 to 2016-08) and 3x the steps (120,000 instead
of 40,000). 120,000 steps x 32 games = 3.84M games, so about 0.77 epoch over roughly 5M games: no repeated data.

## Decide the success criteria now, before seeing any v2 results

Primary (the question being asked):
- [ ] Small-probe accuracy on occupied squares, and whole-board-exactly-right rate (v1: 76.9% and 15.1%)

Secondary:
- [ ] Legal first try, overall and in the late game (v1: 97.0% and 92.9%)
- [ ] Top-1 agreement with the human move (v1: 46.9%)
- [ ] Score against Stockfish skill 0 and ~1320 Elo (v1: 12% and 8%)
- [ ] Validation loss, as a sanity check only (v1: 0.332)

An improvement counts when the difference is larger than the 95% intervals. Report the result either way, including
"no meaningful change". The Stockfish comparison needs more games than v1 had (40 per opponent gives very wide
intervals), so re-run v1 and run v2 with 100 games per opponent.

## Keep the comparison fair

- [ ] **Same held-out games for both models.** Keep v1's `data_big/val.bin` as the shared test set. Build v2's training
  data as v1's training games plus *all* kept games from the four new months, so no v1 test game can leak into v2.
  (`prepare_chess.py` needs a small `--no_val` option so every game from the new months goes to training.)
- [ ] **Same architecture, tokenizer, context length, optimiser settings and schedule shape** (peak LR 6e-4, cosine to
  10%). Only the number of steps and the amount of data change.
- [ ] **Same evaluation protocol**, frozen from v1: `legal --n 3000` (seed 0, temperature 0.5); `stockfish` with
  random, skill:0 and elo:1320, 0.1 s per move, 4 opening plies; `probe --n_train 1500 --n_test 300 --epochs 20 --seed 2`.
- [ ] **Keep v1 untouched:** v2 trains into its own folder (`out_v2/`), never `--resume`s from v1.
- [ ] Tag the current commit `v1` so its exact code and results stay reproducible.

## Build

- [ ] Download the four months and check disk space first (about 4.5 GB raw plus about 7 GB of prepared data, 15 GB with
  checkpoints; 57 GB were free)
- [ ] `prepare_chess.py` on the new months only (`--min_elo 1800`, new `--no_val`), then concatenate onto v1's training
  rows into `data_v2/train.bin`; copy v1's `val.bin` and `meta.json` unchanged
- [ ] `verify_data.py` on the new data: every game must replay legally
- [ ] Short smoke test of the new data path (a few dozen steps) before the long run
- [ ] Launch: `python -u train.py --size base --data_dir data_v2 --out_dir out_v2 --batch_size 16 --grad_accum 2 --max_iters 120000 --lr 6e-4 --warmup 500 --eval_every 2000`
  with output saved to a UTF-8 log (PowerShell's `Tee-Object` wrote UTF-16 last time; Git Bash `tee` avoids that)

## While it trains (about 15 hours at v1's ~445 ms per step)

- [ ] Nothing else on the GPU: no evaluation, no probing, no games. A concurrent run crashed once with a CUDA error
- [ ] PC set not to sleep; power connected
- [ ] If it stops, resume with the same command plus `--resume` (the full state is saved every 2,000 steps)
- [ ] Do not tune anything based on interim numbers

## After training (one job at a time, GPU otherwise idle)

- [ ] `evaluate.py legal` on v2, then `evaluate.py stockfish --games 100` on **both** v1 and v2, then `probe.py` on v2
- [ ] Extend `make_figures.py` with a v1 vs v2 comparison (probe accuracy by layer, headline metrics side by side)
- [ ] README: a v1 vs v2 section with the table, the figure, and a plain statement of what changed and what did not
- [ ] Commit results and figures, tag `v2`

## Open questions for you

- [ ] **Time:** are about 15 hours of GPU (plus about 3 hours of data prep and evaluation) acceptable, and should it start
  before or after your interviews?
- [ ] **Model size:** keep 54.6M for a clean "more compute" comparison (recommended), or also scale the model up and
  accept that two things changed at once?
- [ ] **Data:** the four extra months (recommended), or a different source such as higher-rated games only?
