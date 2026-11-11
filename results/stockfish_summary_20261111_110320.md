Model `base_final.pt` vs Stockfish, 0.1s per move, policy `free`, temperature 0.5, 40 games per opponent (each opening played with both colours). Elo difference is the model's, relative to the opponent, from the score (95% interval); 'Est. Elo' is only given for `elo:` opponents and is on Stockfish's own scale, not Lichess or FIDE.

| Opponent | Games | W / D / L | Score | Elo diff | Est. Elo | Legal first try | Adjudicated |
|---|---|---|---|---|---|---|---|
| Random mover | 40 | 24 / 16 / 0 | 80% | +241 (+109 to +372) | n/a | 79.3% | 0 |
| Stockfish skill 0 | 40 | 2 / 6 / 32 | 12% | -338 (-495 to -181) | n/a | 93.0% | 0 |
| Stockfish ~1320 Elo | 40 | 1 / 4 / 35 | 8% | -436 (-631 to -242) | 884 | 94.7% | 0 |
