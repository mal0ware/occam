# Scoreboard against equal weight

A cell is one rule on one dataset. Every test is two-sided at the 0.05 level. `JK` is the Jobson-Korkie test with Memmel's correction, which assumes independent normal returns. `boot` is the Ledoit-Wolf studentised circular block bootstrap with blocks of 6 months and 4999 draws in the full run. `mv-is` is left out: it is not an investable rule.

## Paper's sample, 120-month window: 197307 to 200411, 4 datasets

Optimizers only (`mv`, `min`, `min-c`, `lw-min`, `erc`, `hrp`): 24 comparisons with `ew`. If every null were true and the comparisons were independent, 1.2 would be significant by chance, 0.6 in each direction. The Holm row controls the chance of any false rejection among the 24 at 0.05.

| test of the Sharpe difference | significantly above ew | significantly below |
|---|---|---|
| Jobson-Korkie with Memmel's correction | 6 | 2 |
| Ledoit-Wolf HAC | 6 | 1 |
| Ledoit-Wolf bootstrap, blocks of 1 | 6 | 1 |
| Ledoit-Wolf bootstrap, blocks of 3 | 6 | 1 |
| Ledoit-Wolf bootstrap, blocks of 6 | 6 | 1 |
| Ledoit-Wolf bootstrap, blocks of 10 | 6 | 1 |
| Ledoit-Wolf bootstrap, blocks of 6, Holm | 3 | 0 |

By rule, out of the datasets:

| rule | Sharpe above ew | JK sig above | JK sig below | boot sig above | boot sig below | CEQ sig above | CEQ sig below | turnover below ew | net Sharpe above ew |
|---|---|---|---|---|---|---|---|---|---|
| mv | 0 | 0 | 1 | 0 | 1 | 0 | 2 | 0 | 0 |
| min | 3 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| min-c | 3 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 3 |
| vw | 0 | 0 | 4 | 0 | 3 | 0 | 2 | 4 | 0 |
| lw-min | 4 | 2 | 0 | 2 | 0 | 0 | 1 | 1 | 4 |
| erc | 4 | 2 | 0 | 2 | 0 | 0 | 0 | 1 | 4 |
| hrp | 3 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 1 |
| bl | 0 | 0 | 4 | 0 | 3 | 0 | 2 | 4 | 0 |

## After the paper's sample, 120-month window: 200412 to 202609, 4 datasets

Optimizers only (`mv`, `min`, `min-c`, `lw-min`, `erc`, `hrp`): 24 comparisons with `ew`. If every null were true and the comparisons were independent, 1.2 would be significant by chance, 0.6 in each direction. The Holm row controls the chance of any false rejection among the 24 at 0.05.

| test of the Sharpe difference | significantly above ew | significantly below |
|---|---|---|
| Jobson-Korkie with Memmel's correction | 2 | 8 |
| Ledoit-Wolf HAC | 2 | 8 |
| Ledoit-Wolf bootstrap, blocks of 1 | 2 | 7 |
| Ledoit-Wolf bootstrap, blocks of 3 | 2 | 6 |
| Ledoit-Wolf bootstrap, blocks of 6 | 1 | 5 |
| Ledoit-Wolf bootstrap, blocks of 10 | 1 | 5 |
| Ledoit-Wolf bootstrap, blocks of 6, Holm | 0 | 5 |

By rule, out of the datasets:

| rule | Sharpe above ew | JK sig above | JK sig below | boot sig above | boot sig below | CEQ sig above | CEQ sig below | turnover below ew | net Sharpe above ew |
|---|---|---|---|---|---|---|---|---|---|
| mv | 2 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| min | 1 | 0 | 2 | 0 | 1 | 0 | 1 | 0 | 0 |
| min-c | 2 | 0 | 2 | 0 | 1 | 0 | 2 | 1 | 2 |
| vw | 4 | 1 | 0 | 0 | 0 | 1 | 0 | 4 | 4 |
| lw-min | 2 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| erc | 3 | 1 | 1 | 0 | 1 | 0 | 1 | 2 | 3 |
| hrp | 2 | 1 | 2 | 1 | 1 | 0 | 2 | 0 | 1 |
| bl | 4 | 1 | 0 | 0 | 0 | 1 | 0 | 4 | 4 |

## Paper's sample, 60-month window: 196807 to 200411, 4 datasets

Optimizers only (`mv`, `min`, `min-c`, `lw-min`, `erc`, `hrp`): 24 comparisons with `ew`. If every null were true and the comparisons were independent, 1.2 would be significant by chance, 0.6 in each direction. The Holm row controls the chance of any false rejection among the 24 at 0.05.

| test of the Sharpe difference | significantly above ew | significantly below |
|---|---|---|
| Jobson-Korkie with Memmel's correction | 7 | 4 |
| Ledoit-Wolf HAC | 7 | 4 |
| Ledoit-Wolf bootstrap, blocks of 1 | 7 | 3 |
| Ledoit-Wolf bootstrap, blocks of 3 | 7 | 3 |
| Ledoit-Wolf bootstrap, blocks of 6 | 6 | 3 |
| Ledoit-Wolf bootstrap, blocks of 10 | 6 | 3 |
| Ledoit-Wolf bootstrap, blocks of 6, Holm | 3 | 0 |

By rule, out of the datasets:

| rule | Sharpe above ew | JK sig above | JK sig below | boot sig above | boot sig below | CEQ sig above | CEQ sig below | turnover below ew | net Sharpe above ew |
|---|---|---|---|---|---|---|---|---|---|
| mv | 0 | 0 | 3 | 0 | 2 | 0 | 3 | 0 | 0 |
| min | 3 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| min-c | 4 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 4 |
| vw | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 4 | 0 |
| lw-min | 4 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 4 |
| erc | 4 | 2 | 0 | 2 | 0 | 1 | 0 | 0 | 4 |
| hrp | 4 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 3 |
| bl | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 4 | 0 |

## After the paper's sample, 60-month window: 200412 to 202609, 4 datasets

Optimizers only (`mv`, `min`, `min-c`, `lw-min`, `erc`, `hrp`): 24 comparisons with `ew`. If every null were true and the comparisons were independent, 1.2 would be significant by chance, 0.6 in each direction. The Holm row controls the chance of any false rejection among the 24 at 0.05.

| test of the Sharpe difference | significantly above ew | significantly below |
|---|---|---|
| Jobson-Korkie with Memmel's correction | 1 | 5 |
| Ledoit-Wolf HAC | 1 | 5 |
| Ledoit-Wolf bootstrap, blocks of 1 | 1 | 5 |
| Ledoit-Wolf bootstrap, blocks of 3 | 1 | 5 |
| Ledoit-Wolf bootstrap, blocks of 6 | 1 | 5 |
| Ledoit-Wolf bootstrap, blocks of 10 | 0 | 5 |
| Ledoit-Wolf bootstrap, blocks of 6, Holm | 0 | 2 |

By rule, out of the datasets:

| rule | Sharpe above ew | JK sig above | JK sig below | boot sig above | boot sig below | CEQ sig above | CEQ sig below | turnover below ew | net Sharpe above ew |
|---|---|---|---|---|---|---|---|---|---|
| mv | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| min | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
| min-c | 2 | 0 | 1 | 0 | 1 | 0 | 2 | 0 | 1 |
| vw | 4 | 1 | 0 | 0 | 0 | 1 | 0 | 4 | 4 |
| lw-min | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
| erc | 2 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 2 |
| hrp | 2 | 1 | 1 | 1 | 1 | 0 | 2 | 0 | 0 |
| bl | 4 | 1 | 0 | 0 | 0 | 1 | 0 | 4 | 4 |
