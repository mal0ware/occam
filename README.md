# Occam

**Does any portfolio optimizer beat putting the same amount in everything?**

On the Ken French portfolios, no optimizer beats equal weight consistently. From 1973 to 2004, the sample of DeMiguel, Garlappi and Uppal, six optimizing rules on four datasets give 24 comparisons: 8 have a Sharpe ratio significantly above 1/N and 2 significantly below. From December 2004 to September 2026, months the paper could not have seen, the count turns over to 3 above and 8 below. None of the six has a certainty-equivalent return significantly above 1/N on any dataset in either period.

The paper's 1/N column is reproduced: monthly Sharpe ratios of 0.1353, 0.1613 and 0.1745 against its 0.1353, 0.1623 and 0.1753 on three datasets, and 0.2347 against 0.2240 on the fourth. Its best rule, minimum variance without short sales, had a Sharpe ratio of 0.36 on the FF-4-factor dataset in its sample (0.3649 here, 0.3580 in the paper) and 0.04 afterwards.

![Sharpe ratio of each rule minus the Sharpe ratio of 1/N, by dataset and period](results/fig_sharpe.png)

Each bar is a rule's monthly out-of-sample Sharpe ratio minus that of 1/N. Solid bars are significant at 5%, one-sided. The rules are defined below.

## Reproduce

```
uv sync
uv run python scripts/fetch_data.py
uv run python scripts/reproduce.py
```

The last command rebuilds everything in `results/` in about four seconds. `--fast` runs two datasets into `results-fast/` for CI.

## The question

Sixty years of portfolio theory against the rule "split it evenly". Out of sample, after turnover, who wins?

## What was built

- The rolling-window test from DeMiguel, Garlappi and Uppal: a 120-month window, weights held for the next month, moved forward one month at a time
- Four of the paper's datasets, all from the Ken French library: ten industries plus the market (N = 11), market, SMB and HML (N = 3), twenty size and book-to-market portfolios plus the market (N = 21), and the same twenty plus four factors (N = 24)
- Five of the paper's rules: `ew` (1/N), `mv` (sample mean-variance), `min` (minimum variance), `min-c` (minimum variance without short sales), `vw` (the market portfolio)
- Four rules the paper did not run: `lw-min` (minimum variance on the Ledoit-Wolf covariance), `erc` (risk parity, as equal risk contribution), `hrp` (hierarchical risk parity), `bl` (Black-Litterman with a neutral view)
- Sharpe ratio, certainty-equivalent return, turnover and return-loss after 50 basis points of cost for each
- The paper's two significance tests of each rule against equal weight
- A second run on the months after the paper's sample

## Results

Monthly Sharpe ratios, with the one-sided p-value of the difference from `ew` in parentheses. Full tables for all four measures are in `results/tables_paper.md` and `results/tables_after.md`.

Paper's sample, July 1973 to November 2004 (377 months):

| rule | Industry | MKT/SMB/HML | FF-1-factor | FF-4-factor |
|---|---|---|---|---|
| ew | 0.1353 | 0.2347 | 0.1613 | 0.1745 |
| mv | -0.0494 (0.01) | 0.2135 (0.34) | 0.0513 (0.05) | 0.1650 (0.45) |
| min | 0.1537 (0.32) | 0.2574 (0.25) | 0.2496 (0.03) | 0.0326 (0.02) |
| min-c | 0.1445 (0.38) | 0.2574 (0.25) | 0.1554 (0.38) | 0.3649 (0.00) |
| vw | 0.1163 (0.02) | 0.1163 (0.00) | 0.1163 (0.02) | 0.1163 (0.01) |
| lw-min | 0.1565 (0.26) | 0.2598 (0.20) | 0.2582 (0.00) | 0.3585 (0.00) |
| erc | 0.1415 (0.07) | 0.2642 (0.10) | 0.1678 (0.00) | 0.3119 (0.00) |
| hrp | 0.1425 (0.16) | 0.2318 (0.47) | 0.1704 (0.03) | 0.2710 (0.00) |

After the paper's sample, December 2004 to September 2026 (262 months):

| rule | Industry | MKT/SMB/HML | FF-1-factor | FF-4-factor |
|---|---|---|---|---|
| ew | 0.1872 | 0.1089 | 0.1439 | 0.1434 |
| mv | 0.0404 (0.06) | 0.1757 (0.22) | 0.1802 (0.32) | 0.1017 (0.31) |
| min | 0.1644 (0.33) | -0.0193 (0.00) | 0.2194 (0.07) | -0.0315 (0.02) |
| min-c | 0.2037 (0.31) | -0.0134 (0.00) | 0.1799 (0.07) | 0.0393 (0.02) |
| vw | 0.1900 (0.41) | 0.1900 (0.02) | 0.1900 (0.03) | 0.1900 (0.04) |
| lw-min | 0.1887 (0.49) | 0.0063 (0.00) | 0.2162 (0.04) | 0.0502 (0.08) |
| erc | 0.1933 (0.17) | 0.0644 (0.00) | 0.1463 (0.02) | 0.1533 (0.36) |
| hrp | 0.1978 (0.19) | 0.0299 (0.00) | 0.1509 (0.01) | 0.1083 (0.02) |

What the tables say:

- The three added risk-based rules look good in the paper's sample. `lw-min` and `erc` have a higher Sharpe ratio than 1/N on all four datasets, significantly on two. After 2004 each of the three is significantly ahead on one dataset and significantly behind on at least one.
- The wins in the paper's sample are largest on FF-4-factor, the dataset that mixes stock portfolios with long-short factors. On average `erc` and `hrp` put 59% and 60% of that portfolio in SMB, HML and UMD in the paper's sample, `min-c` 86% and `lw-min` 95%; 1/N puts 12.5%. Those three factors earned 0.27%, 0.50% and 0.84% a month from 1973 to 2004 and -0.05%, -0.03% and 0.15% a month afterwards, while the market went from 0.55% to 0.84%.
- The market portfolio is significantly worse than 1/N on all four datasets in the paper's sample. After 2004 it is ahead on all four, significantly on three, with no trading.
- Sample mean-variance trades between 2.5 and 10,093 times as much as 1/N per month. After costs it loses to 1/N on every dataset in both periods.
- `erc` is the only optimizer whose turnover stays near that of 1/N: between 0.98 and 1.61 times.
- Black-Litterman with a neutral view is not a new column. With no view the model returns its reference portfolio, here the market, so `bl` equals `vw` to the last digit. The identity is tested.

![Average monthly turnover of each rule, log scale](results/fig_turnover.png)

## Sources

- DeMiguel, Garlappi and Uppal, "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?", Review of Financial Studies 22(5), 2009
- Ledoit and Wolf, "A well-conditioned estimator for large-dimensional covariance matrices", Journal of Multivariate Analysis 88(2), 2004
- López de Prado, "Building Diversified Portfolios that Outperform Out of Sample", Journal of Portfolio Management 42(4), 2016
- Black and Litterman, "Global Portfolio Optimization", Financial Analysts Journal 48(5), 1992
- Maillard, Roncalli and Teiletche, "On the properties of equally-weighted risk contributions portfolios", working paper, May 2009 (added during the build as the definition of risk parity)

## Comparison with the paper

Full detail, with every formula and its equation number, is in `PAPER.md`. All 116 published cells that this project can compute are side by side in `results/paper_comparison.csv`.

- Matches: the 1/N column on three datasets (Sharpe ratio within 0.001), `min-c` on all four (within 0.009), and the paper's significant results for `min` and `min-c`.
- 1/N on MKT/SMB/HML is 0.2347 against the paper's 0.2240, and the market's Sharpe ratio is 0.1163 against 0.1138. The files used here were built from the 202609 CRSP database and the paper's around 2005. Revisions to the factors are the likely cause; this was not verified.
- Sample mean-variance does not match out of sample except on MKT/SMB/HML. On Industry its Sharpe ratio is -0.05 here and 0.07 in the paper, and its turnover is 2,344 times that of 1/N here and 606,594 times in the paper. Its weights divide by a sum that passes near zero in some months, so one month can set the column, and small data revisions move which month.
- The in-sample mean-variance Sharpe ratio is lower than the paper's on every dataset, by 0.009 to 0.059. The cause is not established.
- The paper's return-loss for the market portfolio on two datasets (0.0037 and 0.0157) does not follow from its own Sharpe ratios. This project gets 0.0008 and 0.0053.
- The design promised three added rules published after the paper. Two of the four are (`erc` and `hrp`). Ledoit-Wolf is from 2004 and Black-Litterman from 1992.

## Data

The Kenneth R. French Data Library. Free, no key. Copyright Fama and French; fetched by script, not committed. `results/data_vintage.json` records the SHA-256 of each file and the CRSP vintage it names.

## How it is tested

127 tests, the maths before the results.

- Each optimizer against a hand-solved two-asset case, and mean-variance against the 635% and -535% example in the paper's footnote 8
- Minimum variance without short sales against its optimality conditions and a brute-force grid
- Risk parity against the two closed forms in its paper; hierarchical risk parity against its paper's three-asset worked example
- Ledoit-Wolf against the definition written out with loops
- Both significance tests against the formulas in the paper's footnotes, and a simulation that their statistics have unit variance when the null is true
- Weights sum to one, long-only rules stay long-only, and no rule's weights change when later returns change
- Running the pipeline twice gives byte-identical tables
- The equal-weight column matches the paper, with tolerances set to the gaps reported above

## Layout

```
src/occam/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
results/           committed figures and small tables
PAPER.md           citations, claims tested, comparison with the sources
```

## Roadmap

- [x] Project skeleton, CI, and the reproduce command
- [x] The rolling-window test from DeMiguel, Garlappi and Uppal on the Ken French portfolios
- [x] Sample mean-variance, minimum variance, Ledoit-Wolf shrinkage, risk parity, hierarchical risk parity, and Black-Litterman with a neutral view
- [x] Sharpe ratio, certainty-equivalent return and turnover for each
- [x] A significance test of each method against equal weight
- [x] `PAPER.md` comparison with the sources
- [x] Results and limits written up at the top of this file

CI is written but has not run on GitHub. The steps pass locally on Windows.

## Limits

- Portfolios of stocks, not single stocks. The result may not carry over.
- Four of the paper's seven datasets and five of its fourteen rules. The S&P sector and MSCI country data are not free. The Bayesian rules and the mixtures are not implemented, so nothing here speaks to them.
- All four datasets are cut from the same US stocks over the same months. Twenty-four comparisons are far fewer than 24 independent ones, and counting significant cells overstates the evidence in either direction.
- With 24 comparisons at the 5% level, one or two significant results per period are expected by chance. Nothing is adjusted for that.
- The two tests assume independent, normal monthly returns. The paper says the same of its own.
- The "after" period is 262 months in which SMB and HML averaged slightly below zero. It shows that the 1973 to 2004 winners did not persist over those months. It says nothing about the next 262.
- Costs are a flat 50 basis points on every asset. A long-short factor is charged as if it were one asset, which ignores the trading inside it.
- Black-Litterman is run only with a neutral view, where it reduces to the market portfolio. A version with real views would need a source of views, and none is specified.
- Every number that differs from the paper is listed in `PAPER.md` with the likely reason. The reason offered most often, data revisions, was not verified against 2005 files.
- This is research code. It is not investment advice and it does not trade.

## The name

Occam's razor: do not multiply assumptions beyond need. Equal weight assumes nothing at all.

## Licence

MIT. See `LICENSE`.
