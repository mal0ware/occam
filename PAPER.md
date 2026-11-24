# Sources, formulas and comparison

Every formula in `src/occam/` is listed here with the place it was read. Sources 1 to 5 were read; where only a working-paper copy was available, that is said. Source 6 could not be retrieved and is implemented from the method as published, which is said again where it matters.

## Sources

1. DeMiguel, V., Garlappi, L. and Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" Review of Financial Studies 22(5), 1915-1953. Read in the published version. Called DGU below.
2. Ledoit, O. and Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." Journal of Multivariate Analysis 88(2), 365-411. Read in the published version.
3. López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample." Journal of Portfolio Management 42(4). Read in the SSRN copy (abstract 2708678), which carries the same algorithm and code snippets. Journal page numbers are not given because the journal copy was not read.
4. Black, F. and Litterman, R. (1992). "Global Portfolio Optimization." Financial Analysts Journal 48(5), 28-43. Read from a scan of the published article; the formulas are in the appendix on page 42.
5. Maillard, S., Roncalli, T. and Teiletche, J. "On the properties of equally-weighted risk contributions portfolios." Read in the authors' working paper dated May 2009. This is the definition of "risk parity" used here.
6. Ledoit, O. and Wolf, M. (2008). "Robust performance hypothesis testing with the Sharpe ratio." Journal of Empirical Finance 15(5), 850-859. Not read while this was written: two attempts to fetch a copy failed. The tests in `robust.py` follow the method as the author of this project knows it from the published paper. They are checked by the unit tests and the size simulation described below, and were not compared with the authors' code. Called LW below.

## What is tested

DGU run 14 portfolio rules on 7 datasets with a rolling 120-month window and compare each with 1/N on Sharpe ratio, certainty-equivalent return and turnover. Their conclusion is that none of the 14 is consistently better than 1/N out of sample.

This project reruns that test for the four datasets that come from the Kenneth French library and for five of the paper's rows, then adds four rules the paper did not run, and repeats everything on the months after the paper's sample ends. It also reruns both samples with the 60-month window and tests each Sharpe difference with the two tests of LW next to the paper's.

## Formulas

### The rolling test (DGU section 2)

With `T` months of excess returns and a window of `M = 120`, weights chosen at the end of month `t` use months `t-M+1` to `t` and earn the return of month `t+1`. This gives `T - M` out-of-sample returns. Code: `backtest.rolling_backtest`.

| Quantity | Formula | Source | Code |
|---|---|---|---|
| Relative weights | `w = x / abs(1'x)` | DGU eq. (1) | `optimizers.normalise` |
| Mean-variance, `mv` | `w = Σ⁻¹μ / abs(1'Σ⁻¹μ)` with sample `μ`, `Σ` | DGU eq. (3) with eq. (1) | `optimizers.mean_variance` |
| Minimum variance, `min` | `min w'Σw` subject to `1'w = 1` | DGU eq. (6) | `optimizers.min_variance` |
| Constrained minimum variance, `min-c` | eq. (6) with `w ≥ 0` added | DGU section 1.5 | `optimizers.min_variance_long_only` |
| Market portfolio, `vw` | all wealth in the market asset | DGU section 1.4.2 | `backtest._vw` |
| Sharpe ratio | `mean / std` of out-of-sample excess returns | DGU eq. (12) | `metrics.sharpe_ratio` |
| In-sample Sharpe ratio | eq. (12) for the `mv` portfolio estimated with `M = T` | DGU eq. (13) | `backtest.in_sample_mean_variance` |
| Certainty equivalent | `mean - (γ/2) variance`, `γ = 1` | DGU eq. (14) | `metrics.certainty_equivalent` |
| Turnover | average over the `T - M` months of `Σ_j abs(w_{j,t+1} - w_{j,t+})` | DGU eq. (15) | `metrics.trade_size`, `Backtest.turnover` |
| Return net of costs | `(1 + R_p)(1 - c Σ_j abs(w_{j,t+1} - w_{j,t+})) - 1`, `c = 0.005` | DGU eq. (16) | `metrics.net_returns` |
| Return-loss | `(μ_ew / σ_ew) σ_k - μ_k` on net returns | DGU eq. (17) | `metrics.return_loss` |
| Test of Sharpe ratios | `z = (σ_n μ_i - σ_i μ_n) / sqrt(θ)`, `θ = (1/(T-M)) (2σ_i²σ_n² - 2σ_iσ_nσ_in + μ_i²σ_n²/2 + μ_n²σ_i²/2 - μ_iμ_nσ_in²/(σ_iσ_n))` | DGU footnote 16 (Jobson-Korkie with Memmel's correction) | `metrics.sharpe_difference_test` |
| Test of certainty equivalents | delta method on `v = (μ_i, μ_n, σ_i², σ_n²)` with the covariance matrix `Θ` printed in the footnote | DGU footnote 18 | `metrics.ceq_difference_test` |

Jobson and Korkie (1981) and Memmel (2003) were not read. The statistic is implemented as DGU print it. Both of the paper's tests are reported with two-sided p-values, `2(1 - Φ(abs(z)))`.

### Sharpe tests that allow fat tails and serial dependence (LW)

Each series is first divided by its standard deviation, which leaves Sharpe ratios unchanged. With `v = (μ_i, μ_n, γ_i, γ_n)` and `γ = E[r²]`:

| Quantity | Formula | Code |
|---|---|---|
| Difference | `Δ = μ_i / sqrt(γ_i - μ_i²) - μ_n / sqrt(γ_n - μ_n²)` | `robust.sharpe_difference` |
| Gradient | `(γ_i / (γ_i - μ_i²)^1.5, -γ_n / (γ_n - μ_n²)^1.5, -μ_i / (2(γ_i - μ_i²)^1.5), μ_n / (2(γ_n - μ_n²)^1.5))` | `robust.sharpe_difference` |
| Standard error | `sqrt(∇'Ψ∇ / T)`, `Ψ` the long-run covariance of `y_t = (r_ti - μ_i, r_tn - μ_n, r_ti² - γ_i, r_tn² - γ_n)` | `robust.hac_standard_error` |
| HAC estimate | `Ψ = T/(T-4) Σ_j k(j/S) Γ(j)`, `Γ(j) = (1/T) Σ_t y_t y_{t-j}'`, Parzen kernel `k` | `robust.hac_covariance` |
| Bandwidth | `S = 2.6614 (α T)^(1/5)`, `α` from AR(1) fits to each column of `y` (Andrews 1991, not read) | `robust.andrews_bandwidth` |
| HAC test | `z = Δ / se`, two-sided normal p-value | `robust.hac_sharpe_test` |
| Bootstrap statistic | `d = abs(Δ) / se` on the data; `d* = abs(Δ* - Δ) / se*` on each resample | `robust.bootstrap_sharpe_test` |
| Resampling | circular blocks of `b` months, `l = floor(T/b)` blocks, pairs kept together | same |
| Bootstrap `se*` | `Ψ* = (1/l) Σ_j ζ_j ζ_j'`, `ζ_j = (1/sqrt(b)) Σ` of `y*` over block `j` | same |
| p-value | `(#{d* ≥ d} + 1) / (M + 1)`, `M = 4999` | same |

The bootstrap is run with `b` of 1, 3, 6 and 10 months. Six is the default used in the tables and the figure. Every rule on a dataset is resampled with the same blocks.

### Ledoit-Wolf shrinkage (Ledoit and Wolf, sections 2.1 and 3.3)

With the norm `‖A‖² = tr(AA')/p` and `S = X'X/n`:

| Quantity | Formula | Source |
|---|---|---|
| `m` | `tr(S)/p` | Lemma 3.2 |
| `d²` | `‖S - mI‖²` | Lemma 3.3 |
| `b²` | `min(b̄², d²)`, `b̄² = (1/n²) Σ_k ‖x_k x_k' - S‖²` | Lemma 3.4 |
| `a²` | `d² - b²` | Lemma 3.5 |
| `S*` | `(b²/d²) m I + (a²/d²) S` | eq. (14) |

Code: `estimators.ledoit_wolf`. The rule `lw-min` puts `S*` into DGU eq. (6).

### Equal risk contribution (Maillard, Roncalli and Teiletche, sections 2 and 3)

The risk contribution of asset `i` is `x_i (Σx)_i / sqrt(x'Σx)` (section 2.1). The portfolio is the long-only `x` with `Σ x_i = 1` and equal contributions (their eq. 1). Their problem (7) minimises `sqrt(y'Σy)` subject to `Σ ln y_i ≥ c`, and `x = y / Σ y_i`. Closed forms used as tests: two assets give inverse-volatility weights whatever the correlation (section 3.1), and equal correlations give inverse-volatility weights for any number of assets (their eq. 3). Code: `optimizers.equal_risk_contribution`.

### Hierarchical risk parity (López de Prado, stages 1 to 3)

1. Distance `d_ij = sqrt((1 - ρ_ij)/2)`, then the Euclidean distance between columns of `d`, then single linkage (stage 1). The paper's three-asset example is a test.
2. Reorder the assets by expanding the last cluster into its constituents until only assets remain (stage 2, code snippet 1).
3. Split the ordered list at `int(len/2)`. Each half gets the variance of its inverse-variance portfolio, `V`. The first half receives `α = 1 - V₁/(V₁ + V₂)` of the weight and the second `1 - α`. Repeat inside each half (stage 3, steps 3a to 3e, code snippet 2).

Code: `optimizers.hierarchical_risk_parity`.

### Black-Litterman (appendix, page 42)

Equilibrium premiums `Π = δΣW` (item 6). Posterior mean `E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]` (item 8). With no view the posterior is `Π`, and mean-variance weights on `Π` are `Σ⁻¹δΣW ∝ W`. A neutral view therefore returns the reference portfolio, whatever `δ`, `τ` and the covariance estimate are. The reference portfolio here is the market asset, so the `bl` column is the `vw` column. Code: `optimizers.black_litterman_mean`, `optimizers.black_litterman_neutral`.

## Differences from the sources

1. Four of the paper's seven datasets are run. The S&P sector data came privately from Roberto Wessels and the country indexes from MSCI; neither is free, so both are left out. FF-3-factor is left out because the paper does not report it.
2. Five of the paper's fourteen rules are run: `ew`, `mv`, `min`, `min-c`, `vw`. Bayes-Stein, Data-and-Model, MacKinlay-Pástor, the two other constrained rules, `g-min-c`, and the two mixtures are not implemented.
3. The data are today's files, built from the 202609 CRSP database. The paper used files downloaded around 2005. French revises history when CRSP and Compustat are corrected and when breakpoints change.
4. The paper does not write out `w_{t+}` in eq. (15). This project uses `w_j (1 + R_j) / (1 + w'R)` with the dataset's own returns, which are excess returns for the portfolios and zero-cost returns for SMB, HML and UMD. Adding the bill rate back moves the 1/N turnover on Industry from 0.0221 to 0.0220; the paper has 0.0216.
5. The paper does not say whether its p-values are one-sided. No p-value in its Tables 3 and 4 is above 0.50, and its figures agree with half of the two-sided values computed here (for `min`, 0.30, 0.23, 0.01 and 0.01 in the paper against 0.32, 0.25, 0.03 and 0.02). So the paper's p-values are read as one-sided in the direction of the observed difference. This project reports two-sided p-values everywhere and counts significance two-sided at 5%. Only the cell-by-cell comparison below and `paper_comparison.csv` use the halved value.
6. Ledoit and Wolf assume observations with mean zero. Here each window is demeaned and the divisor stays `n`. The result agrees with scikit-learn's `ledoit_wolf` to 6e-16 on 216 windows of the real data (checked once during the build, with scikit-learn installed for that run only; it is not a dependency).
7. Maillard, Roncalli and Teiletche suggest sequential quadratic programming. Here the first-order condition of their problem (7), `y_i (Σy)_i = 1/n`, is solved one coordinate at a time as a quadratic with one positive root. Tests check the result against their eq. (1) and their closed forms.
8. The sample covariance uses divisor `M - 1`. Every rule normalises its weights, so the divisor cancels.
9. Two of the four added rules were published after the paper (equal risk contribution and hierarchical risk parity). Ledoit-Wolf shrinkage is from 2004 and Black-Litterman from 1992; DGU name Ledoit and Wolf in section 1.5 as an estimator they chose not to test.
10. LW recommend a prewhitened kernel for the HAC estimate. Here the Parzen kernel is used with the Andrews bandwidth and no prewhitening. The same HAC standard error studentises the observed statistic in the bootstrap.
11. LW choose the block length by a calibration on simulated data. Here it is fixed at six months and results for 1, 3 and 10 are shown beside it.
12. Returns are scaled to unit standard deviation before the LW tests. Without this the bandwidth formula weights returns and squared returns by their size, and the test statistic changes with leverage. A unit test caught this.

## Comparison with the paper, cell by cell

Sample: July 1963 to November 2004, out-of-sample months July 1973 to November 2004 (377 months). The full list of 116 cells is in `results/paper_comparison.csv`. P-values in parentheses are one-sided in this section, to match the paper: the two-sided value divided by two.

### Sharpe ratio (DGU Table 3)

| rule | | Industry | MKT/SMB/HML | FF-1-factor | FF-4-factor |
|---|---|---|---|---|---|
| ew | paper | 0.1353 | 0.2240 | 0.1623 | 0.1753 |
| | here | 0.1353 | 0.2347 | 0.1613 | 0.1745 |
| mv, in sample | paper | 0.2124 | 0.2851 | 0.5098 | 0.5364 |
| | here | 0.2034 | 0.2608 | 0.4509 | 0.4999 |
| mv | paper | 0.0679 (0.17) | 0.2186 (0.46) | 0.0128 (0.02) | 0.1841 (0.45) |
| | here | -0.0494 (0.01) | 0.2135 (0.34) | 0.0513 (0.05) | 0.1650 (0.45) |
| min | paper | 0.1554 (0.30) | 0.2493 (0.23) | 0.2778 (0.01) | -0.0183 (0.01) |
| | here | 0.1537 (0.32) | 0.2574 (0.25) | 0.2496 (0.03) | 0.0326 (0.02) |
| min-c | paper | 0.1425 (0.41) | 0.2493 (0.23) | 0.1546 (0.35) | 0.3580 (0.00) |
| | here | 0.1445 (0.38) | 0.2574 (0.25) | 0.1554 (0.38) | 0.3649 (0.00) |
| vw | paper | 0.1138 (0.01) | 0.1138 (0.00) | 0.1138 (0.01) | 0.1138 (0.00) |
| | here | 0.1163 (0.02) | 0.1163 (0.00) | 0.1163 (0.02) | 0.1163 (0.01) |

### Certainty equivalent (DGU Table 4) and turnover (DGU Table 5)

| quantity | rule | | Industry | MKT/SMB/HML | FF-1-factor | FF-4-factor |
|---|---|---|---|---|---|---|
| CEQ | ew | paper | 0.0050 | 0.0039 | 0.0073 | 0.0072 |
| | | here | 0.0050 | 0.0042 | 0.0073 | 0.0072 |
| CEQ | mv | paper | -0.7816 | 0.0045 | -2.7142 | -0.0829 |
| | | here | -0.0480 | 0.0043 | -0.6597 | 0.0133 |
| CEQ | min | paper | 0.0052 | 0.0039 | 0.0100 | -0.0002 |
| | | here | 0.0050 | 0.0040 | 0.0088 | 0.0001 |
| CEQ | min-c | paper | 0.0047 | 0.0039 | 0.0060 | 0.0051 |
| | | here | 0.0047 | 0.0040 | 0.0061 | 0.0052 |
| Turnover, absolute | ew | paper | 0.0216 | 0.0237 | 0.0162 | 0.0198 |
| | | here | 0.0221 | 0.0237 | 0.0164 | 0.0200 |
| Turnover, relative to ew | mv | paper | 606594.36 | 2.83 | 10466.10 | 3553.03 |
| | | here | 2343.71 | 2.50 | 10093.11 | 676.71 |
| Turnover, relative to ew | min | paper | 21.65 | 1.11 | 45.47 | 6.83 |
| | | here | 20.97 | 1.02 | 44.93 | 6.16 |
| Turnover, relative to ew | min-c | paper | 2.58 | 1.11 | 3.93 | 1.76 |
| | | here | 2.34 | 1.02 | 4.20 | 1.63 |

### What matches

- The `ew` column. Sharpe ratio within 0.001 on three datasets and exact to four decimals on Industry. Certainty equivalent within 0.0001 on the same three. Turnover within 0.0005 on all four.
- `min-c` on all four datasets: Sharpe ratio within 0.009, certainty equivalent within 0.0001. The paper's one significant win for this rule, on FF-4-factor, is reproduced (0.3649 against its 0.3580, p 0.00 in both).
- `min` on Industry and MKT/SMB/HML, and the sign and significance of `min` on FF-1-factor (a significant win) and FF-4-factor (a significant loss).
- Most qualitative statements the paper makes about these five rows on these four datasets. The two exceptions are listed below.

### What does not match

- `ew` on MKT/SMB/HML: Sharpe ratio 0.2347 against 0.2240, certainty equivalent 0.0042 against 0.0039. The other three datasets agree, and this is the one made only of factor returns.
- `vw`: 0.1163 against 0.1138 in every column, since it is the same series.
- `mv`, in sample: lower than the paper on all four datasets, by 0.009 to 0.059. The in-sample certainty equivalent is lower as well.
- `mv`, out of sample: nothing matches closely except on MKT/SMB/HML. Sharpe ratio on Industry is -0.0494 here against 0.0679, and relative turnover is 2,344 against 606,594. Some of the direction survives: `mv` has the highest turnover on all four datasets and the worst certainty equivalent on Industry and FF-1-factor.
- Two qualitative differences, both for `mv`. The paper finds its Sharpe ratio below 1/N on FF-1-factor with p = 0.02 and not significant on Industry (p = 0.17); here the one-sided values are 0.05 on FF-1-factor and 0.01 on Industry. On FF-4-factor the paper has a certainty equivalent of -0.0829, significantly below 1/N (p = 0.01); here it is 0.0133, above 1/N and not significant (one-sided p = 0.27).
- `min` on FF-1-factor is 0.2496 against 0.2778, and on FF-4-factor 0.0326 against -0.0183.
- Return-loss of `vw` (Table 5 panel B). On FF-1-factor and FF-4-factor the paper and this project agree (0.0021 and 0.0028 against 0.0020 and 0.0026). On Industry the paper prints 0.0037 and on MKT/SMB/HML 0.0157; here they are 0.0008 and 0.0053. Eq. (17) with the paper's own Table 3 Sharpe ratios and a market volatility of 4.7% a month gives about 0.001 and 0.005, which is what this project finds. No explanation was found for the paper's two figures.

### Could data revisions explain the gaps?

The paper used files downloaded around 2005 and this project uses files built from the 202609 CRSP database. French revises history when CRSP and Compustat are corrected and when the construction of a series changes. That makes revisions a natural hypothesis for the gaps. No 2005 copy of the files was found, so the hypothesis cannot be tested directly.

`sensitivity.py` tests a weaker form of it. Independent normal noise is added to every monthly return of a dataset, the rolling test is rerun, and this is repeated 100 times for each of three noise sizes: 1, 5 and 20 basis points a month. One basis point is the precision the library publishes to. The other two sizes are assumptions; the true size of the revisions is not known. The output is `results/revision_sensitivity.md` and `.csv`. Sharpe ratio on the downloaded files, then the 5th to 95th percentile over the noisy copies, then the paper's figure:

| rule, dataset | here | 1 bp | 20 bp | paper | paper inside the full noisy range |
|---|---|---|---|---|---|
| `ew`, Industry | 0.1353 | 0.1352 to 0.1354 | 0.1340 to 0.1366 | 0.1353 | yes at every size |
| `ew`, MKT/SMB/HML | 0.2347 | 0.2345 to 0.2350 | 0.2296 to 0.2392 | 0.2240 | no at every size |
| `mv` in sample, FF-1-factor | 0.4509 | 0.4505 to 0.4514 | 0.4410 to 0.4576 | 0.5098 | no at every size |
| `mv` in sample, FF-4-factor | 0.4999 | 0.4994 to 0.5006 | 0.4918 to 0.5100 | 0.5364 | no at every size |
| `mv`, Industry | -0.0494 | -0.0530 to -0.0378 | -0.0697 to 0.0569 | 0.0679 | no; the largest noisy value is 0.0615 |
| `mv`, FF-1-factor | 0.0513 | -0.0202 to 0.0666 | -0.0494 to 0.0819 | 0.0128 | yes at every size |
| `mv`, FF-4-factor | 0.1650 | 0.0139 to 0.1773 | -0.0133 to 0.2140 | 0.1841 | yes at every size |
| `min`, FF-1-factor | 0.2496 | 0.2490 to 0.2503 | 0.2321 to 0.2622 | 0.2778 | no at every size |
| `min`, FF-4-factor | 0.0326 | 0.0296 to 0.0357 | 0.0038 to 0.0985 | -0.0183 | only at 20 bp |

What this shows:

- Out-of-sample `mv` cannot be replicated cell by cell on FF-1-factor and FF-4-factor from any file that differs from the authors' in the last published digit. Noise of one basis point moves its Sharpe ratio across a range of 0.09 and 0.16. The weights divide by `1'Σ⁻¹μ`, which passes close to zero in some months, and a single such month can set the whole column. For these two cells the revision hypothesis is enough.
- The gap in `ew` on MKT/SMB/HML is not explained by noise of this kind at any size tried. The same holds for in-sample `mv` on three datasets and for `min` on FF-1-factor. Independent noise barely moves an average, and these gaps are differences in averages. They would need revisions that shift mean returns, for example a change in how SMB and HML are built, or a difference in method that was not found. Neither was verified.
- `min` on FF-4-factor is reachable only with noise of 20 basis points, which is a large error for a published monthly return.

This analysis was designed after the mismatches were seen, and the noise sizes were chosen then.

## The added rules

None of these has a published figure on these datasets to compare with. Results are in `results/tables_paper.md` and `results/tables_after.md`, and for the 60-month window in the two files ending `_m60.md`.

Counts out of four datasets, 120-month window. "Significant" is two-sided at 5% in the bootstrap with blocks of six months.

| rule | Paper's sample: Sharpe above ew, significantly above | After: Sharpe above ew, significantly above | After: significantly below |
|---|---|---|---|
| lw-min | 4 of 4, 2 | 2 of 4, 0 | 1 |
| erc | 4 of 4, 2 | 3 of 4, 0 | 1 |
| hrp | 3 of 4, 1 | 2 of 4, 1 | 1 |
| bl | identical to vw | identical to vw | |

None of `lw-min`, `erc` and `hrp` has a certainty equivalent significantly above 1/N on any dataset in either sample with the 120-month window. The same holds for `mv`, `min` and `min-c`. The market portfolio does once: on MKT/SMB/HML after 2004 (0.0074 against 0.0022, two-sided p 0.00). With the 60-month window `erc` does once, on FF-1-factor in the paper's sample (two-sided p 0.03).

## Size of the tests

`size.py` draws pairs of series with equal Sharpe ratios and counts how often each test rejects at 5%. The series are 377 months long, the length of the paper's out-of-sample period, with 500 pairs for each process, 499 bootstrap draws for each pair and blocks of six months. The standard error of a rate near 5% is 1%.

| process | Jobson-Korkie with Memmel | LW HAC | LW bootstrap |
|---|---|---|---|
| independent normal | 0.042 | 0.048 | 0.026 |
| independent t(5) | 0.048 | 0.052 | 0.038 |
| GARCH(1,1), normal shocks | 0.060 | 0.062 | 0.058 |
| GARCH(1,1), t(5) shocks | 0.078 | 0.080 | 0.060 |
| AR(1) 0.2 with GARCH(1,1), t(5) shocks | 0.142 | 0.074 | 0.060 |

The paper's test is sound when returns are independent, with or without fat tails, and rejects a true null 14% of the time once returns have serial correlation of 0.2. The bootstrap is conservative for independent normal returns and close to 6% in the three GARCH cases. The GARCH parameters are 0.10 and 0.85, shocks have correlation 0.5, and each series has its own variance process.

## Choices and when they were made

Taken from the paper: the five rules, the four datasets, the 120-month window, risk aversion of 1 and the cost of 50 basis points.

Chosen for the first version of this project:

- The four added rules. None has a tuning parameter that was searched. Hierarchical risk parity uses single linkage as in its paper. Black-Litterman's `δ = 2.5` and `τ = 0.05` are placeholders that cancel.
- The second sample: estimation from December 1994, so that the first out-of-sample month is December 2004, the month after the paper's data end. No other split was run.

Choices made after seeing results. All of these are post hoc:

- Test tolerances for the comparison with the paper (`tests/test_data_and_pipeline.py`) were set to the size of the gaps found.
- The first version reported one-sided p-values and counted 8 optimizers significantly above 1/N and 2 below in the paper's sample, then 3 above and 8 below. The switch to two-sided values came after those counts were known. It lowers every count.
- The LW tests were added after the first results. The bootstrap is the headline test because it needs the fewest assumptions; that decision was made when the test was added and before its counts were seen, which the commit history cannot prove. The block length of six months was fixed the same way, and the counts for 1, 3, 6 and 10 months are all in `results/summary.md`.
- Scaling returns to unit variance in the LW tests was added after a first full run, when a unit test showed the statistic changing with leverage. Between that run and the final one the bootstrap counts with blocks of six and the 120-month window did not change. With the 60-month window they moved from 7 to 6 wins in the paper's sample and from 0 to 1 afterwards. Other 120-month rows moved by one: Holm in the paper's sample from 2 to 3 wins, blocks of 10 in the paper's sample from 5 to 6 wins, and after 2004 the HAC count from 9 to 8 losses, blocks of 1 from 6 to 7 and blocks of 3 from 5 to 6.
- The size simulation first used three processes. Two more (independent t(5) and GARCH with normal shocks) were added after seeing that every test over-rejects under GARCH with t(5) shocks, to find which feature causes it. All five are reported.
- The 60-month window was run once, with the same samples: the paper's data from July 1963, and estimation from December 1999 for the months after the paper.
- The six rules counted in the headline leave out the market portfolio. The first version counted the same six.
- Holm's correction was added with the LW tests.

## Not verified

- That data revisions explain any gap. The sensitivity analysis shows the hypothesis is sufficient for out-of-sample `mv` on two datasets and insufficient, in the form tested, for the other gaps. A direct check needs the 2005 files.
- The paper's separate appendix "Implementation Details and Robustness Checks" was not available. Any implementation detail it gives for eq. (15) or for the constrained optimiser is unknown, and its 60-month results could not be compared with the ones here.
- The LW tests against the authors' paper and code. The paper could not be retrieved while this was written. What supports the implementation is in `tests/test_robust.py` and `results/test_size.md`: the gradient matches finite differences, the HAC statistic agrees with the paper's test on long independent normal series, and the bootstrap's rejection rate under a true null is 2.6% to 6.0% across the five simulated processes.
- The certainty-equivalent test is the paper's delta-method test, which assumes independent normal returns. No bootstrap version was run.
