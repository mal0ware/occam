# Occam

**Does any portfolio optimizer beat putting the same amount in everything?**

> **Status: planned.** This repository holds the design. No results exist yet, and nothing below is a finding. Each box in the roadmap gets ticked when the code and its tests are in.

## The question

Sixty years of portfolio theory against the rule "split it evenly". Out of sample, after turnover, who wins?

## What this will contain

- The rolling-window test from DeMiguel, Garlappi and Uppal on the Ken French portfolios
- Sample mean-variance, minimum variance, Ledoit-Wolf shrinkage, risk parity, hierarchical risk parity, and Black-Litterman with a neutral view
- Sharpe ratio, certainty-equivalent return and turnover for each
- A significance test of each method against equal weight

## The result it is built to show

The paper's table reproduced, with three methods published after it added as new columns.

Whatever the answer turns out to be goes at the top of this file, with one chart, before any installation step.

## Sources

- DeMiguel, Garlappi and Uppal, "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?", Review of Financial Studies 22(5), 2009
- Ledoit and Wolf, "A well-conditioned estimator for large-dimensional covariance matrices", Journal of Multivariate Analysis 88(2), 2004
- López de Prado, "Building Diversified Portfolios that Outperform Out of Sample", Journal of Portfolio Management 42(4), 2016
- Black and Litterman, "Global Portfolio Optimization", Financial Analysts Journal 48(5), 1992

Every number that differs from a source will be listed in `PAPER.md` with the likely reason.

## Data

The Kenneth R. French Data Library. Free, no key. Copyright Fama and French; fetched by script, not committed.

## How it is tested

The maths is tested before any result is trusted.

- Each optimizer against a hand-solved two-asset case
- Weights sum to one and respect the stated constraints
- The equal-weight column matches the paper

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

- [ ] Project skeleton, CI, and the reproduce command
- [ ] The rolling-window test from DeMiguel, Garlappi and Uppal on the Ken French portfolios
- [ ] Sample mean-variance, minimum variance, Ledoit-Wolf shrinkage, risk parity, hierarchical risk parity, and Black-Litterman with a neutral view
- [ ] Sharpe ratio, certainty-equivalent return and turnover for each
- [ ] A significance test of each method against equal weight
- [ ] `PAPER.md` comparison with the sources
- [ ] Results and limits written up at the top of this file

## Limits

- Portfolios of stocks, not single stocks. The result may not carry over.
- Every number that differs from the paper is listed with the likely reason.
- This is research code. It is not investment advice and it does not trade.

## The name

Occam's razor: do not multiply assumptions beyond need. Equal weight assumes nothing at all.

## Licence

MIT. See `LICENSE`.
