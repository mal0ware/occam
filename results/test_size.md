# Rejection rates of the Sharpe tests when the null is true

Two series with the same Sharpe ratio and 377 months, the length of the paper's out-of-sample period, drawn 500 times for each process. Each test is two-sided at 0.05. A correctly sized test rejects 5% of the time. With 500 simulations the standard error of a rate near that is 0.010. The bootstrap uses blocks of 6 months and 499 draws for each simulation.

| process | Jobson-Korkie with Memmel | Ledoit-Wolf HAC | Ledoit-Wolf bootstrap |
|---|---|---|---|
| iid normal | 0.042 | 0.048 | 0.026 |
| iid t(5) | 0.048 | 0.052 | 0.038 |
| GARCH(1,1), normal shocks | 0.060 | 0.062 | 0.058 |
| GARCH(1,1), t(5) shocks | 0.078 | 0.080 | 0.060 |
| AR(1) 0.2 with GARCH(1,1), t(5) shocks | 0.142 | 0.074 | 0.060 |
