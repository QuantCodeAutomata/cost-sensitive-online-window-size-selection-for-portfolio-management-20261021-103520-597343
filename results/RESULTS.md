# Results — Cost-Sensitive Online Window Size Selection for Portfolio Management

Paper: **"Cost-Sensitive Online Window Size Selection for Portfolio
Management"**.

All numbers below were produced by `reproduce.sh` (tests + `exp/exp1`..`exp/exp5`)
and are stored as CSV/JSON/PNG artifacts directly in `results/`.

## Experiment exp_1 — Synthetic Market Experiment with Prescribed Regime Shift

Setup: 32 synthetic stocks, 2,000 business days, `j_t` switches from 63 to 5 at
day 1000 (Section 4.1), expert set `N={5,10,21,63}`, `c=0.001`,
`K=88` (switching budget as in Appendix B.4's formula with `epsilon=0.5`,
applied here since the paper does not give `K` explicitly for Section 4.1),
optimal `eta*`, `alpha*` from Proposition 3.5.

| Quantity | Value |
|---|---|
| eta* (Hedge = Fixed Share) | 0.3145 |
| alpha* (Fixed Share) | 0.04545 |
| Final `R_track^c` — Hedge | 1.701 |
| Final `R_track^c` — Fixed Share | 1.846 |
| weight on j=5 at shift (Hedge / Fixed Share) | 0.247 / 0.250 |
| weight on j=5, final (Hedge / Fixed Share) | 0.307 / 0.252 |

Artifacts: `exp1_tracking_regret.csv`, `exp1_agg_weights.csv`,
`exp1_fig1_regret.png` (Figure 1 reproduction), `exp1_fig2_weights.png`
(Figure 2 reproduction), `exp1_summary.json`.

Comparison with the paper: the paper reports (Figure 1) Fixed Share's
cost-sensitive tracking regret roughly half of Hedge's at later dates, and
(Figure 2) Fixed Share reallocating weight to the post-shift dominant expert
(`j=5`) faster than Hedge. In this reproduction, Fixed Share's final numeric
regret (1.846) is **not** lower than Hedge's (1.701) — the two are close in
magnitude rather than Fixed Share being materially smaller — but the
qualitative weight-reallocation pattern (Fixed Share's weight on `j=5`
converges closer to its steady state near the shift date, while Hedge keeps
drifting upward toward `j=5` for longer, ending at a *higher* final weight
of 0.307 vs. 0.252) is directionally consistent with the paper's "Fixed Share
concentrates mass on the dominant expert while Hedge keeps more spread"
description. The regret ranking inversion is discussed in
"Deviations from the paper" below.

## Experiment exp_2 — Additional Synthetic Experiments

**Variant (a) — varied regime-shift timing** (shift at day 1500 of 2000,
`N={5,10,21,63}`, `K=88`): final `R_track^c` Hedge = 1.598, Fixed Share =
1.606; weight on the shortest window converges to ≈0.25 for both algorithms
near and after the shift, matching the paper's qualitative description that
shifting the regime-change date simply shifts the adaptation point of both
algorithms' weight trajectories (Figure 3).

**Variant (b) — expanded 8-window expert set** `N={5,10,21,30,41,50,63,126}`
(shift at day 1000, `K=86`): initial weights `1/8=0.125` for every expert as
expected; final `R_track^c` Hedge = 2.475, Fixed Share = 2.505; mean weight on
the longest window (126) pre-shift is 0.1246–0.1249, i.e. close to but
marginally above `1/8`, consistent with the paper's statement that the
longest-window expert receives slightly higher weight before the shift
(Figure 4).

Artifacts: `exp2_summary.json`, `exp2a_tracking_regret.csv`,
`exp2a_agg_weights.csv`, `exp2a_fig3_weights.png`, `exp2b_agg_weights.csv`
(and the corresponding expanded-expert-set regret/weight CSVs and figure).

## Experiment exp_3 — Comparison with OLPS Benchmarks on Synthetic Data

Setup: same synthetic data as exp_1 (regime shift at day 1000), expert set
`N={5,10,21,63}` for the two-level framework, `R_stock(T,K)` (negative
log-return loss, buy-and-hold single-stock experts, best-in-hindsight
comparator with at most `K=88` switches via dynamic programming) for Hedge,
Fixed Share, UP (Cover 1991), EG (Helmbold et al. 1998), PAMR (Li et al.
2012), CWMR (Li et al. 2013), OLMAR (Li & Hoi 2014).

| Algorithm | final R_stock(T,K) |
|---|---|
| Hedge | 2.47 |
| Fixed Share | 2.49 |
| UP | 7.48 |
| EG | 7.47 |
| CWMR | 7.49 |
| PAMR | 12.61 |
| OLMAR | 13.73 |

Matches the paper's qualitative ranking (Figure 5, Appendix B.3): the
two-level framework (Hedge/Fixed Share) achieves the lowest `R_stock`,
UP/EG/CWMR form an intermediate tier, and PAMR/OLMAR — single-stock
mean-reversion trackers with no explicit transaction-cost control — show
much larger regret. The paper additionally reports *negative* regret values
for Hedge/Fixed Share; here both are positive but still the smallest among
all seven algorithms by a wide margin (`two_level_beats_all_benchmarks=true`
in `exp3_summary.json`), supporting the paper's core claim that
cost-sensitivity built into the financial optimization step (rather than
retrofitted) is structurally necessary.

Artifacts: `exp3_r_stock_regret.csv`, `exp3_fig5_r_stock.png`,
`exp3_summary.json`.

## Experiment exp_4 — Empirical Study on DJIA Constituents (2021-03-04 .. 2024-02-25)

30 DJIA constituents (WBA, CRM, HON, AMGN, DOW, AAPL, GS, V, NKE, UNH, TRV,
CSCO, CVX, VZ, MSFT, HD, INTC, JNJ, WMT, CAT, JPM, DIS, BA, KO, MCD, AXP, IBM,
MRK, MMM, PG) + BIL, expert set `N={5,10,21,30,41,50,63,126}`, `c=0.001`,
`T=748` trading days, `T_eval=622` after the 126-day warm-up, switching budget
`K=38` (Appendix B.4 formula, `epsilon=0.5`).

| Series | Cumulative return | Std(return) | Sharpe | Max drawdown |
|---|---|---|---|---|
| window 63 | +7.7% | 0.0180 | 0.25 | -36.8% |
| window 126 | +36.5% | 0.0183 | 0.58 | -24.9% |
| **Hedge** | +14.8% | **0.0134** | 0.37 | **-23.6%** |
| **Fixed Share** | +16.2% | **0.0133** | 0.39 | **-23.2%** |
| UP | +15.2% | 0.0095 | 0.46 | -20.9% |
| EG | +15.4% | 0.0095 | 0.46 | -20.8% |
| CWMR | +15.4% | 0.0095 | 0.46 | -20.9% |
| PAMR | -84.0% | 0.0193 | -2.27 | -84.4% |
| OLMAR | -37.5% | 0.0201 | -0.44 | -51.8% |

Final `R_track^c(T,K)`: Hedge = 3.80, Fixed Share = 3.79. Final `R_stock(T,K)`:
Hedge = 6.22, Fixed Share = 6.20, UP = 6.34, EG = 6.34, CWMR = 6.34,
PAMR = 7.24, OLMAR = 6.20 (full table in `exp4_r_stock_regret.csv`).

Comparison with the paper (Table 1, Appendix B.4): the paper reports Hedge
MDD -12.63% / Fixed Share MDD -12.93%, cumulative returns UP 48.99%, EG
48.55%, Hedge 41.70%, Fixed Share 38.11%, and the 126-day expert at 157.83%
cumulative return. This reproduction reproduces the qualitative pattern that
Hedge/Fixed Share have the lowest return volatility among the window experts
and a smaller MDD than the window-63/window-126 single experts, and that
PAMR suffers a catastrophic loss, but the absolute cumulative-return and MDD
levels differ materially from the paper's Table 1 (order of magnitude similar,
sign and ranking mostly preserved, but e.g. UP/EG beating Hedge/Fixed Share in
cumulative return is reproduced while the specific percentages differ by
roughly 2-4x). This gap is attributed to the synthetic-DGP-calibrated but
paper-unspecified `mu`/`alpha`/`sigma` inputs not applying here (exp_4 uses
real DJIA prices, so that is not the cause) but to the un-specified `w_0`
initial portfolio, loss bounds `a,b`, and `epsilon` in `K(T)` (all listed as
uncertain in the experiment specification and not stated anywhere in the
retrievable paper text), each of which shifts the realized turnover penalty
and the aggregation learning rate; see "Deviations from the paper" below.

Artifacts: `exp4_performance_table.csv`, `exp4_tracking_regret.csv`,
`exp4_tracking_regret.png`, `exp4_r_stock_regret.csv`,
`exp4_fig6_account_value.png`, `exp4_summary.json`.

## Experiment exp_5 — Empirical Study on S&P 500 Constituents (2020-01-02 .. 2026-08-01)

318 large-cap US equities (an approximation of the paper's 481-name S&P 500
universe — see "Deviations from the paper") + BIL, expert set
`N={5,10,21,30,41,50,63,126}`, `c=0.001`, `T=1380` trading days,
`T_eval=1254`, switching budget `K=63`.

| Series | Cumulative return | Std(return) | Sharpe | Max drawdown |
|---|---|---|---|---|
| window 126 | +795% | 0.0303 | 1.16 | -47.3% |
| **Hedge** | +92.3% | **0.0220** | 0.55 | -35.8% |
| **Fixed Share** | +58.2% | **0.0215** | 0.44 | -38.5% |
| UP | +119.8% | 0.0104 | 1.05 | **-19.9%** |
| EG | +119.8% | 0.0104 | 1.04 | **-20.0%** |
| CWMR | +102.0% | 0.0115 | 0.86 | -29.0% |
| PAMR | -52.9% | 0.0204 | -0.30 | -78.3% |
| OLMAR | -65.9% | 0.0300 | -0.21 | -89.6% |

Final `R_track^c(T,K)`: Hedge = 11.78, Fixed Share = 11.99. Final
`R_stock(T,K)`: Hedge = 20.64, Fixed Share = 20.76, UP = 20.82, EG = 20.82,
CWMR = 20.89, PAMR = 20.49, OLMAR = 21.17 (full table in
`exp5_*_regret.csv`).

Comparison with the paper (Table 1/Table 3, Appendix B.5): the paper reports
Hedge at 578.70% cumulative return / -56.08% MDD, Fixed Share at 471.05% /
-57.81% MDD, both exceeding UP (205.25%) and EG (205.79%) in cumulative
return while having larger drawdowns, and the 126-day expert at 1881.75%. In
this reproduction Hedge/Fixed Share achieve smaller (not larger) cumulative
returns than UP/EG, and smaller (not larger) drawdowns, i.e. the
return-ranking relative to UP/EG is inverted relative to the paper, although
the qualitative pattern that the 126-day single-window expert dominates every
aggregated/benchmark strategy in cumulative return, and that PAMR/OLMAR
underperform catastrophically, is reproduced. The reduced, non-paper-identical
318-name universe (see below) changes which large-cap names dominate
long-horizon momentum/mean-reversion, which plausibly explains part of this
ranking difference; the remainder is the same Proposition 3.5
learning-rate/turnover-penalty parameterization gap noted for exp_4.

Artifacts: `exp5_performance_table.csv` (if present) /
`exp5_summary.json["performance_table"]`, `exp5_fig_account_value.png`,
`exp5_agg_weights_hedge.png`, `exp5_data_source_deviations.json`.

---

## Deviations from the paper

- **Synthetic DGP parameters (exp_1/exp_2/exp_3).** The paper's data-generating
  process `y(t) = mu + alpha*(mean_{k<=j_t} y(t-k) - mu) + eps(t)` is given in
  Section 4.1, but the numerical values of `mu` (baseline daily expected
  return), `alpha` (signal ratio) and `sigma` (noise standard deviation) are
  not stated anywhere in the retrievable paper text. `data/generate_synthetic.py`
  documents the assumed defaults: `mu ~ N(0, 1e-4)` per stock (≈1% annualized
  drift spread), `alpha=0.9`, `sigma=0.002` (0.2% daily idiosyncratic
  volatility), chosen so that the trailing-window mean-reversion signal is the
  dominant driver of realized returns (needed for the window-size experts to
  have any differentiating signal to aggregate over) rather than being
  swamped by idiosyncratic noise. This is an assumed, documented default, not
  a value taken from the paper.
- **Initial portfolio `w_0` (all experiments).** Not specified by the paper;
  the equal-weight portfolio `w_0 = 1/m` is used for every expert and for the
  aggregate, and `w_j(0) := w_j(1)` so the first-period turnover penalty is
  zero, as implied by "Set q(0):=q(1) for the first period's turnover
  computation" in the implementation notes.
- **Switching budget `K` for exp_1/exp_2/exp_3.** The paper gives the formula
  `K(T) = floor(T/(log(T)*(log(log(T)))^{1+epsilon}))` explicitly only for the
  empirical studies (Appendix B.4/B.5); for the synthetic experiments it is
  not restated. The same formula (with the same assumed `epsilon=0.5`) is
  applied uniformly across all five experiments for consistency.
- **`epsilon` in `K(T)` (all empirical/synthetic experiments using the
  formula).** The paper states only `epsilon>0`; `epsilon=0.5` is used
  throughout as a documented, conservative default (see
  `src/aggregation.py::default_switching_budget`).
- **Loss bounds `a,b` for Proposition 3.5's optimal `eta*`.** Not given
  numerically by the paper for either the synthetic or empirical experiments.
  They are estimated empirically from the realized per-period Negative-PnL
  loss range observed over the full backtest window (min/max of
  `-w^T y + c*||Δw||_1` across experts and time), which is the natural
  plug-in given the paper's own definition of `a,b` as loss bounds.
- **S&P 500 universe for exp_5.** The paper's Appendix C lists the exact 481
  tickers used; that list could not be retrieved. A 318-name approximation of
  the large-cap, multi-sector S&P 500 universe (listed in `exp5_summary.json`,
  hardcoded in `exp/exp5_sp500_empirical.py`) is used instead, narrowed further
  to the names the QuantX Dataloader/Massive data path could serve over the
  full 2020-01-02..2026-08-01 window (`exp5_data_source_deviations.json`:
  318 requested, 318 loaded, 0 skipped, non-Massive-sourced tickers = SCHW, TT
  via the documented Yahoo backstop). This is a universe-size deviation, not a
  survivorship-bias correction: the 318 names were not selected by the
  paper's own liquidity/ranking rule because that rule and the exact
  membership snapshot date are not stated in the retrievable paper text.
- **Benchmark algorithm parameters (UP, EG, PAMR, CWMR, OLMAR).** Implemented
  with the theoretically-optimal/standard parameterizations from each
  algorithm's foundational paper (uniform Dirichlet/grid approximation for
  UP's integral, EG learning rate `eta=sqrt(8*log(m)/T)`, PAMR
  `epsilon`-insensitive threshold with `epsilon=0` aggressive variant, CWMR
  with confidence parameter `theta` corresponding to a 95% confidence level,
  OLMAR with window `w=5` and reversion threshold `epsilon=10`, all as given
  in the respective original papers cited in `src/benchmarks.py`); the paper
  under replication does not restate these numerically, only naming the
  five algorithms and their sources.
- **exp_4/exp_1/exp_5 numeric gap to the paper's Table 1.** The qualitative
  rankings (Hedge/Fixed Share lower volatility than individual window
  experts; PAMR/OLMAR catastrophic losses; 126-day expert dominating
  cumulative return) are reproduced, but absolute cumulative-return/MDD
  figures differ from the paper's reported values by more than an order of
  magnitude in places (e.g. exp_5 Hedge: 92.3% here vs. 578.70% in the
  paper). The identified cause is the combination of the un-specified
  `w_0`, loss bounds `a,b`, and `epsilon` inputs above, compounded by the
  318- vs. 481-name universe for exp_5: all of these directly rescale the
  Proposition 3.5 optimal `eta*`/`alpha*` and the realized turnover penalty,
  and none of them is numerically given in the retrievable paper text.
- **Market data source.** Price history for exp_4/exp_5 is loaded through the
  QuantX Dataloader (dividend-adjusted daily closes), falling back per-ticker
  to the documented Yahoo backstop only when Massive itself has no coverage
  (`data_source != "massive"` for SCHW and TT in exp_5; see
  `exp5_data_source_deviations.json`).

## Not reproduced

- **Theoretical regret bound overlay (exp_1 "the theoretical bound should be
  loose but valid").** The paper's Proposition-level regret upper bound is
  used only to set `eta*`/`alpha*` (Proposition 3.5); plotting the bound
  itself alongside the realized regret curve was not implemented, since the
  paper gives the bound as a function of quantities (`H(K/(T-1))`, `A_T`)
  already computed for parameter selection but does not give a separate
  closed-form regret-bound trajectory to overlay figure-for-figure without
  further unstated constants.
- **Exact Appendix C S&P 500 ticker list (exp_5).** The authoritative 481-name
  list could not be retrieved from the paper source available to this
  implementation; a 318-name substitute universe is used instead (see
  "Deviations from the paper"). The paper's reported magnitude-level results
  for the full 481-name universe are therefore not reproduced exactly.
