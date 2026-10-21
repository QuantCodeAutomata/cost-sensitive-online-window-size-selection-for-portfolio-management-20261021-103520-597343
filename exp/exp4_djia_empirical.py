"""Experiment 4 -- Empirical Study on Dow Jones Industrial Average (DJIA)
Constituents (Appendix B.4, Table 1 / Table 3).

Applies the two-level cost-sensitive online window-size aggregation
framework (Hedge, Fixed Share) to the 30 DJIA constituents named in the task
(WBA, CRM, HON, AMGN, DOW, AAPL, GS, V, NKE, UNH, TRV, CSCO, CVX, VZ, MSFT,
HD, INTC, JNJ, WMT, CAT, JPM, DIS, BA, KO, MCD, AXP, IBM, MRK, MMM, PG) plus
BIL (1-3 Month Treasury Bill ETF) as the risk-free asset, over 2021/03/04 to
2024/02/25, expert set N={5,10,21,30,41,50,63,126}, c=0.001. Compares Hedge
and Fixed Share against UP, EG, PAMR, CWMR, OLMAR using cost-sensitive
tracking regret R_track^c, stock tracking regret R_stock, account value, and
the Table 1 / Table 3 trading-performance metrics (cumulative return, mean,
std, annualized Sharpe, max drawdown) for every window expert and every
aggregated/benchmark portfolio.

Uses the same shared pipeline as exp_1-3 (src/pipeline.py), the same
cost-sensitive Mean-Variance experts (src/experts.py), and the same
regret/metric definitions (src/regret.py, src/metrics.py); only the data
source changes (real DJIA bars via data/data_loader.py instead of the
synthetic generator).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import fetch_adjusted_close, DataloaderAuthError, DataloaderRequestError
from src.pipeline import run_two_level_framework, run_benchmark_algorithms
from src.regret import (
    cost_sensitive_tracking_regret,
    portfolio_negative_log_return_loss,
    stock_tracking_regret,
)
from src.metrics import performance_table
from src.aggregation import default_switching_budget

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# 30 DJIA constituents + BIL risk-free asset, as specified in the task
# data_requirements for this experiment.
DJIA_TICKERS = [
    "WBA", "CRM", "HON", "AMGN", "DOW", "AAPL", "GS", "V", "NKE", "UNH",
    "TRV", "CSCO", "CVX", "VZ", "MSFT", "HD", "INTC", "JNJ", "WMT", "CAT",
    "JPM", "DIS", "BA", "KO", "MCD", "AXP", "IBM", "MRK", "MMM", "PG",
]
RISK_FREE_TICKER = "BIL"
ALL_TICKERS = DJIA_TICKERS + [RISK_FREE_TICKER]

START_DATE = "2021-03-04"
END_DATE = "2024-02-25"
WINDOW_SIZES = [5, 10, 21, 30, 41, 50, 63, 126]
C_RATE = 0.001
BENCHMARKS = ["UP", "EG", "PAMR", "CWMR", "OLMAR"]


def load_returns() -> pd.DataFrame:
    """Load adjusted close prices for all 31 assets and build simple daily
    returns y_i(t) = (S_i(t+1)-S_i(t))/S_i(t) (methodology step 2)."""
    result = fetch_adjusted_close(ALL_TICKERS, START_DATE, END_DATE, identity_date=END_DATE)
    prices = result.prices[ALL_TICKERS].dropna(how="any")
    returns = prices.pct_change().dropna(how="any")

    non_massive = [tk for tk, src in result.data_source_by_ticker.items() if src != "massive"]
    print(f"[exp4] loaded prices: rows={prices.shape[0]}, tickers={prices.shape[1]}, "
          f"dates={prices.index[0].date()}..{prices.index[-1].date()}")
    print(f"[exp4] ticker_details data_source != 'massive': {non_massive}")
    if non_massive:
        with open(os.path.join(RESULTS_DIR, "exp4_data_source_deviations.json"), "w") as f:
            json.dump({"non_massive_tickers": non_massive}, f, indent=2)
    return returns


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        returns_df = load_returns()
    except (DataloaderAuthError, DataloaderRequestError) as exc:
        print(f"[exp4] FATAL: QuantX Dataloader failure: {exc}")
        raise

    returns = returns_df.values
    tickers = list(returns_df.columns)
    m = len(tickers)
    T = returns.shape[0]
    start_t = max(WINDOW_SIZES)
    assert start_t < T, "not enough history for the largest expert window"

    T_eval = T - start_t
    K = default_switching_budget(T_eval)
    print(f"[exp4] T={T}, m={m}, T_eval={T_eval}, K={K}, window_sizes={WINDOW_SIZES}")

    w0 = np.full(m, 1.0 / m)
    result = run_two_level_framework(returns, WINDOW_SIZES, C_RATE, K, w0=w0)
    print(f"[exp4] two-level framework done: eta={result.eta}, alpha={result.alpha}")

    benchmark_weights = run_benchmark_algorithms(returns, start_t, BENCHMARKS)
    dates = returns_df.index[start_t:]
    price_relatives = 1.0 + returns[start_t:]

    # --- Cost-sensitive tracking regret R_track^c(t,K), Negative PnL loss --
    regret_track = {
        name: cost_sensitive_tracking_regret(result.agg_losses[name], result.expert_losses, K)
        for name in result.agg_losses
    }
    pd.DataFrame(regret_track, index=dates).to_csv(os.path.join(RESULTS_DIR, "exp4_tracking_regret.csv"))

    # --- Stock tracking regret R_stock(t,K), negative log-return loss ------
    all_portfolio_weights = dict(result.portfolio_weights)
    all_portfolio_weights.update(benchmark_weights)
    r_stock_curves, final_r_stock = {}, {}
    for name, w_path in all_portfolio_weights.items():
        port_losses = np.array([
            portfolio_negative_log_return_loss(w_path[t], price_relatives[t]) for t in range(T_eval)
        ])
        r_stock = stock_tracking_regret(port_losses, price_relatives, K)
        r_stock_curves[name] = r_stock
        final_r_stock[name] = float(r_stock[-1])
    pd.DataFrame(r_stock_curves, index=dates).to_csv(os.path.join(RESULTS_DIR, "exp4_r_stock_regret.csv"))

    # --- Account value / performance table (Appendix B.1, Table 1) ---------
    expert_portfolio_weights = {f"window_{j}": result.expert_weights[j] for j in WINDOW_SIZES}
    perf_inputs = dict(expert_portfolio_weights)
    perf_inputs.update(result.portfolio_weights)
    perf_inputs.update(benchmark_weights)
    perf_table = performance_table(perf_inputs, returns[start_t:], C_RATE)
    perf_table.to_csv(os.path.join(RESULTS_DIR, "exp4_performance_table.csv"))
    print("[exp4] performance table:\n", perf_table)

    # --- Account value curves -----------------------------------------------
    from src.metrics import account_value_path
    account_values = {name: account_value_path(w, returns[start_t:], C_RATE) for name, w in perf_inputs.items()}
    djia_equal_weight = np.full(m - 1, 1.0 / (m - 1))  # DJI-index proxy: equal-weight the 30 DJIA stocks (ex-BIL)
    djia_weights = np.tile(np.append(djia_equal_weight, 0.0), (T_eval, 1))
    account_values["DJI_index_proxy"] = account_value_path(djia_weights, returns[start_t:], 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ["hedge", "fixed_share"] + BENCHMARKS + ["DJI_index_proxy"]:
        ax.plot(dates, account_values[name][1:], label=name)
    ax.set_xlabel("Date")
    ax.set_ylabel("Account value V(t)")
    ax.set_title("DJIA: account value evolution, Hedge/Fixed Share vs. benchmarks")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp4_fig6_account_value.png"), dpi=150)
    plt.close(fig)

    # R_track^c (Problem 2.4) is defined against the window-size expert set
    # N, so it is only evaluated for Hedge/Fixed Share (research_objectives);
    # the benchmark comparison uses R_stock instead (plotted above).
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ["hedge", "fixed_share"]:
        ax.plot(dates, regret_track[name], label=name)
    ax.set_xlabel("Date")
    ax.set_ylabel("$R^c_{track}(t,K)$")
    ax.set_title("DJIA: cost-sensitive tracking regret (Hedge vs Fixed Share)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp4_tracking_regret.png"), dpi=150)
    plt.close(fig)

    summary = {
        "tickers": tickers, "start_date": START_DATE, "end_date": END_DATE,
        "window_sizes": WINDOW_SIZES, "c": C_RATE, "K": K,
        "T": T, "T_eval": T_eval, "eta": result.eta, "alpha": result.alpha,
        "final_r_track_c": {name: float(regret_track[name][-1]) for name in regret_track},
        "final_r_stock": final_r_stock,
        "performance_table": perf_table.to_dict(orient="index"),
    }
    with open(os.path.join(RESULTS_DIR, "exp4_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[exp4] summary written to results/exp4_summary.json")


if __name__ == "__main__":
    main()
