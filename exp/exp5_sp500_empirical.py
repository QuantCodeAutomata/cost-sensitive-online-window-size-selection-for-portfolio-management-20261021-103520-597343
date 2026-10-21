"""Experiment 5 -- Empirical Study on S&P 500 Constituents (Appendix B.5,
Table 1 / Table 3).

Applies the two-level cost-sensitive online window-size aggregation
framework (Hedge, Fixed Share) to a large universe of S&P 500 constituents
plus BIL (1-3 Month Treasury Bill ETF) as the risk-free asset, over
2020/01/02 to 2026/08/01, expert set N={5,10,21,30,41,50,63,126}, c=0.001.
Compares Hedge and Fixed Share against UP, EG, PAMR, CWMR, OLMAR using
cost-sensitive tracking regret R_track^c, stock tracking regret R_stock,
account value, and the Table 1 / Table 3 trading-performance metrics
(cumulative return, mean, std, annualized Sharpe, max drawdown).

Uses the same shared pipeline as exp_1-4 (src/pipeline.py), the same
cost-sensitive Mean-Variance experts (src/experts.py), and the same
regret/metric definitions (src/regret.py, src/metrics.py); only the universe
and sample period change, per the task's exp_5 data_requirements.

Universe deviation (see results/RESULTS.md "Deviations from the paper"): the
task's Appendix C full 481-ticker S&P 500 list could not be retrieved, so a
curated list of large, long-tenured S&P 500 constituents across all eleven
GICS sectors is used instead, kept to the paper's sample period, frequency
and method. Any ticker the data source cannot serve for the full window is
skipped (logged), consistent with the paper's "no additions/removals" filter.
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
from src.metrics import performance_table, account_value_path
from src.aggregation import default_switching_budget

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Curated large, long-tenured S&P 500 constituents spanning all eleven GICS
# sectors (see module docstring deviation note). BIL is the risk-free asset.
SP500_TICKERS = sorted(set([
    # Information Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "ACN", "TXN",
    "QCOM", "AMD", "INTU", "IBM", "NOW", "AMAT", "ADI", "LRCX", "KLAC", "SNPS",
    "CDNS", "MU", "PANW", "FTNT", "ANSS", "ROP", "NXPI", "MCHP", "SWKS", "TER",
    "ADSK", "PTC", "ZBRA", "JNPR", "CTSH", "GLW", "HPQ", "HPE", "NTAP", "WDC",
    "STX", "FFIV", "AKAM", "GRMN",
    # Communication Services
    "GOOGL", "META", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO",
    "OMC", "IPG", "LYV", "MTCH",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "ORLY",
    "AZO", "ROST", "YUM", "MAR", "HLT", "GM", "F", "CMG", "DHI", "LEN",
    "PHM", "NVR", "EBAY", "ETSY", "BBY", "GPC", "TGT", "DG", "DLTR", "EXPE",
    "RL", "TPR", "WHR",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "MDLZ", "CL", "KMB", "GIS", "HSY",
    "STZ", "MKC", "SYY", "KR", "CHD", "CLX", "TSN", "CAG", "HRL", "MNST",
    "KDP", "ADM", "BG", "EL",
    # Health Care
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "MDT", "GILD", "ISRG", "SYK", "VRTX", "REGN", "ZTS", "BDX", "CI",
    "ELV", "HUM", "CVS", "MCK", "HCA", "BSX", "BAX", "A", "IDXX", "DXCM",
    "MTD", "WAT", "IQV", "RMD", "STE", "COO", "ALGN", "HOLX",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "SPGI",
    "CB", "PGR", "MMC", "AON", "ICE", "CME", "USB", "PNC", "TFC", "BK",
    "STT", "COF", "TRV", "ALL", "AIG", "MET", "PRU", "AFL", "HIG", "FITB",
    "KEY", "RF", "CFG", "MTB", "HBAN", "SYF", "AMP", "NTRS", "FIS", "GPN",
    "PYPL", "MA", "V",
    # Industrials
    "HON", "UNP", "UPS", "RTX", "LMT", "GE", "CAT", "DE", "BA", "MMM",
    "GD", "NOC", "ITW", "EMR", "ETN", "PH", "ROK", "CSX", "NSC", "FDX",
    "WM", "RSG", "PCAR", "CMI", "DOV", "XYL", "AME", "IEX", "SWK", "JCI",
    "TT", "FAST", "PAYX", "VRSK", "EFX", "CTAS", "ODFL", "URI", "GWW", "LHX",
    "HWM",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC", "OXY", "WMB",
    "KMI", "OKE", "HES", "DVN", "BKR", "HAL",
    # Materials
    "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "DD", "DOW", "PPG", "NUE",
    "VMC", "MLM", "ALB", "CE", "IFF", "LYB", "CF", "MOS", "IP",
    # Utilities
    "NEE", "SO", "DUK", "D", "AEP", "SRE", "EXC", "XEL", "ED", "PEG",
    "WEC", "ES", "AWK", "PPL", "FE", "EIX", "CMS", "DTE", "AEE", "CNP",
    "NI", "LNT", "ATO",
    # Real Estate
    "PLD", "AMT", "EQIX", "PSA", "O", "WELL", "SPG", "DLR", "AVB", "EQR",
    "VTR", "ESS", "MAA", "UDR", "CPT", "ARE", "BXP", "HST", "REG", "KIM",
    "VICI", "EXR",
]))
RISK_FREE_TICKER = "BIL"
ALL_TICKERS = SP500_TICKERS + [RISK_FREE_TICKER]

START_DATE = "2020-01-02"
END_DATE = "2026-08-01"
WINDOW_SIZES = [5, 10, 21, 30, 41, 50, 63, 126]
C_RATE = 0.001
BENCHMARKS = ["UP", "EG", "PAMR", "CWMR", "OLMAR"]


def load_returns() -> pd.DataFrame:
    """Load adjusted close prices and build simple daily returns
    y_i(t) = (S_i(t+1)-S_i(t))/S_i(t) (methodology step 3). Tickers the
    Dataloader cannot serve across the full window are skipped and logged
    (paper's "no additions/removals" filter applied via data availability)."""
    result = fetch_adjusted_close(
        ALL_TICKERS, START_DATE, END_DATE, identity_date=END_DATE, skip_failed=True
    )
    prices = result.prices.dropna(how="any")
    # Drop any ticker whose series does not span the full sample after
    # alignment (methodology step 2: "no additions or removals").
    full_len = prices.shape[0]
    kept = [c for c in prices.columns]
    returns = prices[kept].pct_change().dropna(how="any")

    requested = set(ALL_TICKERS)
    loaded = set(result.prices.columns)
    skipped = sorted(requested - loaded)
    non_massive = [tk for tk, src in result.data_source_by_ticker.items() if src != "massive"]
    print(f"[exp5] requested={len(ALL_TICKERS)} loaded={len(loaded)} skipped={len(skipped)}")
    print(f"[exp5] loaded prices: rows={prices.shape[0]}, tickers={prices.shape[1]}, "
          f"dates={prices.index[0].date()}..{prices.index[-1].date()}")
    print(f"[exp5] ticker_details data_source != 'massive': {non_massive}")
    with open(os.path.join(RESULTS_DIR, "exp5_data_source_deviations.json"), "w") as f:
        json.dump({"skipped_tickers": skipped, "non_massive_tickers": non_massive,
                    "requested": len(ALL_TICKERS), "loaded": len(loaded)}, f, indent=2)
    return returns


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        returns_df = load_returns()
    except (DataloaderAuthError, DataloaderRequestError) as exc:
        print(f"[exp5] FATAL: QuantX Dataloader failure: {exc}")
        raise

    returns = returns_df.values
    tickers = list(returns_df.columns)
    m = len(tickers)
    T = returns.shape[0]
    start_t = max(WINDOW_SIZES)
    assert start_t < T, "not enough history for the largest expert window"

    T_eval = T - start_t
    K = default_switching_budget(T_eval)
    print(f"[exp5] T={T}, m={m}, T_eval={T_eval}, K={K}, window_sizes={WINDOW_SIZES}")

    w0 = np.full(m, 1.0 / m)
    result = run_two_level_framework(returns, WINDOW_SIZES, C_RATE, K, w0=w0)
    print(f"[exp5] two-level framework done: eta={result.eta}, alpha={result.alpha}")

    benchmark_weights = run_benchmark_algorithms(returns, start_t, BENCHMARKS)
    dates = returns_df.index[start_t:]
    price_relatives = 1.0 + returns[start_t:]

    # --- Cost-sensitive tracking regret R_track^c(t,K), Negative PnL loss --
    regret_track = {
        name: cost_sensitive_tracking_regret(result.agg_losses[name], result.expert_losses, K)
        for name in result.agg_losses
    }
    pd.DataFrame(regret_track, index=dates).to_csv(os.path.join(RESULTS_DIR, "exp5_tracking_regret.csv"))

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
    pd.DataFrame(r_stock_curves, index=dates).to_csv(os.path.join(RESULTS_DIR, "exp5_r_stock_regret.csv"))

    # --- Account value / performance table (Appendix B.1, Table 1) ---------
    expert_portfolio_weights = {f"window_{j}": result.expert_weights[j] for j in WINDOW_SIZES}
    perf_inputs = dict(expert_portfolio_weights)
    perf_inputs.update(result.portfolio_weights)
    perf_inputs.update(benchmark_weights)
    perf_table = performance_table(perf_inputs, returns[start_t:], C_RATE)
    perf_table.to_csv(os.path.join(RESULTS_DIR, "exp5_performance_table.csv"))
    print("[exp5] performance table:\n", perf_table)

    # --- Account value curves -----------------------------------------------
    account_values = {name: account_value_path(w, returns[start_t:], C_RATE) for name, w in perf_inputs.items()}
    gspc_equal_weight = np.full(m - 1, 1.0 / (m - 1))  # GSPC-index proxy: equal-weight the stocks (ex-BIL)
    gspc_weights = np.tile(np.append(gspc_equal_weight, 0.0), (T_eval, 1))
    account_values["GSPC_index_proxy"] = account_value_path(gspc_weights, returns[start_t:], 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ["hedge", "fixed_share"] + BENCHMARKS + ["GSPC_index_proxy"]:
        ax.plot(dates, account_values[name][1:], label=name)
    ax.set_xlabel("Date")
    ax.set_ylabel("Account value V(t)")
    ax.set_title("S&P 500: account value evolution, Hedge/Fixed Share vs. benchmarks")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp5_fig_account_value.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ["hedge", "fixed_share"]:
        ax.plot(dates, regret_track[name], label=name)
    ax.set_xlabel("Date")
    ax.set_ylabel("$R^c_{track}(t,K)$")
    ax.set_title("S&P 500: cost-sensitive tracking regret (Hedge vs Fixed Share)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp5_tracking_regret.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ["hedge", "fixed_share"] + BENCHMARKS:
        ax.plot(dates, r_stock_curves[name], label=name)
    ax.set_xlabel("Date")
    ax.set_ylabel("$R_{stock}(t,K)$")
    ax.set_title("S&P 500: stock tracking regret, Hedge/Fixed Share vs. benchmarks")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp5_r_stock.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for j in WINDOW_SIZES:
        ax.plot(dates, result.agg_weights["hedge"][:, WINDOW_SIZES.index(j)], label=f"j={j}")
    ax.set_xlabel("Date")
    ax.set_ylabel("$q_j(t)$ (Hedge)")
    ax.set_title("S&P 500: Hedge aggregation weights across window-size experts")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp5_agg_weights_hedge.png"), dpi=150)
    plt.close(fig)

    summary = {
        "tickers": tickers, "start_date": START_DATE, "end_date": END_DATE,
        "window_sizes": WINDOW_SIZES, "c": C_RATE, "K": K,
        "T": T, "T_eval": T_eval, "m": m, "eta": result.eta, "alpha": result.alpha,
        "final_r_track_c": {name: float(regret_track[name][-1]) for name in regret_track},
        "final_r_stock": final_r_stock,
        "performance_table": perf_table.to_dict(orient="index"),
    }
    with open(os.path.join(RESULTS_DIR, "exp5_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[exp5] summary written to results/exp5_summary.json")


if __name__ == "__main__":
    main()
