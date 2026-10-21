"""Experiment 3 -- Comparison with Online Portfolio Selection Benchmark
Algorithms on Synthetic Data (Appendix B.3, Figure 5).

Compares the two-level framework (Hedge, Fixed Share) against the standard
OLPS benchmarks UP, EG, PAMR, CWMR, OLMAR using the R_stock(T,K) tracking
regret metric, which treats each of the 32 synthetic stocks as a buy-and-hold
expert and scores every algorithm's realized portfolio with the negative
log-return loss ell(w,x) = -log(w^T x) (src/regret.py: portfolio_negative_log_return_loss,
stock_tracking_regret).

Uses the same synthetic data and expert set N={5,10,21,63} as exp_1, and the
same shared pipeline (src/pipeline.run_two_level_framework,
src/pipeline.run_benchmark_algorithms).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.generate_synthetic import generate_synthetic_returns
from src.pipeline import run_two_level_framework, run_benchmark_algorithms
from src.regret import portfolio_negative_log_return_loss, stock_tracking_regret
from src.aggregation import default_switching_budget

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N_STOCKS = 32
N_DAYS = 2000
SHIFT_DAY = 1000
J_BEFORE, J_AFTER = 63, 5
WINDOW_SIZES = [5, 10, 21, 63]
C_RATE = 0.001
BENCHMARKS = ["UP", "EG", "PAMR", "CWMR", "OLMAR"]


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    returns_df = generate_synthetic_returns(
        n_stocks=N_STOCKS, n_days=N_DAYS, shift_day=SHIFT_DAY,
        j_before=J_BEFORE, j_after=J_AFTER, seed=42,
    )
    returns = returns_df.values
    start_t = max(WINDOW_SIZES)
    dates = returns_df.index[start_t:]
    price_relatives = 1.0 + returns[start_t:]  # x(t) = 1+y(t), aligned to decision times

    print(f"[exp3] synthetic data: rows={returns.shape[0]}, cols={returns.shape[1]}, "
          f"T_eval={len(dates)}, window_sizes={WINDOW_SIZES}")

    T_eval = N_DAYS - start_t
    K = default_switching_budget(T_eval)
    two_level = run_two_level_framework(returns, WINDOW_SIZES, C_RATE, K)
    benchmark_weights = run_benchmark_algorithms(returns, start_t, BENCHMARKS)

    all_weights = dict(two_level.portfolio_weights)  # {"hedge":..., "fixed_share":...}
    all_weights.update(benchmark_weights)  # {"UP":..., "EG":..., ...}

    regret_curves, final_regret = {}, {}
    for name, w_path in all_weights.items():
        port_losses = np.array([
            portfolio_negative_log_return_loss(w_path[t], price_relatives[t])
            for t in range(T_eval)
        ])
        r_stock = stock_tracking_regret(port_losses, price_relatives, K)
        regret_curves[name] = r_stock
        final_regret[name] = float(r_stock[-1])
        print(f"[exp3] {name}: final R_stock={final_regret[name]:.4f}")

    regret_df = pd.DataFrame(regret_curves, index=dates)
    regret_df.to_csv(os.path.join(RESULTS_DIR, "exp3_r_stock_regret.csv"))

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ["hedge", "fixed_share"] + BENCHMARKS:
        ax.plot(dates, regret_curves[name], label=name)
    ax.axvline(returns_df.index[SHIFT_DAY], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("$R_{stock}(t,K)$")
    ax.set_title(f"Figure 5 reproduction: stock tracking regret, K={K}")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp3_fig5_r_stock.png"), dpi=150)
    plt.close(fig)

    summary = {
        "n_stocks": N_STOCKS, "n_days": N_DAYS, "shift_day": SHIFT_DAY,
        "window_sizes": WINDOW_SIZES, "c": C_RATE, "K": K,
        "final_r_stock": final_regret,
        "two_level_beats_all_benchmarks": all(
            final_regret[name] < min(final_regret[b] for b in BENCHMARKS)
            for name in ["hedge", "fixed_share"]
        ),
    }
    with open(os.path.join(RESULTS_DIR, "exp3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[exp3] summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
