"""Experiment 1 -- Synthetic Market Experiment with Prescribed Regime Shift.

Reproduces Section 4.1 ("Synthetic Data") and Figures 1-2 of the paper: a
synthetic 32-stock market whose target window size j_t switches from 63 to 5
at the midpoint of a 2,000 business-day sample. The two-level cost-sensitive
framework (Hedge and Fixed Share) aggregates cost-sensitive Mean-Variance
experts built on window sizes N={5,10,21,63} and is evaluated with the
cost-sensitive tracking regret R_track^c (Problem 2.4) under Negative PnL loss.

Outputs (results/):
  exp1_tracking_regret.csv   -- R_track^c(t,K) trajectories for Hedge/Fixed Share
  exp1_agg_weights.csv       -- q_j(t) aggregation-weight paths for both algorithms
  exp1_fig1_regret.png       -- Figure 1 reproduction: cost-sensitive tracking regret
  exp1_fig2_weights.png      -- Figure 2 reproduction: aggregation-weight evolution
  exp1_summary.json          -- scalar summary (final regret, eta*, alpha*, etc.)
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
from src.pipeline import run_two_level_framework
from src.regret import cost_sensitive_tracking_regret
from src.aggregation import default_switching_budget

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Paper-specified parameters (Section 4.1): 32 stocks, 2000 business days,
# regime shift at the midpoint, j_before=63 -> j_after=5, expert set
# N={5,10,21,63}, transaction-cost rate c=0.001 (10 bps).
N_STOCKS = 32
N_DAYS = 2000
SHIFT_DAY = 1000
J_BEFORE = 63
J_AFTER = 5
WINDOW_SIZES = [5, 10, 21, 63]
C_RATE = 0.001
# Switching budget K: the paper states only that eta/alpha are "optimized ...
# for the given T,K,n,c" for the synthetic experiment and gives no explicit
# numeric K there, while it gives an explicit formula K(T) = floor(T /
# (log(T) * (log(log(T)))^(1+epsilon))) for the empirical studies (Appendix
# B.4/B.5). Since no other value is specified for the synthetic case, the
# same K(T) rule is applied here for consistency across experiments
# (assumed; epsilon=0.5 as in `src.aggregation.default_switching_budget`).


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    returns_df = generate_synthetic_returns(
        n_stocks=N_STOCKS, n_days=N_DAYS, shift_day=SHIFT_DAY,
        j_before=J_BEFORE, j_after=J_AFTER, seed=42,
    )
    print(f"[exp1] synthetic data: rows={returns_df.shape[0]}, "
          f"cols={returns_df.shape[1]}, dates={returns_df.index[0].date()}..{returns_df.index[-1].date()}")
    returns = returns_df.values

    t_eval = N_DAYS - max(WINDOW_SIZES)
    switching_budget_k = default_switching_budget(t_eval)
    print(f"[exp1] T_eval={t_eval}, switching budget K={switching_budget_k}")

    result = run_two_level_framework(
        returns, WINDOW_SIZES, C_RATE, switching_budget_k,
    )
    print(f"[exp1] decision times T_eval={len(result.dates_idx)}, "
          f"eta={result.eta}, alpha={result.alpha}")

    dates = returns_df.index[result.dates_idx]

    # --- Cost-sensitive tracking regret R_track^c(t,K) (Problem 2.4) -------
    regret = {
        name: cost_sensitive_tracking_regret(result.agg_losses[name], result.expert_losses, switching_budget_k)
        for name in result.agg_losses
    }
    regret_df = pd.DataFrame(regret, index=dates)
    regret_df.to_csv(os.path.join(RESULTS_DIR, "exp1_tracking_regret.csv"))

    # --- Aggregation weights q_j(t) -----------------------------------------
    weight_frames = []
    for name, w in result.agg_weights.items():
        df = pd.DataFrame(w, index=dates, columns=[f"j={j}" for j in WINDOW_SIZES])
        df.columns = pd.MultiIndex.from_product([[name], df.columns])
        weight_frames.append(df)
    weights_df = pd.concat(weight_frames, axis=1)
    weights_df.to_csv(os.path.join(RESULTS_DIR, "exp1_agg_weights.csv"))

    # --- Figure 1: cost-sensitive tracking regret ---------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in result.agg_losses:
        ax.plot(dates, regret_df[name], label=name.replace("_", " ").title())
    ax.axvline(returns_df.index[SHIFT_DAY], color="gray", linestyle="--", linewidth=1, label="regime shift")
    ax.set_title("Figure 1 reproduction: Cost-sensitive tracking regret $R^c_{track}(t,K)$\n"
                 "Synthetic market, prescribed regime shift at midpoint")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative tracking regret")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_fig1_regret.png"), dpi=150)
    plt.close(fig)

    # --- Figure 2: aggregation weight evolution -----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, name in zip(axes, ["hedge", "fixed_share"]):
        w = result.agg_weights[name]
        for k, j in enumerate(WINDOW_SIZES):
            ax.plot(dates, w[:, k], label=f"j={j}")
        ax.axvline(returns_df.index[SHIFT_DAY], color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("$q_j(t)$")
        ax.set_title(name.replace("_", " ").title())
        ax.legend(loc="upper right", ncol=4)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Figure 2 reproduction: Aggregation weight evolution across window-size experts")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_fig2_weights.png"), dpi=150)
    plt.close(fig)

    shift_idx_in_eval = SHIFT_DAY - result.dates_idx[0]
    summary = {
        "n_stocks": N_STOCKS, "n_days": N_DAYS, "shift_day": SHIFT_DAY,
        "window_sizes": WINDOW_SIZES, "c": C_RATE, "K": switching_budget_k,
        "eta": result.eta, "alpha": result.alpha,
        "final_regret": {name: float(regret_df[name].iloc[-1]) for name in result.agg_losses},
        "weight_on_j5_post_shift_final": {
            name: float(result.agg_weights[name][-1, WINDOW_SIZES.index(5)]) for name in result.agg_weights
        },
        "weight_on_j5_at_shift": {
            name: float(result.agg_weights[name][max(shift_idx_in_eval, 0), WINDOW_SIZES.index(5)])
            for name in result.agg_weights
        },
    }
    with open(os.path.join(RESULTS_DIR, "exp1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[exp1] summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
