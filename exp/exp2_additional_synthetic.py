"""Experiment 2 -- Additional Synthetic Experiments: Varied Regime Shift
Timing and Expanded Expert Set (Appendix B.2, Figures 3-4).

Two variants of the base synthetic experiment (exp_1), sharing the same DGP
(data/generate_synthetic.py) and the same shared pipeline
(src/pipeline.run_two_level_framework):

  (a) varied regime-shift timing: same expert set N={5,10,21,63}, but the
      shift from j_before=63 to j_after=5 occurs later than the midpoint. The
      paper does not give an exact shift day for Figure 3 (uncertainties);
      the figure's x-axis spans into "2022 Q4/2023 Q1" of a panel starting
      2020-01-01, so a shift at 75% of the sample (day 1500 of 2000) is used
      here as a configurable, explicitly documented assumption.
  (b) expanded expert set N={5,10,21,30,41,50,63,126}, shift at the midpoint
      (day 1000), as in exp_1.

Outputs (results/):
  exp2a_tracking_regret.csv, exp2a_agg_weights.csv, exp2a_fig3_weights.png
  exp2b_tracking_regret.csv, exp2b_agg_weights.csv, exp2b_fig4_weights.png
  exp2_summary.json
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.generate_synthetic import generate_synthetic_returns
from src.pipeline import run_two_level_framework
from src.regret import cost_sensitive_tracking_regret
from src.aggregation import default_switching_budget

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N_STOCKS = 32
N_DAYS = 2000
J_BEFORE = 63
J_AFTER = 5
C_RATE = 0.001

# (a) varied shift timing (assumed: day 1500/2000 = 75%; see module docstring)
SHIFT_DAY_VARIED = 1500
WINDOW_SIZES_A = [5, 10, 21, 63]

# (b) expanded expert set, shift at the midpoint as in exp_1
SHIFT_DAY_BASE = 1000
WINDOW_SIZES_B = [5, 10, 21, 30, 41, 50, 63, 126]


def _run_variant(window_sizes, shift_day, seed, tag, fig_path, fig_title):
    returns_df = generate_synthetic_returns(
        n_stocks=N_STOCKS, n_days=N_DAYS, shift_day=shift_day,
        j_before=J_BEFORE, j_after=J_AFTER, seed=seed,
    )
    print(f"[exp2/{tag}] synthetic data: rows={returns_df.shape[0]}, cols={returns_df.shape[1]}, "
          f"shift_day={shift_day}, window_sizes={window_sizes}")
    returns = returns_df.values

    t_eval = N_DAYS - max(window_sizes)
    K = default_switching_budget(t_eval)
    result = run_two_level_framework(returns, window_sizes, C_RATE, K)
    dates = returns_df.index[result.dates_idx]

    regret = {
        name: cost_sensitive_tracking_regret(result.agg_losses[name], result.expert_losses, K)
        for name in result.agg_losses
    }
    regret_df = pd.DataFrame(regret, index=dates)
    regret_df.to_csv(os.path.join(RESULTS_DIR, f"exp2{tag}_tracking_regret.csv"))

    weight_frames = []
    for name, w in result.agg_weights.items():
        df = pd.DataFrame(w, index=dates, columns=[f"j={j}" for j in window_sizes])
        df.columns = pd.MultiIndex.from_product([[name], df.columns])
        weight_frames.append(df)
    weights_df = pd.concat(weight_frames, axis=1)
    weights_df.to_csv(os.path.join(RESULTS_DIR, f"exp2{tag}_agg_weights.csv"))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, name in zip(axes, ["hedge", "fixed_share"]):
        w = result.agg_weights[name]
        for k, j in enumerate(window_sizes):
            ax.plot(dates, w[:, k], label=f"j={j}")
        ax.axvline(returns_df.index[shift_day], color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("$q_j(t)$")
        ax.set_title(name.replace("_", " ").title())
        ax.legend(loc="upper right", ncol=min(len(window_sizes), 4), fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle(fig_title)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, fig_path), dpi=150)
    plt.close(fig)

    shift_idx = shift_day - result.dates_idx[0]
    return {
        "shift_day": shift_day, "window_sizes": window_sizes, "K": K,
        "eta": result.eta, "alpha": result.alpha,
        "final_regret": {name: float(regret_df[name].iloc[-1]) for name in result.agg_losses},
        "weight_on_shortest_window_post_shift_final": {
            name: float(result.agg_weights[name][-1, 0]) for name in result.agg_weights
        },
        "weight_on_shortest_window_at_shift": {
            name: float(result.agg_weights[name][max(shift_idx, 0), 0]) for name in result.agg_weights
        },
        "weight_on_longest_window_pre_shift_mean": {
            name: float(result.agg_weights[name][:max(shift_idx, 1), -1].mean()) for name in result.agg_weights
        },
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_a = _run_variant(
        WINDOW_SIZES_A, SHIFT_DAY_VARIED, seed=42, tag="a",
        fig_path="exp2a_fig3_weights.png",
        fig_title="Figure 3 reproduction: aggregation weights, varied regime-shift timing\n"
                   f"(assumed shift day={SHIFT_DAY_VARIED}/{N_DAYS}, N={{5,10,21,63}})",
    )
    summary_b = _run_variant(
        WINDOW_SIZES_B, SHIFT_DAY_BASE, seed=42, tag="b",
        fig_path="exp2b_fig4_weights.png",
        fig_title="Figure 4 reproduction: aggregation weights, expanded expert set\n"
                   f"(shift day={SHIFT_DAY_BASE}/{N_DAYS}, N={WINDOW_SIZES_B})",
    )

    summary = {"variant_a_varied_timing": summary_a, "variant_b_expanded_experts": summary_b}
    with open(os.path.join(RESULTS_DIR, "exp2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[exp2] summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
