"""Shared two-level cost-sensitive online window-size selection pipeline.

This module is the single shared implementation of the paper's method used
by every experiment script in exp/. All experiment scripts call
`run_two_level_framework` (and, where relevant, `run_benchmarks` /
`compute_regrets`) with the signatures defined here; they must not redefine
their own experts, aggregation, or regret logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.experts import rolling_moments, solve_cost_sensitive_mv
from src.aggregation import OnlineAggregator, optimal_eta_alpha
from src.regret import negative_pnl_losses
from src.benchmarks import UniversalPortfolio, ExponentiatedGradient, PAMR, CWMR, OLMAR


@dataclass
class TwoLevelResult:
    """Common output contract for the two-level framework.

    Attributes
    ----------
    dates_idx : int array, 0-indexed decision times t (aligned to `returns`).
    expert_weights : {j: array (T_eval, m)} per-expert advice path w_j(t).
    agg_weights : {algo_name: array (T_eval, n_experts)} q_j(t) paths, algo_name in {"hedge","fixed_share"}.
    portfolio_weights : {algo_name: array (T_eval, m)} aggregate portfolio w_hat(t).
    expert_losses : array (T_eval, n_experts) cost-sensitive losses ell_tilde_j(t).
    agg_losses : {algo_name: array (T_eval,)} realized aggregate losses.
    window_sizes : list[int] expert set N, in the order used for all index-aligned arrays.
    eta, alpha : dict[str, float] optimized learning-rate / sharing-rate per algo.
    """

    dates_idx: np.ndarray
    expert_weights: dict[int, np.ndarray]
    agg_weights: dict[str, np.ndarray]
    portfolio_weights: dict[str, np.ndarray]
    expert_losses: np.ndarray
    agg_losses: dict[str, np.ndarray]
    window_sizes: list[int] = field(default_factory=list)
    eta: dict[str, float] = field(default_factory=dict)
    alpha: dict[str, float] = field(default_factory=dict)


def run_two_level_framework(
    returns: np.ndarray,
    window_sizes: list[int],
    c: float,
    K: int,
    w0: np.ndarray | None = None,
    a: float = -0.1,
    b: float = 0.1,
) -> TwoLevelResult:
    """Run the full two-level cost-sensitive aggregation framework (Hedge and
    Fixed Share) over `returns` (array (T, m)) using expert window sizes
    `window_sizes` (expert set N), transaction cost rate `c`, and switching
    budget `K` for the Proposition 3.5 parameter optimization.

    Decision times start at t = max(window_sizes) (0-indexed), matching the
    "initial history of max(N) returns" burn-in required before rolling
    estimators can be computed (data_requirements).

    `a`/`b` are the Negative-PnL loss bounds used in Proposition 3.5's
    optimal eta*/alpha* schedule. The paper does not state numeric values
    for a,b (flagged uncertainty); +/-0.1 is used as a conservative bound on
    one-day portfolio returns (never violated by the long-only, diversified
    mean-variance experts in these experiments) that is tight enough for the
    Hoeffding-style bound to produce a non-degenerate learning rate.
    """
    T, m = returns.shape
    n = len(window_sizes)
    start_t = max(window_sizes)
    assert start_t < T, "not enough history: max(window_sizes) must be < T"
    w0 = w0 if w0 is not None else np.full(m, 1.0 / m)

    T_eval = T - start_t
    expert_weights = {j: np.zeros((T_eval, m)) for j in window_sizes}
    expert_losses = np.zeros((T_eval, n))
    prev_expert_w = {j: w0.copy() for j in window_sizes}

    eta_h, alpha_h = optimal_eta_alpha(T_eval, 0, n, c, a, b)  # Hedge: alpha fixed to 0
    eta_f, alpha_f = optimal_eta_alpha(T_eval, K, n, c, a, b)
    aggregators = {
        "hedge": OnlineAggregator(n, eta_h, 0.0),
        "fixed_share": OnlineAggregator(n, eta_f, alpha_f),
    }
    agg_weights = {name: np.zeros((T_eval, n)) for name in aggregators}
    portfolio_weights = {name: np.zeros((T_eval, m)) for name in aggregators}
    agg_losses = {name: np.zeros(T_eval) for name in aggregators}
    prev_portfolio_w = {name: w0.copy() for name in aggregators}

    for idx, t in enumerate(range(start_t, T)):
        w_j_t = np.zeros((n, m))
        for k, j in enumerate(window_sizes):
            mu_hat, sigma_hat = rolling_moments(returns, t, j)
            w_jt = solve_cost_sensitive_mv(mu_hat, sigma_hat, prev_expert_w[j], c)
            expert_weights[j][idx] = w_jt
            w_j_t[k] = w_jt

        losses_t = negative_pnl_losses(
            w_j_t, returns[t], c, np.array([prev_expert_w[j] for j in window_sizes])
        )
        expert_losses[idx] = losses_t

        for name, agg in aggregators.items():
            q_t = agg.q.copy()
            agg_weights[name][idx] = q_t
            w_hat_t = q_t @ w_j_t
            portfolio_weights[name][idx] = w_hat_t
            turnover = np.abs(w_hat_t - prev_portfolio_w[name]).sum()
            agg_losses[name][idx] = -w_hat_t @ returns[t] + c * turnover
            agg.step(losses_t)
            prev_portfolio_w[name] = w_hat_t

        for j in window_sizes:
            prev_expert_w[j] = expert_weights[j][idx]

    return TwoLevelResult(
        dates_idx=np.arange(start_t, T),
        expert_weights=expert_weights,
        agg_weights=agg_weights,
        portfolio_weights=portfolio_weights,
        expert_losses=expert_losses,
        agg_losses=agg_losses,
        window_sizes=list(window_sizes),
        eta={"hedge": eta_h, "fixed_share": eta_f},
        alpha={"hedge": 0.0, "fixed_share": alpha_f},
    )


BENCHMARK_CONSTRUCTORS = {
    "UP": lambda m: UniversalPortfolio(m),
    "EG": lambda m: ExponentiatedGradient(m),
    "PAMR": lambda m: PAMR(m),
    "CWMR": lambda m: CWMR(m),
    "OLMAR": lambda m: OLMAR(m),
}


def run_benchmark_algorithms(
    returns: np.ndarray,
    start_t: int,
    algos: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Run the OLPS benchmark algorithms (UP, EG, PAMR, CWMR, OLMAR) over the
    same evaluation window [start_t, T) as the two-level framework, using
    price relatives x(t) = 1 + y(t), with theoretically prescribed default
    parameters (Appendix B.3 methodology notes).

    Returns {algo_name: array (T_eval, m)} portfolio weight paths w_hat(t),
    where w_hat(t) is the weight held during period t (decided from
    information through t-1, i.e. before observing returns[t]).
    """
    T, m = returns.shape
    algos = algos or list(BENCHMARK_CONSTRUCTORS.keys())
    models = {name: BENCHMARK_CONSTRUCTORS[name](m) for name in algos}
    T_eval = T - start_t
    paths = {name: np.zeros((T_eval, m)) for name in algos}

    for idx, t in enumerate(range(start_t, T)):
        for name, model in models.items():
            paths[name][idx] = model.advice()
        x_t = 1.0 + returns[t]
        for name, model in models.items():
            model.update(x_t)

    return paths
