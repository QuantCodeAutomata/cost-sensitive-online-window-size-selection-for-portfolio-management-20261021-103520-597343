"""Trading-performance metrics: account value evolution (Appendix B.1),
cumulative return, annualized Sharpe ratio, and maximum drawdown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def account_value_path(
    weights: np.ndarray,
    returns: np.ndarray,
    c: float,
    v0: float = 1.0,
) -> np.ndarray:
    """Account value recursion (Appendix B.1):

        V(t+1) = V(t) * (1 + w_hat(t)^T y(t) - c * ||w_hat(t) - w_hat(t-1)||_1)

    `weights[t]` is the portfolio held during period t (decided using
    information up to t-1), `returns[t]` the realized return vector y(t).
    weights and returns must have matching length T; weights[0] is compared
    against itself (zero turnover) for the first period's cost per the
    implementation note "set q(0):=q(1)" generalized to w_hat(0):=w_hat(1).
    """
    T = returns.shape[0]
    V = np.empty(T + 1)
    V[0] = v0
    prev_w = weights[0]
    for t in range(T):
        turnover = np.abs(weights[t] - prev_w).sum()
        step_ret = weights[t] @ returns[t] - c * turnover
        V[t + 1] = V[t] * (1 + step_ret)
        prev_w = weights[t]
    return V


def cumulative_return(V: np.ndarray) -> float:
    """Total cumulative return V(T)/V(0) - 1."""
    return V[-1] / V[0] - 1.0


def sample_mean_std(step_returns: np.ndarray) -> tuple[float, float]:
    return float(np.mean(step_returns)), float(np.std(step_returns, ddof=1))


def annualized_sharpe(step_returns: np.ndarray, risk_free: np.ndarray | float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio using `periods_per_year` trading days
    (methodology: 252 trading days)."""
    excess = step_returns - risk_free
    mu, sigma = np.mean(excess), np.std(excess, ddof=1)
    if sigma == 0:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


def max_drawdown(V: np.ndarray) -> float:
    """Maximum drawdown = min_t (V(t) - peak(t)) / peak(t), reported as a
    negative fraction (e.g. -0.1263 for -12.63%)."""
    running_max = np.maximum.accumulate(V)
    drawdowns = (V - running_max) / running_max
    return float(drawdowns.min())


def performance_table(
    name_to_weights: dict[str, np.ndarray],
    returns: np.ndarray,
    c: float,
    v0: float = 1.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Build the performance comparison table (paper Table 1 / Table 3 style)
    for a dict of {name: weights path} sharing the same realized returns.

    Returns columns: cumulative_return, mean_return, std_return, sharpe, mdd.
    """
    rows = {}
    for name, weights in name_to_weights.items():
        V = account_value_path(weights, returns, c, v0)
        step_returns = V[1:] / V[:-1] - 1.0
        mu, sigma = sample_mean_std(step_returns)
        rows[name] = {
            "cumulative_return": cumulative_return(V),
            "mean_return": mu,
            "std_return": sigma,
            "sharpe_ratio": annualized_sharpe(step_returns, periods_per_year=periods_per_year),
            "max_drawdown": max_drawdown(V),
        }
    return pd.DataFrame(rows).T
