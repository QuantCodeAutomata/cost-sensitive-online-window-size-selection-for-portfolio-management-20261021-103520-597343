"""Window-size expert advice: rolling moment estimators and the cost-sensitive
Mean-Variance optimization (Problem 2.1 of the paper).

    max_w  mu_hat_j(t)^T w - 0.5 * w^T Sigma_hat_j(t) w - c * ||w - w_j(t-1)||_1
    s.t.   w >= 0, w^T 1 = 1

solved with CVXPY (Diamond & Boyd, 2016) since the objective is a concave
quadratic program with an L1 turnover penalty — a standard convex program with
no closed-form library equivalent among skfolio/pyportfolioopt that natively
exposes a turnover-penalized maximize-utility form with L1 cost against an
*arbitrary* previous weight each period inside an online loop; CVXPY is used
directly as recommended in the paper's implementation notes.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp


def rolling_moments(returns: np.ndarray, t: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean/covariance estimators used at decision time t with window j.

    mu_hat_j(t)    = (1/j)       * sum_{s=t-j}^{t-1} y(s)
    Sigma_hat_j(t) = (1/(j-1))   * sum_{s=t-j}^{t-1} (y(s)-mu_hat_j(t))(y(s)-mu_hat_j(t))^T

    `returns` is indexed 0..T-1 and `t` is the (0-indexed) decision time, so the
    window uses strictly past observations returns[t-j:t] — no look-ahead.
    """
    assert t - j >= 0, "not enough history for the requested window j at time t"
    window = returns[t - j:t]
    mu_hat = window.mean(axis=0)
    if j > 1:
        centered = window - mu_hat
        sigma_hat = (centered.T @ centered) / (j - 1)
    else:
        sigma_hat = np.zeros((returns.shape[1], returns.shape[1]))
    return mu_hat, sigma_hat


def solve_cost_sensitive_mv(
    mu_hat: np.ndarray,
    sigma_hat: np.ndarray,
    w_prev: np.ndarray,
    c: float,
    risk_aversion: float = 1.0,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Solve Problem 2.1: cost-sensitive long-only Mean-Variance optimization
    with an L1 turnover penalty against the previous period's expert weight.

    Returns the optimal simplex weight vector w_j(t).
    """
    m = mu_hat.shape[0]
    w = cp.Variable(m)
    sigma_psd = sigma_hat + ridge * np.eye(m)  # numerical PSD regularization
    objective = (
        mu_hat @ w
        - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma_psd))
        - c * cp.norm1(w - w_prev)
    )
    constraints = [w >= 0, cp.sum(w) == 1]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    try:
        problem.solve(solver=cp.OSQP)
    except cp.error.SolverError:
        problem.solve(solver=cp.SCS)
    if w.value is None:
        # Fall back to equal weight if the solver fails outright (should not
        # happen for a well-posed concave QP on the simplex).
        return np.full(m, 1.0 / m)
    w_val = np.clip(np.asarray(w.value).flatten(), 0.0, None)
    total = w_val.sum()
    return w_val / total if total > 0 else np.full(m, 1.0 / m)


def expert_advice_path(
    returns: np.ndarray,
    window_sizes: list[int],
    c: float,
    start_t: int,
    w0: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Compute the full time path of expert advice w_j(t) for every window
    size j in `window_sizes`, for t = start_t .. T-1 (0-indexed decision
    times). Returns {j: array of shape (T-start_t, m)}.
    """
    T, m = returns.shape
    w0 = w0 if w0 is not None else np.full(m, 1.0 / m)
    paths = {j: np.zeros((T - start_t, m)) for j in window_sizes}
    prev = {j: w0.copy() for j in window_sizes}
    for idx, t in enumerate(range(start_t, T)):
        for j in window_sizes:
            mu_hat, sigma_hat = rolling_moments(returns, t, j)
            w_jt = solve_cost_sensitive_mv(mu_hat, sigma_hat, prev[j], c)
            paths[j][idx] = w_jt
            prev[j] = w_jt
    return paths
