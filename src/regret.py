"""Cost-sensitive tracking regret (Problem 2.4) and the stock-level tracking
regret R_stock, both computed against the best hindsight comparator sequence
restricted to at most K switches.

Custom implementation — Context7 survey found no package computing a
switching-restricted hindsight-optimal comparator (the "best expert sequence
with <=K switches" dynamic program is specific to the online-learning
tracking-regret literature referenced by the paper, not exposed by
vectorbt/skfolio/pyportfolioopt).

The DP (akin to Herbster & Warmuth's "tracking the best expert") computes, for
every prefix length t=1..T simultaneously, the minimal cumulative loss
achievable by a sequence of experts with at most K switches. This gives the
whole regret trajectory R(t, K) in a single O(T*K) sweep (vectorized over the
n experts at each step), rather than O(T) independent DPs.
"""
from __future__ import annotations

import numpy as np


def best_k_switch_prefix_losses(loss_matrix: np.ndarray, K: int) -> np.ndarray:
    """For loss_matrix of shape (T, n) return an array `best[t]` (t=0..T-1)
    giving the minimal cumulative loss of any expert-sequence with at most K
    switches over the prefix loss_matrix[0:t+1].
    """
    T, n = loss_matrix.shape
    assert K >= 0
    K = min(K, T - 1)

    # cost[k, i] = min cumulative loss up to current t, ending at expert i,
    # using at most k switches.
    cost = np.full((K + 1, n), np.inf)
    cost[:, :] = loss_matrix[0]  # at t=0 every budget level costs loss_matrix[0,i], 0 switches used
    best = np.empty(T)
    best[0] = cost[K].min()

    for t in range(1, T):
        new_cost = np.empty_like(cost)
        # k = 0: no switches ever allowed -> must keep the same expert since t=0
        new_cost[0] = cost[0] + loss_matrix[t]
        for k in range(1, K + 1):
            prev_row = cost[k - 1]
            if n == 1:
                min_excl = np.full(n, np.inf)
            else:
                # Compute min_{j != i} prev_row[j] for every i efficiently via
                # the smallest and second-smallest values.
                order = np.argsort(prev_row)
                i1, i2 = order[0], order[1]
                m1, m2 = prev_row[i1], prev_row[i2]
                min_excl = np.full(n, m1)
                min_excl[i1] = m2
            same = cost[k]
            switch = min_excl
            new_cost[k] = loss_matrix[t] + np.minimum(same, switch)
        cost = new_cost
        best[t] = cost[K].min()
    return best


def cost_sensitive_tracking_regret(
    aggregate_losses: np.ndarray,
    expert_losses: np.ndarray,
    K: int,
) -> np.ndarray:
    """R_track^c(t, K) trajectory (Problem 2.4):

        R(t,K) = sum_{s<=t} ell(w_hat(s), y(s)) - min_{seq, <=K switches} sum_{s<=t} ell(w_{i_s}(s), y(s))

    `aggregate_losses` : array (T,) of the online algorithm's realized losses.
    `expert_losses`    : array (T, n) of each expert's realized losses.
    """
    assert aggregate_losses.shape[0] == expert_losses.shape[0]
    best_hindsight = best_k_switch_prefix_losses(expert_losses, K)
    cum_online = np.cumsum(aggregate_losses)
    return cum_online - best_hindsight


def negative_pnl_losses(expert_weights: np.ndarray, y: np.ndarray, c: float, w_prev: np.ndarray) -> np.ndarray:
    """ell_j(t) = -w_j(t)^T y(t) + c*||w_j(t)-w_j(t-1)||_1 for all experts j
    at a single time step. expert_weights, w_prev: (n, m); y: (m,)."""
    pnl = -expert_weights @ y
    turnover = c * np.abs(expert_weights - w_prev).sum(axis=1)
    return pnl + turnover


def negative_log_return_losses(price_relatives: np.ndarray) -> np.ndarray:
    """ell_i(t) = -log(x_i(t)) for buy-and-hold single-stock experts, where
    x_i(t) = 1 + y_i(t) is the gross price relative. `price_relatives` has
    shape (m,) — one value per stock expert at time t."""
    safe = np.clip(price_relatives, 1e-8, None)
    return -np.log(safe)


def portfolio_negative_log_return_loss(w: np.ndarray, price_relatives: np.ndarray) -> float:
    """ell(w,x) = -log(w^T x) for a portfolio weight vector w held during the
    period whose gross price relatives are `price_relatives` (x = 1+y). Used
    for the R_stock comparison (Appendix B.3), which evaluates the two-level
    framework's and the OLPS benchmarks' aggregate portfolios under the same
    negative-log-return loss as the single-stock buy-and-hold experts."""
    gross = float(np.clip(w @ price_relatives, 1e-8, None))
    return -np.log(gross)


def stock_tracking_regret(
    portfolio_losses: np.ndarray,
    price_relatives: np.ndarray,
    K: int,
) -> np.ndarray:
    """R_stock(t,K) trajectory: cumulative realized negative-log-return loss
    of an online algorithm's portfolio, minus the best-hindsight cumulative
    loss achievable by a sequence of single-stock buy-and-hold experts
    (e_i, i.e. the standard basis vectors) with at most K switches.

    `portfolio_losses`  : array (T,) -log(w_hat(t)^T x(t)) for the algorithm.
    `price_relatives`   : array (T, m) gross price relatives x(t) = 1+y(t).
    """
    stock_losses = negative_log_return_losses(price_relatives)  # (T, m), ell_i(t)=-log(x_i(t))
    best_hindsight = best_k_switch_prefix_losses(stock_losses, K)
    cum_online = np.cumsum(portfolio_losses)
    return cum_online - best_hindsight
