"""Unit tests for src/experts.py: rolling moments and cost-sensitive MV (Problem 2.1)."""
import numpy as np
import pytest

from src.experts import rolling_moments, solve_cost_sensitive_mv, expert_advice_path


def test_rolling_moments_no_lookahead():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=(50, 5))
    t, j = 20, 10
    mu_hat, sigma_hat = rolling_moments(returns, t, j)
    expected_mu = returns[t - j:t].mean(axis=0)
    np.testing.assert_allclose(mu_hat, expected_mu)
    assert sigma_hat.shape == (5, 5)
    # Covariance must be symmetric and PSD (up to numerical tolerance).
    np.testing.assert_allclose(sigma_hat, sigma_hat.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(sigma_hat)
    assert eigvals.min() >= -1e-10


def test_rolling_moments_rejects_insufficient_history():
    returns = np.zeros((5, 3))
    with pytest.raises(AssertionError):
        rolling_moments(returns, t=2, j=5)


def test_solve_cost_sensitive_mv_simplex_feasible():
    mu = np.array([0.01, 0.002, -0.001])
    sigma = np.eye(3) * 0.0001
    w_prev = np.full(3, 1 / 3)
    w = solve_cost_sensitive_mv(mu, sigma, w_prev, c=0.001)
    assert w.shape == (3,)
    assert np.all(w >= -1e-8)
    assert abs(w.sum() - 1.0) < 1e-6


def test_solve_cost_sensitive_mv_prefers_higher_mean_asset():
    mu = np.array([0.05, -0.05])
    sigma = np.eye(2) * 0.0001
    w_prev = np.full(2, 0.5)
    w = solve_cost_sensitive_mv(mu, sigma, w_prev, c=0.0)
    assert w[0] > w[1]


def test_turnover_penalty_discourages_large_reallocation():
    """Higher transaction cost c should keep the solution closer to w_prev."""
    mu = np.array([0.05, -0.05])
    sigma = np.eye(2) * 0.0001
    w_prev = np.array([0.5, 0.5])
    w_low_c = solve_cost_sensitive_mv(mu, sigma, w_prev, c=0.0)
    w_high_c = solve_cost_sensitive_mv(mu, sigma, w_prev, c=1.0)
    dist_low = np.abs(w_low_c - w_prev).sum()
    dist_high = np.abs(w_high_c - w_prev).sum()
    assert dist_high <= dist_low + 1e-9


def test_expert_advice_path_shapes():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, size=(30, 4))
    paths = expert_advice_path(returns, window_sizes=[5, 10], c=0.001, start_t=10)
    assert set(paths.keys()) == {5, 10}
    for j, arr in paths.items():
        assert arr.shape == (20, 4)
        np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-6)
