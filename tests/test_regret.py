"""Unit tests for src/regret.py: best-K-switch DP and loss definitions."""
import numpy as np
import pytest

from src.regret import (
    best_k_switch_prefix_losses,
    cost_sensitive_tracking_regret,
    negative_pnl_losses,
    negative_log_return_losses,
)


def test_best_k_switch_k0_equals_best_fixed_expert():
    loss_matrix = np.array([[1.0, 2.0], [0.5, 3.0], [2.0, 0.1]])
    best = best_k_switch_prefix_losses(loss_matrix, K=0)
    # With K=0 no switching allowed: best[t] = min over i of cumsum up to t for fixed i.
    cum = np.cumsum(loss_matrix, axis=0)
    expected = cum.min(axis=1)
    np.testing.assert_allclose(best, expected)


def test_best_k_switch_large_k_equals_greedy_per_step_min():
    rng = np.random.default_rng(2)
    loss_matrix = rng.normal(size=(10, 3))
    best = best_k_switch_prefix_losses(loss_matrix, K=9)
    greedy = np.cumsum(loss_matrix.min(axis=1))
    np.testing.assert_allclose(best, greedy, atol=1e-10)


def test_best_k_switch_monotone_in_K():
    rng = np.random.default_rng(3)
    loss_matrix = rng.normal(size=(15, 4))
    b0 = best_k_switch_prefix_losses(loss_matrix, K=0)
    b1 = best_k_switch_prefix_losses(loss_matrix, K=1)
    b2 = best_k_switch_prefix_losses(loss_matrix, K=2)
    # More switching budget can only weakly decrease the best hindsight loss.
    assert np.all(b1 <= b0 + 1e-9)
    assert np.all(b2 <= b1 + 1e-9)


def test_cost_sensitive_tracking_regret_zero_when_matching_best_expert():
    expert_losses = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    aggregate_losses = expert_losses[:, 0]  # always matches the best (lowest-loss) expert
    regret = cost_sensitive_tracking_regret(aggregate_losses, expert_losses, K=0)
    np.testing.assert_allclose(regret, 0.0, atol=1e-10)


def test_negative_pnl_losses_matches_definition():
    w = np.array([[1.0, 0.0], [0.0, 1.0]])
    w_prev = np.array([[0.5, 0.5], [0.5, 0.5]])
    y = np.array([0.02, -0.01])
    c = 0.1
    losses = negative_pnl_losses(w, y, c, w_prev)
    expected = -w @ y + c * np.abs(w - w_prev).sum(axis=1)
    np.testing.assert_allclose(losses, expected)


def test_negative_log_return_losses_sign_and_zero_at_one():
    x = np.array([1.0, 1.1, 0.9])
    losses = negative_log_return_losses(x)
    assert losses[0] == pytest.approx(0.0, abs=1e-12)
    assert losses[1] < 0  # gain -> negative loss
    assert losses[2] > 0  # loss -> positive loss
