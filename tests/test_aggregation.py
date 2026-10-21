"""Unit tests for src/aggregation.py: Hedge/Fixed Share updates and Proposition 3.5 params."""
import numpy as np
import pytest

from src.aggregation import (
    binary_entropy,
    optimal_eta_alpha,
    default_switching_budget,
    OnlineAggregator,
)


def test_binary_entropy_boundaries():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(np.log(2), rel=1e-6)


def test_optimal_eta_alpha_frictionless_hedge_alpha_zero():
    """C_T=0 (c=0) and K=0 must give alpha*=0 (Hedge)."""
    eta, alpha = optimal_eta_alpha(T=200, K=0, n=4, c=0.0)
    assert alpha == 0.0
    assert eta > 0


def test_optimal_eta_alpha_frictionless_fixed_share_alpha_positive():
    eta, alpha = optimal_eta_alpha(T=200, K=5, n=4, c=0.0)
    assert alpha == pytest.approx(5 / 199)
    assert eta > 0


def test_optimal_eta_alpha_cost_sensitive_hedge():
    """C_T>0 (c>0), K=0 -> alpha*=0 and eta*=sqrt(log n / A_T)."""
    eta, alpha = optimal_eta_alpha(T=200, K=0, n=4, c=0.001)
    assert alpha == 0.0
    assert eta > 0


def test_default_switching_budget_positive_and_monotone_shrinking_share():
    K1 = default_switching_budget(500)
    K2 = default_switching_budget(5000)
    assert K1 >= 0 and K2 >= 0
    # K(T)/T should shrink as T grows (switching budget share decreases).
    assert K2 / 5000 <= K1 / 500 + 1e-9


def test_aggregator_uniform_init_and_simplex_weights():
    agg = OnlineAggregator(n_experts=4, eta=0.5, alpha=0.1)
    np.testing.assert_allclose(agg.q, 0.25)
    losses = np.array([1.0, 0.0, 2.0, 0.5])
    q_used = agg.step(losses)
    np.testing.assert_allclose(q_used, 0.25)
    assert abs(agg.q.sum() - 1.0) < 1e-9
    assert np.all(agg.q >= 0)


def test_hedge_reweights_toward_lower_loss_expert():
    agg = OnlineAggregator(n_experts=2, eta=5.0, alpha=0.0)
    agg.step(np.array([1.0, -1.0]))  # expert 1 (index 1) had much lower loss
    assert agg.q[1] > agg.q[0]


def test_fixed_share_floor_prevents_full_concentration():
    n, alpha = 4, 0.2
    agg = OnlineAggregator(n_experts=n, eta=10.0, alpha=alpha)
    for _ in range(20):
        agg.step(np.array([10.0, -10.0, 10.0, 10.0]))
    # Sharing step guarantees every expert keeps at least alpha/n weight.
    assert np.all(agg.q >= alpha / n - 1e-9)


def test_fixed_share_adapts_faster_than_hedge_after_regime_shift():
    """Reproduces the qualitative claim of Fig. 1/2: after a switch in which
    expert becomes dominant, Fixed Share should give it more relative weight
    than Hedge after the same number of post-shift steps."""
    n = 3
    eta = 2.0
    hedge = OnlineAggregator(n, eta=eta, alpha=0.0)
    fixed_share = OnlineAggregator(n, eta=eta, alpha=0.1)
    # Phase 1: expert 0 dominant.
    for _ in range(30):
        losses = np.array([-1.0, 1.0, 1.0])
        hedge.step(losses)
        fixed_share.step(losses)
    # Phase 2 (post regime shift): expert 2 becomes dominant.
    for _ in range(10):
        losses = np.array([1.0, 1.0, -1.0])
        hedge.step(losses)
        fixed_share.step(losses)
    assert fixed_share.q[2] > hedge.q[2]
