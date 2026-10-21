"""Integration-style tests for src/pipeline.py on small synthetic slices."""
import numpy as np

from src.pipeline import run_two_level_framework, run_benchmark_algorithms
from data.generate_synthetic import generate_synthetic_returns


def test_run_two_level_framework_shapes_and_simplex():
    df = generate_synthetic_returns(n_stocks=6, n_days=120, shift_day=60, j_before=10, j_after=5, seed=0)
    returns = df.values
    window_sizes = [3, 5, 10]
    result = run_two_level_framework(returns, window_sizes, c=0.001, K=1)
    start_t = max(window_sizes)
    T_eval = returns.shape[0] - start_t
    assert result.dates_idx.shape[0] == T_eval
    for algo in ["hedge", "fixed_share"]:
        assert result.agg_weights[algo].shape == (T_eval, len(window_sizes))
        np.testing.assert_allclose(result.agg_weights[algo].sum(axis=1), 1.0, atol=1e-6)
        assert np.all(result.agg_weights[algo] >= -1e-8)
        assert result.portfolio_weights[algo].shape == (T_eval, 6)
        np.testing.assert_allclose(result.portfolio_weights[algo].sum(axis=1), 1.0, atol=1e-6)
    assert result.expert_losses.shape == (T_eval, len(window_sizes))


def test_run_two_level_framework_no_lookahead_first_decision_uses_only_past():
    """The first decision time must be exactly max(window_sizes) (0-indexed),
    i.e. strictly after the required burn-in, never before."""
    df = generate_synthetic_returns(n_stocks=4, n_days=50, shift_day=25, j_before=8, j_after=3, seed=1)
    returns = df.values
    window_sizes = [3, 8]
    result = run_two_level_framework(returns, window_sizes, c=0.001, K=0)
    assert result.dates_idx[0] == 8


def test_run_benchmark_algorithms_returns_simplex_paths():
    df = generate_synthetic_returns(n_stocks=5, n_days=60, shift_day=30, j_before=8, j_after=3, seed=2)
    returns = df.values
    start_t = 8
    paths = run_benchmark_algorithms(returns, start_t, algos=["UP", "EG", "PAMR"])
    T_eval = returns.shape[0] - start_t
    for name, w in paths.items():
        assert w.shape == (T_eval, 5)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(w >= -1e-6)


def test_fixed_share_adapts_to_regime_shift_faster_than_hedge_on_synthetic_dgp():
    """Reproduces the qualitative claim behind Fig. 1/2: after the DGP's
    prescribed regime shift (j_before -> j_after), the short-window expert
    should receive more weight under Fixed Share than under Hedge shortly
    after the shift."""
    df = generate_synthetic_returns(
        n_stocks=8, n_days=400, shift_day=200, j_before=21, j_after=5, alpha=0.7, sigma=0.005, seed=3,
    )
    returns = df.values
    window_sizes = [5, 10, 21]
    result = run_two_level_framework(returns, window_sizes, c=0.001, K=2)
    start_t = max(window_sizes)
    shift_idx = 200 - start_t
    short_window_pos = window_sizes.index(5)
    post_shift_window = 20
    hedge_w = result.agg_weights["hedge"][shift_idx:shift_idx + post_shift_window, short_window_pos].mean()
    fs_w = result.agg_weights["fixed_share"][shift_idx:shift_idx + post_shift_window, short_window_pos].mean()
    assert fs_w >= hedge_w - 1e-6
