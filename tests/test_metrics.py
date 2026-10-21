"""Unit tests for src/metrics.py: account value, Sharpe, drawdown (Appendix B.1)."""
import numpy as np
import pytest

from src.metrics import (
    account_value_path,
    cumulative_return,
    annualized_sharpe,
    max_drawdown,
    performance_table,
)


def test_account_value_path_no_turnover_matches_simple_compounding():
    weights = np.tile(np.array([1.0, 0.0]), (5, 1))  # constant weight -> no turnover cost
    returns = np.array([[0.01, 0.0], [0.02, 0.0], [-0.01, 0.0], [0.0, 0.0], [0.03, 0.0]])
    V = account_value_path(weights, returns, c=0.001, v0=1.0)
    expected = np.cumprod(1 + returns[:, 0])
    np.testing.assert_allclose(V[1:], expected)
    assert V[0] == 1.0


def test_account_value_path_turnover_cost_reduces_value():
    weights_no_turn = np.tile(np.array([0.5, 0.5]), (3, 1))
    weights_turn = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    returns = np.zeros((3, 2))
    V_no_turn = account_value_path(weights_no_turn, returns, c=0.01)
    V_turn = account_value_path(weights_turn, returns, c=0.01)
    assert V_turn[-1] < V_no_turn[-1]


def test_cumulative_return_definition():
    V = np.array([1.0, 1.1, 1.21])
    assert cumulative_return(V) == pytest.approx(0.21)


def test_max_drawdown_is_nonpositive_and_correct():
    V = np.array([1.0, 1.2, 0.9, 1.5, 0.75])
    mdd = max_drawdown(V)
    assert mdd <= 0
    assert mdd == pytest.approx((0.75 - 1.5) / 1.5)


def test_annualized_sharpe_zero_vol_is_zero():
    step_returns = np.zeros(10)
    assert annualized_sharpe(step_returns) == 0.0


def test_annualized_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(0)
    step_returns = rng.normal(0.001, 0.01, size=500)
    sharpe = annualized_sharpe(step_returns, periods_per_year=252)
    assert sharpe > 0


def test_performance_table_columns_and_rows():
    returns = np.random.default_rng(1).normal(0, 0.01, size=(20, 2))
    weights = {"A": np.tile([0.6, 0.4], (20, 1)), "B": np.tile([0.5, 0.5], (20, 1))}
    table = performance_table(weights, returns, c=0.001)
    assert set(table.index) == {"A", "B"}
    for col in ["cumulative_return", "mean_return", "std_return", "sharpe_ratio", "max_drawdown"]:
        assert col in table.columns
