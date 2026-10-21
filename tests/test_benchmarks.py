"""Unit tests for src/benchmarks.py: OLPS benchmark algorithms (Appendix B.3)."""
import numpy as np
import pytest

from src.benchmarks import (
    UniversalPortfolio,
    ExponentiatedGradient,
    PAMR,
    CWMR,
    OLMAR,
    _project_to_simplex,
)


@pytest.mark.parametrize("cls", [
    lambda m: UniversalPortfolio(m, n_samples=200, seed=0),
    lambda m: ExponentiatedGradient(m),
    lambda m: PAMR(m),
    lambda m: CWMR(m),
    lambda m: OLMAR(m),
])
def test_benchmark_advice_is_on_simplex(cls):
    m = 4
    model = cls(m)
    w = model.advice()
    assert w.shape == (m,)
    assert np.all(w >= -1e-8)
    assert abs(w.sum() - 1.0) < 1e-6


@pytest.mark.parametrize("cls", [
    lambda m: UniversalPortfolio(m, n_samples=200, seed=0),
    lambda m: ExponentiatedGradient(m),
    lambda m: PAMR(m),
    lambda m: CWMR(m),
    lambda m: OLMAR(m),
])
def test_benchmark_update_keeps_simplex(cls):
    rng = np.random.default_rng(5)
    m = 4
    model = cls(m)
    for _ in range(15):
        x = 1.0 + rng.normal(0, 0.01, size=m)
        model.update(x)
        w = model.advice()
        assert np.all(w >= -1e-6)
        assert abs(w.sum() - 1.0) < 1e-5


def test_project_to_simplex_feasible():
    v = np.array([0.9, 0.9, -0.5])
    p = _project_to_simplex(v)
    assert abs(p.sum() - 1.0) < 1e-8
    assert np.all(p >= 0)


def test_eg_shifts_weight_to_outperforming_asset():
    model = ExponentiatedGradient(2, eta=0.5)
    for _ in range(50):
        model.update(np.array([1.05, 0.95]))
    assert model.w[0] > model.w[1]


def test_up_wealth_weighting_favors_better_samples():
    model = UniversalPortfolio(2, n_samples=500, seed=1)
    for _ in range(30):
        model.update(np.array([1.05, 0.95]))
    advice = model.advice()
    assert advice[0] > advice[1]
