"""Synthetic data generation for the regime-shift experiments.

Implements the data generating process (DGP) of Section 4.1 ("Synthetic Data")
of the paper:

    y(t) = mu + alpha * ( (1/j_t) * sum_{k=1}^{j_t} y(t-k) - mu ) + eps(t),
    eps(t) ~ N(0, sigma^2 I)

for an artificial market of 32 stocks (SYN_1 .. SYN_32) over 2,000 business
days, with a prescribed regime shift of the target window size j_t halfway
through the sample (j_t=63 for the first 1,000 days, j_t=5 for the remaining
1,000 days).

The paper does not state numerical values for mu (baseline daily expected
return), alpha (signal ratio) or sigma (noise standard deviation); these are
listed as open parameters in Section 4.1 ("all the parameters needed can be
optimized"). We therefore expose them as configurable arguments with
explicitly documented defaults (assumed, not taken from the paper): mu drawn
once from N(0, 1e-4) per stock (~1% annualized drift spread), alpha=0.9 (a
strong, but <1, mean-reversion/momentum loading onto the rolling window
average), sigma=0.002 (0.2% daily idiosyncratic volatility). alpha=0.9 and a
small sigma were chosen, rather than alpha=0.5/sigma=0.01, because a direct
numerical check (comparing each window's realized one-step forecasting MSE
before/after the prescribed shift) showed that the gap between a window
matched to the active j_t and a mismatched window widens only modestly unless
alpha is close to 1 and idiosyncratic noise is small relative to the
mean-reversion signal; with alpha=0.5/sigma=0.01 the idiosyncratic noise term
dominates the rolling-window signal almost entirely. Even at alpha=0.9 the
shortest window (j=5) retains a structural "recency" advantage in raw one-step
MSE in both regimes (because the window formula is a trailing simple moving
average, so a short window always tracks the most recent, most-correlated
lags more tightly than a long window) -- this is a property of the DGP as
written in Section 4.1, not an implementation defect: it was verified that the
*relative* forecasting penalty of the window mismatched to the active j_t
increases after each regime shift (e.g. the j=63 expert's error is lower while
j_t=63 is active than while j_t=5 is active), which is what drives the
aggregation weight reallocation studied in Figures 1-2. This parameter choice
and the underlying diagnostic are recorded in results/RESULTS.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_returns(
    n_stocks: int = 32,
    n_days: int = 2000,
    start_date: str = "2020-01-01",
    shift_day: int = 1000,
    j_before: int = 63,
    j_after: int = 5,
    mu: np.ndarray | None = None,
    alpha: float = 0.9,
    sigma: float = 0.002,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate the synthetic multi-asset return panel with a regime shift.

    Parameters
    ----------
    n_stocks : number of synthetic stocks (paper: 32, SYN_1..SYN_32).
    n_days : number of business days simulated (paper: 2,000).
    start_date : first business day of the panel (paper: 2020-01-01).
    shift_day : 1-indexed day at which j_t switches from j_before to j_after
        (paper: halfway through the dataset, i.e. day 1000 of 2000).
    j_before, j_after : target window sizes j_t before/after the shift
        (paper: 63 then 5).
    mu : baseline daily expected return vector (R^n_stocks). If None, drawn
        once from N(0, 1e-4) (assumed; not specified in the paper).
    alpha : signal ratio in (0, 1) (assumed = 0.9; not specified in the paper).
    sigma : per-period noise std (assumed = 0.002; not specified in the paper).
    seed : RNG seed for reproducibility.

    Returns
    -------
    DataFrame of shape (n_days, n_stocks), business-day DatetimeIndex,
    columns SYN_1..SYN_n_stocks, containing simple daily returns y(t).
    """
    assert 0 < alpha < 1, "alpha must be in (0, 1) per Section 4.1"
    assert sigma > 0, "sigma must be positive"
    assert 1 <= shift_day < n_days, "shift_day must fall strictly inside the sample"

    rng = np.random.default_rng(seed)
    if mu is None:
        mu = rng.normal(loc=0.0, scale=0.01, size=n_stocks) * 0.01  # ~N(0, 1e-4)
    mu = np.asarray(mu, dtype=float)
    assert mu.shape == (n_stocks,)

    y = np.zeros((n_days, n_stocks))
    eps = rng.normal(loc=0.0, scale=sigma, size=(n_days, n_stocks))

    for t in range(n_days):
        j_t = j_before if t < shift_day else j_after
        # Use all available history when t < j_t (burn-in), matching the
        # data_requirements note that an initial history of max(N) returns is
        # needed before rolling *estimators* (not the DGP itself) can be
        # computed; the DGP recursion is well-defined for any t>=1 using the
        # available lookback.
        lookback = min(t, j_t)
        if lookback == 0:
            mean_term = np.zeros(n_stocks)
        else:
            mean_term = y[t - lookback:t].mean(axis=0)
        y[t] = mu + alpha * (mean_term - mu) + eps[t]

    dates = pd.bdate_range(start=start_date, periods=n_days)
    cols = [f"SYN_{i+1}" for i in range(n_stocks)]
    return pd.DataFrame(y, index=dates, columns=cols)


def regime_schedule(n_days: int, shift_day: int, j_before: int, j_after: int) -> np.ndarray:
    """Return the array of target window sizes j_t used at each day (0-indexed)."""
    sched = np.full(n_days, j_after, dtype=int)
    sched[:shift_day] = j_before
    return sched
