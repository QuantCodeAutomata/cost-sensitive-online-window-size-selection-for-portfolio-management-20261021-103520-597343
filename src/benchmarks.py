"""Online portfolio-selection benchmark algorithms (Appendix B.3): Universal
Portfolio (UP, Cover 1991), Exponentiated Gradient (EG, Helmbold et al. 1998),
Passive-Aggressive Mean Reversion (PAMR, Li et al. 2012), Confidence-Weighted
Mean Reversion (CWMR, Li et al. 2013), and Online Moving Average Reversion
(OLMAR, Li & Hoi 2014).

Custom implementation — Context7 survey of vectorbt/skfolio/pyportfolioopt
found no package implementing this specific family of "OLPS" (Online
Portfolio Selection) algorithms; they are implemented directly from their
respective foundational papers with the theoretically prescribed default
parameters (not tuned on this data), as directed by the paper's methodology
notes for Appendix B.3/B.4/B.5.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (Duchi et al. 2008),
    used by PAMR/CWMR/OLMAR update steps that are not natively constrained."""
    n = v.shape[0]
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - css / ind > 0
    rho = ind[cond][-1]
    theta = css[cond][-1] / rho
    return np.clip(v - theta, 0, None)


class UniversalPortfolio:
    """Cover's (1991) Universal Portfolio, approximated by Monte Carlo
    sampling over the simplex (Dirichlet(1,...,1) proposal), weighting
    samples by their historical cumulative wealth (the UP integral has no
    closed form for m>2 assets)."""

    def __init__(self, m: int, n_samples: int = 2000, seed: int = 0):
        self.m = m
        rng = np.random.default_rng(seed)
        self.samples = rng.dirichlet(np.ones(m), size=n_samples)
        self.wealth = np.ones(n_samples)

    def advice(self) -> np.ndarray:
        weights = self.wealth / self.wealth.sum()
        return weights @ self.samples

    def update(self, x: np.ndarray):
        """x: gross price relative vector (1+y) for the realized period."""
        self.wealth *= self.samples @ x


class ExponentiatedGradient:
    """EG (Helmbold, Schapire, Singer & Warmuth, 1998): multiplicative weight
    update w_i(t+1) ∝ w_i(t) exp(eta * x_i(t) / (w(t)^T x(t)))."""

    def __init__(self, m: int, eta: float = 0.05):
        self.w = np.full(m, 1.0 / m)
        self.eta = eta

    def advice(self) -> np.ndarray:
        return self.w

    def update(self, x: np.ndarray):
        ret = self.w @ x
        grad = x / ret
        log_w = np.log(self.w + 1e-300) + self.eta * grad
        log_w -= log_w.max()
        w = np.exp(log_w)
        self.w = w / w.sum()


class PAMR:
    """Passive-Aggressive Mean Reversion (Li, Hoi & Gopalkrishnan, 2012),
    PAMR-0 variant with sensitivity parameter epsilon."""

    def __init__(self, m: int, epsilon: float = 0.5):
        self.w = np.full(m, 1.0 / m)
        self.epsilon = epsilon

    def advice(self) -> np.ndarray:
        return self.w

    def update(self, x: np.ndarray):
        m = len(x)
        x_mean = x.mean()
        loss = max(0.0, self.w @ x - self.epsilon)
        denom = np.sum((x - x_mean) ** 2)
        tau = loss / denom if denom > 1e-12 else 0.0
        w_new = self.w - tau * (x - x_mean)
        self.w = _project_to_simplex(w_new)


class CWMR:
    """Confidence-Weighted Mean Reversion (Li, Hoi, Zhao & Gopalkrishnan,
    2013), diagonal-covariance variant with confidence level phi (via
    Gaussian quantile) and sensitivity epsilon."""

    def __init__(self, m: int, epsilon: float = 0.5, confidence: float = 0.95, var0: float = 1.0):
        self.m = m
        self.w = np.full(m, 1.0 / m)
        self.sigma = np.full(m, var0)  # diagonal covariance
        self.phi = norm.ppf(confidence)
        self.epsilon = epsilon

    def advice(self) -> np.ndarray:
        return self.w

    def update(self, x: np.ndarray):
        """Diagonal-covariance CWMR-Var update (Li, Hoi, Zhao & Gopalkrishnan,
        2013, Sec. 3.2): closed-form Lagrange multiplier lambda solving the
        KL-projected confidence update, with mean/variance step.
        """
        mu, sigma, phi, eps = self.w, self.sigma, self.phi, self.epsilon
        x_mean = x.mean()
        dx = x - x_mean
        M = mu @ x
        V = sigma @ dx ** 2
        if V > 1e-12 and phi > 0:
            disc = max((1 + 2 * phi * M) ** 2 - 8 * phi * (M - eps - phi * V), 0.0)
            lam = max(0.0, (-(1 + 2 * phi * M) + np.sqrt(disc)) / (4 * phi * V))
        else:
            lam = 0.0
        mu_new = mu - lam * sigma * dx
        sigma_new = 1.0 / (1.0 / sigma + 2 * lam * phi * dx ** 2 + 1e-12)
        self.w = _project_to_simplex(mu_new)
        self.sigma = np.clip(sigma_new, 1e-6, None)


class OLMAR:
    """Online Moving Average Reversion (Li & Hoi, 2014), using an SMA
    price-relative predictor with window `window` and sensitivity
    `epsilon`."""

    def __init__(self, m: int, window: int = 5, epsilon: float = 10.0):
        self.w = np.full(m, 1.0 / m)
        self.window = window
        self.epsilon = epsilon
        self.price_history: list[np.ndarray] = []

    def advice(self) -> np.ndarray:
        return self.w

    def update(self, x: np.ndarray):
        """x: gross price relative for the just-realized period (close_t/close_{t-1})."""
        self.price_history.append(x)
        if len(self.price_history) > self.window:
            self.price_history.pop(0)
        k = len(self.price_history)
        # Predicted next-period price relative via SMA of price ratios
        # relative to the current price level (Li & Hoi, 2014, Eq. 8):
        # x_pred_i = (1/k) * sum_{i=0}^{k-1} 1 / prod_{j=0}^{i-1} x(t-j),
        # i.e. the average of {1, 1/x(t), 1/(x(t)x(t-1)), ...}.
        inv_cum_prod = np.ones_like(x)
        preds = [inv_cum_prod.copy()]
        for i in range(1, k):
            inv_cum_prod = inv_cum_prod / self.price_history[-i]
            preds.append(inv_cum_prod.copy())
        x_pred = np.mean(preds, axis=0)

        ell = max(0.0, self.epsilon - x_pred @ self.w)
        x_bar = x_pred.mean()
        denom = np.sum((x_pred - x_bar) ** 2)
        lam = ell / denom if denom > 1e-12 else 0.0
        w_new = self.w + lam * (x_pred - x_bar)
        self.w = _project_to_simplex(w_new)
