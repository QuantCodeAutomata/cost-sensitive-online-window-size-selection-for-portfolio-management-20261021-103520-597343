"""Two-level cost-sensitive online aggregation: Hedge and Fixed Share updates
over window-size experts, with the optimal learning-rate / sharing-rate
schedule of Proposition 3.5.

Custom implementation — Context7/library survey found no package implementing
cost-sensitive Hedge/Fixed-Share tracking-regret aggregation with a turnover
penalty (this is the paper's own Problem 2.1-2.4 / Proposition 3.5
construction); standard "expert aggregation" packages (e.g. in scikit-learn,
vectorbt) do not expose Hedge/Fixed-Share weight updates with a turnover-cost
loss, so the exponential-weights and sharing-step recursions are implemented
directly from the paper's equations.
"""
from __future__ import annotations

import numpy as np


def binary_entropy(x: float) -> float:
    """H(x) = -x*log(x) - (1-x)*log(1-x), with H(0)=H(1)=0 by continuity."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log(x) - (1 - x) * np.log(1 - x)


def compute_C_T(T: int, c: float) -> float:
    """C_T, the cumulative worst-case turnover-cost contribution over the
    horizon (implementation notes: C_T=0 reduces to the frictionless case).
    We use C_T = c * (T - 1), the maximal possible turnover cost accumulated
    by the L1 penalty term across T-1 transitions (each bounded by c * 2,
    i.e. a full simplex reallocation at each step contributes at most 2c);
    C_T=0 iff c=0.
    """
    return 0.0 if c == 0 else c * (T - 1)


def compute_A_T(T: int, a: float, b: float, c: float) -> float:
    """A_T per the implementation notes for the C_T>0, K=0 case:
    A_T = sum_t (b-a+2c)^2/8 + sum_{t>=2} c*(b-a+2c*(t-1))
    """
    term1 = T * ((b - a + 2 * c) ** 2) / 8.0
    if T >= 2:
        t_idx = np.arange(2, T + 1)
        term2 = np.sum(c * (b - a + 2 * c * (t_idx - 1)))
    else:
        term2 = 0.0
    return term1 + term2


def optimal_eta_alpha(T: int, K: int, n: int, c: float, a: float = -1.0, b: float = 1.0) -> tuple[float, float]:
    """Optimal (eta, alpha) per Proposition 3.5.

    - If C_T == 0 (c == 0): alpha* = K/(T-1), eta* = (1/(b-a)) *
      sqrt(8/T * ((K+1)*log(n) + (T-1)*H(K/(T-1)))).
    - If C_T > 0 and K == 0: alpha* = 0, eta* = sqrt(log(n)/A_T).
    - If C_T > 0 and K >= 1: no closed form is given; we use the natural
      interpolation alpha* = K/(T-1) (as in the frictionless case) together
      with eta* = sqrt(log(n)/A_T) (replacing the frictionless per-step
      bound (b-a) with the cost-inflated bound), since both reduce to the
      stated formulas in the K=0 and C_T=0 limits respectively.
    """
    assert T > 1, "T must exceed 1 for Proposition 3.5"
    assert 0 <= K <= T - 1, "switching budget K must satisfy 0<=K<=T-1"
    C_T = compute_C_T(T, c)

    if C_T == 0.0:
        alpha_star = K / (T - 1)
        H = binary_entropy(alpha_star)
        eta_star = (1.0 / (b - a)) * np.sqrt(8.0 / T * ((K + 1) * np.log(n) + (T - 1) * H))
        return eta_star, alpha_star

    A_T = compute_A_T(T, a, b, c)
    if K == 0:
        return np.sqrt(np.log(n) / A_T), 0.0

    alpha_star = K / (T - 1)
    eta_star = np.sqrt(np.log(n) / A_T)
    return eta_star, alpha_star


def default_switching_budget(T: int, epsilon: float = 0.5) -> int:
    """K(T) = floor( T / ( log(T) * (log(log(T)))^{1+epsilon} ) ), used for the
    empirical studies (Appendix B.4/B.5). epsilon is not numerically specified
    in the paper beyond 'epsilon>0'; epsilon=0.5 is used as a conservative
    default (assumed) and is configurable.
    """
    assert T > np.e ** np.e, "T too small for log(log(T)) to be defined/positive"
    log_T = np.log(T)
    loglog_T = np.log(log_T)
    denom = log_T * (loglog_T ** (1 + epsilon))
    return max(0, int(np.floor(T / denom)))


class OnlineAggregator:
    """Hedge (alpha=0) / Fixed Share (alpha>0) weight updates over n experts.

    q(1) = 1/n (uniform). At each step:
      ell_tilde_j(t) = -w_j(t)^T y(t) + c*||w_j(t)-w_j(t-1)||_1   (Negative PnL + turnover)
      q_j^exp(t) = q_j(t) exp(-eta * ell_tilde_j(t)) / sum_k q_k(t) exp(-eta * ell_tilde_k(t))
      q_j(t+1)   = (1-alpha) q_j^exp(t) + alpha/n                  (alpha=0 recovers Hedge)
    """

    def __init__(self, n_experts: int, eta: float, alpha: float):
        assert n_experts > 0
        assert eta > 0
        assert 0 <= alpha <= 1
        self.n = n_experts
        self.eta = eta
        self.alpha = alpha
        self.q = np.full(n_experts, 1.0 / n_experts)

    def step(self, losses: np.ndarray) -> np.ndarray:
        """Update q given the realized per-expert losses ell_tilde(t); returns
        the aggregation weight vector q(t) used to form this period's
        aggregate portfolio (i.e. the pre-update q used to weight advice),
        while internally advancing self.q to q(t+1) via log-sum-exp-stable
        exponential weighting followed by the sharing step."""
        q_t = self.q.copy()
        # log-sum-exp trick for numerical stability
        log_w = np.log(q_t + 1e-300) - self.eta * losses
        log_w -= log_w.max()
        w = np.exp(log_w)
        q_exp = w / w.sum()
        self.q = (1 - self.alpha) * q_exp + self.alpha / self.n
        return q_t
