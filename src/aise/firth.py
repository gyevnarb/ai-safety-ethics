"""Firth's penalized logistic regression with profile-likelihood intervals.

Firth's penalty (the Jeffreys prior, 0.5 log det of the Fisher information) removes the
first-order small-sample bias of maximum likelihood and keeps estimates finite under
separation, which matters when events are rare and some predictors have few or no
events (Heinze & Schemper 2002, Statistics in Medicine 21:2409). Wald intervals are
unreliable in that setting, so intervals and tests use the penalized profile likelihood,
as R's logistf does.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


@dataclass
class _Fit:
    beta: np.ndarray
    loglik: float  # penalized log likelihood
    cov: np.ndarray  # inverse Fisher information at beta


def _fit(X: np.ndarray, y: np.ndarray, fixed: dict[int, float] | None = None,
         beta: np.ndarray | None = None, max_iter: int = 200, tol: float = 1e-9) -> _Fit:
    """Maximize the penalized log likelihood, holding the coefficients in fixed."""
    fixed = fixed or {}
    beta = np.zeros(X.shape[1]) if beta is None else beta.copy()
    for j, v in fixed.items():
        beta[j] = v
    free = np.array([j not in fixed for j in range(X.shape[1])])

    def evaluate(b):
        eta = X @ b
        p = special.expit(eta)
        w = p * (1 - p)
        info = X.T @ (X * w[:, None])
        sign, logdet = np.linalg.slogdet(info)
        ll = (np.sum(y * special.log_expit(eta) + (1 - y) * special.log_expit(-eta))
              + 0.5 * logdet) if sign > 0 else -np.inf
        return p, w, info, ll

    p, w, info, ll = evaluate(beta)
    for _ in range(max_iter):
        cov = np.linalg.inv(info)
        h = w * np.einsum("ij,jk,ik->i", X, cov, X)  # leverages
        score = X.T @ (y - p + h * (0.5 - p))  # Firth-modified score
        step = np.zeros_like(beta)
        step[free] = np.linalg.solve(info[np.ix_(free, free)], score[free])
        step *= min(1.0, 5.0 / max(np.abs(step).max(), 1e-12))  # cap large steps
        for _ in range(30):  # step halving until the penalized likelihood improves
            new = beta + step
            p_new, w_new, info_new, ll_new = evaluate(new)
            if ll_new >= ll - 1e-12:
                break
            step /= 2
        converged = np.abs(new - beta).max() < tol
        beta, p, w, info, ll = new, p_new, w_new, info_new, ll_new
        if converged:
            break
    return _Fit(beta, ll, np.linalg.inv(info))


def firth_logit(X: pd.DataFrame, y: pd.Series, level: float = 0.95) -> pd.DataFrame:
    """Coefficients, profile-likelihood intervals and penalized likelihood-ratio p-values.

    X must include an intercept column if one is wanted.
    """
    Xa, ya = X.to_numpy(float), y.to_numpy(float)
    full = _fit(Xa, ya)
    crit = stats.chi2.ppf(level, 1) / 2
    rows = []
    for j, name in enumerate(X.columns):
        b, se = full.beta[j], np.sqrt(full.cov[j, j])

        def drop(v, j=j):  # > 0 once the profile likelihood at beta_j = v falls below
            return full.loglik - crit - _fit(Xa, ya, {j: v}, full.beta).loglik  # the cut

        bounds = []
        for direction in (-1, 1):
            width, end = 2 * se, np.nan
            for _ in range(12):  # widen the bracket until the profile crosses crit
                if drop(b + direction * width) > 0:
                    end = optimize.brentq(drop, b, b + direction * width, xtol=1e-6)
                    break
                width *= 2
            bounds.append(end if not np.isnan(end) else direction * np.inf)
        lr = 2 * (full.loglik - _fit(Xa, ya, {j: 0.0}, full.beta).loglik)
        rows.append({"term": name, "coef": b, "se": se, "lo": bounds[0], "hi": bounds[1],
                     "p": stats.chi2.sf(max(lr, 0.0), 1)})
    return pd.DataFrame(rows).set_index("term")


def holm(p: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    order = p.sort_values().index
    m = len(p)
    adj = np.minimum(1, np.maximum.accumulate(
        [(m - i) * p[t] for i, t in enumerate(order)]))
    return pd.Series(adj, index=order).reindex(p.index)
