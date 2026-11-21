"""
Small statistics toolkit used by every analysis script in reproduce/.

Written with numpy and scipy only, so the analysis runs anywhere the benchmark
runs, including Windows on ARM where statsmodels wheels are not always
available. Every function is deterministic: the same input always gives the
same output, to the last bit, on any machine with the same numpy version.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def ols(y, columns, names):
    """
    Ordinary least squares with the quantities a full residual critique needs.

    columns: list of 1-D arrays, the first normally a column of ones.
    """
    X = np.column_stack([np.asarray(c, dtype=float) for c in columns])
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    fitted = X @ beta
    resid = y - fitted
    rss = float(resid @ resid)
    df = n - p
    s2 = rss / df
    se = np.sqrt(np.diag(xtx_inv) * s2)
    t = beta / se
    p_values = 2 * stats.t.sf(np.abs(t), df)
    t_crit = stats.t.ppf(0.975, df)
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    r_std = resid / np.sqrt(s2 * (1 - leverage))
    cooks = r_std ** 2 * leverage / (p * (1 - leverage))
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    adj_r2 = 1 - (1 - r2) * (n - 1) / df if df > 0 else float("nan")
    aic = n * np.log(rss / n) + 2 * p
    coefs = {
        name: {
            "estimate": float(b), "se": float(s), "t": float(tv), "p": float(pv),
            "ci_low": float(b - t_crit * s), "ci_high": float(b + t_crit * s),
        }
        for name, b, s, tv, pv in zip(names, beta, se, t, p_values)
    }
    return {
        "coefs": coefs, "rss": rss, "df": df, "n": n, "p": p,
        "r2": float(r2), "adj_r2": float(adj_r2), "aic": float(aic),
        "resid": resid, "fitted": fitted, "leverage": leverage,
        "r_std": r_std, "cooks": cooks, "X": X,
    }


def nested_f(small, big):
    """F-test that the extra terms in `big` are zero."""
    extra = small["df"] - big["df"]
    f = ((small["rss"] - big["rss"]) / extra) / (big["rss"] / big["df"])
    return float(f), float(stats.f.sf(f, extra, big["df"]))


def breusch_pagan(model):
    """Koenker's studentised Breusch-Pagan test on the model's own regressors."""
    u = model["resid"] ** 2
    X = model["X"]
    gamma = np.linalg.lstsq(X, u, rcond=None)[0]
    r2 = 1 - ((u - X @ gamma) ** 2).sum() / ((u - u.mean()) ** 2).sum()
    lm = model["n"] * r2
    return float(lm), float(stats.chi2.sf(lm, model["p"] - 1))


def durbin_watson(resid):
    resid = np.asarray(resid, dtype=float)
    return float((np.diff(resid) ** 2).sum() / (resid ** 2).sum())


def residual_checks(model):
    """The numeric half of a residual critique; plots are drawn by the caller."""
    sw = stats.shapiro(model["r_std"]) if model["n"] >= 3 else (np.nan, np.nan)
    bp = breusch_pagan(model)
    worst = int(np.argmax(model["cooks"]))
    return {
        "shapiro_w": float(sw[0]), "shapiro_p": float(sw[1]),
        "breusch_pagan_lm": bp[0], "breusch_pagan_p": bp[1],
        "durbin_watson": durbin_watson(model["resid"]),
        "max_cooks_d": float(model["cooks"][worst]), "max_cooks_index": worst,
        "max_leverage": float(model["leverage"].max()),
    }


def paired_mean_ci(x):
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    t_crit = stats.t.ppf(0.975, len(x) - 1)
    return {"mean": float(mean), "ci_low": float(mean - t_crit * se),
            "ci_high": float(mean + t_crit * se), "n": int(len(x))}


def tost_ratio(treated, control, margin=0.01):
    """
    Two one-sided tests for equivalence of two group means, as a ratio.

    Equivalent when the 90% CI of (treated/control - 1) lies inside
    [-margin, +margin]. Welch standard error, delta method for the ratio.
    """
    a = np.asarray(treated, dtype=float)
    b = np.asarray(control, dtype=float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    diff = ma - mb
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    bound = margin * mb
    p_low = stats.t.sf((diff + bound) / se, df)
    p_high = stats.t.cdf((diff - bound) / se, df)
    t90 = stats.t.ppf(0.95, df)
    return {
        "ratio": float(ma / mb),
        "ci90_low": float((diff - t90 * se) / mb + 1),
        "ci90_high": float((diff + t90 * se) / mb + 1),
        "margin": margin,
        "p_tost": float(max(p_low, p_high)),
        "equivalent": bool(max(p_low, p_high) < 0.05),
    }


def poisson_rate_glm(counts, exposure, design, names):
    """
    Poisson GLM with log link and log(exposure) offset, fitted by IRLS.

    Returns rate ratios with both Poisson and quasi-Poisson 95% CIs, the
    dispersion estimate and the likelihood-ratio test against the
    intercept-only model.
    """
    y = np.asarray(counts, dtype=float)
    offset = np.log(np.asarray(exposure, dtype=float))
    X = np.column_stack([np.asarray(c, dtype=float) for c in design])

    def fit(Xm):
        beta = np.zeros(Xm.shape[1])
        beta[0] = np.log(y.sum() / np.exp(offset).sum())
        for _ in range(100):
            mu = np.exp(Xm @ beta + offset)
            z = Xm @ beta + (y - mu) / mu
            new = np.linalg.solve(Xm.T @ (mu[:, None] * Xm), Xm.T @ (mu * z))
            if np.max(np.abs(new - beta)) < 1e-12:
                beta = new
                break
            beta = new
        mu = np.exp(Xm @ beta + offset)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(y / mu), 0.0)
        deviance = float(2 * np.sum(term - (y - mu)))
        pearson = float(((y - mu) ** 2 / mu).sum())
        cov = np.linalg.inv(Xm.T @ (mu[:, None] * Xm))
        return beta, cov, deviance, pearson

    beta, cov, dev, pearson = fit(X)
    _, _, dev0, _ = fit(X[:, :1])
    df_resid = len(y) - X.shape[1]
    phi = pearson / df_resid
    se = np.sqrt(np.diag(cov))
    se_q = se * np.sqrt(phi)
    lr = dev0 - dev
    k = X.shape[1] - 1
    f_quasi = (lr / k) / phi if k else float("nan")
    out = {
        "n": int(len(y)), "deviance": dev, "null_deviance": dev0,
        "dispersion": float(phi),
        "lr_chi2": float(lr), "lr_p": float(stats.chi2.sf(lr, k)) if k else float("nan"),
        "quasi_f": float(f_quasi),
        "quasi_f_p": float(stats.f.sf(f_quasi, k, df_resid)) if k else float("nan"),
        "terms": {},
    }
    for name, b, s, sq in zip(names, beta, se, se_q):
        out["terms"][name] = {
            "rate_ratio": float(np.exp(b)),
            "ci_low": float(np.exp(b - 1.96 * s)), "ci_high": float(np.exp(b + 1.96 * s)),
            "quasi_ci_low": float(np.exp(b - 1.96 * sq)),
            "quasi_ci_high": float(np.exp(b + 1.96 * sq)),
        }
    return out


def coefs_table(model):
    """Plain-text table of an OLS fit, for console output."""
    lines = [f"{'term':<16}{'estimate':>11}{'95% CI':>24}{'p':>9}"]
    for name, c in model["coefs"].items():
        ci = f"{c['ci_low']:+.4f} to {c['ci_high']:+.4f}"
        lines.append(f"{name:<16}{c['estimate']:>+11.4f}{ci:>24}{c['p']:>9.4f}")
    lines.append(f"R2 {model['r2']:.3f}  adj R2 {model['adj_r2']:.3f}  "
                 f"AIC {model['aic']:.2f}  n {model['n']}")
    return "\n".join(lines)
