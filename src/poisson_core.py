"""Core mathematics: crash episodes, Poisson probabilities, exact confidence intervals, dispersion test,
compound-Poisson moments and a two-state Hamilton (Markov-switching) model.  Pure functions, unit-tested in tests/."""
import math
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- crash episodes
def episodes(level: pd.Series, threshold: float):
    """Peak-to-recovery episodes in a price/total-return index sampled at month-end.
    An episode starts at a running peak, is counted once the fall from that peak reaches `threshold`,
    and ends when the index regains the peak.  Returns a list of dicts."""
    out, peak, pdate, in_ev = [], level.iloc[0], level.index[0], False
    trough = tdate = cross = None
    for d, v in level.items():
        if not in_ev:
            if v >= peak:
                peak, pdate = v, d
            elif v / peak - 1 <= -threshold:
                in_ev, trough, tdate, cross = True, v, d, d
        else:
            if v < trough:
                trough, tdate = v, d
            if v >= peak:
                out.append(dict(peak=pdate, cross=cross, trough=tdate, recovered=d, depth=trough / peak - 1))
                in_ev, peak, pdate = False, v, d
    if in_ev:
        out.append(dict(peak=pdate, cross=cross, trough=tdate, recovered=None, depth=trough / peak - 1))
    return out


def max_drawdown(level: pd.Series) -> float:
    return float((level / level.cummax() - 1).min())


# ---------------------------------------------------------------- Poisson
def pmf(k, lam_t):
    """P(N = k) = e^(-lam t) (lam t)^k / k!"""
    return math.exp(-lam_t) * lam_t ** k / math.factorial(k)


def p_at_least_one(lam, t):
    return 1 - math.exp(-lam * t)


def garwood_ci(n, T, conf=0.90):
    """Exact (Garwood 1936) confidence interval for a Poisson rate from n events in T years."""
    a = 1 - conf
    lo = stats.chi2.ppf(a / 2, 2 * n) / (2 * T) if n > 0 else 0.0
    hi = stats.chi2.ppf(1 - a / 2, 2 * n + 2) / (2 * T)
    return lo, hi


def dispersion_test(counts):
    """Index of dispersion: under Poisson, (n-1) s^2 / mean ~ chi2(n-1).  Returns D = var/mean and the p-value
    for over-dispersion (clustering)."""
    c = np.asarray(counts, float); n = len(c); m = c.mean(); v = c.var(ddof=1)
    stat = (n - 1) * v / m
    return dict(mean=m, var=v, D=v / m, chi2=stat, df=n - 1, p_overdispersed=float(1 - stats.chi2.cdf(stat, n - 1)))


def summarise_rate(n, T, windows):
    lam = n / T; lo, hi = garwood_ci(n, T)
    row = dict(n=n, years=T, lam=lam, lam_ci90=[lo, hi], mean_wait_years=(1 / lam if lam > 0 else None))
    row["p_at_least_one"] = {w: [p_at_least_one(lam, t), p_at_least_one(lo, t), p_at_least_one(hi, t)] for w, t in windows.items()}
    row["pmf_6y"] = [pmf(k, 6 * lam) for k in range(3)] + [1 - sum(pmf(k, 6 * lam) for k in range(3))]
    return row


def compound_moments(lam_t, loss_samples):
    """Aggregate loss S = L_1 + ... + L_N with N ~ Poisson(lam t):  E[S] = lam t E[L],  Var[S] = lam t E[L^2]."""
    L = np.asarray(loss_samples, float)
    return lam_t * L.mean(), lam_t * (L ** 2).mean()


# ---------------------------------------------------------------- Markov switching (Hamilton 1989), fitted by EM
def hamilton_em(x, iters=300):
    x = np.asarray(x, float)
    mu = np.array([0.01, -0.01]); sd = np.array([0.035, 0.08])
    P = np.array([[0.97, 0.03], [0.10, 0.90]]); pi = np.array([0.8, 0.2])
    for _ in range(iters):
        B = stats.norm.pdf(x[:, None], mu, sd)
        a = np.zeros((len(x), 2)); c = np.zeros(len(x))
        a[0] = pi * B[0]; c[0] = a[0].sum(); a[0] /= c[0]
        for t in range(1, len(x)):                       # forward filter = Bayes' theorem each month
            a[t] = (a[t - 1] @ P) * B[t]; c[t] = a[t].sum(); a[t] /= c[t]
        b = np.ones((len(x), 2))
        for t in range(len(x) - 2, -1, -1):
            b[t] = (P @ (B[t + 1] * b[t + 1])) / c[t + 1]
        g = a * b; g /= g.sum(1, keepdims=True)
        xi = (a[:-1, :, None] * P[None] * (B[1:] * b[1:])[:, None, :]) / c[1:, None, None]
        P = xi.sum(0) / xi.sum(0).sum(1, keepdims=True); pi = g[0]
        mu = (g * x[:, None]).sum(0) / g.sum(0)
        sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / g.sum(0))
    if sd[0] > sd[1]:                                     # state 0 = calm, state 1 = turbulent
        mu, sd, P, g, a = mu[::-1], sd[::-1], P[::-1, ::-1], g[:, ::-1], a[:, ::-1]
    stationary = np.linalg.solve(np.vstack([(P.T - np.eye(2))[0], [1, 1]]), [0, 1])
    return dict(mu=mu, sd=sd, P=P, smoothed=g, filtered=a, stationary=stationary, loglik=float(np.log(c).sum()))


def mmpp_p_at_least_one(P, lam_calm, lam_turb, p_turb0, years, n=200_000, seed=7):
    """Markov-modulated Poisson: crash rate depends on the regime, regime follows the monthly chain P."""
    rng = np.random.default_rng(seed)
    s = (rng.random(n) < p_turb0).astype(int); hit = np.zeros(n, bool); out = []
    for m in range(int(round(12 * years))):
        lam = np.where(s == 1, lam_turb, lam_calm) / 12
        hit |= rng.random(n) < 1 - np.exp(-lam)
        s = np.where(rng.random(n) < P[s, 1], 1, 0)
        out.append(hit.mean())
    return out                                            # cumulative P(>=1) after each month
