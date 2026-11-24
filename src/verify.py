"""Independent verification. Recomputes every headline number a second way (different algorithm or closed form) and
compares it with results/*.json, and checks the case-study arithmetic against the printed tables in the competition guide
(WGHS 2026-27 Team Study Pack, sections 3.1-3.2).  Exit code 1 if any check fails.  Output: results/verification.json"""
import json
import math
import numpy as np
import pandas as pd
from scipy import stats, optimize
from config import PROC, RES, HOLDINGS, CASH, INITIAL, TOP_UP, PAYMENT, N_PAYMENTS, FUNDS, EQUITY

checks = []


def chk(group, name, got, want, tol, note=""):
    ok = bool(abs(got - want) <= tol)
    checks.append(dict(group=group, check=name, computed=float(got), reference=float(want), tolerance=tol, ok=ok, note=note))


# ------------------------------------------------------------------ A. case-study arithmetic vs the competition guide
def reserve(r, n=N_PAYMENTS):       # annuity-due present value at the start of 2033
    return PAYMENT * n if r == 0 else PAYMENT * (1 - (1 + r) ** -n) / r * (1 + r)


GUIDE_RESERVE = {0.00: 500000, 0.02: 458112, 0.03: 439305, 0.04: 421767, 0.05: 405391, 0.06: 390085}
for r, v in GUIDE_RESERVE.items():
    chk("A guide s3.1", f"reserve at {r:.0%}", reserve(r), v, 1, "annuity-due, n=10")
GUIDE_PORT = {0.00: 450000, 0.02: 503461, 0.03: 532107, 0.04: 562094, 0.05: 593471, 0.06: 626290, 0.07: 660602, 0.08: 696462}
for r, v in GUIDE_PORT.items():
    chk("A guide s3.2", f"portfolio at start of 2033, {r:.0%}", INITIAL * (1 + r) ** 6 + TOP_UP * (1 + r) ** 5, v, 1)
GUIDE_REM = {0.00: 28233, 0.02: 81694, 0.03: 110340, 0.04: 140327, 0.05: 171704, 0.06: 204523, 0.07: 238835, 0.08: 274695}
for r, v in GUIDE_REM.items():
    chk("A guide s3.2", f"remaining after $421,767 reserve, {r:.0%}", INITIAL * (1 + r) ** 6 + TOP_UP * (1 + r) ** 5 - 421767, v, 1)
for r, v in {0.03: 501600, 0.05: 538300, 0.07: 577000}.items():
    chk("A guide s3.3", f"portfolio at start of 2031, {r:.0%}", INITIAL * (1 + r) ** 4 + TOP_UP * (1 + r) ** 3, v, 100, "guide says 'about'")
chk("A holdings", "holdings + cash = $300,000", sum(HOLDINGS.values()) + CASH, INITIAL, 0.005)
chk("A holdings", "equity share (VTI+VXUS+EQAL)", sum(HOLDINGS[f] for f in EQUITY) / INITIAL, 0.39683816666666666, 1e-9)
chk("A guide s3.1", "gap between $500,000 and the 4% reserve is about $78,000", 500000 - reserve(0.04), 78000, 300, "guide: 'roughly $78,000'")

# ------------------------------------------------------------------ B. market-wide Poisson
E = json.load(open(RES / "crash_events.json")); L = json.load(open(RES / "laura_poisson.json"))
r = pd.read_csv(PROC / "us_market_long.csv", index_col=0, parse_dates=True).iloc[:, 0]
lv = np.concatenate([[1.0], (1 + r.values).cumprod()]); T = (len(lv) - 1) / 12


def count_falls(level, thr):
    """Second algorithm: running peak vectorised; an episode = a maximal run of months below the peak whose minimum
    drawdown reaches the threshold."""
    peak = np.maximum.accumulate(level); dd = level / peak - 1; below = dd < 0
    n, i, N = 0, 0, len(level)
    while i < N:
        if below[i]:
            j = i
            while j < N and below[j]:
                j += 1
            if dd[i:j].min() <= -thr:
                n += 1
            i = j
        else:
            i += 1
    return n


n20, n30 = count_falls(lv, 0.20), count_falls(lv, 0.30)
chk("B market", "years of history", T, E["years"], 1e-9)
chk("B market", "number of falls >= 20%", n20, E["equity"]["20"]["n"], 0, "independent episode counter")
chk("B market", "number of falls >= 30%", n30, E["equity"]["30"]["n"], 0, "independent episode counter")
lam = n20 / T
chk("B market", "lambda = n/T (20%)", lam, E["equity"]["20"]["lam"], 1e-12)
chk("B market", "lambda = n/T (30%)", n30 / T, E["equity"]["30"]["lam"], 1e-12)
for key, t in {"2027 (first year)": 1, "2031-32 (critical years)": 2, "2027-32 (build-up)": 6, "2027-42 (whole plan)": 16}.items():
    chk("B market", f"P(>=1) {key}: 1-exp(-lt)", 1 - math.exp(-lam * t), E["equity"]["20"]["p_at_least_one"][key][0], 1e-9)
    chk("B market", f"P(>=1) {key}: scipy poisson.sf(0)", stats.poisson.sf(0, lam * t), E["equity"]["20"]["p_at_least_one"][key][0], 1e-9)
chk("B market", "binomial limit, 24 months p=lam/12 vs Poisson 2 years", 1 - (1 - lam / 12) ** 24, 1 - math.exp(-lam * 2), 1e-3, "Poisson = limit of binomial: 18.14% vs 18.07%")
n = n20
lo = stats.chi2.ppf(0.05, 2 * n) / (2 * T); hi = stats.chi2.ppf(0.95, 2 * (n + 1)) / (2 * T)
chk("B market", "Garwood 90% lower bound (chi-square)", lo, E["equity"]["20"]["lam_ci90"][0], 1e-9)
chk("B market", "Garwood 90% upper bound (chi-square)", hi, E["equity"]["20"]["lam_ci90"][1], 1e-9)
# Garwood by root-finding on the Poisson cdf itself (no chi-square shortcut)
f_hi = optimize.brentq(lambda m: stats.poisson.cdf(n, m) - 0.05, 1e-6, 100) / T
f_lo = optimize.brentq(lambda m: stats.poisson.sf(n - 1, m) - 0.05, 1e-6, 100) / T
chk("B market", "Garwood upper bound by inverting the Poisson cdf", f_hi, hi, 1e-6)
chk("B market", "Garwood lower bound by inverting the Poisson cdf", f_lo, lo, 1e-6)
yr = pd.Series(0, index=range(1926, 2027))   # dispersion test of yearly counts
ev = E["equity"]["20"]["episodes"]
for e in ev:
    yr[int(e["peak"][:4])] += 1
D = yr.var(ddof=1) / yr.mean()
chk("B market", "dispersion index var/mean", D, E["equity"]["20"]["dispersion"]["D"], 0.01)
chk("B market", "dispersion test p (chi-square, df=N-1)", stats.chi2.sf((len(yr) - 1) * D, len(yr) - 1), E["equity"]["20"]["dispersion"]["p_overdispersed"], 0.01)
chk("B market", "type counts add to 10", sum(v["n"] for k, v in E["by_type"].items() if k[:2] in ("T1", "T2", "T3", "T4", "T5") and "+" not in k), 10, 0)
chk("B market", "mean wait 1/lambda (years)", 1 / lam, E["equity"]["20"]["mean_wait_years"], 1e-9)

# ------------------------------------------------------------------ C. Markov regime model
mk = E["markov"]; P = np.array(mk["P"])
w, v = np.linalg.eig(P.T); pi = np.real(v[:, np.argmax(np.real(w))]); pi /= pi.sum()
chk("C Markov", "rows of P sum to 1", float(np.abs(P.sum(1) - 1).max()), 0, 1e-12)
chk("C Markov", "stationary distribution (eigenvector) calm", pi[0], mk["stationary"][0], 1e-9)
chk("C Markov", "stationary distribution (closed form) turbulent", P[0, 1] / (P[0, 1] + P[1, 0]), mk["stationary"][1], 1e-9)
chk("C Markov", "expected calm stay = 1/(1-p00) months", 1 / (1 - P[0, 0]), mk["expected_months"][0], 1e-9)
chk("C Markov", "expected turbulent stay = 1/(1-p11) months", 1 / (1 - P[1, 1]), mk["expected_months"][1], 1e-9)
chk("C Markov", "12-month transition P^12 [0,1]", np.linalg.matrix_power(P, 12)[0, 1], mk["P12"][0][1], 1e-12)
lc, lt = mk["lam_calm"], mk["lam_turbulent"]


def mmpp(p0, yrs):
    """P(no crash) = p' prod (diag(exp(-lam/12)) P) ... month by month, then 1 - that."""
    D = np.diag([math.exp(-lc / 12), math.exp(-lt / 12)]); M = D @ P; s = np.array([1 - p0, p0])
    return 1 - float((s @ np.linalg.matrix_power(M, yrs * 12)).sum())


for lab, p0 in (("from today's state", mk["p_turbulent_now"]), ("from the long-run mix", mk["stationary"][1]), ("just after turbulence starts", 1.0)):
    for yrs in (2, 6):
        chk("C Markov", f"MMPP P(>=1) {lab}, {yrs} years", mmpp(p0, yrs), mk["mmpp_curves"][lab][yrs - 1], 0.02,
            "tolerance 2 points: the code steps the chain at a different point in the month")

# ------------------------------------------------------------------ D. fund-level Poisson
M = pd.read_csv(PROC / "monthly.csv", index_col=0, parse_dates=True)[FUNDS].dropna()
Tf = len(M) / 12
for row in L["fund_level"]["rows"]:
    f = row["fund"]; lvl = np.concatenate([[1.0], (1 + M[f].values).cumprod()])
    n_f = count_falls(lvl, 0.20); chk("D funds", f"{f} falls >= 20%", n_f, row["n20"], 0)
    chk("D funds", f"{f} lambda", n_f / Tf, row["lam20"], 1e-9)
    chk("D funds", f"{f} P(>=1) in 2031-32", 1 - math.exp(-2 * n_f / Tf), row["p20_2031_32"], 1e-9)
    chk("D funds", f"{f} worst drawdown", float((lvl / np.maximum.accumulate(lvl) - 1).min()), row["max_drawdown"], 1e-9)
    chk("D funds", f"{f} Garwood upper bound", stats.chi2.ppf(0.95, 2 * (n_f + 1)) / (2 * Tf), row["lam20_hi90"], 1e-9)
    chk("D funds", f"{f} dollars held", row["dollars"], {**HOLDINGS, "SGOV": HOLDINGS["SGOV"] + CASH}[f], 0.01, "SGOV includes the $6,319.55 cash row")
chk("D funds", "fund weights add to 100%", sum(L["weights"].values()), 1.0, 1e-9)
chk("D funds", "history length (years)", Tf, L["fund_level"]["years"], 0.01)

# ------------------------------------------------------------------ E. compound Poisson: E[S] = lambda t E[L]
bt = L["loss_given_crash"]["by_type"]
share = {t: v["share_of_crashes"] for t, v in bt.items()}
chk("E compound", "type shares add to 1", sum(share.values()), 1.0, 1e-9)
EL = sum(share[t] * bt[t]["dollars_300k_mean"] for t in bt)
chk("E compound", "mean loss per crash, $300k portfolio", EL, L["loss_given_crash"]["mean_dollars_300k"], 40.0, "stored value is a Monte Carlo mean; formula value is exact")
for t, v in bt.items():
    chk("E compound", f"{t[:2]} fund dollars add to portfolio loss", sum(v["fund_dollars_300k"].values()), v["dollars_300k_mean"], 1.0)
lam_c = L["lambda"]["ge20"]
exp_mult = {"2027 only": 1.0, "2031-32": 2 * 1.5, "2027-32": 1.0 + 5 * 1.5}   # exposure 300k in 2027, 450k from 2028
for k, mlt in exp_mult.items():
    a = L["aggregate_poisson"][k]
    chk("E compound", f"E[S] {k}: lambda * E[L] * exposure multiple", lam_c * EL * mlt, a["expected_loss_formula"], 0.01 * a["expected_loss_formula"] + 5)
    chk("E compound", f"E[S] {k}: simulation vs formula", a["expected_loss"], a["expected_loss_formula"], 0.03 * a["expected_loss_formula"] + 50, "within 3% (Monte Carlo)")
    pN = [stats.poisson.pmf(j, a["lam_t"]) for j in range(3)]
    chk("E compound", f"P(N=0) {k}: Poisson pmf", pN[0], a["pN_exact"][0], 1e-9)
    chk("E compound", f"P(N=1) {k}: Poisson pmf", pN[1], a["pN_exact"][1], 1e-9)
    chk("E compound", f"{k}: simulated P(N=0) vs exact", a["pN"][0], pN[0], 0.01, "Monte Carlo")
    chk("E compound", f"{k}: P(loss>=$50k) <= P(N>=1)", min(a["p_loss_ge"]["50000"], 1 - pN[0]), a["p_loss_ge"]["50000"], 1e-12, "a loss needs at least one crash")
    chk("E compound", f"{k}: VaR95 <= VaR99 <= CVaR99", float(a["var95"] <= a["var99"] <= a["cvar99"]), 1.0, 0)
# ------------------------------------------------------------------ F. probability that each crash type affects Laura
for a in L["affects_laura"]:
    p_occ = 1 - math.exp(-2 * a["lam"])
    chk("F affects", f"{a['type']} P(occurs 2031-32) = 1-exp(-2 lambda)", p_occ, a["p_occurs_2031_32"], 1e-9)
    for s in ("S9", "S10", "S10C"):
        chk("F affects", f"{a['type']} {s} shortfall = P(occurs) x P(short | crash)", p_occ * a[f"{s}_shortfall_given_crash"], a[f"{s}_p_shortfall"], 1e-9)
        chk("F affects", f"{a['type']} {s} promise broken = P(occurs) x P(broken | crash)", p_occ * a[f"{s}_promise_broken_given_crash"], a[f"{s}_p_promise_broken"], 1e-9)
        chk("F affects", f"{a['type']} {s}: shortfall <= promise broken", float(a[f"{s}_p_shortfall"] <= a[f"{s}_p_promise_broken"] + 1e-12), 1.0, 0)

bad = [c for c in checks if not c["ok"]]
json.dump(dict(n_checks=len(checks), n_failed=len(bad), checks=checks), open(RES / "verification.json", "w"), indent=1)
by = {}
for c in checks:
    by.setdefault(c["group"], [0, 0]); by[c["group"]][0] += 1; by[c["group"]][1] += c["ok"]
for g, (n_, ok_) in by.items():
    print(f"{g:16s} {ok_}/{n_} pass")
print(f"TOTAL {len(checks) - len(bad)}/{len(checks)} pass")
for c in bad:
    print("FAIL", c["group"], c["check"], "computed", round(c["computed"], 6), "reference", round(c["reference"], 6))
raise SystemExit(1 if bad else 0)
