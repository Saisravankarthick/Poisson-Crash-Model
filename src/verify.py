"""Independent verification. Recomputes every headline number a second way (different algorithm or closed form) and
compares it with results/*.json, and checks the case-study arithmetic against the printed tables in the competition guide
(WGHS 2026-27 Team Study Pack, sections 3.1-3.2).  Exit code 1 if any check fails.  Output: results/verification.json"""
import json
import math
import numpy as np
import pandas as pd
from scipy import stats, optimize
from config import ROOT, PROC, RES, HOLDINGS, CASH, INITIAL, TOP_UP, PAYMENT, N_PAYMENTS, FUNDS, EQUITY

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

# ------------------------------------------------------------------ G. liability maths (Research Pack Section 8 / IPS)
SR = json.load(open(RES / "strategy_results.json")); meta = SR["meta"]
chk("G liability", "R4 stored in simulation = guide's $421,767", meta["R4"], 421767, 1, "guide s3.1")
chk("G liability", "R3 stored in simulation = guide's $439,305", meta["R3"], 439305, 1, "guide s3.1")
src_ = (ROOT / "src" / "strategy_sim.py").read_text()
chk("G liability", "simulation set-up in the code is 10,000 paths x 72 months x 6-month blocks", float("PATHS, MON, BLK = 10000, 72, 6" in src_), 1.0, 0, "72 months = 2027-01 to 2033-01")
chk("G liability", "stress case subtracts 3.25 points a year", float("RH - 0.0325 / 12" in src_), 1.0, 0)
chk("G liability", "bootstrap pool = 274 months (2004-01 to 2026-10)", meta["months"], 274, 0)
r_ = 0.04
for yr_, d in SR["liab"].items():
    t_ = int(yr_) - 2027                      # years since the start of 2027
    ks = np.arange(6, 16)                      # payments at 2033..2042 = year index 6..15
    cf = 50000.0; pvs = cf / (1 + r_) ** (ks - t_); pv = pvs.sum()
    mac = (pvs * (ks - t_)).sum() / pv
    mod = mac / (1 + r_)
    conv = (pvs * (ks - t_) * (ks - t_ + 1)).sum() / pv / (1 + r_) ** 2
    pv01 = pv * mod * 1e-4
    chk("G liability", f"{yr_} PV of the ten payments at 4%", pv, d["pv"], 0.01)
    chk("G liability", f"{yr_} Macaulay duration", mac, d["mac"], 1e-6)
    chk("G liability", f"{yr_} modified duration", mod, d["mod"], 1e-6)
    chk("G liability", f"{yr_} convexity", conv, d["conv"], 1e-4)
    chk("G liability", f"{yr_} PV01 ($ per 1bp)", pv01, d["pv01"], 1e-4)
chk("G liability", "PV grows 4% a year: PV(2032)/PV(2031)", SR["liab"]["2032"]["pv"] / SR["liab"]["2031"]["pv"], 1.04, 1e-9)
chk("G liability", "equity share as built", meta["E0"], 0.39683816666666666, 1e-9)

# ------------------------------------------------------------------ H. independent re-simulation of S1 and S4 (separate code, different seed)
Mo = pd.read_csv(PROC / "monthly.csv", index_col=0, parse_dates=True).loc["2004-01-01":"2026-10-01"]
rng_ = np.random.default_rng(2026); NP_, MON_, BLK_ = 20000, 72, 6
for fund, strat in (("VTI", "S4"), ("SGOV", "S1")):
    x = Mo[fund].to_numpy(); starts_ = rng_.integers(0, len(x) - BLK_ + 1, size=(NP_, MON_ // BLK_))
    path = np.concatenate([x[st_:st_ + BLK_] for st_ in starts_.T], axis=0) if False else np.stack([x[starts_[:, b][:, None] + np.arange(BLK_)] for b in range(MON_ // BLK_)], 1).reshape(NP_, MON_)
    for case, shift in (("H", 0.0), ("S", -0.0325 / 12)):
        g = 1 + path + shift
        V33 = 300000 * g.prod(1) + 150000 * g[:, 12:].prod(1)
        c = V33 - 421766.58
        ref = SR[case][strat]
        chk("H re-sim", f"{strat} case {case}: funding certainty", float((c >= 0).mean()), ref["f4"], 0.012, "20,000 paths, own bootstrap; Monte Carlo error about 0.5 pt")
        chk("H re-sim", f"{strat} case {case}: median contribution", float(np.median(c)), ref["p50"], 0.04 * abs(ref["p50"]) + 4000, "within 4% (Monte Carlo)")
        chk("H re-sim", f"{strat} case {case}: 5th percentile contribution", float(np.percentile(c, 5)), ref["p5"], 0.06 * abs(ref["p5"]) + 8000, "within 6% (Monte Carlo)")
        chk("H re-sim", f"{strat} case {case}: 95th percentile contribution", float(np.percentile(c, 95)), ref["p95"], 0.05 * abs(ref["p95"]) + 8000, "within 5% (Monte Carlo)")

# ------------------------------------------------------------------ I. invariants of the Monte Carlo output
pk = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
for c_ in "HSCJ":
    for s_, d in SR[c_].items():
        chk("I invariants", f"{c_} {s_}: percentiles non-decreasing", float(all(d[a] <= d[b] + 1e-9 for a, b in zip(pk, pk[1:]))), 1.0, 0)
        chk("I invariants", f"{c_} {s_}: P(fund at 3%) <= P(fund at 4%)", float(d["f3"] <= d["f4"] + 1e-12), 1.0, 0, "a higher reserve is harder to meet")
        ks_ = [(float(p[1:]), d[p]) for p in pk]
        k_pos = [k for k, v in ks_ if v >= 0]; k_neg = [k for k, v in ks_ if v < 0]
        lower = 1 - min(k_pos) / 100 if k_pos else 0.0       # all paths above the first non-negative percentile are funded
        upper = 1 - max(k_neg) / 100 if k_neg else 1.0       # no path below the last negative percentile is funded
        chk("I invariants", f"{c_} {s_}: funding certainty lies between the bounds implied by the percentiles", float(lower - 0.011 <= d["f4"] <= upper + 0.011), 1.0, 0, f"bounds {lower:.2f} to {upper:.2f}")
        chk("I invariants", f"{c_} {s_}: promise_kept within [0,1]", float(0 <= d["promise_kept"] <= 1), 1.0, 0)
for s_ in SR["H"]:
    chk("I invariants", f"all-equity paths do not depend on yield case ({s_} H vs C)" if s_ == "S4" else f"{s_}: S stress lowers funding certainty vs H",
        float(SR["H"]["S4"]["f4"]) if s_ == "S4" else float(SR["S"][s_]["f4"] <= SR["H"][s_]["f4"] + 1e-12),
        float(SR["C"]["S4"]["f4"]) if s_ == "S4" else 1.0, 1e-12)

# ------------------------------------------------------------------ J. 'affects Laura' uses the replay results
for a in L["affects_laura"]:
    key = a["scenario"]; w_ = SR["crash"][key]["timing"]["worst"]
    for s_ in ("S9", "S10", "S10C"):
        chk("J replay link", f"{a['type']} {s_}: P(shortfall | crash) = 1 - replay funding certainty", 1 - w_[s_]["f4"], a[f"{s_}_shortfall_given_crash"], 5e-4)
        chk("J replay link", f"{a['type']} {s_}: P(promise broken | crash) = 1 - replay promise kept", 1 - w_[s_]["promise_kept"], a[f"{s_}_promise_broken_given_crash"], 5e-4)

# ------------------------------------------------------------------ K. tables on disk equal the JSON they claim to come from
T_ = pd.read_csv(RES / "tables" / "strategy_summary.csv")
for _, row in T_.iterrows():
    chk("K tables", f"strategy_summary {row['case']} {row['strategy']} funding certainty", row["funding_certainty_4pct"], SR[row["case"]][row["strategy"]]["f4"], 1e-12)
Ta = pd.read_csv(RES / "tables" / "probability_crash_affects_laura.csv")
for a, (_, row) in zip(L["affects_laura"], Ta.iterrows()):
    chk("K tables", f"affects_laura csv {a['type']} S9 promise broken", row["S9_p_promise_broken"], a["S9_p_promise_broken"], 1e-12)
Te = pd.read_csv(RES / "tables" / "crash_episodes.csv")
chk("K tables", "crash_episodes.csv has the ten falls", len(Te), 10, 0)
Tw = pd.read_csv(RES / "tables" / "poisson_by_window.csv"); r2 = Tw[(Tw.threshold == "20%") & (Tw.window == "2031-32 (critical years)")].iloc[0]
chk("K tables", "poisson_by_window 2031-32 = 1-exp(-2 lambda)", r2["p_at_least_one"], 1 - math.exp(-2 * lam), 1e-9)

# ------------------------------------------------------------------ L. regime model against the crash dates (supports the clustering sentence in the report)
reg = pd.read_csv(PROC / "regime_probability.csv", index_col=0, parse_dates=True).iloc[:, 0]
inside = 0
for e in E["equity"]["20"]["episodes"]:
    seg = reg.loc[pd.Timestamp(e["peak"] + "-01") - pd.DateOffset(months=3): pd.Timestamp(e["trough"] + "-01")]
    inside += int(seg.max() > 0.5)
chk("L regimes", "falls that overlap a turbulent spell (probability > 50%), of 10", inside, E["equity"]["20"]["n"], 3, "reported in REPORT.md section 10")


# ------------------------------------------------------------------ M. bucket ladder (S0) and the robustness tests (src/robustness.py)
import strategy_sim as ss_
Z_ = np.zeros((4, 72, 8)); D_ = np.zeros((4, 72))
chk("M S0 engine", "S0 with zero returns ends at $300k + $150k", ss_.simulate_s0(Z_, D_, "S0")["V33"].mean(), 450000, 1e-6)
G_ = np.full((4, 72, 8), 0.004)
chk("M S0 engine", "S0 with 0.4% a month on every fund = 300k*1.004^72 + 150k*1.004^60", ss_.simulate_s0(G_, D_, "S0")["V33"].mean(),
    300000 * 1.004 ** 72 + 150000 * 1.004 ** 60, 1e-6)
for t_ in range(6):
    b_ = ss_.bucket_pv(t_)
    chk("M S0 engine", f"buckets B1+B2+B3 = liability PV at start of {2027 + t_}", sum(b_), ss_.pv_liab(t_), 1e-6)
    for V0 in (250e3, 460e3, 900e3):
        Vv = np.full(3, V0); Eq_ = np.minimum(np.maximum(Vv - sum(b_), 0), ss_.E0 * Vv); H_ = ss_.s0_alloc(Vv, t_, Eq_)
        chk("M S0 engine", f"S0 allocation sums to V, no negatives, equity <= cap (t={t_}, V={V0:.0f})",
            float(abs(H_.sum(1) - Vv).max() < 1e-6 and (H_ >= -1e-9).all() and (H_[:, :3].sum(1) <= ss_.E0 * Vv + 1e-6).all()), 1.0, 0)
RB = json.load(open(RES / "robustness.json"))
crit_ = RB["criteria"]; S_ = RB["set"]
for c_ in crit_:
    vals = RB["scoreboard"]["values"][c_]
    for s_ in S_:
        chk("M scoreboard", f"rank of {s_} on '{c_}' = 1 + number strictly better", RB["scoreboard"]["ranks"][c_][s_], 1 + sum(vals[x] > vals[s_] for x in S_), 0)
for s_ in S_:
    for w_, ws_ in {"equal": [1] * 6, "safety": [2, 2, 1, 1, 2, 2], "growth": [1, 1, 3, 1, 1, 1]}.items():
        chk("M scoreboard", f"{s_} {w_} average rank recomputed", RB["scoreboard"]["totals"][w_][s_],
            sum(a * RB["scoreboard"]["ranks"][c_][s_] for a, c_ in zip(ws_, crit_)) / sum(ws_), 1e-12)
PUB = {"S10C": (4.0, 3.8, 4.75), "S9A": (4.0, 3.7, 4.25), "S7": (4.1667, 3.7, 5.125), "S10": (4.1667, 4.0, 4.625), "S1": (5.5, 5.5, 6.625), "S4": (7.3333, 7.7, 5.75)}
for s_, v_ in PUB.items():
    for w_, x_ in zip(("equal", "safety", "growth"), v_):
        chk("M scoreboard", f"ten-strategy Table 8.21 {s_} {w_} reproduced", RB["full10_totals_published"][w_][s_], x_, 5e-4,
            "Research Pack p.13; LDCS run without the 2031 yield condition, as printed")
chk("M scoreboard", "base seed re-run through robustness.py reproduces the scoreboard", RB["base_rerun_matches"], 0, 1e-12)
chk("M weights", "P(ranked 1st) sums to 1 over strategies", sum(RB["weights"]["p_first"].values()), 1, 1e-9)
chk("M weights", "P(top 3) sums to 3 over strategies", sum(RB["weights"]["p_top3"].values()), 3, 1e-9)
chk("M seeds", "twenty fresh seeds", RB["seeds"]["n"], 20, 0)
PT_ = pd.read_csv(RES / "tables" / "paired_tests.csv")
for _, r_ in PT_.iterrows():
    chk("M paired", f"{r_['case']} {r_['rival']}: (LDCS-only minus rival-only funded paths)/n = difference in funding certainty",
        (r_["ldcs_only_funded"] - r_["rival_only_funded"]) / 10000, r_["f4_ldcs"] - r_["f4_rival"], 1e-12)
    chk("M paired", f"{r_['case']} {r_['rival']}: LDCS funding certainty equals the stored Monte Carlo value", r_["f4_ldcs"], SR[r_["case"]]["S10C"]["f4"], 1e-12)
    chk("M paired", f"{r_['case']} {r_['rival']}: rival funding certainty equals the stored Monte Carlo value", r_["f4_rival"], SR[r_["case"]][r_["rival"]]["f4"], 1e-12)
CT_ = pd.read_csv(RES / "tables" / "crash_type_promise.csv")
for _, r_ in CT_.iterrows():
    n_, T__ = r_["events"], r_["years"]
    chk("M joint Poisson", f"{r_['type']} lambda 90% lower bound by gamma quantile", r_["lambda_lo90"], stats.gamma.ppf(0.05, n_) / T__ if n_ else 0, 1e-9)
    chk("M joint Poisson", f"{r_['type']} lambda 90% upper bound by gamma quantile", r_["lambda_hi90"], stats.gamma.ppf(0.95, n_ + 1) / T__, 1e-9)
    chk("M joint Poisson", f"{r_['type']} P(in 2031-32) = 1-exp(-2 lambda)", r_["p_2031_32"], 1 - math.exp(-2 * r_["lambda"]), 1e-12)
JP_ = pd.read_csv(RES / "tables" / "joint_poisson_shortfall.csv").set_index("strategy")
for s_ in S_:
    q_ = sum(r_["p_2031_32"] * (1 - r_[f"funded_{s_}"]) for _, r_ in CT_.iterrows())
    chk("M joint Poisson", f"{s_} joint shortfall recomputed from the crash table", JP_.loc[s_, "shortfall_sum"], q_, 1e-12)
    chk("M joint Poisson", f"{s_} union <= sum <= upper-lambda sum", float(JP_.loc[s_, "shortfall_union"] <= JP_.loc[s_, "shortfall_sum"] + 1e-15 <= JP_.loc[s_, "shortfall_sum_hi"] + 2e-15), 1.0, 0)
pt_ = RB["poisson_tests"]
chk("M Poisson tests", "decade counts add to the ten crashes", sum(pt_["decade"]["counts"]), 10, 0)
chk("M Poisson tests", "no overdispersion by decade at 5%", float(pt_["decade"]["p_overdispersed"] > 0.05), 1.0, 0)
mk_ = RB["markov"]; P_ = np.array(mk_["P"]); pi_ = np.array(mk_["stationary"])
chk("M Markov", "stationary distribution solves pi P = pi", float(np.abs(pi_ @ P_ - pi_).max()), 0, 1e-9)
chk("M Markov", "P(any turbulent month in 2031-32): matrix powers vs simulated paths", mk_["p_any_turbulent_2031_32_exact"], mk_["p_any_turbulent_2031_32_sim"], 0.02)
MC_ = pd.read_csv(RES / "tables" / "markov_conditional_shortfall.csv")
for s_ in S_:
    q_ = MC_[MC_.strategy == s_]
    chk("M Markov", f"{s_}: sum of share x conditional shortfall = case J shortfall", (q_.share_of_paths * q_.p_shortfall).sum(), 1 - SR["J"][s_]["f4"], 1e-9)
    chk("M Markov", f"{s_}: sum of share x conditional broken promise = case J broken promise", (q_.share_of_paths * q_.p_promise_broken).sum(), 1 - SR["J"][s_]["promise_kept"], 1e-9)
chk("M Markov", "regime-bucket shares sum to 1", MC_.drop_duplicates("turbulent_months_2031_32").share_of_paths.sum(), 1, 1e-12)

# ------------------------------------------------------------------ N. crash impact table and the IPS-consistency variants
IM_ = pd.read_csv(RES / "tables" / "crash_impact_by_strategy.csv")
CTi = CT_.set_index("type")
for _, r_ in IM_.iterrows():
    scen_ = next(a["scenario"] for a in L["affects_laura"] if a["type"] == r_["type"])
    w_ = SR["crash"][scen_]["timing"]["worst"][r_["strategy"]]
    chk("N crash impact", f"{r_['type']} {r_['strategy']}: P(in 2031-32) matches the crash-type table", r_["p_2031_32"], CTi.loc[r_["type"], "p_2031_32"], 1e-12)
    chk("N crash impact", f"{r_['type']} {r_['strategy']}: joint broken promise = P(type) x (1 - replay promise kept)",
        r_["joint_promise_broken"], r_["p_2031_32"] * (1 - w_["promise_kept"]), 1e-12)
    chk("N crash impact", f"{r_['type']} {r_['strategy']}: contribution hit = no-crash median - replay median",
        r_["contribution_hit"], SR["H"][r_["strategy"]]["p50"] - w_["p50"], 1e-6)
CI_ = RB["crash_impact"]
for s_ in S_:
    chk("N crash impact", f"{s_}: total joint broken promise equals the joint Poisson table", CI_["totals"][s_]["joint_promise_broken"], JP_.loc[s_, "broken_sum"], 1e-12)
    chk("N crash impact", f"{s_}: expected hit = sum of P(type) x hit", CI_["totals"][s_]["expected_hit"],
        (IM_[IM_.strategy == s_].p_2031_32 * IM_[IM_.strategy == s_].contribution_hit).sum(), 1e-6)
    chk("N crash impact", f"{s_}: stock-crash expected hit = sum over T1-T4 of P(type) x hit", CI_["totals"][s_]["expected_hit_stock_crashes"],
        (IM_[(IM_.strategy == s_) & (IM_.type != "T5")].p_2031_32 * IM_[(IM_.strategy == s_) & (IM_.type != "T5")].contribution_hit).sum(), 1e-6)
for t_, w__ in CI_["worst"].items():
    q_ = IM_[IM_.type == t_]
    chk("N crash impact", f"{t_}: worst strategy (all nine) has the highest joint broken-promise probability",
        q_.set_index("strategy").loc[w__["all"]["strategy"], "joint_promise_broken"], q_.joint_promise_broken.max(), 1e-15)
IV_ = RB["ips_check"]
for k_, v_ in IV_.items():
    chk("N IPS variants", f"{k_}: lock-in share within [0, 1]", float(0 <= v_["lock_share_H"] <= 1), 1.0, 0)
    chk("N IPS variants", f"{k_}: crash resilience and promise reliability within [0, 1]",
        float(0 <= v_["crit"][crit_[4]] <= 1 and 0 <= v_["crit"][crit_[5]] <= 1), 1.0, 0)
    chk("N IPS variants", f"{k_}: LDCS keeps first place under equal weights", v_["place"]["equal"], 1, 0)
chk("N IPS variants", "published version locks in every path", IV_["published"]["lock_share_H"], 1.0, 0)
chk("N IPS variants", "headline version equals the stored S10C results (promise reliability)", IV_["headline"]["crit"][crit_[5]],
    RB["scoreboard"]["values"][crit_[5]]["S10C"], 1e-12)
Zr = np.zeros((3, 72, 8)); Dr = np.zeros((3, 72)); Dr[0, 10] = -0.02                # path 0: yields fall 2 pts before 2031
lk_ = ss_.simulate(Zr, Dr, "S10C")["lock"]
chk("N IPS variants", "yield condition: lock-in off when the 2031 2-year yield is below 4%", float(lk_[0]), 0.0, 0)
chk("N IPS variants", "yield condition: lock-in on when the 2031 2-year yield is at least 4%", float(lk_[1]), 1.0, 0)

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
