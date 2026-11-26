"""Step 3 - strategy simulation (Research Pack Section 8): strategy simulation, monthly, 2027-01 to 2033-01 (72 months).
Cash flows (Case Study): $300k start 2027, $150k start 2028, ten $50k payments start 2033-2042.
Reviews: annual (start of each year) + extra review when the equity sleeve is 20% below its peak since the last review.
Cases: H = 6-month block bootstrap of 2004-01..2026-10 monthly data; S = H minus 3.25 pts a year on every fund;
J = Markov-modulated jump-diffusion (2-state Hamilton regimes, regime normals, compound-Poisson equity jumps, Poisson rate-shock episodes)."""
import json, os, numpy as np, pandas as pd
from config import RAW, PROC, RES
D = None
M = pd.read_csv(PROC / "monthly.csv", index_col=0, parse_dates=True)
F = ["VTI", "VXUS", "EQAL", "IEF", "SCHP", "SGOV", "SHY", "VGIT"]
W = M.loc["2004-01-01":"2026-10-01"]
X = W[F + ["dy5"]].to_numpy()                      # 274 x 9
NM = len(X)
EV = json.load(open(RES / "crash_events.json"))
Y_NOW = EV["now"]["y5"] / 100
cost = {"VTI": 64962.10, "VXUS": 39786.00, "EQAL": 14303.35, "IEF": 35736.00, "SCHP": 27698.00, "SGOV": 30159.00, "SHY": 40590.00, "VGIT": 40446.00}
CASH = 300000 - sum(cost.values())
eqt = sum(cost[k] for k in F[:3]); bdt = sum(cost[k] for k in F[3:]) + CASH
w_eq = np.array([cost[k] / eqt for k in F[:3]] + [0] * 5)
w_bd = np.array([0] * 3 + [(cost[k] + (CASH if k == "SGOV" else 0)) / bdt for k in F[3:]])
w_sgov = np.eye(8)[5]; w_vti = np.eye(8)[0]
E0 = eqt / 300000                                  # 0.3968 strategic equity share (as built)
DUR = np.array([0, 0, 0, 7.4, 6.6, 0.1, 1.9, 5.1])
PMT, NP, R = 50000, 10, 0.04
def ann_due(r): return PMT * NP if r <= 0 else PMT * (1 - (1 + r) ** -NP) / r * (1 + r)
R4, R3 = ann_due(.04), ann_due(.03)
def pv_liab(t, r=R):        # t in years from start-2027; payments at 6..15
    return sum(PMT / (1 + r) ** (k - t) for k in range(6, 16) if k >= t)
def pv_contrib(t): return 150000 / (1 + R) ** (1 - t) if t < 1 else 0.0
PATHS, MON, BLK = 10000, 72, 6
vol_e = (X[:, :8] @ w_eq).std() * np.sqrt(12); vol_b = (X[:, :8] @ w_bd).std() * np.sqrt(12)
RP = (1 / vol_e) / (1 / vol_e + 1 / vol_b)
STRATS = ["S1", "S2", "S4", "S6", "S7", "S8", "S9", "S9A", "S10", "S10C", "S0", "S0b"]
NAMES = {"S1": "T-bills only (SGOV)", "S2": "Pure LDI (bond sleeve)", "S4": "All-equity (VTI)", "S6": "Target-date glide path 60%->30%",
         "S7": "Risk parity (inverse volatility)", "S8": "CPPI, multiplier 3", "S9": "Core-satellite as built (LDCS holdings, static, no floor rule)",
         "S9A": "LDCS + calendar glide path (Option A)", "S10": "LDCS + IPS cushion rule (Option B)", "S10C": "LDCS + cushion rule + 2031 lock-in (Option B+C)",
         "S0": "Bucket ladder (refill in up years only)", "S0b": "Bucket ladder (refill every January)"}

def target(name, t, V, L=None):
    """equity share and weight matrix at time t (years) given portfolio value V (array)"""
    n = len(V)
    if name in ("S0", "S0b"): return np.zeros(n), None   # S0 is simulated by simulate_s0; this is its start (no equity in 2027)
    if name == "S1": return np.zeros(n), None
    if name == "S2": return np.zeros(n), None
    if name == "S4": return np.ones(n), None
    if name == "S6": return np.full(n, 0.6 - 0.3 * min(t, 5) / 5), None
    if name == "S7": return np.full(n, RP), None
    if name == "S9": return np.full(n, E0), None
    if name == "S9A": return np.full(n, E0 - (E0 - 0.20) * min(t, 5) / 5), None
    floor = pv_liab(t) - pv_contrib(t)
    if name == "S10C" and L is not None and t >= 4: floor = floor + L / (1 + R) ** (6 - t)
    cush = np.maximum(V - floor, 0)
    if name == "S8": return np.minimum(3 * cush / V, 1.0), None
    return np.minimum(cush / V, E0), None          # S10 and S10C: equity <= cushion, capped at strategic weight

def weights(name, e):
    if name == "S1": return np.tile(w_sgov, (len(e), 1))
    if name == "S4": return np.tile(w_vti, (len(e), 1))
    return e[:, None] * w_eq + (1 - e[:, None]) * w_bd

LOCK_Y2 = 0.04                                     # S10C adds the promise to the floor only if the 2-year yield is >= 4% in Jan 2031
SPREAD_52 = (EV["now"]["y5"] - EV["now"]["y2"]) / 100   # today's 5y-2y spread (0.08 pts); the model tracks the 5y yield only


LDCS_BAND = 0.02                                   # the IPS tolerance band, applied to LDCS (S10C) by default


def simulate(RET, DY, name, y0=Y_NOW, lock_cond=True, band="ips", trigger=True):
    """RET: PATHS x MON x 8 monthly returns; DY: PATHS x MON yield changes.
    lock_cond: S10C locks in the promise only where the January 2031 2-year yield (5-year yield minus today's 5y-2y spread) is >= 4%.
    band: "ips" = the IPS +/-2% band for S10C and none for the rivals; a number (e.g. 0.02) or None overrides. If set, check every month-end and rebalance to the rule's target when the equity share is more than `band`
          away from it (the IPS tolerance band).  trigger: extra review after a 20% fall in the equity sleeve."""
    if name in ("S0", "S0b"): return simulate_s0(RET, DY, name, y0)
    if band == "ips": band = LDCS_BAND if name == "S10C" else None   # LDCS = cushion rule + IPS band + 20% review + 2031 lock-in
    n = RET.shape[0]
    V = np.full(n, 300000.0); L = None; Lf = None; lock = np.zeros(n, bool)
    e, _ = target(name, 0, V); w = weights(name, e)
    H = w * V[:, None]                              # dollar holdings
    eq_peak = (H[:, :3].sum(1)).copy(); twr = np.ones(n); twr_pk = np.ones(n); mdd = np.zeros(n)
    eshare = {0: np.median(e)}; Vy = {0: V.copy()}; reviews = np.zeros(n)
    for m in range(MON):
        if m > 0 and m % 12 == 0:
            t = m // 12; V = H.sum(1)
            if t == 1: V = V + 150000
            if t == 4:
                Vy["pre2031"] = V.copy()
                L = 0.5 * np.maximum(V - pv_liab(4), 0) * (1 + R) ** 2
                y2_31 = y0 + DY[:, :m].sum(1) - SPREAD_52
                lock = (y2_31 >= LOCK_Y2) if lock_cond else np.ones(n, bool)
                Lf = np.where(lock, L, 0.0)                         # the part of the promise added to the floor
            e, _ = target(name, t, V, Lf); H = weights(name, e) * V[:, None]; eq_peak = H[:, :3].sum(1).copy()
            eshare[t] = np.median(e); Vy[t] = V.copy()
        r = RET[:, m, :]
        Vb = H.sum(1); H = H * (1 + r); Va = H.sum(1)
        twr *= Va / Vb; twr_pk = np.maximum(twr_pk, twr); mdd = np.maximum(mdd, 1 - twr / twr_pk)
        if (m + 1) % 12 == 0:                                        # the January review follows
            continue
        eqv = H[:, :3].sum(1); eq_peak = np.maximum(eq_peak, eqv)
        trig = (eqv < 0.8 * eq_peak) & (eq_peak > 1) if trigger else np.zeros(n, bool)
        if band is not None:
            t = (m + 1) / 12; Vn = H.sum(1); Ln = None if Lf is None else Lf
            e_t, _ = target(name, t, Vn, Ln)
            trig = trig | (np.abs(eqv / Vn - e_t) > band)
        if trig.any():                                              # extra review
            t = (m + 1) / 12; Vt = H[trig].sum(1)
            Lt = None if Lf is None else Lf[trig]
            e2, _ = target(name, t, Vt, Lt); H[trig] = weights(name, e2) * Vt[:, None]
            eq_peak[trig] = H[trig][:, :3].sum(1); reviews += trig
    V33 = H.sum(1)
    y33 = np.maximum(y0 + DY.sum(1), 0.0025)
    ladder = PMT * (1 - (1 + y33) ** -NP) / y33 * (1 + y33)
    if L is None: L = 0.5 * np.maximum(Vy["pre2031"] - pv_liab(4), 0) * (1 + R) ** 2
    return dict(V33=V33, ladder=ladder, y33=y33, L=L, V31=Vy["pre2031"], mdd=mdd, eshare=eshare, reviews=reviews, lock=lock)

# ---------- S0: bucket ladder ----------
# Three Treasury buckets matched to the payment years (B1 2033-35, B2 2036-39, B3 2040-42), filled in that order; whatever is
# left above the liability PV is the equity bucket, capped at the strategic 39.7%. Annual January review only (no 20% trigger,
# no lock-in). S0 refills the buckets from equity only when the equity bucket rose since the last review; S0b refills every year.
def _w(names, wts):
    v = np.zeros(8)
    for nm, x in zip(names, wts): v[F.index(nm)] = x
    return v / v.sum()
wB1 = _w(["SGOV", "SHY"], [cost["SGOV"] + CASH, cost["SHY"]])      # B1: payments 2033-35, bills and short Treasuries
wB2 = _w(["VGIT"], [1.0])                                            # B2: payments 2036-39, intermediate Treasuries
wB3 = _w(["IEF", "SCHP"], [cost["IEF"], cost["SCHP"]])               # B3: payments 2040-42, longest sleeve
def bucket_pv(t):
    f = lambda ks: sum(PMT / (1 + R) ** (k - t) for k in ks if k >= t)
    return f(range(6, 9)), f(range(9, 13)), f(range(13, 16))
def s0_alloc(V, t, Eq):
    b1, b2, b3 = bucket_pv(t); tot = b1 + b2 + b3; avail = V - Eq
    over = avail >= tot
    B1 = np.where(over, b1 * avail / tot, np.minimum(avail, b1))
    B2 = np.where(over, b2 * avail / tot, np.minimum(np.maximum(avail - b1, 0), b2))
    B3 = np.where(over, b3 * avail / tot, np.maximum(avail - b1 - b2, 0))
    return Eq[:, None] * w_eq + B1[:, None] * wB1 + B2[:, None] * wB2 + B3[:, None] * wB3
def simulate_s0(RET, DY, name, y0=Y_NOW):
    n = RET.shape[0]; up_only = (name == "S0")
    V = np.full(n, 300000.0); L = None
    Eq = np.zeros(n); H = s0_alloc(V, 0, Eq); eq_prior = Eq.copy()
    twr = np.ones(n); twr_pk = np.ones(n); mdd = np.zeros(n)
    eshare = {0: 0.0}; Vy = {0: V.copy()}; reviews = np.zeros(n); sold = np.zeros(n); held = np.zeros(n)
    for m in range(MON):
        if m > 0 and m % 12 == 0:
            t = m // 12; V = H.sum(1)
            if t == 1: V = V + 150000
            if t == 4:
                Vy["pre2031"] = V.copy()
                L = 0.5 * np.maximum(V - pv_liab(4), 0) * (1 + R) ** 2
            Eq_cur = H[:, :3].sum(1); tot = sum(bucket_pv(t))
            Eq_tgt = np.minimum(np.maximum(V - tot, 0), E0 * V)
            if up_only:
                need_sell = Eq_tgt < Eq_cur; up = Eq_cur > eq_prior
                Eq_new = np.where(need_sell & ~up, Eq_cur, Eq_tgt)
                held += (need_sell & ~up)
            else:
                Eq_new = Eq_tgt
            Eq_new = np.minimum(Eq_new, V)
            H = s0_alloc(V, t, Eq_new); eq_prior = Eq_new.copy()
            eshare[t] = float(np.median(Eq_new / V)); Vy[t] = V.copy()
        r = RET[:, m, :]
        Vb = H.sum(1); H = H * (1 + r); Va = H.sum(1)
        twr *= Va / Vb; twr_pk = np.maximum(twr_pk, twr); mdd = np.maximum(mdd, 1 - twr / twr_pk)
    V33 = H.sum(1)
    y33 = np.maximum(y0 + DY.sum(1), 0.0025)
    ladder = PMT * (1 - (1 + y33) ** -NP) / y33 * (1 + y33)
    return dict(V33=V33, ladder=ladder, y33=y33, L=L, V31=Vy["pre2031"], mdd=mdd, eshare=eshare, reviews=held)

def summarise(res):
    V = res["V33"]; c = V - R4; L = res["L"]
    q = lambda p: float(np.percentile(c, p))
    worst5 = np.sort(c)[: len(c) // 20].mean()
    # 2031 conditional range: paths whose 2031 value is within 2.5% of the median 2031 value
    v31 = res["V31"]; med = np.median(v31); nb = np.abs(v31 / med - 1) < 0.025
    cc = c[nb]
    a, lam = 0.88, 2.25
    gain = np.maximum(c - L, 0) ** a; loss = np.maximum(L - c, 0) ** a
    return dict(f4=float((V >= R4).mean()), f3=float((V >= R3).mean()), fl=float((V >= res["ladder"]).mean()),
                p1=q(1), p5=q(5), p10=q(10), p25=q(25), p50=q(50), p75=q(75), p90=q(90), p95=q(95), p99=q(99), mean=float(c.mean()),
                cvar5=float(worst5), g50=float((c >= 50000).mean()), g100=float((c >= 100000).mean()), g150=float((c >= 150000).mean()),
                g200=float((c >= 200000).mean()),
                mdd50=float(np.median(res["mdd"])), mdd95=float(np.percentile(res["mdd"], 95)),
                eshare={str(k): float(v) for k, v in res["eshare"].items()},
                r31=[float(np.percentile(cc, 10)), float(np.percentile(cc, 50)), float(np.percentile(cc, 90))], r31_n=int(nb.sum()), v31_med=float(med),
                promise_kept=float((c >= L).mean()), L_med=float(np.median(L)), short_vs_L=float(np.maximum(L - c, 0).mean()),
                pt_value=float(gain.mean() - lam * loss.mean()), reviews=float(res["reviews"].mean()),
                ladder_med=float(np.median(res["ladder"])), y33_p10=float(np.percentile(res["y33"], 10)), y33_p50=float(np.median(res["y33"])),
                y33_p90=float(np.percentile(res["y33"], 90)))

rng = np.random.default_rng(42)
def boot(shift=0.0):
    nb = -(-MON // BLK)
    st = rng.integers(0, NM - BLK + 1, size=(PATHS, nb))
    idx = (st[:, :, None] + np.arange(BLK)).reshape(PATHS, -1)[:, :MON]
    Z = X[idx]
    return Z[:, :, :8] + shift, Z[:, :, 8]

# ---------- case J: Markov-modulated jump diffusion ----------
reg = pd.read_csv(PROC / "regime_probability.csv", index_col=0, parse_dates=True).iloc[:, 0]
P = np.array(EV["markov"]["P"]); mer = EV["merton"]
turb = (reg.reindex(W.index).fillna(0) > 0.5).to_numpy()
jumpm = (W["VTI"].to_numpy() <= -0.095)
mu_r, cov_r = [], []
for s in (False, True):
    sel = (turb == s) & ~jumpm
    mu_r.append(X[sel].mean(0)); cov_r.append(np.cov(X[sel].T))
# jump responses of the other funds: slope through the origin on 2004-2026 jump months
Jm = X[jumpm]; beta_j = (Jm * Jm[:, [0]]).sum(0) / (Jm[:, 0] ** 2).sum()
# rate-shock episodes (R1 starts): 12-month 5y-yield rise and equity excess return, from FRED + CRSP
y5m = M["y5"]; lm = pd.read_csv(PROC / "us_market_long.csv", index_col=0, parse_dates=True).iloc[:, 0]
g5 = pd.read_csv(RAW / "fred" / "GS5.csv"); g5.columns = ["d", "v"]; g5 = g5.set_index(pd.to_datetime(g5["d"]))["v"].astype(float) / 100
shocks = []
for s in EV["other"]["R1 Rate shock: 10y yield +2 pts within 12 months"]["starts"]:
    d = pd.Timestamp(s); d0 = d - pd.DateOffset(months=11)
    dy = g5.asof(d) - g5.asof(d0 - pd.DateOffset(months=1))
    er = (1 + lm.loc[d0:d]).prod() - 1 - (1 + lm.mean()) ** 12 + 1
    shocks.append((float(dy), float(er), str(d.date())[:7]))
LAMR = EV["other"]["R1 Rate shock: 10y yield +2 pts within 12 months"]["lam"]
LAMJ = (mer["lam_J_calm"], mer["lam_J_turb"])
def jumpdiff(seed=7, record=False):
    g = np.random.default_rng(seed)
    s = (g.random(PATHS) < EV["markov"]["p_turbulent_now"]).astype(int)
    RET = np.zeros((PATHS, MON, 8)); DY = np.zeros((PATHS, MON)); Ls = [np.linalg.cholesky(c) for c in cov_r]
    shock_left = np.zeros(PATHS, int); shock_dy = np.zeros(PATHS); shock_er = np.zeros(PATHS)
    njump = np.zeros(PATHS); nshock = np.zeros(PATHS); TM = np.zeros((PATHS, MON), np.int8)
    for m in range(MON):
        TM[:, m] = s
        z = g.standard_normal((PATHS, 9)); out = np.zeros((PATHS, 9))
        for k in (0, 1):
            ix = s == k; out[ix] = mu_r[k] + z[ix] @ Ls[k].T
        lam = np.where(s == 1, LAMJ[1], LAMJ[0]) / 12
        jmp = g.random(PATHS) < 1 - np.exp(-lam)
        J = np.exp(g.normal(mer["muJ"], mer["sdJ"], PATHS)) - 1
        out[jmp] = (beta_j[None, :] * J[jmp, None])
        njump += jmp
        new = (shock_left == 0) & (g.random(PATHS) < 1 - np.exp(-LAMR / 12))
        pick = g.integers(0, len(shocks), PATHS)
        shock_dy[new] = np.array([shocks[i][0] for i in pick[new]]); shock_er[new] = np.array([shocks[i][1] for i in pick[new]])
        shock_left[new] = 12; nshock += new
        on = shock_left > 0
        ddy = shock_dy[on] / 12
        out[on, 3:8] += -DUR[3:8] * ddy[:, None] * np.array([1, 0.6, 1, 1, 1])     # TIPS real yields move ~0.6x nominal
        out[on, 4] += 0.02 / 12                                                    # extra CPI accrual in an inflation shock
        out[on, 5] += ddy * 6                                                      # bills reprice upward (carry)
        out[on, :3] += ((1 + shock_er[on]) ** (1 / 12) - 1)[:, None]
        out[on, 8] += ddy
        shock_left[on] -= 1
        RET[:, m] = out[:, :8]; DY[:, m] = out[:, 8]
        s = np.where(g.random(PATHS) < P[s, 1], 1, 0)
    info = dict(jumps_per_path=float(njump.mean()), p_any_jump=float((njump > 0).mean()), p_any_shock=float((nshock > 0).mean()))
    return (RET, DY, info, TM) if record else (RET, DY, info)

def fm_(i):
    t = pd.read_csv(RAW / "fred" / f"{i}.csv"); t.columns = ["d", "v"]; t = t.set_index(pd.to_datetime(t["d"]))["v"]
    return pd.to_numeric(t, errors="coerce").dropna()
win = lambda z: z.loc["2004-01-01":"2026-10-31"].mean()
carry = np.zeros(8)
for j, ids in ((3, ["DGS7", "DGS10"]), (4, ["DFII7"]), (5, ["DTB3"]), (6, ["DGS2"]), (7, ["DGS5"])):
    carry[j] = np.mean([fm_(i).iloc[-1] - win(fm_(i)) for i in ids]) / 100
# ---------- crash-type stress definitions (episode replay with the trough at the 2033 valuation) ----------
LONG = None
def long_synth():
    """1970-2026 synthetic monthly set for the 1973-74 replay (CRSP + FRED; no TIPS existed: bills + CPI as a proxy)."""
    def fm(i):
        s = pd.read_csv(RAW / "fred" / f"{i}.csv"); s.columns = ["d", "v"]; s = s.set_index(pd.to_datetime(s["d"]))["v"]
        return pd.to_numeric(s, errors="coerce").dropna().resample("ME").last().to_period("M").to_timestamp() / 100
    g10, g5_, g1, tb = fm("DGS10"), fm("DGS5"), fm("DGS1"), fm("DTB3")
    cpi = pd.read_csv(RAW / "fred" / "CPIAUCSL.csv"); cpi = cpi.set_index(pd.to_datetime(cpi.iloc[:, 0])).iloc[:, 1].astype(float)
    b = lambda y, Dm, C: y.shift(1) / 12 - Dm * y.diff() + 0.5 * C * y.diff() ** 2
    ew = pd.read_csv(PROC / "monthly.csv", index_col=0, parse_dates=True)
    df = pd.DataFrame({"VTI": lm, "VXUS": lm, "EQAL": lm, "IEF": b(g10, 7.4, 65), "SCHP": cpi.pct_change() + 0.01 / 12,
                       "SGOV": tb.shift(1) / 12, "SHY": b(g1, 0.95, 1.5), "VGIT": b(g5_, 4.3, 23)}).loc["1970-01-01":"1990-07-01"]
    df["dy5"] = g5_.diff()
    return df
EPIS = {"T1 Recession bear (2007-11 to 2009-02)": ("2007-11-01", "2009-02-01", M),
        "T2 Exogenous V-shock (2020-02 to 2020-03)": ("2020-02-01", "2020-03-01", M),
        "T3 Inflation/rate shock (2022-01 to 2022-09)": ("2022-01-01", "2022-09-01", M),
        "T3+ Severe inflation shock (1973-01 to 1974-09)": ("1973-01-01", "1974-09-01", None),
        "T4 Valuation bust (2000-09 to 2002-09)": ("2000-09-01", "2002-09-01", M),
        "T5 Falling-rate shock (2019-01 to 2020-07)": ("2019-01-01", "2020-07-01", M)}
LS = long_synth()

def make_cases(seed_boot=42, seed_j=7, record=False):
    """The four return cases, rebuilt from fresh seeds (used by robustness.py; the main run below uses 42 and 7)."""
    global rng
    rng = np.random.default_rng(seed_boot)
    RH_, DH_ = boot()
    out = jumpdiff(seed_j, record)
    cs = {"H": (RH_, DH_), "S": (RH_ - 0.0325 / 12, DH_), "C": (RH_ + carry / 12, DH_), "J": out[:2]}
    return (cs, out[3]) if record else cs

def crash_replay(RH_, DH_, key, timing="worst"):
    """Paths with one historical episode inserted at the end (worst) or start (early) of the 72 months."""
    a, b, src = EPIS[key]; src = LS if src is None else src
    ep = src.loc[a:b, F + ["dy5"]].to_numpy(); L_ = len(ep)
    R0, D0 = RH_.copy(), DH_.copy()
    if timing == "worst": R0[:, MON - L_:, :] = ep[None, :, :8]; D0[:, MON - L_:] = ep[None, :, 8]
    else: R0[:, :L_, :] = ep[None, :, :8]; D0[:, :L_] = ep[None, :, 8]
    return R0, D0

if __name__ == "__main__":
    results = {"meta": dict(R4=R4, R3=R3, E0=E0, RP=RP, y_now=Y_NOW, w_eq=w_eq.tolist(), w_bd=w_bd.tolist(), months=NM,
                            window=[str(W.index[0].date()), str(W.index[-1].date())], beta_jump=beta_j.tolist(), shocks=shocks,
                            lamJ=LAMJ, lamR=LAMR, mu_calm=(mu_r[0] * 12).tolist(), mu_turb=(mu_r[1] * 12).tolist(),
                            vol_calm=(np.sqrt(np.diag(cov_r[0]) * 12)).tolist(), vol_turb=(np.sqrt(np.diag(cov_r[1]) * 12)).tolist())}
    cases = {}
    RH, DH = boot(); cases["H"] = (RH, DH)
    cases["S"] = (RH - 0.0325 / 12, DH)
    results["meta"]["carry_shift"] = carry.tolist()
    cases["C"] = (RH + carry / 12, DH)
    RJ, DJ, jinfo = jumpdiff(); cases["J"] = (RJ, DJ); results["meta"]["J"] = jinfo
    for c, (RR, DD) in cases.items():
        ann = ((1 + RR).prod(1) ** (1 / 6) - 1)
        results.setdefault("case_returns", {})[c] = {f: [float(np.median(ann[:, i])), float(np.std(np.log1p(RR[:, :, i])) * np.sqrt(12))] for i, f in enumerate(F)}
        results[c] = {}
        for s in STRATS:
            results[c][s] = summarise(simulate(RR, DD, s))
        print(c, {s: (round(results[c][s]["f4"], 3), round(results[c][s]["p50"] / 1000)) for s in STRATS}, flush=True)

    crash = {}
    for k, (a, b, src) in EPIS.items():
        src = LS if src is None else src
        ep = src.loc[a:b, F + ["dy5"]].to_numpy(); L = len(ep)
        out = {}
        for timing in ("worst", "early"):
            R0, D0 = RH.copy(), DH.copy()
            if timing == "worst": R0[:, MON - L:, :] = ep[None, :, :8]; D0[:, MON - L:] = ep[None, :, 8]
            else: R0[:, :L, :] = ep[None, :, :8]; D0[:, :L] = ep[None, :, 8]
            out[timing] = {s: summarise(simulate(R0, D0, s)) for s in STRATS}
        # recovery on actual history from the episode start (static weights at each strategy's typical 2032 equity share)
        full = src.loc[a:, F].to_numpy()
        rec = {}
        for s in STRATS:
            es = results["H"][s]["eshare"]["5"]
            w = w_sgov if s == "S1" else (w_vti if s == "S4" else es * w_eq + (1 - es) * w_bd)
            idx = np.cumprod(1 + full @ w); pk = 1.0
            trough_i = int(np.argmin(idx[:max(L, 1) + 24])); depth = float(idx[trough_i] - 1) if idx[trough_i] < 1 else 0.0
            after = np.where(idx[trough_i:] >= 1.0)[0]
            rec[s] = dict(depth=min(depth, 0.0), m_trough=trough_i + 1 if depth < 0 else 0,
                          m_recover=(int(trough_i + after[0] + 1) if len(after) else None) if depth < 0 else 0,
                          episode_ret=float(np.prod(1 + ep[:, :8] @ w) - 1))
        crash[k] = dict(months=L, timing=out, recovery=rec, data_end=str(src.index[-1].date()))
        print(k, {s: round(out["worst"][s]["f4"], 3) for s in STRATS}, flush=True)
    results["crash"] = crash

    # ---------- deterministic liability analytics ----------
    def liab_stats(t, r=R):
        ks = [k for k in range(6, 16) if k >= t]; cf = np.array([PMT / (1 + r) ** (k - t) for k in ks]); tt = np.array([k - t for k in ks], float)
        pv = cf.sum(); mac = (tt * cf).sum() / pv; mod = mac / (1 + r); conv = (tt * (tt + 1) * cf).sum() / pv / (1 + r) ** 2
        return dict(pv=pv, mac=mac, mod=mod, conv=conv, pv01=pv * mod * 0.0001)
    results["liab"] = {str(2027 + t): liab_stats(t) for t in range(0, 7)}
    results["sleeve_duration"] = float(w_bd @ DUR)
    # fund statistics for the syllabus links (2004-2026 monthly)
    st = {}
    from scipy import stats as ss
    for i, f in enumerate(F):
        x = X[:, i]; mu, sd = x.mean(), x.std(ddof=1)
        jb = ss.jarque_bera(x)
        beta = np.cov(x, X[:, 0])[0, 1] / X[:, 0].var(ddof=1)
        rfm = W["SGOV"].mean()
        st[f] = dict(ann=float((1 + x).prod() ** (12 / NM) - 1), vol=float(sd * np.sqrt(12)), skew=float(ss.skew(x)), kurt=float(ss.kurtosis(x)),
                     jb_p=float(jb.pvalue), beta=float(beta), sharpe=float((mu - rfm) / sd * np.sqrt(12)), worst=float(x.min()),
                     z_worst=float((x.min() - mu) / sd), p_norm_worst=float(ss.norm.cdf((x.min() - mu) / sd)))
    results["fund_stats"] = st
    C = np.corrcoef(X[:, :8].T); results["corr"] = C.tolist()
    sb = pd.DataFrame({"e": W[F[:3]].to_numpy() @ w_eq[:3], "b": W[F[3:]].to_numpy() @ w_bd[3:]}, index=W.index)
    results["stock_bond_corr"] = {"2004-2020": float(sb.loc[:"2020-12-31"].corr().iloc[0, 1]), "2021-2026": float(sb.loc["2021-01-01":].corr().iloc[0, 1]),
                                  "full": float(sb.corr().iloc[0, 1])}
    # strategy-level volatility via w' Sigma w (initial weights) and normal vs historical 1-year VaR
    Sig = np.cov(X[:, :8].T) * 12; mu_a = X[:, :8].mean(0) * 12
    sv = {}
    for s in STRATS:
        e0 = target(s, 0, np.array([300000.0]))[0][0]; w = weights(s, np.array([e0]))[0]
        port = W[F].to_numpy() @ w; r12 = pd.Series(port).rolling(12).apply(lambda z: np.prod(1 + z) - 1).dropna()
        sv[s] = dict(e0=float(e0), mu=float(w @ mu_a), vol=float(np.sqrt(w @ Sig @ w)), var95_norm=float(w @ mu_a - 1.645 * np.sqrt(w @ Sig @ w)),
                     var95_hist=float(np.percentile(r12, 5)), worst12=float(r12.min()), dur=float(w @ DUR))
    results["strategy_stats"] = sv
    json.dump(results, open(RES / "strategy_results.json", "w"), indent=1)
    print(json.dumps(results["meta"], indent=0)[:2500])
