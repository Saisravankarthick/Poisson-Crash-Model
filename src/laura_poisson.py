"""Step 4 - the Poisson crash model applied to Laura Gao's actual holdings and dollar allocation.

Level 1 (each fund):  count every >=20% / >=30% fall of each fund since 1990 (month-end closes), estimate lambda_i,
                      exact 90% interval and window probabilities, and the dollars of Laura's money at stake.
Level 2 (portfolio):  compound Poisson.  Number of market crashes N ~ Poisson(lambda t) with lambda from 100 years of
                      CRSP data.  Each crash j is given a type (recession, V-shock, inflation/rate, valuation) with the
                      historical type frequencies, a depth drawn from the same type's history, and the fund-by-fund
                      response of the matching 1990-2026 episode.  Dollar loss of crash j = exposure x sum_i w_i r_i.
                      Aggregate crash damage S = L_1 + ... + L_N  (before any recovery).
Level 3 (Markov):     the same with lambda switching between calm and turbulent regimes (Markov-modulated Poisson).
Level 4 (affects Laura): P(type occurs in 2031-32) x P(shortfall or broken promise | that crash ends at the 2033
                      valuation), using the strategy simulation results if available.
Output: results/laura_poisson.json and results/tables/*.csv"""
import json
import math
import numpy as np
import pandas as pd
from config import PROC, RES, HOLDINGS, CASH, FUNDS, EQUITY, INITIAL, TOP_UP, WINDOWS, SEED
import poisson_core as pc

TABLES = RES / "tables"; TABLES.mkdir(exist_ok=True)
N_SIM = 200_000
# 1990-2026 template episode for each crash type (peak month, trough month on CRSP month-end data)
TEMPLATES = {"T1 Recession bear": ("2007-10-01", "2009-02-01"), "T2 Exogenous V-shock": ("2020-01-01", "2020-03-01"),
             "T3 Inflation / rate shock": ("2021-12-01", "2022-09-01"), "T4 Valuation bust": ("2000-08-01", "2002-09-01")}


def weights():
    h = dict(HOLDINGS); h["SGOV"] += CASH                # cash is held like T-bills
    w = np.array([h[f] for f in FUNDS]) / INITIAL
    return h, w


def fund_level(M, h):
    T = len(M) / 12; rows = []
    for f in FUNDS:
        lvl = (1 + M[f]).cumprod(); lvl = pd.concat([pd.Series([1.0], index=[M.index[0] - pd.DateOffset(months=1)]), lvl])
        row = dict(fund=f, dollars=h[f], weight=h[f] / INITIAL, max_drawdown=pc.max_drawdown(lvl))
        for thr in (0.10, 0.20, 0.30):
            ev = pc.episodes(lvl, thr); lam = len(ev) / T; lo, hi = pc.garwood_ci(len(ev), T)
            k = f"{int(thr*100)}"
            row[f"n{k}"] = len(ev); row[f"lam{k}"] = lam; row[f"lam{k}_hi90"] = hi
            row[f"p{k}_2031_32"] = pc.p_at_least_one(lam, 2); row[f"p{k}_2027_32"] = pc.p_at_least_one(lam, 6)
            row[f"p{k}_2031_32_hi90"] = pc.p_at_least_one(hi, 2)
            row[f"episodes{k}"] = [f"{str(e['peak'].date())[:7]} to {str(e['trough'].date())[:7]} ({e['depth']:.0%})" for e in ev]
        row["dollars_at_worst_fall"] = h[f] * row["max_drawdown"]
        rows.append(row)
    return rows, T


def type_responses(M):
    """fund returns from peak month-end to trough month-end of each template episode"""
    out = {}
    for typ, (a, b) in TEMPLATES.items():
        seg = M.loc[pd.Timestamp(a) + pd.DateOffset(months=1):b, FUNDS]
        out[typ] = (1 + seg).prod().values - 1
    return out


def crash_sampler(EV, resp, w, rng):
    rows = EV["equity"]["20"]["episodes"]
    tmpl_depth = {}
    for typ, (a, b) in TEMPLATES.items():
        tmpl_depth[typ] = next(r["depth"] for r in rows if r["peak"] == a[:7])
    pool = {t: [] for t in TEMPLATES}
    for r in rows:                                       # 'T5 Other' (1962, a 6-month fall) uses the V-shock template
        t = r["type"] if r["type"] in TEMPLATES else "T2 Exogenous V-shock"
        pool[t].append(r["depth"])
    types = list(TEMPLATES); freq = np.array([len(pool[t]) for t in types], float); freq /= freq.sum()
    eq = np.array([f in EQUITY for f in FUNDS])

    def draw(n):
        ti = rng.choice(len(types), size=n, p=freq)
        ret = np.zeros(n); fund_ret = np.zeros((n, len(FUNDS)))
        for j, t in enumerate(types):
            m = ti == j
            if not m.any():
                continue
            d = rng.choice(pool[t], size=m.sum()); s = d / tmpl_depth[t]
            r = np.tile(resp[t], (m.sum(), 1))
            r[:, eq] = np.maximum(r[:, eq] * s[:, None], -0.95)
            fund_ret[m] = r; ret[m] = r @ w
        return ti, ret, fund_ret
    return types, freq, pool, tmpl_depth, draw


def aggregate(draw, lam_t_fn, exposure_fn, years, rng, n=N_SIM, regime=None):
    """Monte Carlo of S = sum of crash losses.  If regime is given, crash times come from a Markov-modulated process."""
    if regime is None:
        N = rng.poisson(lam_t_fn * years, n)
        times = [rng.random(k) * years for k in N]
    else:
        P, lc, lt, p0 = regime; s = (rng.random(n) < p0).astype(int); times = [[] for _ in range(n)]
        for m in range(int(12 * years)):
            lam = np.where(s == 1, lt, lc) / 12
            hit = np.where(rng.random(n) < 1 - np.exp(-lam))[0]
            for i in hit:
                times[i].append((m + 0.5) / 12)
            s = np.where(rng.random(n) < P[s, 1], 1, 0)
        N = np.array([len(t) for t in times])
    tot = N.sum(); _, ret, _ = draw(tot)
    flat_t = np.concatenate([np.asarray(t, float) for t in times]) if tot else np.zeros(0)
    loss = -ret * exposure_fn(flat_t)                    # positive = dollars lost
    S = np.zeros(n); idx = np.repeat(np.arange(n), N); np.add.at(S, idx, loss)
    return N, S, loss


def main():
    rng = np.random.default_rng(SEED)
    M = pd.read_csv(PROC / "monthly.csv", index_col=0, parse_dates=True)[FUNDS].dropna()
    EV = json.load(open(RES / "crash_events.json"))
    h, w = weights()
    res = dict(holdings=h, weights=dict(zip(FUNDS, w.tolist())), equity_share=float(sum(h[f] for f in EQUITY) / INITIAL))
    # ---- Level 1
    rows, Tf = fund_level(M, h)
    res["fund_level"] = dict(period=[str(M.index[0].date()), str(M.index[-1].date())], years=Tf, rows=rows)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("episodes")} for r in rows]).to_csv(TABLES / "fund_level_poisson.csv", index=False)
    # ---- Level 2
    lam = EV["equity"]["20"]["lam"]; lam30 = EV["equity"]["30"]["lam"]
    resp = type_responses(M)
    types, freq, pool, tdepth, draw = crash_sampler(EV, resp, w, rng)
    ti, ret, fr = draw(N_SIM)
    by_type = {}
    for j, t in enumerate(types):
        m = ti == j
        by_type[t] = dict(share_of_crashes=float(freq[j]), depths_used=pool[t], template_depth=tdepth[t],
                          portfolio_return_mean=float(ret[m].mean()), portfolio_return_worst=float(ret[m].min()),
                          dollars_300k_mean=float(-ret[m].mean() * INITIAL), dollars_450k_mean=float(-ret[m].mean() * (INITIAL + TOP_UP)),
                          fund_dollars_300k={f: float(-fr[m, i].mean() * h[f]) for i, f in enumerate(FUNDS)},
                          template_fund_returns=dict(zip(FUNDS, resp[t].tolist())))
    res["loss_given_crash"] = dict(mean_return=float(ret.mean()), p5_return=float(np.percentile(ret, 5)), worst_return=float(ret.min()),
                                   mean_dollars_300k=float(-ret.mean() * INITIAL), by_type=by_type,
                                   fund_dollars_300k={f: float(-(fr[:, i] * h[f]).mean()) for i, f in enumerate(FUNDS)})
    pd.DataFrame({t: v["fund_dollars_300k"] for t, v in by_type.items()}).to_csv(TABLES / "dollar_loss_by_fund_and_type.csv")
    exposure = lambda t: np.where(t < 1, INITIAL, INITIAL + TOP_UP)
    wins = {"2027 only": (1, lambda t: np.full_like(t, INITIAL)), "2031-32": (2, lambda t: np.full_like(t, INITIAL + TOP_UP)),
            "2027-32": (6, exposure)}
    agg = {}
    for lab, (yrs, ex) in wins.items():
        N, S, L = aggregate(draw, lam, ex, yrs, rng)
        mean_an, var_an = pc.compound_moments(lam * yrs, L if len(L) else [0])
        agg[lab] = dict(years=yrs, lam_t=lam * yrs, pN=[float((N == k).mean()) for k in range(3)] + [float((N >= 3).mean())],
                        pN_exact=[pc.pmf(k, lam * yrs) for k in range(3)], expected_loss=float(S.mean()), expected_loss_formula=float(mean_an),
                        sd_loss=float(S.std()), sd_formula=float(math.sqrt(var_an)),
                        p_loss_ge={str(x): float((S >= x).mean()) for x in (10_000, 25_000, 50_000, 75_000, 100_000, 150_000)},
                        var95=float(np.percentile(S, 95)), var99=float(np.percentile(S, 99)),
                        cvar99=float(S[S >= np.percentile(S, 99)].mean()),
                        exceedance=[[float(x), float((S >= x).mean())] for x in np.linspace(0, 250_000, 51)])
    res["aggregate_poisson"] = agg
    # ---- Level 3: Markov-modulated
    mk = EV["markov"]; P = np.array(mk["P"])
    N, S, L = aggregate(draw, None, exposure, 6, np.random.default_rng(SEED + 1), n=100_000,
                        regime=(P, mk["lam_calm"], mk["lam_turbulent"], mk["p_turbulent_now"]))
    res["aggregate_mmpp_2027_32"] = dict(pN=[float((N == k).mean()) for k in range(3)] + [float((N >= 3).mean())], expected_loss=float(S.mean()),
                                         p_loss_ge={str(x): float((S >= x).mean()) for x in (10_000, 25_000, 50_000, 75_000, 100_000, 150_000)},
                                         var99=float(np.percentile(S, 99)),
                                         exceedance=[[float(x), float((S >= x).mean())] for x in np.linspace(0, 250_000, 51)])
    res["lambda"] = dict(ge20=lam, ge30=lam30, ge20_ci90=EV["equity"]["20"]["lam_ci90"])
    # ---- Level 4: probability each crash type affects Laura
    try:
        SR = json.load(open(RES / "strategy_results.json"))
        keymap = {"T1": "T1 Recession bear", "T2": "T2 Exogenous V-shock", "T3": "T3 Inflation/rate shock", "T3+": "T3+ Severe inflation shock",
                  "T4": "T4 Valuation bust", "T5": "T5 Falling-rate shock"}
        bt = EV["equity"]["20"]["episodes"]; T = EV["years"]
        n3 = sum(r["type"].startswith("T3") for r in bt); n3p = sum(r["type"].startswith("T3") and r["depth"] <= -0.30 for r in bt)
        lam_t = {"T1": sum(r["type"].startswith("T1") for r in bt) / T, "T2": sum(r["type"].startswith(("T2", "T5")) for r in bt) / T,
                 "T3": (n3 - n3p) / T, "T3+": n3p / T, "T4": sum(r["type"].startswith("T4") for r in bt) / T,
                 "T5": EV["other"]["R2 Falling-rate shock: 5y yield -2 pts within 24 months"]["lam"]}
        aff = []
        for k, lab in keymap.items():
            ck = next(c for c in SR["crash"] if c.startswith(lab))
            wst = SR["crash"][ck]["timing"]["worst"]; p_occ = pc.p_at_least_one(lam_t[k], 2)
            row = dict(type=k, scenario=ck, lam=lam_t[k], p_occurs_2031_32=p_occ)
            for s in ("S9", "S10", "S10C"):
                row[f"{s}_p_shortfall"] = p_occ * (1 - wst[s]["f4"]); row[f"{s}_p_promise_broken"] = p_occ * (1 - wst[s]["promise_kept"])
                row[f"{s}_shortfall_given_crash"] = 1 - wst[s]["f4"]; row[f"{s}_promise_broken_given_crash"] = 1 - wst[s]["promise_kept"]
            aff.append(row)
        res["affects_laura"] = aff
        pd.DataFrame(aff).to_csv(TABLES / "probability_crash_affects_laura.csv", index=False)
    except FileNotFoundError:
        res["affects_laura"] = "run strategy_sim.py first"
    json.dump(res, open(RES / "laura_poisson.json", "w"), indent=1, default=float)
    a6 = agg["2027-32"]
    print(f"lambda(>=20%) = {lam:.4f}; E[crash damage 2027-32] = ${a6['expected_loss']:,.0f} (formula ${a6['expected_loss_formula']:,.0f}); "
          f"P(damage >= $50k) = {a6['p_loss_ge']['50000']:.1%}; 99% VaR = ${a6['var99']:,.0f}")


if __name__ == "__main__":
    main()
