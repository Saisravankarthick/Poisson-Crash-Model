"""Step 4 - is the strategy ranking robust?  Joint Poisson, Markov and Monte Carlo tests of the Table 8.21 scoreboard.

Strategy set for the Final Report (team decision, 26 Nov 2026): LDCS = S10C (cushion rule + 2031 lock-in), against
S9 core-satellite as built (same ETFs, static), S7 risk parity, S0 bucket ladder, S1 T-bills, S2 pure LDI, S6 target-date,
S8 CPPI m=3 and S4 all-equity.  S9A and S10 are earlier versions of our own rule and are left out of the ranking (they
remain in strategy_results.json; the full ten-strategy scoreboard is reproduced as a check).

Tests
  A  Scoreboard: six criteria, ranks and three weightings (equal / safety / growth), as in Research Pack Table 8.21.
  B  Weight robustness: 20,000 random weightings of the six criteria (flat Dirichlet).
  C  Seed stability: the whole Monte Carlo (four cases + six crash replays) re-run on 20 fresh random seeds.
  D  Paired tests: every rival against LDCS on identical paths (McNemar exact test for funding and promise,
     paired mean difference for the contribution).
  E  Joint Poisson shortfall: sum over crash types of P(type occurs in 2031-32) x P(shortfall | worst-timed replay),
     with the union form and the 90% Garwood interval of each lambda.
  F  Poisson assumptions: dispersion by year and by decade, and exponential gaps between crashes (KS test).
  G  Markov: transition matrix, stationary shares, expected spell lengths, P(turbulence in 2031-32), and shortfall
     conditional on the months the jump-diffusion paths spend in the turbulent state during 2031-32.
Outputs: results/robustness.json, results/tables/{scoreboard_ranked,weight_robustness,seed_stability,paired_tests,
joint_poisson_shortfall,crash_type_promise,markov_conditional_shortfall}.csv, figures/16-19, report/ROBUSTNESS.md"""
import json
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
from config import RES, FIG, ROOT, PALETTE, INK, MUTED_INK, GRID
import strategy_sim as ss

RANKSET = ["S10C", "S9", "S7", "S0", "S1", "S2", "S6", "S8", "S4"]
FULL10 = ["S1", "S2", "S4", "S6", "S7", "S8", "S9", "S9A", "S10", "S10C"]
SHORT = {"S10C": "LDCS (ours)", "S9": "Core-satellite", "S7": "Risk parity", "S0": "Bucket ladder", "S1": "T-bills",
         "S2": "Pure LDI", "S6": "Target-date", "S8": "CPPI m=3", "S4": "All-equity", "S9A": "LDCS calendar", "S10": "LDCS rule only"}
CRIT = ["Minimum funding certainty", "Average funding certainty", "Median contribution", "Bad-case contribution (P5, J)",
        "Crash resilience", "Promise reliability"]
CRIT_DEF = ["Lowest chance, across the four cases, that the 2033 portfolio covers the $421,767 reserve",
            "Chance the 2033 portfolio covers the reserve, averaged over the four cases",
            "Typical 2033 amount above the reserve (average of the history and current-yield cases)",
            "Amount above the reserve in the worst 5% of jump-diffusion paths",
            "Chance the reserve is still covered when each of six historical crashes is replayed to end in January 2033",
            "How often the 2033 contribution meets the minimum announced to co-sponsors in 2031 (history, jumps, replays)"]
WEIGHTS = {"equal": [1, 1, 1, 1, 1, 1], "safety": [2, 2, 1, 1, 2, 2], "growth": [1, 1, 3, 1, 1, 1]}
TYPES = ["T1", "T2", "T3", "T3+", "T4", "T5"]
TNAME = {"T1": "2008-type recession", "T2": "2020-type V-shock", "T3": "2022-type inflation", "T3+": "1973-74-type inflation",
         "T4": "2000-02-type bust", "T5": "2019-20 falling rates"}
NSEED, NDIR = 20, 20000
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED_INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED_INK, "ytick.color": MUTED_INK, "axes.spines.top": False, "axes.spines.right": False})


# ------------------------------------------------------------------ helpers
def light(res):
    c = res["V33"] - ss.R4
    return dict(f4=float((res["V33"] >= ss.R4).mean()), p50=float(np.median(c)), p5=float(np.percentile(c, 5)),
                pk=float((c >= res["L"]).mean()))


def from_results(R, S):
    st = {c: {s: dict(f4=R[c][s]["f4"], p50=R[c][s]["p50"], p5=R[c][s]["p5"], pk=R[c][s]["promise_kept"]) for s in S} for c in "HSJC"}
    st["crash"] = {k: {s: dict(f4=R["crash"][k]["timing"]["worst"][s]["f4"], pk=R["crash"][k]["timing"]["worst"][s]["promise_kept"])
                       for s in S} for k in R["crash"]}
    return st


def criteria(st, S):
    ks = list(st["crash"])
    return [{s: min(st[c][s]["f4"] for c in "HSJC") for s in S},
            {s: float(np.mean([st[c][s]["f4"] for c in "HSJC"])) for s in S},
            {s: float(np.mean([st[c][s]["p50"] for c in "HC"])) for s in S},
            {s: st["J"][s]["p5"] for s in S},
            {s: float(np.mean([st["crash"][k][s]["f4"] for k in ks])) for s in S},
            {s: float(np.mean([st["H"][s]["pk"], st["J"][s]["pk"]] + [st["crash"][k][s]["pk"] for k in ks])) for s in S}]


def rank(cr, S):
    rk = [{s: sorted(S, key=lambda x: -c[x]).index(s) + 1 for s in S} for c in cr]
    tot = {w: {s: sum(a * r[s] for a, r in zip(ws, rk)) / sum(ws) for s in S} for w, ws in WEIGHTS.items()}
    return rk, tot


def garwood(n, T, conf=0.90):
    a = 1 - conf
    lo = 0.0 if n == 0 else stats.chi2.ppf(a / 2, 2 * n) / 2 / T
    return lo, stats.chi2.ppf(1 - a / 2, 2 * n + 2) / 2 / T


def main():
    R = json.load(open(RES / "strategy_results.json")); E = json.load(open(RES / "crash_events.json"))
    LP = json.load(open(RES / "laura_poisson.json")); out = {"set": RANKSET, "names": SHORT, "criteria": CRIT, "criteria_def": CRIT_DEF}
    T_ = RES / "tables"

    # ---------------- A. scoreboard
    base = from_results(R, RANKSET); cr = criteria(base, RANKSET); rk, tot = rank(cr, RANKSET)
    rows = []
    for s in RANKSET:
        rows.append({"strategy": s, "name": SHORT[s], **{CRIT[i]: cr[i][s] for i in range(6)}, **{f"rank {CRIT[i]}": rk[i][s] for i in range(6)},
                     **{f"avg rank {w}": tot[w][s] for w in WEIGHTS}})
    sb = pd.DataFrame(rows); sb.to_csv(T_ / "scoreboard_ranked.csv", index=False)
    out["scoreboard"] = dict(values={CRIT[i]: cr[i] for i in range(6)}, ranks={CRIT[i]: rk[i] for i in range(6)}, totals=tot,
                             place={w: sorted(RANKSET, key=lambda s: tot[w][s]) for w in WEIGHTS})
    cr10 = criteria(from_results(R, FULL10), FULL10); _, tot10 = rank(cr10, FULL10)
    out["full10_totals"] = tot10

    # ---------------- B. random weights
    g = np.random.default_rng(42); RK = np.array([[rk[i][s] for i in range(6)] for s in RANKSET], float)
    W = g.dirichlet(np.ones(6), NDIR); TT = RK @ W.T                         # strategies x draws
    order = np.argsort(TT, axis=0, kind="stable")
    first = {s: float((order[0] == j).mean()) for j, s in enumerate(RANKSET)}
    top3 = {s: float((order[:3] == j).any(0).mean()) for j, s in enumerate(RANKSET)}
    out["weights"] = dict(draws=NDIR, p_first=first, p_top3=top3)
    pd.DataFrame({"strategy": RANKSET, "name": [SHORT[s] for s in RANKSET], "p_rank_1": [first[s] for s in RANKSET],
                  "p_top_3": [top3[s] for s in RANKSET]}).to_csv(T_ / "weight_robustness.csv", index=False)

    # ---------------- C. seed stability (and D/G inputs from the base seed)
    seeds = []; keys = list(ss.EPIS)
    for k in range(-1, NSEED):
        sb_, sj = (42, 7) if k < 0 else (1000 + k, 2000 + k)
        if k < 0:
            cases, TM = ss.make_cases(sb_, sj, record=True)
        else:
            cases = ss.make_cases(sb_, sj)
        st = {c: {} for c in "HSJC"}; st["crash"] = {kk: {} for kk in keys}; paths = {}
        for c, (RR, DD) in cases.items():
            for s in RANKSET:
                res = ss.simulate(RR, DD, s); st[c][s] = light(res)
                if k < 0: paths[(c, s)] = (res["V33"], res["L"])
        RH_, DH_ = cases["H"]
        for kk in keys:
            R0, D0 = ss.crash_replay(RH_, DH_, kk)
            for s in RANKSET:
                st["crash"][kk][s] = light(ss.simulate(R0, D0, s))
        c_ = criteria(st, RANKSET); r_, t_ = rank(c_, RANKSET)
        rec = dict(seed=[sb_, sj], crit={CRIT[i]: c_[i] for i in range(6)}, ranks={CRIT[i]: r_[i] for i in range(6)}, totals=t_)
        if k < 0:
            base_check = rec; base_paths = paths; base_TM = TM
        else:
            seeds.append(rec)
        print("seed", sb_, {w: sorted(RANKSET, key=lambda s: t_[w][s])[0] for w in WEIGHTS}, flush=True)
    out["base_rerun_matches"] = float(max(abs(base_check["crit"][CRIT[i]][s] - cr[i][s]) for i in range(6) for s in RANKSET))
    stab = {}
    for w in WEIGHTS:
        places = {s: [sorted(RANKSET, key=lambda x: z["totals"][w][x]).index(s) + 1 for z in seeds] for s in RANKSET}
        stab[w] = {s: dict(first=float(np.mean([p == 1 for p in places[s]])), best=min(places[s]), worst=max(places[s]),
                           median=float(np.median(places[s]))) for s in RANKSET}
    spread = {CRIT[i]: {s: [float(np.mean([z["crit"][CRIT[i]][s] for z in seeds])), float(np.std([z["crit"][CRIT[i]][s] for z in seeds], ddof=1))]
                        for s in RANKSET} for i in range(6)}
    rank_hold = {CRIT[i]: {s: float(np.mean([z["ranks"][CRIT[i]][s] == rk[i][s] for z in seeds])) for s in RANKSET} for i in range(6)}
    out["seeds"] = dict(n=NSEED, place=stab, criterion_mean_sd=spread, criterion_rank_unchanged=rank_hold, runs=seeds)
    srows = []
    for s in RANKSET:
        row = {"strategy": s, "name": SHORT[s]}
        for w in WEIGHTS:
            row[f"{w}: share of seeds ranked 1st"] = stab[w][s]["first"]; row[f"{w}: best-worst place"] = f"{stab[w][s]['best']}-{stab[w][s]['worst']}"
        for i in range(6):
            row[f"{CRIT[i]} sd across seeds"] = spread[CRIT[i]][s][1]
        srows.append(row)
    pd.DataFrame(srows).to_csv(T_ / "seed_stability.csv", index=False)

    # ---------------- D. paired tests against LDCS on the same paths
    prow = []; n = ss.PATHS
    for c in "HSJC":
        V0, L0 = base_paths[(c, "S10C")]; c0 = V0 - ss.R4; ok0 = V0 >= ss.R4; pk0 = c0 >= L0
        for s in RANKSET[1:]:
            V1, L1 = base_paths[(c, s)]; c1 = V1 - ss.R4; ok1 = V1 >= ss.R4; pk1 = c1 >= L1
            b, cc = int((ok0 & ~ok1).sum()), int((~ok0 & ok1).sum())
            bp, cp = int((pk0 & ~pk1).sum()), int((~pk0 & pk1).sum())
            p_f = stats.binomtest(b, b + cc, 0.5).pvalue if b + cc else 1.0
            p_p = stats.binomtest(bp, bp + cp, 0.5).pvalue if bp + cp else 1.0
            d = c0 - c1
            prow.append(dict(case=c, rival=s, rival_name=SHORT[s], f4_ldcs=float(ok0.mean()), f4_rival=float(ok1.mean()),
                             se_f4_ldcs=float(math.sqrt(ok0.mean() * (1 - ok0.mean()) / n)), ldcs_only_funded=b, rival_only_funded=cc,
                             p_funding=float(p_f), pk_ldcs=float(pk0.mean()), pk_rival=float(pk1.mean()), ldcs_only_kept=bp,
                             rival_only_kept=cp, p_promise=float(p_p), mean_contrib_diff=float(d.mean()),
                             se_contrib_diff=float(d.std(ddof=1) / math.sqrt(n)), share_paths_ldcs_higher=float((d > 0).mean())))
    PT = pd.DataFrame(prow); PT.to_csv(T_ / "paired_tests.csv", index=False)
    verdict = {}
    for s in RANKSET[1:]:
        q = PT[PT.rival == s]
        verdict[s] = dict(funding=[int(((q.p_funding < 0.05) & (q.ldcs_only_funded > q.rival_only_funded)).sum()),
                                   int(((q.p_funding < 0.05) & (q.ldcs_only_funded < q.rival_only_funded)).sum())],
                          promise=[int(((q.p_promise < 0.05) & (q.ldcs_only_kept > q.rival_only_kept)).sum()),
                                   int(((q.p_promise < 0.05) & (q.ldcs_only_kept < q.rival_only_kept)).sum())],
                          contribution_higher_cases=int((q.mean_contrib_diff - 1.96 * q.se_contrib_diff > 0).sum()),
                          contribution_lower_cases=int((q.mean_contrib_diff + 1.96 * q.se_contrib_diff < 0).sum()))
    out["paired"] = dict(verdict=verdict, note="counts of the four cases where LDCS is significantly better / worse (p<0.05)")

    # ---------------- E. joint Poisson shortfall (Table 8.14 extended)
    AL = {a["type"]: a for a in LP["affects_laura"]}
    nT = {"T1": (1, E["years"]), "T2": (3, E["years"]), "T3": (2, E["years"]), "T3+": (2, E["years"]), "T4": (2, E["years"])}
    r2 = E["other"]["R2 Falling-rate shock: 5y yield -2 pts within 24 months"]; nT["T5"] = (r2["n"], r2["years"])
    jrows, trows = [], []
    lam = {t: AL[t]["lam"] for t in TYPES}; ci = {t: garwood(*nT[t]) for t in TYPES}
    for t in TYPES:
        assert abs(nT[t][0] / nT[t][1] - lam[t]) < 1e-9, t
        trows.append({"type": t, "name": TNAME[t], "events": nT[t][0], "years": nT[t][1], "lambda": lam[t], "lambda_lo90": ci[t][0],
                      "lambda_hi90": ci[t][1], "p_2031_32": 1 - math.exp(-2 * lam[t]), "p_2027_32": 1 - math.exp(-6 * lam[t]),
                      "p_2027_42": 1 - math.exp(-16 * lam[t]),
                      **{f"promise_kept_{s}": R["crash"][AL[t]["scenario"]]["timing"]["worst"][s]["promise_kept"] for s in RANKSET},
                      **{f"funded_{s}": R["crash"][AL[t]["scenario"]]["timing"]["worst"][s]["f4"] for s in RANKSET}})
    for s in RANKSET:
        q = {t: 1 - R["crash"][AL[t]["scenario"]]["timing"]["worst"][s]["f4"] for t in TYPES}
        b_ = {t: 1 - R["crash"][AL[t]["scenario"]]["timing"]["worst"][s]["promise_kept"] for t in TYPES}
        P = lambda l: {t: 1 - math.exp(-2 * l[t]) for t in TYPES}
        p, plo, phi = P(lam), P({t: ci[t][0] for t in TYPES}), P({t: ci[t][1] for t in TYPES})
        jrows.append(dict(strategy=s, name=SHORT[s],
                          shortfall_sum=sum(p[t] * q[t] for t in TYPES), shortfall_union=1 - np.prod([1 - p[t] * q[t] for t in TYPES]),
                          shortfall_sum_lo=sum(plo[t] * q[t] for t in TYPES), shortfall_sum_hi=sum(phi[t] * q[t] for t in TYPES),
                          broken_sum=sum(p[t] * b_[t] for t in TYPES), broken_union=1 - np.prod([1 - p[t] * b_[t] for t in TYPES]),
                          broken_sum_lo=sum(plo[t] * b_[t] for t in TYPES), broken_sum_hi=sum(phi[t] * b_[t] for t in TYPES)))
    JP = pd.DataFrame(jrows); JP.to_csv(T_ / "joint_poisson_shortfall.csv", index=False)
    CT = pd.DataFrame(trows); CT.to_csv(T_ / "crash_type_promise.csv", index=False)
    out["joint_poisson"] = JP.set_index("strategy").to_dict("index"); out["crash_types"] = trows

    # ---------------- F. Poisson assumptions
    ep = pd.read_csv(T_ / "crash_episodes.csv"); yrs = ep["peak"].str[:4].astype(int)
    dec = np.array([((yrs >= y) & (yrs < y + 10)).sum() for y in range(1926, 2026, 10)])
    Dd = dec.var(ddof=1) / dec.mean(); chi_d = (len(dec) - 1) * Dd
    gaps = pd.read_csv(T_ / "gaps_between_crashes.csv")
    gcol = [c for c in gaps.columns if "year" in c.lower() or "gap" in c.lower()][-1]; gv = gaps[gcol].dropna().to_numpy(float)
    ks = stats.kstest(gv, "expon", args=(0, 1 / E["equity"]["20"]["lam"]))
    out["poisson_tests"] = dict(annual=E["equity"]["20"]["dispersion"],
                                decade=dict(counts=dec.tolist(), D=float(Dd), chi2=float(chi_d), df=len(dec) - 1,
                                            p_overdispersed=float(1 - stats.chi2.cdf(chi_d, len(dec) - 1))),
                                gaps=dict(column=gcol, n=len(gv), mean=float(gv.mean()), ks_stat=float(ks.statistic), ks_p=float(ks.pvalue)))

    # ---------------- G. Markov: matrix facts and conditional shortfall
    Pm = np.array(E["markov"]["P"]); pi = np.array(E["markov"]["stationary"]); p0 = E["markov"]["p_turbulent_now"]
    start = np.array([1 - p0, p0]); s48 = start @ np.linalg.matrix_power(Pm, 48)
    p_calm24 = s48[0] * Pm[0, 0] ** 23
    tm = base_TM[:, 48:72].sum(1); bins = [(0, 0), (1, 6), (7, 12), (13, 24)]
    mrows = []
    for lo, hi in bins:
        sel = (tm >= lo) & (tm <= hi)
        for s in RANKSET:
            V, L = base_paths[("J", s)]; c = V - ss.R4
            mrows.append(dict(turbulent_months_2031_32=f"{lo}" if lo == hi else f"{lo}-{hi}", share_of_paths=float(sel.mean()),
                              strategy=s, name=SHORT[s], p_shortfall=float((V[sel] < ss.R4).mean()) if sel.any() else np.nan,
                              p_promise_broken=float((c[sel] < L[sel]).mean()) if sel.any() else np.nan,
                              median_contribution=float(np.median(c[sel])) if sel.any() else np.nan))
    MC = pd.DataFrame(mrows); MC.to_csv(T_ / "markov_conditional_shortfall.csv", index=False)
    out["markov"] = dict(P=Pm.tolist(), stationary=pi.tolist(), pi_check=float(np.abs(pi @ Pm - pi).max()),
                         expected_spell_months=[float(1 / (1 - Pm[0, 0])), float(1 / (1 - Pm[1, 1]))], p_turbulent_now=p0,
                         p_any_turbulent_2031_32_exact=float(1 - p_calm24), p_any_turbulent_2031_32_sim=float((tm > 0).mean()),
                         conditional=mrows)

    json.dump(out, open(RES / "robustness.json", "w"), indent=1, default=float)
    figures(out, CT, JP, MC, R)
    write_md(out, sb, PT, JP, CT, MC)
    return out


# ------------------------------------------------------------------ figures
COL = {"S10C": PALETTE[0], "S9": PALETTE[1], "S7": PALETTE[2], "S0": PALETTE[3]}
NEUTRAL = "#a9a9a9"


def figures(out, CT, JP, MC, R=None):
    # 16: promise kept by crash type (small multiples), LDCS vs the rivals named in the report, + joint Poisson panel
    show = ["S10C", "S9", "S0", "S7", "S1", "S2", "S6"]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.7), sharey=True)
    y = np.arange(len(show))[::-1]
    for ax, (_, row) in zip(axs.flat, CT.iterrows()):
        v = [row[f"promise_kept_{s}"] for s in show]
        ax.barh(y, v, height=0.68, color=[COL["S10C"] if s == "S10C" else NEUTRAL for s in show])
        for yy, vv in zip(y, v):
            ax.text(min(vv, 1) + 0.03, yy, f"{vv:.0%}", va="center", fontsize=6.3, color=INK)
        ax.set_xlim(0, 1.32); ax.set_xticks([0, 0.5, 1]); ax.xaxis.set_major_formatter(PercentFormatter(1, 0))
        ax.set_title(f"{row['name']}\nP(in 2031-32) = {row['p_2031_32']:.1%}", fontsize=7, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=6.5)
    ax = axs.flat[6]; v = [out["joint_poisson"][s]["broken_sum"] for s in show]
    ax.barh(y, v, height=0.68, color=[COL["S10C"] if s == "S10C" else NEUTRAL for s in show])
    for yy, vv in zip(y, v):
        ax.text(vv + 0.002, yy, f"{vv:.2%}", va="center", fontsize=6.3, color=INK)
    ax.set_xlim(0, max(v) * 1.45); ax.xaxis.set_major_formatter(PercentFormatter(1, 0)); ax.tick_params(labelsize=6.5)
    ax.set_title("Joint Poisson risk of a\nbroken promise, 2031-32", fontsize=7, color=INK, loc="left"); ax.grid(axis="x", color=GRID, lw=0.6)
    axs.flat[7].axis("off")
    axs.flat[7].text(0, 0.95, "Bars 1-6: share of paths that keep\nthe 2031 promise when the crash is\nreplayed to end in January 2033.\n\n"
                     "Bar 7: sum over types of\nP(type occurs, Poisson) x\nP(promise broken | replay).\nAn upper bound.",
                     fontsize=6.5, color=MUTED_INK, va="top")
    for a in axs[:, 0]:
        a.set_yticks(y, [SHORT[s] for s in show], fontsize=6.8)
    fig.suptitle("Promise kept under each crash type, LDCS (blue) against the alternatives", fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=0.6, h_pad=0.9); fig.savefig(FIG / "16_crash_type_promise_by_strategy.png", dpi=300); plt.close(fig)

    # 17: ogive of the 2033 contribution, case H and case J
    import strategy_sim as s_
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, c, ttl in ((axs[0], "H", "History (case H)"), (axs[1], "J", "Crash-jump model (case J)")):
        cases = s_.make_cases(42, 7)
        RR, DD = cases[c]
        order = ["S1", "S2", "S6", "S8", "S4", "S0", "S7", "S9", "S10C"]
        dash = {"S1": (0, (1, 1.2)), "S2": "-.", "S6": "--", "S8": (0, (6, 1.5, 1, 1.5)), "S4": (0, (8, 2))}
        for s in order:
            cc = np.sort(s_.simulate(RR, DD, s)["V33"] - s_.R4) / 1000; pp = np.arange(1, len(cc) + 1) / len(cc)
            col = COL.get(s, "#8c8c8c"); lw = 2.0 if s == "S10C" else (1.3 if s in COL else 0.9)
            ax.plot(cc, pp, color=col, lw=lw, ls=dash.get(s, "-"), zorder=3 if s in COL else 2, label=SHORT[s])
        ax.axvline(0, color=INK, lw=0.7, ls=(0, (3, 2))); ax.set_xlim(-250, 650); ax.set_title(ttl, fontsize=8, loc="left", color=INK)
        ax.set_xlabel("2033 contribution above the \\$421,767 reserve (\\$ thousand)", fontsize=7); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(PercentFormatter(1, 0)); ax.tick_params(labelsize=6.8)
    axs[0].set_ylabel("Share of paths at or below", fontsize=7)
    h, l = axs[0].get_legend_handles_labels(); idx = [l.index(SHORT[x]) for x in ["S10C", "S9", "S7", "S0", "S6", "S8", "S4", "S2", "S1"]]
    fig.legend([h[i] for i in idx], [l[i] for i in idx], fontsize=6.5, frameon=False, loc="lower center", ncol=5, handlelength=2.8, columnspacing=1.4)
    fig.suptitle("Ogive: cumulative distribution of the 2033 contribution (left of the dashed line = reserve not covered)",
                 fontsize=8.2, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.11, 1, 0.95)); fig.savefig(FIG / "17_contribution_ogive.png", dpi=300); plt.close(fig)

    # 18: rank robustness under random weights (seed stability reported in the note: every seed gives the same first place)
    S = out["set"]; fig, ax = plt.subplots(figsize=(5.2, 2.9)); y = np.arange(len(S))[::-1]
    w1 = [out["weights"]["p_first"][s] for s in S]; w3 = [out["weights"]["p_top3"][s] for s in S]
    ax.barh(y + 0.18, w3, height=0.34, color="#b9c9de", label="In the top 3"); ax.barh(y - 0.18, w1, height=0.34, color=PALETTE[0], label="Ranked 1st")
    for yy, a_, b_ in zip(y, w1, w3):
        ax.text(b_ + 0.015, yy + 0.18, f"{b_:.0%}", va="center", fontsize=6.2); ax.text(a_ + 0.015, yy - 0.18, f"{a_:.0%}", va="center", fontsize=6.2)
    ax.set_title(f"Share of {out['weights']['draws']:,} random weightings of the six criteria", fontsize=7.5, loc="left")
    st = out["seeds"]["place"]; n1 = sum(st[w]["S10C"]["first"] == 1.0 for w in WEIGHTS)
    ax.text(1.0, 0.02, f"Seed check: LDCS ranked 1st in {pct(min(st[w]['S10C']['first'] for w in WEIGHTS), 0)} of\n{out['seeds']['n']} re-run Monte Carlo seeds under all {n1} weightings",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color=MUTED_INK)
    ax.legend(fontsize=6.5, frameon=False, loc="center right")
    ax.set_xlim(0, 1.15); ax.xaxis.set_major_formatter(PercentFormatter(1, 0)); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.8); ax.set_yticks(y, [SHORT[s] for s in S], fontsize=6.8)
    fig.tight_layout(); fig.savefig(FIG / "18_rank_robustness.png", dpi=300); plt.close(fig)

    # 19: Markov conditional shortfall
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    bl = list(dict.fromkeys(MC.turbulent_months_2031_32)); x = np.arange(len(bl))
    for s in ["S10C", "S9", "S7", "S0"]:
        q = MC[MC.strategy == s]
        axs[0].plot(x, q.p_shortfall, marker="o", ms=4, lw=1.6 if s == "S10C" else 1.1, color=COL[s], label=SHORT[s])
        axs[1].plot(x, q.p_promise_broken, marker="o", ms=4, lw=1.6 if s == "S10C" else 1.1, color=COL[s], label=SHORT[s])
    shares = MC.drop_duplicates("turbulent_months_2031_32").share_of_paths.tolist()
    for ax, t in ((axs[0], "P(reserve not covered in 2033)"), (axs[1], "P(2031 promise broken)")):
        ax.set_xticks(x, [f"{b}\n({sh:.0%} of paths)" for b, sh in zip(bl, shares)], fontsize=6.3)
        ax.set_xlabel("Months in the turbulent regime during 2031-32", fontsize=7); ax.set_title(t, fontsize=7.5, loc="left")
        ax.yaxis.set_major_formatter(PercentFormatter(1, 0)); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=6.8)
    axs[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "19_markov_conditional_shortfall.png", dpi=300); plt.close(fig)


# ------------------------------------------------------------------ report page
def pct(x, d=1): return f"{100 * x:.{d}f}%"
def usd(x): return f"{'-' if x < 0 else ''}${abs(x) / 1000:.0f}k"


def md(h, rows):
    return "\n".join(["| " + " | ".join(h) + " |", "|" + "---|" * len(h)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write_md(o, sb, PT, JP, CT, MC):
    S = o["set"]; sc = o["scoreboard"]; L = []
    L.append("# Robustness of the strategy ranking\n")
    L.append("Generated by `src/robustness.py` from `results/strategy_results.json` and `results/crash_events.json`. "
             "**LDCS** in this page is S10C: hold Treasury, TIPS and T-bill funds worth the present value of Laura's ten payments; "
             "invest only the amount above that floor in equities, capped at 39.7%; reset each January and after any 20% fall in the "
             "equity sleeve; from the 2031 co-sponsor announcement add the announced minimum to the floor if two-year Treasury yields "
             "are at least 4%. S9A and S10 (earlier versions of the same rule) are left out of the ranking; S9 is the same ETFs held "
             "static, i.e. core-satellite without the floor rule.\n")
    L.append("## A. Scoreboard (Research Pack Table 8.21, nine strategies)\n")
    L.append(md(["Criterion", "Definition"], list(zip(CRIT, CRIT_DEF))) + "\n")
    rows = []
    for s in S:
        v = [pct(sc["values"][CRIT[0]][s]), pct(sc["values"][CRIT[1]][s]), usd(sc["values"][CRIT[2]][s]), usd(sc["values"][CRIT[3]][s]),
             pct(sc["values"][CRIT[4]][s]), pct(sc["values"][CRIT[5]][s])]
        rows.append([f"{s} {SHORT[s]}"] + [f"{a} ({sc['ranks'][CRIT[i]][s]})" for i, a in enumerate(v)] +
                    [" / ".join(f"{sc['totals'][w][s]:.2f}" for w in WEIGHTS)])
    L.append(md(["Strategy", "Min. funding certainty", "Avg. funding certainty", "Median contribution", "Bad-case P5 (J)", "Crash resilience",
                 "Promise reliability", "Avg rank equal / safety / growth"], rows) + "\n")
    L.append("Order: " + "; ".join(f"**{w}** " + ", ".join(sc["place"][w]) for w in WEIGHTS) + ".\n")
    t10 = o["full10_totals"]
    L.append(f"With all ten original strategies (Table 8.21) LDCS scores {t10['equal']['S10C']:.2f} / {t10['safety']['S10C']:.2f} / "
             f"{t10['growth']['S10C']:.2f}; the calendar glide path S9A scores {t10['equal']['S9A']:.2f} / {t10['safety']['S9A']:.2f} / "
             f"{t10['growth']['S9A']:.2f}. Leaving out our own two earlier versions is what turns that tie into a clear first place.\n")
    L.append("## B. Random weights\n")
    L.append(md(["Strategy", "P(ranked 1st)", "P(top 3)"], [[f"{s} {SHORT[s]}", pct(o['weights']['p_first'][s]), pct(o['weights']['p_top3'][s])] for s in S]) + "\n")
    L.append(f"{o['weights']['draws']:,} weightings drawn from a flat Dirichlet distribution (every mix of the six criteria equally likely).\n")
    L.append("## C. Seed stability\n")
    st = o["seeds"]["place"]
    L.append(md(["Strategy"] + [f"{w}: 1st in / place range" for w in WEIGHTS],
                [[f"{s} {SHORT[s]}"] + [f"{pct(st[w][s]['first'], 0)} / {st[w][s]['best']}-{st[w][s]['worst']}" for w in WEIGHTS] for s in S]) + "\n")
    L.append(f"The whole Monte Carlo (four cases and six crash replays, 10,000 paths each) was re-run on {o['seeds']['n']} fresh seeds. "
             f"The base seed was also re-run through the same code and reproduces the scoreboard to {o['base_rerun_matches']:.1e}.\n")
    sd = o["seeds"]["criterion_mean_sd"]
    L.append(md(["Strategy"] + [f"{c} (mean ± sd)" for c in CRIT],
                [[f"{s} {SHORT[s]}"] + [(f"{usd(sd[c][s][0])} ± {sd[c][s][1] / 1000:.1f}k" if "contribution" in c.lower() else
                                        f"{pct(sd[c][s][0])} ± {100 * sd[c][s][1]:.2f}") for c in CRIT] for s in S]) + "\n")
    L.append("## D. Paired tests against LDCS (same 10,000 paths, base seed)\n")
    v = o["paired"]["verdict"]
    L.append(md(["Rival", "Funding: LDCS better / worse (of 4 cases)", "Promise: better / worse", "Contribution: LDCS higher / lower"],
                [[f"{s} {SHORT[s]}", f"{v[s]['funding'][0]} / {v[s]['funding'][1]}", f"{v[s]['promise'][0]} / {v[s]['promise'][1]}",
                  f"{v[s]['contribution_higher_cases']} / {v[s]['contribution_lower_cases']}"] for s in S[1:]]) + "\n")
    L.append("Funding and promise use McNemar's exact test on the paths where exactly one of the two strategies succeeds; the contribution "
             "uses the paired mean difference ± 1.96 standard errors. Full detail: `results/tables/paired_tests.csv`.\n")
    L.append("## E. Joint Poisson shortfall in 2031-32\n")
    L.append(md(["Crash type", "Events / years", "λ (90% range)", "P(in 2031-32)"] + [SHORT[s] for s in ["S10C", "S9", "S0", "S7", "S1", "S2", "S6"]],
                [[r["name"], f"{r['events']} / {r['years']:.0f}", f"{r['lambda']:.3f} ({r['lambda_lo90']:.3f}-{r['lambda_hi90']:.3f})", pct(r["p_2031_32"])] +
                 [pct(r[f"promise_kept_{s}"], 0) for s in ["S10C", "S9", "S0", "S7", "S1", "S2", "S6"]] for r in o["crash_types"]]) + "\n")
    L.append("Strategy columns: share of paths keeping the 2031 promise when the crash is replayed to end in January 2033.\n")
    L.append(md(["Strategy", "Reserve shortfall: sum (union) [λ range]", "Broken promise: sum (union) [λ range]"],
                [[f"{s} {SHORT[s]}", f"{pct(JP.set_index('strategy').loc[s, 'shortfall_sum'], 2)} ({pct(JP.set_index('strategy').loc[s, 'shortfall_union'], 2)}) "
                  f"[{pct(JP.set_index('strategy').loc[s, 'shortfall_sum_lo'], 2)}-{pct(JP.set_index('strategy').loc[s, 'shortfall_sum_hi'], 2)}]",
                  f"{pct(JP.set_index('strategy').loc[s, 'broken_sum'], 2)} ({pct(JP.set_index('strategy').loc[s, 'broken_union'], 2)}) "
                  f"[{pct(JP.set_index('strategy').loc[s, 'broken_sum_lo'], 2)}-{pct(JP.set_index('strategy').loc[s, 'broken_sum_hi'], 2)}]"] for s in S]) + "\n")
    L.append("Sum = Σ P(type occurs in 2031-32) × P(failure | worst-timed replay), an upper bound; union = 1 − Π(1 − each term), "
             "treating types as independent; the bracket uses each λ's 90% Garwood interval.\n")
    pt = o["poisson_tests"]
    L.append("## F. Are crashes Poisson?\n")
    L.append(f"- By year: index of dispersion {pt['annual']['D']:.2f}, p(overdispersed) = {pt['annual']['p_overdispersed']:.2f}.\n"
             f"- By decade (counts {pt['decade']['counts']}): index {pt['decade']['D']:.2f}, p = {pt['decade']['p_overdispersed']:.2f}.\n"
             f"- Gaps between crashes ({pt['gaps']['n']}, mean {pt['gaps']['mean']:.1f} years) against an exponential with mean 1/λ: "
             f"KS = {pt['gaps']['ks_stat']:.2f}, p = {pt['gaps']['ks_p']:.2f}.\n"
             "None rejects a constant-rate Poisson at the 5% level. Ten events give these tests little power, which is why the "
             "Markov layer is kept as the clustering alternative.\n")
    m = o["markov"]
    L.append("## G. Markov regimes and conditional shortfall\n")
    L.append(f"Monthly transition matrix (calm, turbulent): [[{m['P'][0][0]:.3f}, {m['P'][0][1]:.3f}], [{m['P'][1][0]:.3f}, {m['P'][1][1]:.3f}]]; "
             f"long-run shares {pct(m['stationary'][0])} calm / {pct(m['stationary'][1])} turbulent; expected spells "
             f"{m['expected_spell_months'][0]:.0f} and {m['expected_spell_months'][1]:.1f} months. Starting from today's "
             f"{pct(m['p_turbulent_now'])} turbulent probability, the chance of at least one turbulent month in 2031-32 is "
             f"{pct(m['p_any_turbulent_2031_32_exact'])} (matrix powers) against {pct(m['p_any_turbulent_2031_32_sim'])} in the simulated paths.\n")
    rows = []
    for b in dict.fromkeys(MC.turbulent_months_2031_32):
        q = MC[MC.turbulent_months_2031_32 == b].set_index("strategy")
        rows.append([b, pct(q.share_of_paths.iloc[0])] + [f"{pct(q.loc[s, 'p_shortfall'])} / {pct(q.loc[s, 'p_promise_broken'])}" for s in ["S10C", "S9", "S0", "S7", "S2"]])
    L.append(md(["Turbulent months in 2031-32", "Share of paths"] + [f"{SHORT[s]}: shortfall / broken" for s in ["S10C", "S9", "S0", "S7", "S2"]], rows) + "\n")
    L.append("Case J paths, base seed. Total shortfall = Σ share × conditional shortfall (checked in verify.py).\n")
    L.append("## Limits of these tests\n")
    L.append("- Seed stability measures Monte Carlo sampling noise only. Every seed resamples the same 2004-2026 history, uses the same six "
             "crash replays and the same model, so it says nothing about model risk; the four cases (H, S, C, J) and the λ ranges are the model-risk checks.\n"
             "- The six criteria and the three weightings are team judgements. Random weights test the weights, not the choice of criteria.\n"
             "- With 10,000 paths the paired tests detect very small differences. Statistically significant is not the same as large enough to matter to Laura.\n"
             "- Crash replays are deterministic worst-timing stresses. The joint Poisson figures are upper bounds and the union form assumes crash types are independent.\n"
             "- The simulation ends in January 2033; the payout years 2033-42 are not simulated.\n"
             "- S0 is one bucket design (a liability ladder refilled only after up years). Other bucket designs could rank differently.\n")
    L.append("## Figures\n\n![](../figures/16_crash_type_promise_by_strategy.png)\n\n![](../figures/17_contribution_ogive.png)\n\n"
             "![](../figures/18_rank_robustness.png)\n\n![](../figures/19_markov_conditional_shortfall.png)\n")
    (ROOT / "report" / "ROBUSTNESS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
