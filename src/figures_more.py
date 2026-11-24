"""Step 5b - the other charts and tables behind the final report: crash history, goodness of fit, Markov regimes and the
Monte Carlo strategy layer.  Same muted palette as figures.py.  Output: figures/07-14 and results/tables/*.csv"""
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from config import RES, FIG, PROC, PALETTE, INK, MUTED_INK, GRID
import figures as F

B, O, V, G = PALETTE
GREY = "#9a9a9a"
TYPE_COL = {"T1": B, "T2": O, "T3": V, "T4": G, "T5": GREY}
TYPE_NAME = {"T1": "Recession", "T2": "V-shock", "T3": "Inflation/rate", "T4": "Valuation bust", "T5": "Other"}
usd_k = FuncFormatter(lambda x, _: f"{'−' if x < 0 else ''}${abs(x)/1000:,.0f}k")
TAB = RES / "tables"; TAB.mkdir(exist_ok=True)
STRATS = ["S1", "S2", "S4", "S6", "S7", "S8", "S9", "S9A", "S10", "S10C"]
SNAME = {"S1": "S1 T-bills", "S2": "S2 pure LDI", "S4": "S4 all-equity", "S6": "S6 target-date", "S7": "S7 risk parity", "S8": "S8 CPPI m=3",
         "S9": "S9 LDCS static", "S9A": "S9A LDCS + glide", "S10": "S10 LDCS + cushion", "S10C": "S10C cushion + lock-in"}
CASES = [("H", "H historical", B, "o"), ("S", "S stressed (−3.25 pts/yr)", O, "s"), ("C", "C current yields", G, "^"), ("J", "J jump-diffusion", V, "D")]


def ymd(s):
    return pd.Timestamp(s + "-01")


def main():
    E = json.load(open(RES / "crash_events.json")); L = json.load(open(RES / "laura_poisson.json")); S = json.load(open(RES / "strategy_results.json"))
    e20 = E["equity"]["20"]; lam = e20["lam"]; T = e20["years"]; eps = e20["episodes"]
    r = pd.read_csv(PROC / "us_market_long.csv", index_col=0, parse_dates=True).iloc[:, 0]
    lv = (1 + r).cumprod(); dd = lv / lv.cummax() - 1

    # ---- 07 drawdown history with the ten falls coloured by type
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.fill_between(dd.index, dd.values, 0, color=GRID, lw=0)
    ax.plot(dd.index, dd.values, color=MUTED_INK, lw=0.7)
    for e in eps:
        c = TYPE_COL[e["type"][:2]]
        a, b = ymd(e["peak"]), ymd(e["recovered"] or e["trough"])
        ax.axvspan(a, b, color=c, alpha=0.28, lw=0)
        ax.text(ymd(e["trough"]), e["depth"] - 0.045, f"{e['depth']*100:.0f}%", ha="center", fontsize=7, color=INK)
    ax.axhline(-0.20, color=MUTED_INK, lw=0.7, ls="--"); ax.text(pd.Timestamp("1990-03-01"), -0.215, "−20% line", fontsize=7.5, color=MUTED_INK, va="top")
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0)); ax.set_ylim(-0.95, 0.03); ax.set_ylabel("Fall from previous peak"); F.grid(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TYPE_COL[k], alpha=0.5, label=v) for k, v in TYPE_NAME.items() if any(x["type"][:2] == k for x in eps)], frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.52, 0.0), ncol=3)
    ax.set_title("US total market 1926–2026: the ten falls of 20% or more, peak to recovery, by type", loc="left")
    fig.savefig(FIG / "07_drawdown_history_by_type.png"); plt.close(fig)

    # ---- 08 waiting times between crash starts vs the exponential law (Poisson <-> exponential)
    peaks = sorted(ymd(e["peak"]) for e in eps); gaps = np.diff([p.year + (p.month - 1) / 12 for p in peaks])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    x = np.sort(gaps); ax = axs[0]
    ax.step(np.concatenate([[0], x]), np.arange(0, len(x) + 1) / len(x), where="post", color=B, lw=2, label="Observed gaps (9)")
    t = np.linspace(0, 45, 200); ax.plot(t, 1 - np.exp(-lam * t), color=O, lw=2, ls="--", label=f"Exponential, λ = {lam:.3f}")
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0)); ax.set_xlabel("Years between crash starts"); ax.set_ylabel("Share of gaps ≤ t"); F.grid(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_title("Waiting time between crashes", loc="left", fontsize=9.5)
    ks = stats.kstest(gaps, "expon", args=(0, 1 / lam)); ax.text(0.03, 0.93, f"KS test p = {ks.pvalue:.2f}: exponential not rejected", transform=ax.transAxes, ha="left", fontsize=8, color=INK)
    ax = axs[1]
    yr = pd.Series(0, index=range(1926, 2027))
    for e in eps:
        yr[int(e["peak"][:4])] += 1
    obs = yr.value_counts().reindex([0, 1, 2], fill_value=0).values; n_y = len(yr); expd = [n_y * stats.poisson.pmf(k, lam) for k in range(3)]
    k = np.arange(3); ax.bar(k - 0.19, obs, 0.36, color=B, label="Observed years"); ax.bar(k + 0.19, expd, 0.36, color=O, label="Poisson expects")
    for xi, v in zip(k - 0.19, obs): ax.text(xi, v + 1, f"{v}", ha="center", fontsize=8)
    for xi, v in zip(k + 0.19, expd): ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(k, ["0 crashes", "1 crash", "2 crashes"]); ax.set_ylabel(f"Calendar years (of {n_y})"); F.grid(ax)
    ax.legend(frameon=False, fontsize=8); ax.set_title("Crash starts per year", loc="left", fontsize=9.5)
    ax.text(0.97, 0.45, f"var/mean = {e20['dispersion']['D']}\np = {e20['dispersion']['p_overdispersed']:.2f}", transform=ax.transAxes, ha="right", fontsize=8, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "08_poisson_goodness_of_fit.png"); plt.close(fig)
    pd.DataFrame({"gap_years": gaps}).to_csv(TAB / "gaps_between_crashes.csv", index=False)

    # ---- 09 fat tails: observed monthly falls against what a normal model expects
    nc = E["normal_check"]; keys = [k_ for k_ in nc if k_ != "moments"]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    xs = np.arange(len(keys)); ob = [nc[k_]["observed"] for k_ in keys]; ne = [nc[k_]["normal_expected"] for k_ in keys]
    ax.bar(xs - 0.19, ob, 0.36, color=B, label="Observed (1,204 months)"); ax.bar(xs + 0.19, ne, 0.36, color=O, label="Normal model expects")
    ax.set_yscale("log"); ax.set_xticks(xs, [f"Month falls ≥ {k_.replace('<= -', '')}" for k_ in keys]); ax.set_ylabel("Number of months (log scale)"); F.grid(ax)
    for xi, a_, b_ in zip(xs, ob, ne):
        ax.text(xi - 0.19, a_ * 1.25, f"{a_}", ha="center", fontsize=8); ax.text(xi + 0.19, max(b_, 0.012) * 1.25, f"{b_:.2f}", ha="center", fontsize=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right"); mo = nc["moments"]
    ax.set_title(f"Why jumps are needed: skew {mo['skew']:.2f}, excess kurtosis {mo['excess_kurtosis']:.1f}", loc="left")
    fig.savefig(FIG / "09_fat_tails_normal_vs_observed.png"); plt.close(fig)

    # ---- 10 Markov regimes through time
    reg = pd.read_csv(PROC / "regime_probability.csv", index_col=0, parse_dates=True).iloc[:, 0]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.fill_between(reg.index, reg.values, 0, color=V, alpha=0.55, lw=0)
    for e in eps:
        ax.axvline(ymd(e["peak"]), color=MUTED_INK, lw=0.7, ls=":")
    ax.set_ylim(0, 1.02); ax.yaxis.set_major_formatter(PercentFormatter(1, 0)); ax.set_ylabel("P(turbulent state)"); F.grid(ax)
    ax.text(0.01, 0.9, "dotted lines = start of each 20% fall", transform=ax.transAxes, fontsize=8, color=MUTED_INK)
    ax.set_title("Two-state Markov model: smoothed probability that the market is turbulent", loc="left")
    fig.savefig(FIG / "10_markov_regime_probability.png"); plt.close(fig)

    # ---- 11 transition matrices
    mk = E["markov"]; P = np.array(mk["P"]); P12 = np.array(mk["P12"])
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for ax, M, ttl in [(axs[0], P, "One month ahead"), (axs[1], P12, "Twelve months ahead (P¹²)")]:
        ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("m", ["#ffffff", B]), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]*100:.1f}%", ha="center", va="center", fontsize=10, color="white" if M[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["to calm", "to turbulent"]); ax.set_yticks([0, 1], ["from calm", "from turbulent"]); ax.set_title(ttl, fontsize=9.5)
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle(f"Switching probabilities; long-run share calm {mk['stationary'][0]*100:.0f}% / turbulent {mk['stationary'][1]*100:.0f}%", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "11_markov_transition_matrix.png"); plt.close(fig)

    # ---- 12 Monte Carlo: funding certainty by strategy and case
    fig, ax = plt.subplots(figsize=(6.6, 3.9)); yy = np.arange(len(STRATS))[::-1]
    for ci, (c, lab, col, mkr) in enumerate(CASES):
        ax.plot([S[c][s]["f4"] for s in STRATS], yy + (ci - 1.5) * 0.12, mkr, color=col, ms=6.2, label=lab, ls="none", mec="white", mew=0.5)
    for y_ in yy: ax.axhline(y_, color=GRID, lw=0.6, zorder=0)
    ax.set_yticks(yy, [SNAME[s] for s in STRATS]); ax.xaxis.set_major_formatter(PercentFormatter(1, 0)); ax.set_xlim(0.2, 1.01)
    ax.set_xlabel("Funding certainty: P(portfolio at the start of 2033 ≥ $421,767 reserve)"); F.grid(ax, "x")
    ax.legend(frameon=False, fontsize=8, loc="lower left", title="Return case", title_fontsize=8.5)
    ax.set_title("Monte Carlo layer: how often each strategy funds the reserve (10,000 paths)", loc="left")
    fig.savefig(FIG / "12_mc_funding_certainty.png"); plt.close(fig)

    # ---- 13 Monte Carlo: facility contribution range, case H and case S
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    for ax, c, ttl in [(axs[0], "H", "H historical"), (axs[1], "S", "S stressed")]:
        for i, s in enumerate(STRATS):
            y = len(STRATS) - 1 - i; d = S[c][s]; col = V if s in ("S9", "S9A", "S10", "S10C") else B
            ax.plot([d["p5"], d["p95"]], [y, y], color=col, lw=1.2, alpha=0.7); ax.plot([d["p25"], d["p75"]], [y, y], color=col, lw=5, solid_capstyle="butt")
            ax.plot([d["p50"]], [y], "o", color="white", ms=4, mec=INK, mew=0.8, zorder=3)
        ax.axvline(0, color=MUTED_INK, lw=0.8); ax.xaxis.set_major_formatter(usd_k); ax.tick_params(axis="x", labelsize=7.5); F.grid(ax, "x"); ax.set_title(ttl, fontsize=9.5)
    axs[0].set_yticks(np.arange(len(STRATS))[::-1], [SNAME[s] for s in STRATS])
    fig.suptitle("Facility contribution in 2033 (portfolio − reserve): 5th–95th percentile line, 25th–75th bar, median dot; LDCS family in violet", x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "13_mc_contribution_range.png"); plt.close(fig)

    # ---- 14 Monte Carlo crash replays: funding certainty when each historical crash is replayed at the worst moment
    ck = [k for k in S["crash"] if "T3+" not in k]
    fig, ax = plt.subplots(figsize=(7.0, 3.5)); xs = np.arange(len(ck)); w = 0.2; picks = [("S4", B), ("S7", G), ("S9", O), ("S10C", V)]
    for i, (s, col) in enumerate(picks):
        vals = [S["crash"][k]["timing"]["worst"][s]["f4"] for k in ck]
        ax.bar(xs + (i - 1.5) * w, vals, w, color=col, label=SNAME[s], edgecolor="white", lw=0.6)
    ax.set_xticks(xs, [k.split(" (")[0].replace("Exogenous ", "").replace(" bear", "") + "\n(" + k.split("(")[1].rstrip(")") + ")" for k in ck], fontsize=7.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0)); ax.set_ylim(0, 1.12); ax.set_ylabel("Funding certainty"); F.grid(ax)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center"); ax.set_title("Replaying a historical crash at the worst timing: funding certainty by strategy", loc="left")
    fig.savefig(FIG / "14_mc_crash_replay_certainty.png"); plt.close(fig)

    # ---- 15 pipeline diagram (same flow as the mermaid chart in INTERCONNECTIONS.md)
    fig, ax = plt.subplots(figsize=(7.0, 4.3)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6.4)
    boxes = {"data": (0.2, 5.3, "Data: CRSP 1926–2026,\nFRED, French, team returns", GREY), "ep": (3.6, 5.3, "Crash episodes\n(falls ≥ 20%, ≥ 30%)", B), "reg": (7.0, 5.3, "Two-state Markov\nregime model (Hamilton)", V),
             "poi": (0.2, 3.4, "Poisson rate λ = n/T\nexact 90% range", B), "typ": (3.6, 3.4, "Crash types T1–T5\nand loss by fund", O), "mm": (7.0, 3.4, "Markov-modulated\nPoisson (clustering)", V),
             "cmp": (0.2, 1.5, "Compound Poisson:\ndollar damage to Laura", B), "mc": (3.6, 1.5, "Monte Carlo strategies:\nblock bootstrap, 10,000 paths", G), "fin": (7.0, 1.5, "P(type affects Laura) =\nP(occurs, from λ) × P(harm)", O)}
    for k, (x, y, t, c) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), 2.8, 0.85, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec="none", alpha=0.22)); ax.add_patch(FancyBboxPatch((x, y), 0.07, 0.85, boxstyle="square,pad=0", fc=c, ec="none"))
        ax.text(x + 1.45, y + 0.43, t, ha="center", va="center", fontsize=8.3, color=INK)
    arrows = [("data", "ep"), ("ep", "reg"), ("ep", "poi"), ("ep", "typ"), ("reg", "mm"), ("poi", "cmp"), ("typ", "cmp"), ("mm", "cmp"), ("typ", "mc"), ("mc", "fin")]
    cen = {k: (x + 1.4, y + 0.43) for k, (x, y, _, _) in boxes.items()}
    for a_, b_ in arrows:
        (x1, y1), (x2, y2) = cen[a_], cen[b_]
        if abs(y1 - y2) < 0.1: x1, x2 = (x1 + 1.4, x2 - 1.4) if x2 > x1 else (x1 - 1.4, x2 + 1.4)
        else: y1, y2 = (y1 - 0.43, y2 + 0.43) if y2 < y1 else (y1 + 0.43, y2 - 0.43)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED_INK, lw=0.9, shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=0.0"))
    ax.text(0.2, 0.55, "Markov-Model repo idea: regimes and transition matrix · Monte-Carlo-Simulations- repo idea: block-bootstrap of fund returns", fontsize=8, color=MUTED_INK)
    ax.set_title("How the Poisson, Markov and Monte Carlo layers connect", loc="left", fontsize=10.5, fontweight="bold")
    fig.savefig(FIG / "15_model_interconnections.png"); plt.close(fig)

    # ================================================================= tables (CSV)
    rows = [dict(peak=e["peak"], trough=e["trough"], recovered=e["recovered"], depth=e["depth"], type=e["type"], d10y=e["d10y"], cpi_at_trough=e["cpi_at_trough"], cape_at_peak=e["cape"], nber_recession=e["nber_recession"], months_to_trough=e["months_to_trough"]) for e in eps]
    pd.DataFrame(rows).to_csv(TAB / "crash_episodes.csv", index=False)
    rows = []
    for thr, e in (("20%", E["equity"]["20"]), ("30%", E["equity"]["30"])):
        for k, v in e["p_at_least_one"].items():
            rows.append(dict(threshold=thr, window=k, lam=e["lam"], p_at_least_one=v[0], lo90=v[1], hi90=v[2]))
    pd.DataFrame(rows).to_csv(TAB / "poisson_by_window.csv", index=False)
    pd.DataFrame([dict(type=k, n=v["n"], lam=v["lam"], lam_lo90=v["lam_ci90"][0], lam_hi90=v["lam_ci90"][1], mean_wait_years=v["mean_wait_years"], p_2031_32=v["p_at_least_one"]["2031-32 (critical years)"][0]) for k, v in E["by_type"].items()]).to_csv(TAB / "poisson_by_type.csv", index=False)
    pd.DataFrame([dict(state="calm", stationary=mk["stationary"][0], expected_months=mk["expected_months"][0], lam=mk["lam_calm"], to_calm=P[0, 0], to_turbulent=P[0, 1]),
                  dict(state="turbulent", stationary=mk["stationary"][1], expected_months=mk["expected_months"][1], lam=mk["lam_turbulent"], to_calm=P[1, 0], to_turbulent=P[1, 1])]).to_csv(TAB / "markov_states.csv", index=False)
    pd.DataFrame({"years": range(1, 17), "plain_poisson": [1 - math.exp(-lam * y) for y in range(1, 17)], **{k: v[:16] for k, v in mk["mmpp_curves"].items()}}).to_csv(TAB / "markov_modulated_curves.csv", index=False)
    rows = []
    for k, a in L["aggregate_poisson"].items():
        rows.append(dict(window=k, lam_t=a["lam_t"], expected=a["expected_loss"], expected_formula=a["expected_loss_formula"], sd=a["sd_loss"], p_ge_50k=a["p_loss_ge"]["50000"], p_ge_100k=a["p_loss_ge"]["100000"], var95=a["var95"], var99=a["var99"], cvar99=a["cvar99"]))
    m = L["aggregate_mmpp_2027_32"]; rows.append(dict(window="2027-32 Markov-modulated", expected=m["expected_loss"], p_ge_50k=m["p_loss_ge"]["50000"], p_ge_100k=m["p_loss_ge"]["100000"], var99=m["var99"]))
    pd.DataFrame(rows).to_csv(TAB / "compound_poisson_damage.csv", index=False)
    rows = []
    for c in "HSCJ":
        for s in STRATS:
            d = S[c][s]; rows.append(dict(case=c, strategy=s, funding_certainty_4pct=d["f4"], funding_certainty_3pct=d["f3"], funding_certainty_ladder=d["fl"], contribution_p5=d["p5"], contribution_p50=d["p50"], contribution_p95=d["p95"], promise_kept=d["promise_kept"], cvar5=d["cvar5"], median_max_drawdown=d["mdd50"]))
    pd.DataFrame(rows).to_csv(TAB / "strategy_summary.csv", index=False)
    rows = []
    for k, v in S["crash"].items():
        for timing in ("worst", "early"):
            for s in STRATS:
                rows.append(dict(crash=k, timing=timing, strategy=s, funding_certainty_4pct=v["timing"][timing][s]["f4"], contribution_p50=v["timing"][timing][s]["p50"]))
    pd.DataFrame(rows).to_csv(TAB / "crash_replay_summary.csv", index=False)
    V_ = json.load(open(RES / "verification.json")) if (RES / "verification.json").exists() else None
    if V_: pd.DataFrame(V_["checks"]).to_csv(TAB / "verification_checks.csv", index=False)
    dl = pd.read_csv(TAB / "dollar_loss_by_fund_and_type.csv")
    if dl.columns[0].startswith("Unnamed"): dl.rename(columns={dl.columns[0]: "fund"}).to_csv(TAB / "dollar_loss_by_fund_and_type.csv", index=False)
    print("figures 07-15 and tables written")


if __name__ == "__main__":
    main()
