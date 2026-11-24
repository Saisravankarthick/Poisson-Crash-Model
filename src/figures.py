"""Step 5 - charts (muted palette from config.PALETTE; identity is also carried by labels, markers and line styles)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
from config import RES, FIG, PALETTE, INK, MUTED_INK, GRID, EQUITY, FUNDS

B, O, V, G = PALETTE
plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED_INK, "axes.linewidth": 0.6, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED_INK, "ytick.color": MUTED_INK, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "savefig.dpi": 220, "savefig.bbox": "tight"})
usd_k = FuncFormatter(lambda x, _: f"{'−' if x < 0 else ''}${abs(x)/1000:,.0f}k")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)


def main():
    E = json.load(open(RES / "crash_events.json")); L = json.load(open(RES / "laura_poisson.json"))
    lam = E["equity"]["20"]["lam"]
    from math import exp, factorial
    # 1. Poisson pmf for three windows
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    k = np.arange(4); wins = [("2031–32 (t = 2)", 2, B), ("2027–32 (t = 6)", 6, O), ("2027–42 (t = 16)", 16, V)]
    for i, (lab, t, c) in enumerate(wins):
        p = [exp(-lam * t) * (lam * t) ** j / factorial(j) for j in range(3)]; p.append(1 - sum(p))
        x = k + (i - 1) * 0.26
        ax.bar(x, p, 0.24, color=c, label=lab, edgecolor="white", linewidth=0.8)
        for xi, pi in zip(x, p):
            ax.text(xi, pi + 0.012, f"{pi:.0%}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(k, ["0 crashes", "1", "2", "3 or more"]); ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
    ax.set_ylabel("Probability"); ax.set_ylim(0, 0.95); grid(ax); ax.legend(frameon=False, title="Window", title_fontsize=8.5, fontsize=8.5)
    ax.set_title(f"Poisson probability of k US equity falls of 20% or more (λ = {lam:.3f} per year)", loc="left")
    fig.savefig(FIG / "01_poisson_pmf_windows.png"); plt.close(fig)
    # 2. fund-level lambda with 90% upper bound, dollars allocated
    rows = L["fund_level"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        c = B if r["fund"] in EQUITY else G
        ax.barh(yi, r["lam20"], 0.55, color=c)
        ax.plot([r["lam20"], r["lam20_hi90"]], [yi, yi], color=MUTED_INK, lw=0.9)
        ax.plot([r["lam20_hi90"]], [yi], "|", color=MUTED_INK, ms=7)
        ax.text(r["lam20_hi90"] + 0.006, yi, f"{r['n20']} fall{'s' if r['n20'] != 1 else ''} · ${r['dollars']/1000:,.1f}k held · P(2031–32) {r['p20_2031_32']:.0%}",
                va="center", fontsize=7.8, color=INK)
    ax.set_yticks(y, [r["fund"] for r in rows]); ax.set_xlim(0, 0.62)
    ax.set_xlabel("Falls of ≥20% per year (λ), bar = estimate, line = 90% upper bound"); grid(ax, "x")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=B, label="Equity sleeve"), Patch(color=G, label="Bond sleeve")], frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title(f"Each fund's own crash rate, {L['fund_level']['period'][0][:4]}–2026 (month-end closes)", loc="left")
    fig.savefig(FIG / "02_fund_level_lambda.png"); plt.close(fig)
    # 3. exceedance curves: Poisson vs Markov-modulated
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for lab, d, c, ls in [("Poisson, 2027–32", L["aggregate_poisson"]["2027-32"], B, "-"), ("Markov-modulated, 2027–32", L["aggregate_mmpp_2027_32"], V, "--"),
                          ("Poisson, 2031–32 only", L["aggregate_poisson"]["2031-32"], O, "-")]:
        e = np.array(d["exceedance"])[1:]; ax.plot(e[:, 0], e[:, 1], color=c, lw=2, ls=ls, label=lab)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.xaxis.set_major_formatter(usd_k); ax.yaxis.set_major_formatter(PercentFormatter(1, 0)); grid(ax)
    ax.set_xlabel("Crash damage to Laura's portfolio at least this large (before any recovery)"); ax.set_ylabel("Probability")
    ax.set_title("Compound Poisson: chance that crashes cost Laura at least $X", loc="left"); ax.set_ylim(0, 0.5)
    fig.savefig(FIG / "03_crash_damage_exceedance.png"); plt.close(fig)
    # 4. expected dollar change per fund, by crash type
    bt = L["loss_given_crash"]["by_type"]; types = list(bt)
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 3.0), sharey=True)
    for ax, t in zip(axs, types):
        vals = [-bt[t]["fund_dollars_300k"][f] for f in FUNDS]
        cols = [B if f in EQUITY else G for f in FUNDS]
        ax.barh(np.arange(len(FUNDS))[::-1], vals, color=cols, height=0.6)
        ax.axvline(0, color=MUTED_INK, lw=0.6); ax.set_title(t.split(" ", 1)[1].replace(" / ", "/\n").replace("Exogenous ", "").replace(" bear", ""), fontsize=8.8)
        ax.xaxis.set_major_formatter(usd_k); ax.tick_params(axis="x", labelsize=7); grid(ax, "x")
        tot = sum(vals); ax.text(0.02, -0.18, f"Total {'−' if tot < 0 else '+'}${abs(tot)/1000:,.0f}k", transform=ax.transAxes, fontsize=8, color=INK)
    axs[0].set_yticks(np.arange(len(FUNDS))[::-1], FUNDS)
    fig.suptitle("Average dollar change in each holding during one crash of each type ($300k portfolio)", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "04_dollar_change_by_fund_and_type.png"); plt.close(fig)
    # 5. Markov-modulated probability of >=1 crash over the horizon
    mk = E["markov"]["mmpp_curves"]; yrs = np.arange(1, 17)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(yrs, 1 - np.exp(-lam * yrs), color=MUTED_INK, lw=1.5, ls=":", marker="o", ms=3, label="Plain Poisson")

    for (lab, c, m) in [("just after turbulence starts", O, "s"), ("from the long-run mix", V, "^"), ("from today's state", B, "o")]:
        v = mk[lab]; ax.plot(yrs, v[:16], color=c, lw=2, marker=m, ms=3.5, label=f"Markov: {lab}")
    ax.axvspan(4, 6, color=GRID, alpha=0.6); ax.text(5, 0.04, "2031–32", ha="center", fontsize=8, color=MUTED_INK)
    ax.legend(frameon=False, fontsize=8.3, loc="lower right"); ax.set_xlim(0.5, 16.5); ax.set_ylim(0, 0.9); ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16]); ax.set_xlabel("Years from January 2027")
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0)); grid(ax); ax.set_ylabel("P(at least one ≥20% fall)")
    ax.set_title("Clustering: the Markov-modulated Poisson model against plain Poisson", loc="left")
    fig.savefig(FIG / "05_markov_modulated_horizon.png"); plt.close(fig)
    # 6. probability each crash type affects Laura (2031-32)
    aff = L.get("affects_laura")
    if isinstance(aff, list):
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
        lab = [a["type"] for a in aff]; yy = np.arange(len(aff))[::-1]
        for ax, key, ttl in [(axs[0], "p_shortfall", "Misses the $421,767 reserve"), (axs[1], "p_promise_broken", "Breaks the 2031 promise")]:
            for i, (s, c, nm) in enumerate([("S9", O, "S9 static"), ("S10", B, "S10 rule (B)"), ("S10C", V, "S10C B+C")]):
                ax.barh(yy + (1 - i) * 0.26, [a[f"{s}_{key}"] for a in aff], 0.24, color=c, label=nm)
            ax.set_title(ttl, fontsize=9.5); ax.xaxis.set_major_formatter(PercentFormatter(1, 1)); grid(ax, "x")
        axs[0].set_yticks(yy, [f"{a['type']}  (occurs {a['p_occurs_2031_32']:.1%})" for a in aff])
        axs[1].legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle("P(crash type occurs in 2031–32) × P(it hurts Laura | it ends at the 2033 valuation)", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
        fig.tight_layout(); fig.savefig(FIG / "06_probability_crash_affects_laura.png"); plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
