"""Step 2 - market-wide crash history (CRSP US market, month-end total return, 1926-2026) and other shock types (FRED).
Estimates Poisson rates with exact intervals, tests the Poisson assumption, fits the Markov regime model and
computes Markov-modulated probabilities.  Output: results/crash_events.json"""
import json
import numpy as np
import pandas as pd
from config import RAW, PROC, RES, THRESHOLDS, WINDOWS
import poisson_core as pc


def fred(sid):
    s = pd.read_csv(RAW / "fred" / f"{sid}.csv"); s.columns = ["d", "v"]
    s["d"] = pd.to_datetime(s["d"]); s["v"] = pd.to_numeric(s["v"], errors="coerce")
    return s.set_index("d")["v"].dropna()


def mdiff(a, b):
    return (b.year - a.year) * 12 + b.month - a.month


def starts(flag, gap):
    out, last = [], None
    for d, f in flag.items():
        if f:
            if last is None or mdiff(last, d) > gap:
                out.append(d)
            last = d
    return out


def main():
    r = pd.read_csv(PROC / "us_market_long.csv", index_col=0, parse_dates=True).iloc[:, 0]
    level = (1 + r).cumprod(); level.loc[r.index[0] - pd.DateOffset(months=1)] = 1.0; level = level.sort_index()
    T = (len(level) - 1) / 12
    sh = pd.read_csv(RAW / "sp500_shiller.csv", parse_dates=["Date"]).set_index("Date")
    cape = sh["PE10"].replace(0, np.nan)
    y10 = pd.concat([sh["Long Interest Rate"].replace(0, np.nan).loc[:"1953-03-01"], fred("GS10")])
    cpi = pd.concat([sh["Consumer Price Index"].replace(0, np.nan).dropna().loc[:"1946-12-01"], fred("CPIAUCSL")])
    # 12-month change matched by date, not by row: FRED has no CPI for October 2025 (not published), so a row shift
    # would compare October 2026 with September 2025 (a 13-month change).
    infl = (cpi / cpi.reindex(cpi.index - pd.DateOffset(months=12)).set_axis(cpi.index) - 1) * 100
    usrec = fred("USREC")

    def classify(e):
        p, t = e["peak"], e["trough"]
        dy = y10.asof(t) - y10.asof(p); n = mdiff(p, t); c = cape.asof(p); it = infl.asof(t)
        rec = usrec[(usrec.index >= p) & (usrec.index <= t)].sum() > 0
        if dy >= 1.0 or it >= 5.0: typ = "T3 Inflation / rate shock"
        elif n <= 4: typ = "T2 Exogenous V-shock"
        elif c >= 25 and n >= 18: typ = "T4 Valuation bust"
        elif rec: typ = "T1 Recession bear"
        else: typ = "T5 Other"
        return typ, dict(d10y=round(float(dy), 2), cpi_at_trough=round(float(it), 1), cape=None if c != c else round(float(c), 1),
                         nber_recession=bool(rec), months_to_trough=n)

    out = dict(source="CRSP US total market, month-end total return (Kenneth R. French Data Library)",
               period=[str(level.index[0].date()), str(level.index[-1].date())], years=T, equity={}, by_type={}, other={})
    for thr in THRESHOLDS:
        ev = pc.episodes(level, thr); rows = []
        for e in ev:
            typ, info = classify(e)
            rows.append(dict(peak=str(e["peak"].date())[:7], trough=str(e["trough"].date())[:7],
                             recovered=str(e["recovered"].date())[:7] if e["recovered"] is not None else None,
                             depth=round(float(e["depth"]), 4), type=typ, **info))
        key = f"{int(thr*100)}"
        counts = pd.Series(0, index=range(level.index[0].year, level.index[-1].year + 1))
        for e in ev:
            counts[e["peak"].year] += 1
        out["equity"][key] = dict(episodes=rows, **pc.summarise_rate(len(ev), T, WINDOWS), dispersion=pc.dispersion_test(counts.values))
        if key == "20":
            for typ in sorted({x["type"] for x in rows}):
                n = sum(x["type"] == typ for x in rows)
                out["by_type"][typ] = pc.summarise_rate(n, T, WINDOWS)
            sev = [x for x in rows if x["type"].startswith("T3") and x["depth"] <= -0.30]
            out["by_type"]["T3+ Severe inflation shock (>=30%)"] = pc.summarise_rate(len(sev), T, WINDOWS)
    # ---- other shock types from FRED
    gs10, gs5 = fred("GS10"), fred("GS5")
    twd = fred("DEXTAUS").resample("MS").mean()
    defs = {
        "R1 Rate shock: 10y yield +2 pts within 12 months": (starts((gs10 - gs10.rolling(13).min()) >= 2.0, 12), (len(gs10) - 12) / 12),
        "R2 Falling-rate shock: 5y yield -2 pts within 24 months": (starts((gs5.rolling(25).max() - gs5) >= 2.0, 24), (len(gs5) - 24) / 12),
        "R4 TWD appreciation >= 10% vs USD in 12 months": (starts(((twd / twd.shift(12) - 1) <= -0.10).dropna(), 12), (twd.notna().sum() - 12) / 12),
        "R5 US CPI inflation >= 5% year on year": (starts((infl.loc["1948-01-01":] >= 5).dropna(), 24), infl.loc["1948-01-01":].notna().sum() / 12)}
    for k, (st, TT) in defs.items():
        out["other"][k] = dict(starts=[str(s.date())[:7] for s in st], **pc.summarise_rate(len(st), TT, WINDOWS))
    # ---- normal model versus fat tails
    lr = np.log(level / level.shift(1)).dropna(); mu, sd = lr.mean(), lr.std()
    from scipy import stats
    out["normal_check"] = {f"<= -{int(100*q)}%": dict(observed=int((lr <= np.log(1 - q)).sum()),
                                                       normal_expected=float(stats.norm.cdf(np.log(1 - q), mu, sd) * len(lr))) for q in (0.10, 0.15, 0.20)}
    out["normal_check"]["moments"] = dict(mean=float(mu), sd=float(sd), skew=float(stats.skew(lr)), excess_kurtosis=float(stats.kurtosis(lr)), months=len(lr))
    # ---- Markov regimes and Markov-modulated Poisson
    hm = pc.hamilton_em(lr.values)
    reg = pd.Series(hm["smoothed"][:, 1], index=lr.index)
    cross = [e["cross"] for e in pc.episodes(level, 0.20)]
    t_turb, t_calm = reg.sum() / 12, (1 - reg).sum() / 12
    n_turb = float(sum(reg.asof(c) for c in cross))
    lam_c, lam_t = (len(cross) - n_turb) / t_calm, n_turb / t_turb
    p_now = float(hm["filtered"][-1, 1])
    curves = {lab: pc.mmpp_p_at_least_one(hm["P"], lam_c, lam_t, p0, 16) for lab, p0 in
              (("from today's state", p_now), ("from the long-run mix", hm["stationary"][1]), ("just after turbulence starts", 1.0))}
    out["markov"] = dict(mu=hm["mu"].tolist(), sd=hm["sd"].tolist(), P=hm["P"].tolist(), P12=np.linalg.matrix_power(hm["P"], 12).tolist(),
                         stationary=hm["stationary"].tolist(), expected_months=(1 / (1 - np.diag(hm["P"]))).tolist(),
                         p_turbulent_now=p_now, lam_calm=lam_c, lam_turbulent=lam_t, crossings_in_turbulence=n_turb,
                         mmpp_curves={k: v[11::12] for k, v in curves.items()})
    jm = lr <= np.log(0.9)
    out["merton"] = dict(definition="jump = month with total return <= -10%", n_jumps=int(jm.sum()), lam_J=float(jm.sum() / (len(lr) / 12)),
                         muJ=float(lr[jm].mean()), sdJ=float(lr[jm].std()),
                         lam_J_calm=float((jm.sum() - reg[jm].sum()) / t_calm), lam_J_turb=float(reg[jm].sum() / t_turb))
    out["now"] = dict(date=str(fred("DGS5").index[-1].date()), y5=float(fred("DGS5").iloc[-1]), y2=float(fred("DGS2").iloc[-1]),
                      y10=float(fred("DGS10").iloc[-1]), tbill=float(fred("DTB3").iloc[-1]), cpi_yoy=float(infl.dropna().iloc[-1]), cpi_yoy_month=str(infl.dropna().index[-1].date()),
                      fed_funds=float(fred("FEDFUNDS").iloc[-1]), twd_per_usd=float(fred("DEXTAUS").iloc[-1]))
    reg.to_csv(PROC / "regime_probability.csv")
    json.dump(out, open(RES / "crash_events.json", "w"), indent=1, default=str)
    e20 = out["equity"]["20"]
    print(f"{len(e20['episodes'])} falls >=20% in {T:.1f} years: lambda = {e20['lam']:.4f}/yr; "
          f"P(>=1 in 2031-32) = {e20['p_at_least_one']['2031-32 (critical years)'][0]:.1%}")


if __name__ == "__main__":
    main()
