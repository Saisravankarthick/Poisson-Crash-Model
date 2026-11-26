"""Figures quoted in the Final Report that are derived from the LDCS (S10C) simulation paths
rather than read directly from strategy_results.json.

Writes results/tables/final_report_figures.csv (one row per figure: section, figure, value, how it is computed).
Run from the repository root:  python src/final_report_figures.py

Every path-based figure uses the same cases and seeds as strategy_sim.py (make_cases + simulate),
so the per-case numbers reproduce strategy_results.json exactly; this script only adds the
pooled, conditional and deterministic figures built on top of them.
"""
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import strategy_sim as S  # noqa: E402

R4, R3 = S.R4, S.R3                      # reserve at 4% (IPS rate) and at 3%
BUFFER = R3 - R4                          # flexibility buffer
PAY, N, R = 50_000.0, 10, 0.04
PV_PAY_2031 = sum(PAY / (1 + R) ** (k + 2) for k in range(N))   # floor in Jan 2031 (payments start Jan 2033)
BAND = 0.025                              # "similar 2031 value": within +/-2.5% of the typical value
LO, HI = 85_000, 185_000                  # announced range
IPS_HIGH = 236_000                        # IPS estimate

rows = []
def add(section, figure, value, how):
    rows.append(dict(section=section, figure=figure, value=value, how=how))

# ---------------------------------------------------------------- simulate the four cases (LDCS)
cases = S.make_cases()
res = {c: S.simulate(*cases[c], "S10C") for c in "HSCJ"}
V31 = {c: res[c]["V31"] for c in res}
C33 = {c: res[c]["V33"] - R4 for c in res}            # contribution above the 4% reserve
pool31 = np.concatenate([V31[c] for c in "HSCJ"])
pool33 = np.concatenate([C33[c] for c in "HSCJ"])
pct = lambda a, q: float(np.percentile(a, q))

# ---------------------------------------------------------------- Table 4, pooled row ("All four, equally weighted")
v_typ = float(np.median(pool31))
m = np.abs(pool31 / v_typ - 1) < BAND
L_typ = 0.5 * max(v_typ - PV_PAY_2031, 0) * (1 + R) ** 2
add("Table 6", "Typical January 2031 value, four cases pooled", round(v_typ), "median of pooled V31")
add("Table 6", "Pooled range P10 (gross of buffer)", round(pct(pool33[m], 10)), "P10 of 2033 contribution, paths within 2.5% of typical V31")
add("Table 6", "Pooled median (gross of buffer)", round(pct(pool33[m], 50)), "as above, P50")
add("Table 6", "Pooled range P90 (gross of buffer)", round(pct(pool33[m], 90)), "as above, P90")
add("Table 6", "Minimum at typical 2031 value", round(L_typ), "0.5 x (V31 - PV of payments in 2031) x 1.04^2")
add("Table 6", "Minimum met, average of four cases", round(np.mean([float((C33[c] >= res[c]["L"]).mean()) for c in "HSCJ"]), 4),
    "mean over cases of P(contribution >= own minimum); per-case values = promise_kept in strategy_results.json")
add("Table 6 note", "Flexibility buffer", round(BUFFER, 2), "reserve at 3% minus reserve at 4%")
for q in (10, 50, 90):
    add("Table 6 note", f"Pooled P{q} net of buffer", round(pct(pool33[m] - BUFFER, q)), "gross figure minus buffer")

# ---------------------------------------------------------------- confidence in the announced range
for c in "HSCJ":
    mc = np.abs(V31[c] / v_typ - 1) < BAND
    x = C33[c][mc]
    add("Confidence", f"Share inside ${LO:,}-${HI:,} at the typical 2031 value, case {c}", round(float(((x >= LO) & (x <= HI)).mean()), 4),
        "paths of this case whose V31 is within 2.5% of the pooled typical value")
for c in "HSCJ":
    add("Confidence", f"Share inside ${LO:,}-${HI:,} seen from today, case {c}", round(float(((C33[c] >= LO) & (C33[c] <= HI)).mean()), 4), "all paths")
add("Confidence", f"Share inside ${LO:,}-${HI:,} seen from today, pooled", round(float(((pool33 >= LO) & (pool33 <= HI)).mean()), 4), "all pooled paths")
lock = np.concatenate([res[c]["lock"] for c in "HSCJ"])
add("Confidence", "Share of paths where the 2031 lock-in is switched off (2-year yield < 4%)", round(1 - float(lock.mean()), 4), "pooled")

# ---------------------------------------------------------------- the IPS's $236,000
add("IPS range", f"Share of paths with contribution >= ${IPS_HIGH:,}, pooled", round(float((pool33 >= IPS_HIGH).mean()), 4), "all pooled paths")
add("IPS range", f"Share of paths with contribution >= ${IPS_HIGH:,}, current-yield case", round(float((C33['C'] >= IPS_HIGH).mean()), 4), "case C")

def steady(r):
    v31 = 300_000 * (1 + r) ** 4 + 150_000 * (1 + r) ** 3
    v33 = v31 * (1 + r) ** 2
    return v31, 0.5 * max(v31 - PV_PAY_2031, 0) * (1 + R) ** 2, v33, v33 - R4
lo_, hi_ = 0.0, 0.2
for _ in range(100):
    mid = (lo_ + hi_) / 2
    lo_, hi_ = (mid, hi_) if steady(mid)[3] < IPS_HIGH else (lo_, mid)
add("IPS range", f"Steady annual return giving ${IPS_HIGH:,}", round(mid, 4), "solve V33 - reserve = 236,000 with constant returns")
for r in (0.0, 0.02, 0.04, 0.06, 0.08):
    v31, L, v33, c = steady(r)
    add("Steady-return check", f"{r:.0%} a year: V31 / minimum / V33 / contribution",
        f"{v31:,.0f} / {L:,.0f} / {v33:,.0f} / {c:,.0f}", "$300k Jan 2027 + $150k Jan 2028, constant return")

# ---------------------------------------------------------------- Table 7 (history case, yearly values)
Vy = res["H"]["Vy"]
floor = {0: sum(PAY / (1 + R) ** (k + 6) for k in range(N)) - 150_000 / (1 + R),
         1: sum(PAY / (1 + R) ** (k + 5) for k in range(N)), 4: PV_PAY_2031, 6: R4}
for t, yr in ((0, 2027), (1, 2028), (4, 2031), (6, 2033)):
    typ, p10 = float(np.median(Vy[t])), pct(Vy[t], 10)
    add("Table 9", f"{yr}: floor / typical / 10th percentile / funding ratio",
        f"{floor[t]:,.0f} / {typ:,.0f} / {p10:,.0f} / {typ / floor[t]:.2f}", "history case; floor excludes any locked-in minimum")

# ---------------------------------------------------------------- deterministic figures (Section 2, Table 3, inflation)
pv27 = sum(PAY / (1 + R) ** (k + 6) for k in range(N)); pv150 = 150_000 / (1 + R)
add("Section 2", "PV of payments, Jan 2027 / PV of 2028 contribution / floor / cushion",
    f"{pv27:,.0f} / {pv150:,.0f} / {pv27 - pv150:,.0f} / {300_000 - (pv27 - pv150):,.0f}", "4% discount rate")
add("Section 2", "Cushion as share of $300,000", round((300_000 - (pv27 - pv150)) / 300_000, 4), "most the rule allows in equities on day one")
for name, ks in (("2033-2035", range(0, 3)), ("2036-2039", range(3, 7)), ("2040-2042", range(7, 10))):
    add("Table 5", f"Reserve slice {name}", round(sum(PAY / (1 + R) ** k for k in ks)), "PV at Jan 2033, 4%")
add("Inflation", "Cost rise 2027-2033 at 4% a year", round(1.04 ** 6 - 1, 4), "1.04^6 - 1")
add("Inflation", "$137,000 in 2027 dollars", round(137_000 / 1.04 ** 6), "137,000 / 1.04^6")

# ---------------------------------------------------------------- expected contribution lost to stock crashes (Table 5 note)
import pandas as pd  # noqa: E402
ci = pd.read_csv(ROOT / "results/tables/crash_impact_by_strategy.csv")
stock = ci[ci["type"] != "T5"]
for sid, nm in (("S10C", "LDCS"), ("S9", "Core-satellite")):
    add("Table 7 note", f"Expected contribution lost to the five stock-crash types, {nm}",
        round(float(stock.loc[stock.strategy == sid, "expected_hit_stock_crashes"].sum())), "sum of expected_hit_stock_crashes, crash_impact_by_strategy.csv")

# ---------------------------------------------------------------- Section 2, Table 2: risk profile of the final WInS portfolio
monthly = pd.read_csv(ROOT / "data/processed/monthly.csv", index_col=0, parse_dates=True).loc["2004-01-01":"2026-10-01"]
W1 = dict(VTI=20.82, VXUS=13.00, EQAL=4.76, IEF=12.43, SCHP=9.53, SGOV=10.20, SHY=13.81, VGIT=13.67)   # Table 1 weights (%)
CASH = 1.78                                                                                         # cash earns nothing in WInS
def profile(ws, cash=0.0):
    tot = sum(ws.values()) + cash
    r = sum(monthly[k] * v for k, v in ws.items()) / tot                                            # rebalanced monthly
    r12 = (1 + r).rolling(12).apply(np.prod, raw=True) - 1
    return dict(ret=(1 + r).prod() ** (12 / len(r)) - 1, vol=r.std() * np.sqrt(12), worst12=r12.min(),
                worst12_end=str(r12.idxmin().date()), y2022=(1 + r.loc["2022"]).prod() - 1)
import json  # noqa: E402
dur = json.load(open(ROOT / "results/strategy_results.json"))["sleeve_duration"]
eq = {k: W1[k] for k in ("VTI", "VXUS", "EQAL")}; bd = {k: W1[k] for k in ("IEF", "SCHP", "SGOV", "SHY", "VGIT")}
for nm, ws, cash, d_ in (("Equity funds", eq, 0.0, None), ("Treasury funds", bd, 0.0, dur), ("Whole portfolio", W1, CASH, dur * sum(bd.values()) / 100)):
    st = profile(ws, cash)
    add("Table 2", f"{nm}: return a year / volatility / worst 12 months (end) / 2022 / duration",
        f"{st['ret']:.1%} / {st['vol']:.1%} / {st['worst12']:.1%} ({st['worst12_end']}) / {st['y2022']:.1%} / " + ("-" if d_ is None else f"{d_:.1f} years"),
        "monthly returns Jan 2004-Oct 2026, Table 1 weights, compound annual return")

# ---------------------------------------------------------------- inflation figures (FRED series stored in data/raw/fred)
def fred_series(code):
    x = pd.read_csv(ROOT / f"data/raw/fred/{code}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(x, errors="coerce").dropna()
cpi = fred_series("CPIAUCSL")
yoy = (cpi / cpi.reindex(cpi.index - pd.DateOffset(months=12)).set_axis(cpi.index) - 1).dropna()
add("Inflation", f"CPI 12-month change, latest available ({yoy.index[-1]:%B %Y})", round(float(yoy.iloc[-1]), 4),
    "CPIAUCSL matched by date; October 2025 CPI was not published, so October 2026 has no 12-month figure")
be = fred_series("T5YIE")
add("Inflation", f"Five-year breakeven inflation, latest ({be.index[-1].date()})", round(float(be.iloc[-1]), 2), "T5YIE, percent a year")
add("Inflation", "Five-year breakeven inflation, October 2026 average", round(float(be.loc["2026-10"].mean()), 2), "T5YIE monthly mean")

out = ROOT / "results/tables/final_report_figures.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["section", "figure", "value", "how"]); w.writeheader(); w.writerows(rows)
print(f"wrote {out} ({len(rows)} rows)")
