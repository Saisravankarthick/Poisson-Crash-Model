"""Step 1 - build the monthly (month-end) data set used by every other script.

Equities : Kenneth R. French Data Library (CRSP US market for VTI; 78% developed ex-US + 22% emerging for VXUS;
           equal-weighted top-30% size portfolio for EQAL).
Bonds    : FRED constant-maturity yields turned into total returns with the duration-convexity approximation
           R_t = y_{t-1}/12 - D*dy + 0.5*C*dy^2   (+ 3-month-lagged CPI accrual for TIPS).
Actual   : the team's own daily fund returns (Monte-Carlo-Simulations- repo) replace every proxy from Dec 2016.
Outputs  : data/processed/monthly.csv, us_market_long.csv (1926-2026), proxy_validation.json
Refresh  : download newer FRED/French files into data/raw and re-run.
"""
import json
import numpy as np
import pandas as pd
from config import RAW, PROC, FUNDS


def french(fn, marker=None):
    lines = (RAW / "french" / fn).read_text().splitlines()
    i0 = next(i for i, l in enumerate(lines) if marker in l) if marker else 0
    hdr = next(i for i in range(i0, len(lines)) if lines[i].startswith(","))
    names = [c.strip() for c in lines[hdr].split(",")]
    rows = []
    for l in lines[hdr + 1:]:
        p = [x.strip() for x in l.split(",")]
        if len(p) < 2 or not p[0].isdigit() or len(p[0]) != 6:
            break
        rows.append(p)
    df = pd.DataFrame(rows, columns=["d"] + names[1:])
    df["d"] = pd.to_datetime(df["d"], format="%Y%m")
    return df.set_index("d").astype(float).replace(-99.99, np.nan) / 100


def fred_month_end(sid):
    s = pd.read_csv(RAW / "fred" / f"{sid}.csv")
    s.columns = ["d", "v"]
    s["d"] = pd.to_datetime(s["d"]); s["v"] = pd.to_numeric(s["v"], errors="coerce")
    s = s.dropna().set_index("d")["v"]
    return s.resample("ME").last().to_period("M").to_timestamp() / 100


def bond_return(y, dur, conv):
    dy = y.diff()
    return y.shift(1) / 12 - dur * dy + 0.5 * conv * dy ** 2


def main():
    us = french("F-F_Research_Data_Factors.csv"); us_mkt = us["Mkt-RF"] + us["RF"]
    dev = french("Developed_ex_US_3_Factors.csv"); em = french("Emerging_5_Factors.csv")
    ew = french("Portfolios_Formed_on_ME.csv", "Average Equal Weighted Returns -- Monthly")["Hi 30"]
    y2, y5, y7, y10, tb, r7 = (fred_month_end(s) for s in ("DGS2", "DGS5", "DGS7", "DGS10", "DTB3", "DFII7"))
    cpi = pd.read_csv(RAW / "fred" / "CPIAUCSL.csv"); cpi = cpi.set_index(pd.to_datetime(cpi.iloc[:, 0])).iloc[:, 1].astype(float)
    syn = pd.DataFrame({
        "VTI": us_mkt, "VXUS": 0.78 * (dev["Mkt-RF"] + dev["RF"]) + 0.22 * (em["Mkt-RF"] + em["RF"]), "EQAL": ew,
        "IEF": bond_return((y7 + y10) / 2, 7.4, 65), "SCHP": bond_return(r7, 6.6, 55) + cpi.pct_change().shift(3).reindex(r7.index),
        "SGOV": tb.shift(1) / 12, "SHY": bond_return(y2, 1.9, 5), "VGIT": bond_return(y5, 5.1, 32)})
    pre = syn["SCHP"].isna() & syn.index.isin(y7.index)          # before TIPS real yields exist (pre-2003): nominal proxy
    syn.loc[pre, "SCHP"] = bond_return(y7, 6.6, 55)[syn.index[pre]]
    syn = syn.loc["1990-08-01":]
    # actual fund returns from the team repository (daily -> calendar month)
    rd = pd.read_csv(RAW / "team" / "returns_used.csv", index_col=0, parse_dates=True)[FUNDS]
    act = (1 + rd.fillna(0)).groupby(rd.index.to_period("M")).prod() - 1
    act.index = act.index.to_timestamp()
    ok = rd["SGOV"].notna().groupby(rd.index.to_period("M")).mean(); ok.index = ok.index.to_timestamp()
    act.loc[ok < 0.9, "SGOV"] = np.nan
    act = act.loc["2016-12-01":]
    val = {}
    for c in FUNDS:
        a = act[c].dropna(); s = syn[c].reindex(a.index).dropna(); a = a.reindex(s.index)
        val[c] = dict(months=len(a), corr=float(np.corrcoef(a, s)[0, 1]), tracking_error=float((a - s).std() * np.sqrt(12)),
                      actual_return=float((1 + a).prod() ** (12 / len(a)) - 1), proxy_return=float((1 + s).prod() ** (12 / len(s)) - 1))
    data = syn.copy()
    for c in FUNDS:
        m = act[c].dropna(); data.loc[m.index, c] = m
    data = pd.concat([data, act.loc[act.index.difference(data.index)]]).sort_index()
    data["dy5"] = y5.diff().reindex(data.index); data["y5"] = y5.reindex(data.index)
    data = data.loc[:"2026-10-01"]
    data.to_csv(PROC / "monthly.csv")
    lm = us_mkt.copy()
    if pd.Timestamp("2026-10-01") not in lm.index:
        lm.loc[pd.Timestamp("2026-10-01")] = act.loc["2026-10-01", "VTI"]
    lm.rename("us_market_total_return").to_csv(PROC / "us_market_long.csv")
    json.dump(val, open(PROC / "proxy_validation.json", "w"), indent=1)
    print(f"monthly.csv {data.index[0].date()} .. {data.index[-1].date()}; long market {lm.index[0].date()} .. {lm.index[-1].date()}")
    for c, v in val.items():
        print(f"  {c:5s} corr {v['corr']:.3f}  TE {v['tracking_error']:.2%}")


if __name__ == "__main__":
    main()
