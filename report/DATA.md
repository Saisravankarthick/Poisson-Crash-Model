# Data catalogue

Every file the model reads or writes. Raw snapshots were downloaded on 24 Nov 2026.

## Raw inputs (`data/raw/`)

| File | Source | Rows | First to last row | Columns |
|---|---|---|---|---|
| `data/raw/fred/BAMLH0A0HYM2.csv` | FRED, Federal Reserve Bank of St. Louis | 792 | 2023-11-24 to 2026-11-20 | observation_date, BAMLH0A0HYM2 |
| `data/raw/fred/CPIAUCSL.csv` | FRED, Federal Reserve Bank of St. Louis | 958 | 1947-01-01 to 2026-10-01 | observation_date, CPIAUCSL |
| `data/raw/fred/DEXTAUS.csv` | FRED, Federal Reserve Bank of St. Louis | 11255 | 1983-10-03 to 2026-11-20 | observation_date, DEXTAUS |
| `data/raw/fred/DFII10.csv` | FRED, Federal Reserve Bank of St. Louis | 6232 | 2003-01-02 to 2026-11-20 | observation_date, DFII10 |
| `data/raw/fred/DFII5.csv` | FRED, Federal Reserve Bank of St. Louis | 6232 | 2003-01-02 to 2026-11-20 | observation_date, DFII5 |
| `data/raw/fred/DFII7.csv` | FRED, Federal Reserve Bank of St. Louis | 6232 | 2003-01-02 to 2026-11-20 | observation_date, DFII7 |
| `data/raw/fred/DGS1.csv` | FRED, Federal Reserve Bank of St. Louis | 16929 | 1962-01-02 to 2026-11-20 | observation_date, DGS1 |
| `data/raw/fred/DGS10.csv` | FRED, Federal Reserve Bank of St. Louis | 16929 | 1962-01-02 to 2026-11-20 | observation_date, DGS10 |
| `data/raw/fred/DGS2.csv` | FRED, Federal Reserve Bank of St. Louis | 13169 | 1976-06-01 to 2026-11-20 | observation_date, DGS2 |
| `data/raw/fred/DGS5.csv` | FRED, Federal Reserve Bank of St. Louis | 16929 | 1962-01-02 to 2026-11-20 | observation_date, DGS5 |
| `data/raw/fred/DGS7.csv` | FRED, Federal Reserve Bank of St. Louis | 14974 | 1969-07-01 to 2026-11-20 | observation_date, DGS7 |
| `data/raw/fred/DTB3.csv` | FRED, Federal Reserve Bank of St. Louis | 19015 | 1954-01-04 to 2026-11-20 | observation_date, DTB3 |
| `data/raw/fred/FEDFUNDS.csv` | FRED, Federal Reserve Bank of St. Louis | 868 | 1954-07-01 to 2026-10-01 | observation_date, FEDFUNDS |
| `data/raw/fred/GS10.csv` | FRED, Federal Reserve Bank of St. Louis | 883 | 1953-04-01 to 2026-10-01 | observation_date, GS10 |
| `data/raw/fred/GS2.csv` | FRED, Federal Reserve Bank of St. Louis | 605 | 1976-06-01 to 2026-10-01 | observation_date, GS2 |
| `data/raw/fred/GS5.csv` | FRED, Federal Reserve Bank of St. Louis | 883 | 1953-04-01 to 2026-10-01 | observation_date, GS5 |
| `data/raw/fred/T5YIE.csv` | FRED, Federal Reserve Bank of St. Louis | 6233 | 2003-01-02 to 2026-11-23 | observation_date, T5YIE |
| `data/raw/fred/TB3MS.csv` | FRED, Federal Reserve Bank of St. Louis | 1114 | 1934-01-01 to 2026-10-01 | observation_date, TB3MS |
| `data/raw/fred/USREC.csv` | FRED, Federal Reserve Bank of St. Louis | 2063 | 1854-12-01 to 2026-10-01 | observation_date, USREC |
| `data/raw/french/Developed_ex_US_3_Factors.csv` | Kenneth R. French Data Library | 435 | 199007 to 202609 | YYYYMM |
| `data/raw/french/Emerging_5_Factors.csv` | Kenneth R. French Data Library | 447 | 198907 to 202609 | YYYYMM |
| `data/raw/french/F-F_Research_Data_Factors.csv` | Kenneth R. French Data Library | 1203 | 192607 to 202609 | YYYYMM |
| `data/raw/french/Portfolios_Formed_on_ME.csv` | Kenneth R. French Data Library | 4812 | 192607 to 202609 | YYYYMM |
| `data/raw/sp500_shiller.csv` | Robert Shiller's S&P 500 data | 1869 | 1871-01-01 to 2026-09-01 | Date, SP500, Dividend, Earnings … |
| `data/raw/team/returns_used.csv` | Team's own daily fund returns (from Monte-Carlo-Simulations-) | 2512 | 2016-11-02 to 2026-10-30 | Date, VOO, VTI, QQQM … |

## Processed data (`data/processed/`)

| File | What it holds | Rows | Span |
|---|---|---|---|
| `monthly.csv` | Month-end total returns for the eight funds (proxies before Dec 2016, the team's actual returns after) plus the 5-year yield and its change | 435 | 1990-08-01 to 2026-10-01 |
| `proxy_validation.json` | Correlation of each proxy with the real fund over the overlap |  |  |
| `regime_probability.csv` | Smoothed probability of the turbulent state, each month | 1204 | 1926-07-01 to 2026-10-01 |
| `us_market_long.csv` | CRSP US total-market monthly return, 1926–2026: the series the Poisson rate is counted on | 1204 | 1926-07-01 to 2026-10-01 |

## Proxy validation

Before Dec 2016 the funds did not exist (or have no usable history), so proxies are used: French CRSP-market for VTI; a developed ex-US and emerging mix for VXUS; an equal-weighted large-cap series for EQAL; yields turned into returns with the duration–convexity formula for the bond funds. Each proxy is checked against the real fund where both exist.

| Fund | Months compared | Correlation | Tracking error (monthly) | Real return (annualised) | Proxy return (annualised) |
|---|---|---|---|---|---|
| VTI | 118 | 0.999 | 0.71% | 14.7% | 15.0% |
| VXUS | 118 | 0.990 | 2.11% | 9.8% | 9.9% |
| EQAL | 118 | 0.980 | 3.54% | 9.9% | 11.5% |
| IEF | 119 | 0.995 | 0.63% | 1.0% | 1.0% |
| SCHP | 119 | 0.945 | 1.69% | 2.6% | 2.6% |
| SGOV | 77 | 0.991 | 0.08% | 3.0% | 3.0% |
| SHY | 119 | 0.991 | 0.26% | 1.8% | 1.9% |
| VGIT | 119 | 0.993 | 0.55% | 1.5% | 1.2% |

## Results (`results/`)

| File | Produced by | Holds |
|---|---|---|
| `crash_events.json` | `src/crash_events.py` | The ten crashes with types, λ and 90% ranges, dispersion test, Markov model, Merton jump parameters, current yields |
| `laura_poisson.json` | `src/laura_poisson.py` | Fund-level λ, loss per crash by type, compound-Poisson damage distribution, Markov-modulated version, probability each type affects Laura |
| `strategy_results.json` | `src/strategy_sim.py` | 10,000-path simulation of the ten strategies under four cases and six crash replays (the Monte Carlo layer) |
| `verification.json` | `src/verify.py` | All 199 independent checks with computed and reference values |
| `tables/*.csv` | `src/laura_poisson.py` | Fund-level rates, dollar loss by fund and type, probability each type affects Laura |

## Figures (`figures/`)

| File | Shows | Data behind it |
|---|---|---|
| `01_poisson_pmf_windows.png` | Probability of 0, 1, 2, 3+ crashes in three windows | `crash_events.json` → equity → 20 |
| `02_fund_level_lambda.png` | Each fund's crash rate | `tables/fund_level_poisson.csv` |
| `03_crash_damage_exceedance.png` | Chance crashes cost Laura at least $X | `laura_poisson.json` → aggregate_poisson, aggregate_mmpp_2027_32 |
| `04_dollar_change_by_fund_and_type.png` | Dollar change per holding by crash type | `tables/dollar_loss_by_fund_and_type.csv` |
| `05_markov_modulated_horizon.png` | Plain against Markov-modulated Poisson | `crash_events.json` → markov |
| `06_probability_crash_affects_laura.png` | Probability each type affects Laura | `tables/probability_crash_affects_laura.csv` |

All charts use one muted palette (blue `#3b6ea8`, orange `#c8733f`, violet `#8a64b0`, green `#2e9474`), checked for adjacent-pair separation under colour-vision deficiency. Labels, markers and line styles carry identity as well, so colour is never the only cue.
