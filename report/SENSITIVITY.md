# Sensitivity of the Poisson rate

Ten crashes is a small sample, so the headline λ depends on how a crash is defined and how much history is used. Everything below is recomputed from `data/processed/us_market_long.csv`.

## Depth threshold (full 1926–2026 history)

| Fall of at least | Crashes | λ per year | λ 90% range | P(≥1) 2031–32 | P(≥1) 2027–32 |
|---|---|---|---|---|---|
| 10% | 19 | 0.1894 | 0.124 – 0.278 | 31.5% | 67.9% |
| 15% | 14 | 0.1395 | 0.084 – 0.218 | 24.4% | 56.7% |
| 20% | 10 | 0.0997 | 0.054 – 0.169 | 18.1% | 45.0% |
| 25% | 6 | 0.0598 | 0.026 – 0.118 | 11.3% | 30.1% |
| 30% | 5 | 0.0498 | 0.020 – 0.105 | 9.5% | 25.8% |
| 35% | 4 | 0.0399 | 0.014 – 0.091 | 7.7% | 21.3% |
| 40% | 4 | 0.0399 | 0.014 – 0.091 | 7.7% | 21.3% |

The chosen 20% and 30% thresholds sit in the middle of this range. Because the mean wait between 20% falls is about ten years, a different cut-off changes the 2031–32 probability by several points, which is why the 90% range matters more than the point estimate.

## Length of history (20% threshold)

| Sample | Years | Crashes | λ per year | λ 90% range | P(≥1) 2031–32 |
|---|---|---|---|---|---|
| 1926–2026 | 100.3 | 10 | 0.0997 | 0.054 – 0.169 | 18.1% |
| 1950–2026 | 76.8 | 8 | 0.1041 | 0.052 – 0.188 | 18.8% |
| 1970–2026 | 56.8 | 7 | 0.1232 | 0.058 – 0.231 | 21.8% |
| 1990–2026 | 36.8 | 4 | 0.1086 | 0.037 – 0.249 | 19.5% |

The four estimates (0.100 to 0.123) all sit inside each other's 90% ranges, so the history length does not change the conclusion; it only widens the range as the sample shrinks. A crash whose peak falls before a sample starts is not counted in that sample.

## Rate at the ends of the 90% range

| λ | Value | P(≥1) 2031–32 | P(≥1) 2027–32 |
|---|---|---|---|
| Lower bound | 0.0541 | 10.3% | 27.7% |
| Estimate | 0.0997 | 18.1% | 45.0% |
| Upper bound | 0.1691 | 28.7% | 63.7% |

## Normal distribution against the data

A normal model badly underestimates large monthly falls, which is the reason a jump (Poisson) component is used at all.

| Monthly fall of at least | Observed | Normal model expects |
|---|---|---|
| 10% | 31 | 19.00 |
| 15% | 11 | 0.74 |
| 20% | 5 | 0.01 |

Monthly skew -0.55, excess kurtosis 6.5 over 1204 months.
