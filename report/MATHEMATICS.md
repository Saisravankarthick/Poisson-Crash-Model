# Mathematics and methods

This document explains, from first principles, every calculation in the Poisson Crash Model repository: the data we built, the cases we simulated, the strategies we compared, the statistics we used to rank them, and the tests we ran to challenge that ranking. For each tool we state the formula, derive it where a derivation is short, show how we applied it to Laura Gao's case with the repository's own numbers, and cite the source. Every number quoted here comes from the files in `results/`; `src/verify.py` recomputes 1,044 of them independently.

Contents

1. [Notation and the client case](#1-notation-and-the-client-case)
2. [Data](#2-data)
3. [Laura's liability: present value, duration and convexity](#3-lauras-liability-present-value-duration-and-convexity)
4. [The strategies](#4-the-strategies)
5. [The Monte Carlo engine and the four return cases](#5-the-monte-carlo-engine-and-the-four-return-cases)
6. [The crash replays](#6-the-crash-replays)
7. [The six scoreboard criteria](#7-the-six-scoreboard-criteria)
8. [Ranking, weightings and random weights](#8-ranking-weightings-and-random-weights)
9. [Seed stability and Monte Carlo error](#9-seed-stability-and-monte-carlo-error)
10. [Paired tests on identical paths](#10-paired-tests-on-identical-paths)
11. [The Poisson crash model](#11-the-poisson-crash-model)
12. [Testing the Poisson assumption](#12-testing-the-poisson-assumption)
13. [The Markov regime model](#13-the-markov-regime-model)
14. [How the layers connect](#14-how-the-layers-connect)
15. [Verification](#15-verification)
16. [Assumptions and limits](#16-assumptions-and-limits)
17. [References](#17-references)

---

## 1. Notation and the client case

The Gao Case Study fixes the cash flows. Laura contributes \$300,000 at the start of 2027 and \$150,000 at the start of 2028. From the start of 2033 to the start of 2042 the fund pays \$50,000 a year to operate the facility, ten payments in all. Whatever the portfolio holds in January 2033 above the money needed for those payments is available as Laura's contribution to the facility, and in 2031 she announces a minimum contribution to co-sponsors.

| Symbol | Meaning | Value |
|---|---|---|
| $t$ | Time in years from 1 January 2027 | 0 to 6 |
| $m$ | Month index | 0 to 71; valuation after month 71 (January 2033) |
| $V_t$ | Portfolio value | $V_0 = 300{,}000$; $150{,}000$ added at $t=1$ |
| $PMT$, $n$ | Operating payment, number of payments | 50,000; 10 |
| $r$ | IPS discount rate | 4% (stress 3%) |
| $R_4$ | Operating reserve in January 2033 at 4% | $421{,}766.58$ |
| $c$ | Contribution: $V_{2033} - R_4$ | Money above the reserve |
| $L$ | Minimum contribution announced in January 2031 | Defined in Section 4.3 |
| $E_0$ | Strategic equity share of the portfolio as built | $119{,}051.45 / 300{,}000 = 39.68\%$ |
| $N$ | Simulated paths per case | 10,000 |
| $\lambda$ | Poisson rate of market crashes per year | 0.0997 |
| $P$ | Monthly Markov transition matrix (calm, turbulent) | Section 13 |

**Holdings.** We built the WInS portfolio from eight ETFs at cost basis and held the \$6,319.55 cash balance like SGOV in every model.

| Fund | Asset | Cost basis | Sleeve weight |
|---|---|---|---|
| VTI | US total market equity | \$64,962.10 | 54.6% of equity |
| VXUS | International equity | \$39,786.00 | 33.4% of equity |
| EQAL | US equal-weight equity | \$14,303.35 | 12.0% of equity |
| IEF | 7–10 year Treasuries | \$35,736.00 | 19.7% of bonds |
| SCHP | TIPS | \$27,698.00 | 15.3% of bonds |
| SGOV | 0–3 month T-bills (plus cash) | \$30,159.00 + \$6,319.55 | 20.2% of bonds |
| SHY | 1–3 year Treasuries | \$40,590.00 | 22.4% of bonds |
| VGIT | 3–10 year Treasuries | \$40,446.00 | 22.4% of bonds |

We write $w_E$ for the equity-sleeve weight vector and $w_B$ for the bond-sleeve weight vector. Every strategy (except S0, S1 and S4) holds $e\,w_E + (1-e)\,w_B$, so a strategy is a rule for one number, the equity share $e$.

---

## 2. Data

We built one monthly data set, `data/processed/monthly.csv`, with `src/build_data.py`.

**Equities.** We took monthly total returns from the Kenneth R. French Data Library (French, 2026): the CRSP US market (market excess return plus the risk-free rate) for VTI; 78% developed ex-US plus 22% emerging markets for VXUS; and the equal-weighted top-30% size portfolio for EQAL. We used the CRSP market from June 1926 to October 2026 as the long history for crash counting.

**Treasuries.** We turned FRED constant-maturity yields (Federal Reserve Bank of St. Louis, 2026) into monthly total returns. A bond's price $P(y)$ has the second-order Taylor expansion

```math
\frac{\Delta P}{P} \approx -D\,\Delta y + \tfrac12\, C\,(\Delta y)^2 ,
```

where $D$ is modified duration and $C$ convexity (Fabozzi, 2016). Adding one month of income at last month's yield gives the return we used:

```math
r_t = \frac{y_{t-1}}{12} - D\,\Delta y_t + \tfrac12\, C\,(\Delta y_t)^2 .
```

| Fund | Yield series | $D$ | $C$ |
|---|---|---|---|
| IEF | average of 7- and 10-year yields | 7.4 | 65 |
| SCHP | 7-year TIPS real yield, plus CPI inflation lagged 3 months | 6.6 | 55 |
| SGOV | 3-month bill yield / 12 | 0.1 | – |
| SHY | 2-year yield | 1.9 | 5 |
| VGIT | 5-year yield | 5.1 | 32 |

Before TIPS existed we used the nominal 7-year yield for SCHP.

**Actual fund returns.** From December 2016 we replaced every proxy with the team's own daily fund returns compounded to calendar months. Over the overlap the proxies track the real funds closely:

| Fund | Months | Correlation | Tracking error a year |
|---|---|---|---|
| VTI | 118 | 0.999 | 0.71% |
| VXUS | 118 | 0.990 | 2.11% |
| EQAL | 118 | 0.980 | 3.54% |
| IEF | 119 | 0.995 | 0.63% |
| SCHP | 119 | 0.945 | 1.69% |
| SGOV | 77 | 0.991 | 0.08% |
| SHY | 119 | 0.991 | 0.26% |
| VGIT | 119 | 0.993 | 0.55% |

**Other inputs.** We used Shiller's CAPE series (Shiller, 2026) and the NBER recession indicator and CPI from FRED to classify crashes (Section 11.6), and the 5-year yield change $\Delta y_5$ to track rates inside the simulation.

**Windows.** The Monte Carlo resamples January 2004 to October 2026 (274 months). The crash model uses June 1926 to October 2026 (100.33 years). The stock–bond correlation in our sleeves was −0.19 over 2004–2020 and +0.60 over 2021–2026; that change is why we stress joint stock–bond falls (case J rate shocks and the 2022 replay).

---

## 3. Laura's liability: present value, duration and convexity

### 3.1 The reserve as an annuity-due

The ten payments fall at the start of each year, so their value at the start of 2033 is the present value of an annuity-due. With discount factor $v = 1/(1+r)$:

```math
R(r) = PMT\,(1 + v + v^2 + \dots + v^{n-1}) = PMT\,\frac{1-v^n}{1-v} = PMT\cdot\frac{1-(1+r)^{-n}}{r}\cdot(1+r).
```

The second step is the sum of a finite geometric series; the third uses $1 - v = r/(1+r)$.

| Rate | Annuity-due factor | Reserve | Role |
|---|---|---|---|
| 4% (IPS) | 8.4353 | \$421,766.58 | The hurdle in every funding-certainty figure |
| 3% (stress) | 8.7861 | \$439,305.45 | Reported as a stress |
| 0% | 10.0000 | \$500,000.00 | The undiscounted total |

### 3.2 The liability before 2033

At any time $t$ before 2033 we value the payments at years $k = 6,\dots,15$:

```math
PV_L(t) = \sum_{k=6}^{15} \frac{PMT}{(1+r)^{\,k-t}}, \qquad PV_L(0) = \$333{,}328, \qquad PV_L(6) = R_4 .
```

Until the \$150,000 arrives in January 2028 we count it as an asset worth $150{,}000/(1+r)^{1-t}$.

### 3.3 Interest-rate sensitivity

Macaulay (1938) duration is the present-value-weighted average time to the payments. Modified duration converts it into the percentage price change for a one-point yield change, convexity is the second-order term, and PV01 is the dollar change for one basis point:

```math
D_{Mac} = \frac{\sum_k (k-t)\,CF_k\,v^{k-t}}{PV_L}, \quad D_{mod} = \frac{D_{Mac}}{1+r}, \quad C = \frac{\sum_k (k-t)(k-t+1)\,CF_k\,v^{k-t}}{PV_L\,(1+r)^2}, \quad PV01 = PV_L\, D_{mod} \times 0.0001 .
```

| | January 2027 | January 2033 |
|---|---|---|
| Liability value | \$333,328 | \$421,767 |
| Macaulay duration | 10.18 years | 4.18 years |
| Modified duration | 9.79 | 4.02 |
| Convexity | 112.7 | 27.6 |
| PV01 | \$326 | \$169 |

The bond sleeve's weighted duration is $w_B^\top D = 4.06$ years. Redington (1952) immunisation matches the durations of assets and liabilities so that small parallel yield changes move both equally. We matched the sleeve to the liability as it stands at the 2033 valuation, not to its 2027 duration; Section 16 states what that leaves exposed.

---

## 4. The strategies

### 4.1 The rival strategies

| Code | Strategy | Rule | Finance basis |
|---|---|---|---|
| S0 | Bucket ladder | Section 4.4 | Evensky's bucket method, as tested by Estrada (2019) |
| S1 | T-bills | 100% SGOV | Risk-free benchmark |
| S2 | Pure LDI | $e = 0$: 100% bond sleeve | Liability-driven investing (Redington, 1952) |
| S4 | All-equity | 100% VTI | Growth benchmark |
| S6 | Target-date | $e_t = 0.60 - 0.30\,\min(t,5)/5$ | Calendar glide path, 60% to 30% |
| S7 | Risk parity | $e = \dfrac{1/\sigma_E}{1/\sigma_E + 1/\sigma_B} = 17.9\%$ | Equal risk contribution (Maillard et al., 2010) |
| S8 | CPPI | $e_t = \min(3\,C_t/V_t,\ 1)$ | Black and Perold (1992) |
| S9 | Core-satellite | $e = E_0 = 39.7\%$ fixed | The portfolio as built, with no floor rule |
| S10C | **LDCS** | Section 4.2 | Contingent immunisation (Leibowitz and Weinberger, 1982) |

S9A (our rule with a calendar glide path) and S10 (our rule without the 2031 lock-in) are earlier versions of LDCS. We keep them in `strategy_results.json` and in the ten-strategy check, and leave them out of the nine-strategy ranking.

**Why inverse volatility is risk parity for two sleeves.** A sleeve's risk contribution is $RC_i = w_i\,(\Sigma w)_i$. For two sleeves with correlation $\rho$,

```math
RC_E = w_E^2\sigma_E^2 + w_E w_B\,\rho\,\sigma_E\sigma_B, \qquad RC_B = w_B^2\sigma_B^2 + w_E w_B\,\rho\,\sigma_E\sigma_B .
```

The cross terms are identical, so $RC_E = RC_B$ exactly when $w_E\sigma_E = w_B\sigma_B$, whatever $\rho$ is. We measured $\sigma_E = 15.08\%$ and $\sigma_B = 3.28\%$ a year over 2004–2026, which gives $e = 17.9\%$.

### 4.2 LDCS: the cushion rule

LDCS holds bonds worth the floor and invests only the cushion above it in equities, up to the strategic share:

```math
F_t = PV_L(t) \;-\; \frac{150{,}000}{(1+r)^{1-t}}\,\mathbf 1_{\{t<1\}} \;+\; \frac{L\,\mathbf 1_{\{lock\}}}{(1+r)^{6-t}}\,\mathbf 1_{\{t\ge 4\}}, \qquad C_t = \max(V_t - F_t,\ 0), \qquad e_t = \min\!\Big(\frac{C_t}{V_t},\ E_0\Big).
```

**Proposition (why the cushion rule protects the reserve).** Suppose that at a review the equity holding is $E \le C = V - F$, so the bond holding is $B = V - E \ge F$. Suppose further that until the next review the bond sleeve grows at least as fast as the floor. Then the portfolio stays at or above the floor, however far equities fall.

*Proof.* After any equity return $\rho_E \ge -100\%$ the equity holding is $E(1+\rho_E) \ge 0$, so $V' = B' + E(1+\rho_E) \ge B' \ge F'$. ∎

Two consequences follow, and the simulations show both.

- **CPPI breaks the proposition.** With multiplier 3, $E = 3C$. An equity fall of more than one third before the next review leaves $V' < F'$. That is the gap risk of CPPI (Black and Perold, 1992), and it is why S8 scores only 47.0% on promise reliability (Section 7).
- **The proposition needs the bonds.** If bonds fall while the floor does not, the second condition fails. The 2022-type replay, where stocks and bonds fell together, is exactly this case, and it is LDCS's weakest result (45% of paths keep the promise).

The cap $E_0$ makes LDCS more conservative than a pure $m = 1$ rule, and the 20% extra review (Section 4.5) resets the cushion during a fall.

### 4.3 The 2031 promise and the lock-in

In January 2031 ($t = 4$) we set the minimum contribution Laura announces to co-sponsors at half the surplus over the liability, grown at 4% to 2033:

```math
L = 0.5 \cdot \max\big(V_{2031} - PV_L(4),\ 0\big)\cdot(1+r)^2 .
```

For every strategy we record whether the 2033 contribution meets it ($c \ge L$). LDCS alone adds the discounted promise to its floor from January 2031, so it protects the promise as well as the reserve, but only when safe assets can grow it at the 4% rate: the lock-in applies if the two-year Treasury yield in January 2031 is at least 4%. Our engine simulates the 5-year yield, so we estimate the 2-year yield as the simulated 5-year yield minus today's 0.08-point spread between them:

```math
\text{lock} = \mathbf 1\big\{\,y_5^{2031} - 0.0008 \ge 0.04\,\big\}, \qquad y_5^{2031} = y_5^{now} + \sum_{m<48} \Delta y_{5,m}.
```

The condition holds in 68% of the history-case paths and 73% of the jump-diffusion paths. Where it fails, LDCS keeps the cushion rule alone and the promise stays unprotected.

### 4.4 S0: the bucket ladder

We split the liability into three Treasury buckets matched to payment dates and valued each at 4%:

| Bucket | Payments | Funds | Value in January 2027 | Value in January 2033 |
|---|---|---|---|---|
| B1 | 2033–2035 | SGOV, SHY | \$114,046 | \$144,305 |
| B2 | 2036–2039 | VGIT | \$132,616 | \$167,802 |
| B3 | 2040–2042 | IEF, SCHP | \$86,666 | \$109,660 |

At each January review we fill B1, then B2, then B3 from the money not in equities; if there is more than enough, we scale all three pro rata. The equity bucket is what remains above the three buckets, capped at the strategic share:

```math
E^{target}_t = \min\Big(\max\big(V_t - (B_1 + B_2 + B_3),\ 0\big),\ E_0 V_t\Big).
```

S0 sells equity to refill the buckets only after a year in which the equity bucket rose; otherwise it holds the equity and lets the buckets run short. S0b, a variant, refills every January. S0 has one review a year and no 20% trigger or lock-in.

### 4.5 Reviews

For every strategy except S0 we reset the holdings to the rule each January (adding the \$150,000 in January 2028) and also in any month in which the equity sleeve stands 20% below its peak since the last review. For LDCS and CPPI the extra review cuts equity in a fall; for the fixed-mix strategies (S6, S7, S9) it rebalances back to target, which means buying equity after the fall.

### 4.6 Consistency with the IPS: the tolerance band and the yield condition

Our IPS commits to the cushion rule and to "a strict ±2% tolerance band" that triggers rebalancing. The 20% extra review and the 2031 lock-in come from our later research (Research Pack, Section 8). To check that the ranking does not depend on those later choices, we ran four other versions of LDCS through the same paths and put each into the nine-strategy scoreboard in place of S10C. For the band we check the equity share every month-end and rebalance to the rule's target whenever it is more than 2 points away:

```math
\big|\,e^{actual}_m - e^{rule}_m\,\big| > 0.02 \;\Rightarrow\; \text{rebalance to } e^{rule}_m .
```

| Version of LDCS | Lock-in applied | Min. funding certainty | Avg. funding certainty | Median contribution | Promise reliability | Place (equal / safety / growth) |
|---|---|---|---|---|---|---|
| Lock-in in every path (Research Pack) | 100% | 83.0% | 94.1% | \$184k | 91.2% | 1 / 1 / 1 |
| **Lock-in if 2-year yield ≥ 4% (headline)** | 68% | 83.5% | 94.2% | \$187k | 89.0% | 1 / 1 / 1 |
| IPS ±2% band, no 20% trigger | 68% | 82.1% | 93.8% | \$185k | 89.6% | 1 / 1 / 1 |
| IPS ±2% band and 20% trigger | 68% | 82.3% | 93.9% | \$185k | 89.6% | 1 / 1 / 1 |
| IPS cushion rule and band only, no lock-in | none | 83.1% | 94.0% | \$194k | 85.5% | 1 / 1 / 1 |

Every version ranks first under all three weightings, and no criterion moves by more than 1.5 points or \$10k. The lock-in's job shows in the last column: without it, promise reliability falls from 89–91% to 85.5%.

---

## 5. The Monte Carlo engine and the four return cases

### 5.1 The principle

We cannot know which market path 2027–2033 will follow, so we simulate many and read probabilities as shares of paths (Glasserman, 2003). If $X_i = 1$ when path $i$ covers the reserve and 0 otherwise, the law of large numbers makes $\hat p = \frac1N\sum X_i$ converge to the true probability $p$, and because each $X_i$ is Bernoulli with variance $p(1-p)$,

```math
SE(\hat p) = \sqrt{\frac{p(1-p)}{N}} .
```

With $N = 10{,}000$ and $p = 0.94$, $SE = 0.24$ percentage points. We simulate each strategy monthly for 72 months: each month every holding grows by its return, and the reviews of Section 4.5 reset the weights.

### 5.2 Case H: history by block bootstrap

We build each path from 12 blocks of 6 consecutive months, each block starting at a uniformly random month of the 274-month 2004–2026 history:

```math
s_b \sim \text{Uniform}\{1,\dots,269\}, \quad b = 1,\dots,12; \qquad \text{path} = (x_{s_1},\dots,x_{s_1+5},\ x_{s_2},\dots,x_{s_{12}+5}).
```

Each draw is a whole month of all eight funds plus $\Delta y_5$, so the correlations between funds survive. Drawing blocks rather than single months also keeps short-run momentum and volatility clustering (the moving-block bootstrap of Künsch, 1989).

### 5.3 Case S: stress

We take the H paths and subtract 3.25 percentage points a year from every fund ($0.0325/12$ a month). This asks whether the ranking holds in a decade clearly worse than 2004–2026.

### 5.4 Case C: current bond carry

A bond fund's long-run return is close to its starting yield. Yields in November 2026 (5-year 4.71%, 3-month bill 4.13%) are well above their 2004–2026 averages, so history understates bond income. We take the H paths and add to each bond fund, every month, one-twelfth of its current yield minus its 2004–2026 average yield:

| IEF | SCHP | SGOV | SHY | VGIT |
|---|---|---|---|---|
| +1.90 pts | +1.86 pts | +2.36 pts | +2.54 pts | +2.16 pts |

### 5.5 Case J: Markov-modulated jump-diffusion

Case J generates returns from a model rather than from resampled history, so it can produce crashes and rate shocks that 2004–2026 did not contain. Each month it does four things.

1. **Regime.** The market is calm or turbulent, moving between the two by the transition matrix of Section 13. Each path starts turbulent with probability 1.76%, today's filtered probability.
2. **Normal returns.** Given the regime $s$, the nine variables (eight funds and $\Delta y_5$) are multivariate normal, $x = \mu_s + L_s z$ with $z$ standard normal and $L_s L_s^\top = \Sigma_s$ (Cholesky factor). We estimated $\mu_s$ and $\Sigma_s$ from the 2004–2026 months that the regime model classifies as calm (smoothed probability ≤ 0.5) or turbulent, leaving out jump months. For VTI the regimes give 16.1% a year mean and 12.0% volatility when calm, and −14.0% mean and 26.2% volatility when turbulent.
3. **Jumps (Merton, 1976).** In each month a jump occurs with probability $1 - e^{-\lambda_{J,s}/12}$. We defined a jump as a month of −10% or worse in the CRSP market: 31 in 1926–2026. Their log returns have mean −0.159 and standard deviation 0.058, so VTI's jump return is

   ```math
   J = e^{Y} - 1, \qquad Y \sim \mathcal N(-0.159,\ 0.058^2).
   ```

   Weighting each jump month by the regime probability gives $\lambda_J = 0.013$ a year when calm and 1.80 a year when turbulent. In a jump month every other fund returns $\beta_i J$, with $\beta_i$ the least-squares slope through the origin on the 2004–2026 jump months, $\beta_i = \sum_t x_{i,t}x_{VTI,t} / \sum_t x_{VTI,t}^2$: VXUS 1.19, EQAL 1.24, SCHP 0.37, IEF −0.06, SHY −0.06, VGIT −0.09.
4. **Rate shocks.** A rate shock is a rise of 2 points or more in the 10-year yield within 12 months: 5 in 72.6 years of FRED data, $\lambda = 0.069$ a year. A shock starts in a month with probability $1 - e^{-0.069/12}$, lasts 12 months and replays one of the five historical episodes (1980, 1984, 1987, 1994, 2022), drawn at random. During the shock each bond fund loses $D_i\,\Delta y/12$ a month (TIPS 0.6 times that, plus 2% a year of extra inflation accrual), bills earn more as their yield rises, and equities earn the episode's 12-month excess return spread evenly.

Across the 10,000 J paths we get 1.6 jump months per path on average; 63% of paths contain at least one jump and 34% a rate shock.

### 5.6 The valuation

At the end of each path we record $V_{2033}$ and $c = V_{2033} - R_4$. We also record the January 2033 5-year yield $y_{33} = \max(y_{now} + \sum \Delta y,\ 0.25\%)$ and the market cost of the ten payments at that yield (the "ladder" value), which we report but do not score.

---

## 6. The crash replays

The four cases describe typical and bad markets in general. The replays ask a sharper question: what if a specific historical crash happened at the worst moment for Laura? For each type we take the actual monthly returns of the episode and overwrite the last months of every H path with them, so the trough falls exactly on the January 2033 valuation and no time remains to recover. We also ran each episode at the start of the plan ("early" timing) for comparison; the scoreboard uses the worst timing.

| Replay | Type | Episode | Months |
|---|---|---|---|
| T1 | Recession bear | November 2007 – February 2009 | 16 |
| T2 | Exogenous V-shock | February – March 2020 | 2 |
| T3 | Inflation and rate shock | January – September 2022 | 9 |
| T3+ | Severe inflation shock | January 1973 – September 1974 | 21 |
| T4 | Valuation bust | September 2000 – September 2002 | 25 |
| T5 | Falling-rate shock | January 2019 – July 2020 | 19 |

No ETFs or TIPS existed in 1973–74, so for T3+ we rebuilt every fund from CRSP equity returns and FRED yields with the formula of Section 2, and proxied TIPS by CPI inflation plus 1% a year.

---

## 7. The six scoreboard criteria

We score each strategy from two path-level events, covering the reserve and keeping the promise. For a case or replay $k$:

```math
F_k = \frac1N\sum_{i=1}^N \mathbf 1\{V_{2033,i} \ge R_4\}, \qquad PK_k = \frac1N\sum_{i=1}^N \mathbf 1\{c_i \ge L_i\}.
```

| Criterion | Formula | What it tells Laura |
|---|---|---|
| Minimum funding certainty | $\min(F_H, F_S, F_J, F_C)$ | The worst of the four worlds: the maximin principle (Wald, 1950), because Laura cannot renegotiate the operating promise |
| Average funding certainty | $\frac14(F_H + F_S + F_J + F_C)$ | The overall chance the reserve is fully funded in 2033 |
| Median contribution | $\frac12[\text{median}_H(c) + \text{median}_C(c)]$ | The typical money for the facility; we left out the stress case because it is a stress, not an expectation |
| Bad-case contribution | 5th percentile of $c$ in case J | A Value-at-Risk-style tail figure (Jorion, 2007): the worst 1 path in 20 |
| Crash resilience | $\frac16\sum_{j=1}^6 F_{T_j}$ | The reserve after each historical crash at the worst moment |
| Promise reliability | mean of $PK_H$, $PK_J$ and the six replay $PK$s | Whether the portfolio delivers in 2033 the figure Laura announces in 2031 |

**Worked example (LDCS).** $F_H = 99.58\%$, $F_S = 83.53\%$, $F_J = 93.67\%$, $F_C = 99.99\%$. The minimum is 83.5% and the average 94.2%.

---

## 8. Ranking, weightings and random weights

### 8.1 Weighted average rank

For each criterion we rank the nine strategies from 1 (highest value) to 9; no criterion has tied values. A strategy's score is its weighted average rank, and the lowest score wins. This Borda-style rank sum puts percentages and dollars on one scale:

```math
A_s = \frac{\sum_{c=1}^6 w_c\, r_{s,c}}{\sum_{c=1}^6 w_c}.
```

| Weighting | Weights (min FC, avg FC, median $c$, P5, crash, promise) | LDCS ranks 4, 2, 5, 3, 2, 2 give |
|---|---|---|
| Equal | 1, 1, 1, 1, 1, 1 | 18 / 6 = 3.00 |
| Safety | 2, 2, 1, 1, 2, 2 | 28 / 10 = 2.80 |
| Growth | 1, 1, 3, 1, 1, 1 | 28 / 8 = 3.50 |

Ranks keep order and drop size, so we read them alongside the values and test the actual differences in Section 10.

### 8.2 Random weights

Our three weightings are judgements, so we drew 20,000 weight vectors from the flat Dirichlet distribution, which makes every mix of six non-negative weights summing to 1 equally likely (Kotz et al., 2000). We drew each vector by taking six independent exponential numbers $G_1,\dots,G_6$ and setting $w_c = G_c / \sum G$.

**Why that is uniform.** The joint density of the $G$'s is $e^{-\sum g_c}$. Changing variables to the total $S = \sum G_c$ and the shares $w$ has Jacobian $S^{5}$, so the joint density becomes $S^{5}e^{-S}$, which does not depend on $w$. Integrating $S$ out leaves a constant density over the simplex: the flat Dirichlet. ∎

For each draw we recomputed every strategy's weighted rank. LDCS came first in 68.2% of draws and in the top three in 98.2%. The 6,368 draws it lost put on average 27% of the weight on minimum funding certainty (12% in the draws it won); risk parity, first on that criterion, won 4,707 of them and T-bills 1,193.

---

## 9. Seed stability and Monte Carlo error

To show the ranking is not an accident of one set of random numbers, we re-ran the whole simulation (four cases and six replays, 10,000 paths each) with 20 new seed pairs: bootstrap seeds 1000–1019 and jump-diffusion seeds 2000–2019. LDCS came first under all three weightings in all 20 runs.

The spread across seeds matches the formula of Section 5.1. For LDCS's minimum funding certainty the formula gives $\sqrt{0.835 \times 0.165 / 10{,}000} = 0.37$ points; we observed 0.38. The narrowest gaps next to an LDCS rank are 0.5 points (crash resilience against risk parity), 0.7 points (average funding certainty against the bucket ladder) and \$6k (median contribution against the bucket ladder), against seed-to-seed spreads of 0.04–0.13 points and \$0.8–0.9k.

We also re-ran the base seeds (42 for the bootstrap, 7 for case J) through the same code; it reproduced every scoreboard value exactly.

---

## 10. Paired tests on identical paths

Every strategy runs on the same 10,000 paths, so we compare LDCS with each rival path by path. Using common random numbers cancels the market luck both strategies share and isolates the effect of the rule (Glasserman, 2003).

### 10.1 McNemar's exact test for funding and promise

Paths where both strategies succeed or both fail carry no information about which is better. Let $b$ count paths where only LDCS covers the reserve and $c$ paths where only the rival does. If the two strategies were equally good, each of these $b + c$ discordant paths would be equally likely to fall either way, so under the null hypothesis

```math
b \sim \text{Binomial}\big(b+c,\ \tfrac12\big), \qquad p = \min\!\Big(1,\ 2\sum_{k=0}^{\min(b,c)}\binom{b+c}{k}2^{-(b+c)}\Big)
```

(McNemar, 1947). We applied the same test to keeping the promise.

**Example.** Against core-satellite in case J, LDCS alone covers the reserve on 992 paths and core-satellite alone on 92, so $p \approx 10^{-191}$. In case S the result reverses, 351 against 1,031: with 3.25 points a year less return, LDCS's smaller equity share leaves it short more often. Table D in `ROBUSTNESS.md` counts, out of the four cases, how often LDCS is significantly better or worse at $p < 0.05$.

### 10.2 Paired mean difference for the contribution

For each path we take $d_i = c_{LDCS,i} - c_{rival,i}$ and call the difference significant when the 95% interval excludes zero:

```math
\bar d \pm 1.96\,\frac{s_d}{\sqrt N}.
```

Against core-satellite in case H, $\bar d = -\$43{,}447 \pm \$879$: LDCS gives up about \$43,000 of typical facility money for its safety.

### 10.3 Multiple testing

Table D summarises 96 tests (8 rivals × 4 cases × 3 measures), so about five could pass $p < 0.05$ by chance. We re-checked every count at the Bonferroni level $0.05/96 = 0.00052$. All counts stay the same except two, which drop out: one funding win over pure LDI and one promise loss to risk parity. With 10,000 paths very small differences become significant, so we read significance as "real in the model" and use the sizes in Table A to judge whether a difference matters to Laura.

---

## 11. The Poisson crash model

### 11.1 Counting crashes

We define a crash as a fall of 20% or more from a running peak in the CRSP total-return index at month-ends. We start an episode at a running peak, count it once the fall reaches 20%, follow it to the trough, and end it when the index regains the old peak, so we count each bear market once. June 1926 to October 2026 contains 10 crashes (5 of them 30% or deeper).

| Peak | Trough | Depth | Type |
|---|---|---|---|
| 1929-08 | 1932-06 | −83.7% | T4 Valuation bust |
| 1946-05 | 1947-05 | −24.3% | T3 Inflation |
| 1961-12 | 1962-06 | −23.1% | Other (grouped with T2) |
| 1968-11 | 1970-06 | −33.6% | T3+ Severe inflation |
| 1972-12 | 1974-09 | −46.5% | T3+ Severe inflation |
| 1987-08 | 1987-11 | −29.9% | T2 V-shock |
| 2000-08 | 2002-09 | −45.0% | T4 Valuation bust |
| 2007-10 | 2009-02 | −50.3% | T1 Recession bear |
| 2020-01 | 2020-03 | −20.2% | T2 V-shock |
| 2021-12 | 2022-09 | −24.8% | T3 Inflation |

### 11.2 The Poisson process from first principles

A Poisson process with rate $\lambda$ assumes that in a short interval $dt$ one event occurs with probability $\lambda\,dt$, two or more with negligible probability, and that disjoint intervals are independent (Ross, 2014). Let $P_0(t)$ be the probability of no event by time $t$. Then

```math
P_0(t+dt) = P_0(t)(1 - \lambda\,dt) \;\Rightarrow\; P_0'(t) = -\lambda P_0(t) \;\Rightarrow\; P_0(t) = e^{-\lambda t},
```

and repeating the argument for $k$ events gives $\Pr(N_t = k) = e^{-\lambda t}(\lambda t)^k / k!$. Two results follow that we use throughout:

```math
\Pr(N_t \ge 1) = 1 - e^{-\lambda t}, \qquad \Pr(T > t) = e^{-\lambda t}\ \text{(the wait for the next crash is exponential with mean } 1/\lambda).
```

### 11.3 Estimating the rate

Observing $n$ crashes in $T$ years gives the likelihood $\mathcal L(\lambda) \propto \lambda^n e^{-\lambda T}$. Setting the derivative of $\log\mathcal L = n\log\lambda - \lambda T$ to zero gives the maximum-likelihood estimate

```math
\hat\lambda = \frac nT = \frac{10}{100.33} = 0.0997 \text{ a year}, \qquad \Pr(\ge 1 \text{ crash in 2031–32}) = 1 - e^{-0.0997\times2} = 18.1\%.
```

| Window | Years | P(at least one 20% crash) |
|---|---|---|
| 2027 | 1 | 9.5% |
| 2031–32 (critical years) | 2 | 18.1% |
| 2027–32 (build-up) | 6 | 45.0% |
| 2027–42 (whole plan) | 16 | 79.7% |

### 11.4 The exact confidence interval

The identity $\Pr(N \le n \mid \mu) = \Pr(\chi^2_{2n+2} > 2\mu)$ links the Poisson and chi-square distributions. The upper 90% limit is the mean at which observing $n$ or fewer events has probability 5%, and the lower limit the mean at which observing $n$ or more has probability 5%. Solving both and dividing by $T$ gives Garwood's (1936) exact interval:

```math
\lambda \in \left[\frac{\chi^2_{0.05}(2n)}{2T},\ \frac{\chi^2_{0.95}(2n+2)}{2T}\right] = [0.054,\ 0.169], \qquad \Pr(\ge1 \text{ in 2031–32}) \in [10.3\%,\ 28.7\%].
```

`verify.py` re-derives these limits by root-finding on the Poisson distribution itself, without the chi-square identity.

### 11.5 Crash types and the splitting theorem

We classified each crash from its peak and trough with fixed rules, applied in order:

1. 10-year yield up 1 point or more, or CPI inflation 5% or more at the trough → inflation shock (T3); T3+ if the fall is 30% or deeper.
2. Trough within 4 months → exogenous V-shock (T2).
3. CAPE 25 or more at the peak and 18 months or more to the trough → valuation bust (T4).
4. An NBER recession between peak and trough → recession bear (T1).
5. Otherwise "other"; the single case (1962) we grouped with V-shocks.

If each event of a Poisson process is independently assigned type $j$ with probability $\pi_j$, the type counts are independent Poisson processes with rates $\lambda\pi_j$, and their rates add back to $\lambda$ (the splitting, or thinning, theorem; Ross, 2014). That lets us estimate each type's rate from its own count:

| Type | Count in 100.33 years | $\lambda_j$ | 90% interval | P(in 2031–32) |
|---|---|---|---|---|
| T1 Recession | 1 | 0.010 | 0.001–0.047 | 2.0% |
| T2 V-shock | 3 | 0.030 | 0.008–0.077 | 5.8% |
| T3 Inflation | 2 | 0.020 | 0.004–0.063 | 3.9% |
| T3+ Severe inflation | 2 | 0.020 | 0.004–0.063 | 3.9% |
| T4 Valuation bust | 2 | 0.020 | 0.004–0.063 | 3.9% |
| T5 Falling rates | 9 in 71.6 years | 0.126 | 0.066–0.219 | 22.2% |

T5 is a different event from the stock crashes: a fall of 2 points or more in the 5-year yield within 24 months (FRED, from 1953). It matters because falling yields raise the value of Laura's liability.

### 11.6 Joint Poisson shortfall

Let $p_j = 1 - e^{-2\lambda_j}$ be the chance that a type-$j$ crash occurs in 2031–32, and $q_j$ the share of paths in which that crash, replayed at the worst moment, breaks the promise (or leaves the reserve short). Let $A_j$ be the event that type $j$ occurs and breaks the promise. Boole's inequality, $\Pr(\bigcup A_j) \le \sum \Pr(A_j)$, gives the bound we report, and assuming the types occur independently gives the union form:

```math
\Pr(\text{broken promise}) \le \sum_j p_j\,q_j, \qquad \Pr(\text{broken promise}) \approx 1 - \prod_j (1 - p_j q_j).
```

The sum is conservative twice over: Boole's inequality is an upper bound, and $q_j$ comes from the worst timing. For the bracket we replaced each $\lambda_j$ by its Garwood limits.

**Worked example (LDCS, broken promise).**

| Type | $p_j$ | $q_j$ | $p_j q_j$ |
|---|---|---|---|
| T1 | 1.97% | 7.45% | 0.147% |
| T2 | 5.80% | 2.52% | 0.146% |
| T3 | 3.91% | 55.26% | 2.160% |
| T3+ | 3.91% | 3.05% | 0.119% |
| T4 | 3.91% | 0.00% | 0.000% |
| T5 | 22.23% | 0.01% | 0.002% |
| **Sum** | | | **2.57%** |

Almost all the risk (2.16 of 2.57 points) is the 2022-type shock. The same arithmetic gives core-satellite 9.81%, because without a floor rule it breaks the promise in 92% of 1973–74-type paths, 95% of 2008-type, 54% of 2022-type and 35% of 2000–02-type.

### 11.7 Compound Poisson damage to Laura's holdings

To put a dollar figure on crash exposure, we modelled total crash damage as a random sum: $N$ crashes in the window, each with loss $L_j$ (Klugman et al., 2012). By the laws of total expectation and total variance,

```math
E[S] = E\big[E[S\mid N]\big] = E[N]\,E[L] = \lambda t\,E[L], \qquad \operatorname{Var}[S] = E[N]\operatorname{Var}(L) + \operatorname{Var}(N)\,E[L]^2 = \lambda t\,E[L^2],
```

using $E[N] = \operatorname{Var}(N) = \lambda t$.

We drew each crash's type with its historical frequency (T1 10%, T2 30%, T3 40%, T4 20%), its depth from the same type's history, and its fund-by-fund returns from the matching 1990–2026 episode, scaling equity returns by depth relative to that episode. The dollar loss is the exposure (\$300,000 in 2027, \$450,000 afterwards) times the portfolio return. The average crash costs the portfolio as built 13.3%. Over 2027–32, with 200,000 simulations, the expected damage is \$33,676 against \$33,797 from the formula, and the standard deviation \$48,136 against \$48,205. This is damage before any recovery: a measure of exposure, not a forecast.

We also counted falls of 10%, 20% and 30% in each of the eight funds since 1990 (`fund_level_poisson.csv`) to show which holdings carry the crash risk.

### 11.8 Which crash hurts Laura most, and under which strategy

This is where the Poisson layer and the Monte Carlo replays meet. For each crash type $j$ and strategy $s$ we multiplied the Poisson chance that the crash happens in 2031–32 by the share of replayed paths in which it breaks the 2031 promise:

```math
\text{Risk}_{j,s} = \underbrace{\big(1 - e^{-2\lambda_j}\big)}_{\text{Poisson: crash occurs in 2031–32}} \times \underbrace{\Pr(c < L \mid \text{crash } j \text{ ends in January 2033},\ s)}_{\text{Monte Carlo replay}} .
```

We then read across each row for the strategy hit hardest, and summed down each column (Section 11.6) for the strategy most exposed overall. Cells are percentages; the bold cell is the hardest-hit of the seven strategies compared in the Final Report.

| Crash type | $\lambda_j$ a year | P(in 2031–32) | LDCS (ours) | Bucket ladder | T-bills | Pure LDI | Target-date | Risk parity | Core-satellite | Worst of all nine |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 2008-type recession | 0.010 | 2.0% | 0.15% | 0.52% | 0.00% | 0.00% | 1.16% | 0.30% | **1.87%** | All-equity |
| T2 2020-type V-shock | 0.030 | 5.8% | 0.15% | 0.31% | 0.00% | 0.18% | 0.51% | 0.25% | **0.80%** | CPPI m=3 |
| T3 2022-type inflation | 0.020 | 3.9% | 2.16% | 2.81% | 0.00% | **2.97%** | 1.65% | 2.55% | 2.12% | Pure LDI |
| T3+ 1973–74-type inflation | 0.020 | 3.9% | 0.12% | 0.96% | 0.00% | 0.00% | 1.95% | 0.44% | **3.60%** | All-equity |
| T4 2000–02-type bust | 0.020 | 3.9% | 0.00% | 0.00% | 0.00% | 0.00% | 0.53% | 0.00% | **1.37%** | All-equity |
| T5 2019–20 falling rates | 0.126 | 22.2% | 0.00% | 0.01% | 0.00% | 0.00% | **0.09%** | 0.00% | 0.04% | CPPI m=3 |
| **All six types** | | | **2.57%** | **4.60%** | **0.00%** | **3.15%** | **5.90%** | **3.54%** | **9.81%** | CPPI m=3 (13.71%) |

To show the damage in money, we also measured how far each crash pulls the median 2033 contribution below the no-crash history case, and weighted each fall by the crash's Poisson probability:

```math
\text{Expected fall}_s = \sum_{j \ne T5} \big(1 - e^{-2\lambda_j}\big)\,\big[\text{median}_H(c_s) - \text{median}_{j}(c_s)\big].
```

| Crash type | LDCS (ours) | Bucket ladder | T-bills | Pure LDI | Target-date | Risk parity | Core-satellite |
|---|---|---|---|---|---|---|---|
| T1 2008-type recession | \$44k | \$72k | \$1k | −\$38k | \$124k | \$48k | \$158k |
| T2 2020-type V-shock | \$13k | \$24k | \$0k | −\$13k | \$37k | \$14k | \$51k |
| T3 2022-type inflation | \$80k | \$92k | \$3k | \$52k | \$107k | \$80k | \$117k |
| T3+ 1973–74-type inflation | \$44k | \$72k | −\$51k | −\$29k | \$115k | \$48k | \$147k |
| T4 2000–02-type bust | \$3k | \$20k | −\$17k | −\$92k | \$83k | −\$3k | \$109k |
| T5 2019–20 falling rates | −\$50k | −\$46k | \$1k | −\$42k | −\$58k | −\$48k | −\$54k |
| **Expected fall, five stock-crash types** | **\$7k** | **\$10k** | **−\$2k** | **−\$4k** | **\$16k** | **\$7k** | **\$21k** |

A negative figure is a gain. We left the falling-rate replay out of the total: it raises every bond-holding strategy's contribution because we value the reserve at a fixed 4%, while at market yields the same fall in rates would also raise the cost of the ten payments (Section 16).

What the two tables show:

- **The 2022-type stock-and-bond inflation shock is the crash that hurts LDCS most.** It supplies 2.16% of LDCS's 2.57% risk and the largest dollar fall (\$80k). This is the failure the proposition of Section 4.2 predicts: when bonds fall with stocks, the floor stops protecting the promise. Pure LDI is hit hardest of all by this crash type, because it holds only bonds.
- **Without the floor rule the same ETFs are far more exposed.** Core-satellite holds LDCS's funds without the rule and carries 9.81% risk, 3.8 times LDCS's; it is hit hardest by the 2008-type (1.87%) and 1973–74-type (3.60%) crashes and loses \$158k of median contribution in a 2008-type crash against \$44k for LDCS.
- **The most exposed strategies overall** (all nine, by summed risk): CPPI m=3 13.71%, All-equity 13.69%, Core-satellite 9.81%, Target-date 5.90%, Bucket ladder 4.60%, Risk parity 3.54%, Pure LDI 3.15%, LDCS (ours) 2.57%, T-bills 0.00%.
- **T-bills carry no crash risk to the promise** but give up the most facility money in every normal market (Section 7), which is why they rank fourth rather than first.

---

## 12. Testing the Poisson assumption

A Poisson process has two fingerprints we can test: counts whose variance equals their mean, and exponential waiting times.

### 12.1 Index of dispersion

If crashes cluster, counts per period vary more than Poisson allows. Under the Poisson hypothesis the statistic follows a chi-square distribution (Cochran, 1954):

```math
I = \frac{s^2}{\bar x}, \qquad (k-1)\,I \sim \chi^2_{k-1}.
```

| Counted by | Periods $k$ | Mean | Variance | $I$ | $\chi^2$ | p (clustering) |
|---|---|---|---|---|---|---|
| Year of peak | 101 | 0.099 | 0.090 | 0.91 | 91.0 on 100 df | 0.73 |
| Decade | 10 | 1.0 | 0.44 | 0.44 | 4.0 on 9 df | 0.91 |

Both indices are below 1: crashes are, if anything, more evenly spaced than random.

### 12.2 Waiting times

The nine gaps between crash peaks (1.9 to 16.8 years, mean 10.3) should be exponential with mean $1/\lambda = 10.0$ years. The Kolmogorov–Smirnov statistic is the largest distance between the observed and the theoretical cumulative distributions (Massey, 1951):

```math
D = \sup_x \big|F_9(x) - (1 - e^{-\lambda x})\big| = 0.28, \qquad p = 0.42.
```

We estimated $\lambda$ from the same data, which makes the standard KS p-value too generous; Lilliefors (1969) gives the corrected test for an exponential with an estimated mean. With only nine gaps, all three tests can detect only a gross departure, so we keep the Markov layer as the clustering alternative.

### 12.3 Why we added jumps

Monthly CRSP log returns have excess kurtosis 6.5 and skew −0.55. A normal distribution with the same mean and volatility predicts 0.007 months of −20% or worse in 1,204 months; history has 5. Heavy tails are a standard property of asset returns (Cont, 2001), and they are why case J adds Merton jumps rather than relying on normal returns alone.

---

## 13. The Markov regime model

### 13.1 The model

Markets alternate between calm and turbulent stretches, and turbulent months tend to follow each other. We fitted a two-state Markov-switching model (Hamilton, 1989) to the 1,204 monthly CRSP log returns. The hidden state $s_t$ sets the return distribution and moves from month to month by a transition matrix:

```math
r_t \mid s_t \sim \mathcal N(\mu_{s_t}, \sigma_{s_t}^2), \qquad P = \begin{pmatrix} p_{cc} & p_{ct}\\ p_{tc} & p_{tt}\end{pmatrix} = \begin{pmatrix} 0.975 & 0.025\\ 0.123 & 0.877\end{pmatrix}.
```

| Regime | Mean monthly log return | Monthly volatility | Expected spell | Long-run share |
|---|---|---|---|---|
| Calm | +1.35% | 3.59% | 40.7 months | 83.4% |
| Turbulent | −1.83% | 9.74% | 8.1 months | 16.6% |

### 13.2 Fitting by the EM algorithm

We fitted the model with the expectation–maximisation algorithm (Dempster et al., 1977), 300 iterations from fixed starting values.

- **Forward pass (filter).** Bayes' theorem updates the state probabilities each month: $\alpha_t(j) \propto \big[\sum_i \alpha_{t-1}(i)\,P_{ij}\big]\, f_j(r_t)$, where $f_j$ is the normal density of state $j$.
- **Backward pass (smoother).** $\beta_t(i) \propto \sum_j P_{ij}\, f_j(r_{t+1})\, \beta_{t+1}(j)$, and the smoothed probability is $\gamma_t(j) \propto \alpha_t(j)\beta_t(j)$, which uses the whole sample.
- **Update.** With $\xi_t(i,j) \propto \alpha_{t-1}(i)\,P_{ij}\,f_j(r_t)\,\beta_t(j)$, we re-estimate

```math
P_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t\sum_k \xi_t(i,k)}, \qquad \mu_j = \frac{\sum_t \gamma_t(j)\,r_t}{\sum_t\gamma_t(j)}, \qquad \sigma_j^2 = \frac{\sum_t \gamma_t(j)(r_t-\mu_j)^2}{\sum_t\gamma_t(j)}.
```

The filtered probability for October 2026, using data up to that month only, is 1.76% turbulent; this is the starting state of case J.

### 13.3 Long-run shares and spell lengths

The stationary distribution satisfies $\pi P = \pi$ with $\pi_c + \pi_t = 1$. For two states the flows between them must balance, $\pi_c\,p_{ct} = \pi_t\,p_{tc}$, which gives

```math
\pi_t = \frac{p_{ct}}{p_{ct} + p_{tc}} = \frac{0.0246}{0.0246 + 0.1234} = 16.6\%.
```

A spell in state $i$ lasts $k$ months with probability $p_{ii}^{k-1}(1-p_{ii})$, a geometric distribution with mean $1/(1-p_{ii})$: 40.7 months calm, 8.1 months turbulent.

### 13.4 Turbulence in 2031–32

We moved today's state vector forward 48 months by matrix powers to January 2031, then required all 24 months of 2031–32 to stay calm:

```math
\pi_{48} = (0.982,\ 0.018)\,P^{48}, \qquad \Pr(\ge 1 \text{ turbulent month in 2031–32}) = 1 - \pi_{48}(c)\,p_{cc}^{23} = 52.9\%.
```

The case J paths give 53.7%, which checks the simulator against the algebra.

### 13.5 Markov-modulated Poisson

Crash rates also depend on the regime. We weighted each of the 10 crash crossings by the smoothed probability that its month was turbulent (9.25 crossings in turbulence) and divided by the expected years spent in each state:

```math
\lambda_{turb} = \frac{\sum_{\text{crashes}} \gamma_t(\text{turb})}{\sum_t \gamma_t(\text{turb})/12} = 0.556, \qquad \lambda_{calm} = 0.009 \text{ a year}.
```

A crash is about 60 times as likely in a turbulent year as in a calm one. We used this Markov-modulated Poisson process (`mmpp_p_at_least_one`) for the clustering version of the crash probabilities and the 2027–32 damage.

### 13.6 Conditional shortfall

We grouped the case J paths by the number of months of 2031–32 they spent turbulent and measured shortfall within each group. The groups recombine by the law of total probability, which `verify.py` checks:

```math
\Pr(\text{shortfall}) = \sum_b \Pr(b)\,\Pr(\text{shortfall}\mid b).
```

| Turbulent months in 2031–32 | Share of paths | LDCS shortfall | Core-satellite | Bucket ladder | Risk parity |
|---|---|---|---|---|---|
| 0 | 46.3% | 4.4% | 5.5% | 3.9% | 2.3% |
| 1–6 | 27.6% | 5.4% | 12.4% | 6.3% | 5.6% |
| 7–12 | 15.5% | 8.8% | 26.2% | 11.4% | 11.6% |
| 13–24 | 10.5% | 13.8% | 50.1% | 22.7% | 26.5% |

The cushion rule cuts equity as turbulence erodes the surplus, so LDCS's shortfall grows slowly with turbulence, while fixed-mix strategies rebalance back into a falling market. In calm paths risk parity does better. Because turbulence drives case J returns by construction, this table shows how each rule responds to a bad regime; it is not separate evidence that regimes exist.

---

## 14. How the layers connect

| Layer | Question it answers | Feeds |
|---|---|---|
| Liability maths (Section 3) | How much money does the operating commitment need? | The floor of LDCS, CPPI and S0; the $R_4$ hurdle |
| Strategy rules (Section 4) | How much equity does each strategy hold, given the floor? | The simulation |
| Monte Carlo cases H, S, C (Section 5) | How do the strategies fare across ordinary, poor and higher-yield markets? | Criteria 1–3 and promise reliability |
| Markov regimes (Section 13) | When are crashes and volatility more likely? | Case J; conditional shortfall |
| Jump-diffusion, case J (Section 5.5) | How do the strategies fare with crashes and rate shocks history did not contain? | Criteria 1, 2, 4 and 6 |
| Crash replays (Section 6) | What does each historical crash do at the worst moment? | Criteria 5 and 6; the $q_j$ of the joint Poisson model |
| Poisson rates (Section 11) | How likely is each crash type in 2031–32? | The $p_j$ of the joint Poisson model |
| Scoreboard and tests (Sections 7–10) | Which strategy is best, and how sure are we? | The choice of LDCS |

---

## 15. Verification

`src/verify.py` recomputes the main results by separate code paths and compares them with the stored results within stated tolerances. The latest run passes 1,044 of 1,044 checks.

| Group | What we check | Checks |
|---|---|---|
| A | Case arithmetic against the Competition Guide and the holdings totals | 28 |
| B | Crash counting from the raw index, λ, window probabilities, Garwood limits by root-finding | 22 |
| C | Markov matrix, stationary shares, Markov-modulated probabilities | 12 |
| D | Fund-level crash counts and rates | 50 |
| E | Compound Poisson: simulated mean and variance against $\lambda t E[L]$ and $\lambda t E[L^2]$ | 27 |
| F, J | Crash-type probabilities and their link to the replay results | 96 |
| G | Liability value, duration, convexity and PV01 | 42 |
| H | Independent re-simulation of S1 and S4 with different code and seed | 16 |
| I | Invariants of the Monte Carlo output (shares between 0 and 1, ordered percentiles) | 204 |
| K | Every table on disk equals the JSON it comes from | 48 |
| L | Regime model against the crash dates | 1 |
| M | S0 engine, scoreboard recomputation and the published ten-strategy Table 8.21, weights, seeds, paired tests, joint Poisson, Poisson tests, Markov | 284 |
| N | Every cell of the crash-impact table (Section 11.8) recomputed from the replays; the yield condition switches the lock-in off and on correctly; every IPS version keeps LDCS first | 214 |

To reproduce everything from the raw data: `cd src && python run_all.py && python verify.py`.

---

## 16. Assumptions and limits

1. **The 2031 yield condition.** We model the two-year-yield test path by path, but the engine simulates the 5-year yield only, so we estimate the 2-year yield from it with today's 0.08-point spread. A flatter or steeper curve in 2031 would change which paths lock in.
2. **A fixed 4% hurdle.** We compare $V_{2033}$ with $R_4 = \$421{,}767$, as the IPS defines the reserve, and do not reprice the payments at 2033 market yields. If yields are below 4% in 2033 the payments cost more, so the falling-rate replay (T5) looks safer than it would at market value. The bond sleeve's 4.06-year duration is shorter than the liability's 9.8 years in 2027, so the portfolio carries this rate risk before 2033.
3. **Sampling noise only.** The seed test resamples the same 2004–2026 history with the same model; cases S, C and J and the λ intervals are the model-risk checks.
4. **Judgements.** The six criteria and the three weightings are team choices; random weights test the weights, not the criteria.
5. **Ranks and significance.** Ranks discard size, and 10,000 paths make tiny differences significant.
6. **Worst-timing replays and independence.** The joint Poisson figures are upper bounds; the union form assumes crash types occur independently, and the splitting theorem assumes a crash's type is independent of its timing.
7. **Small samples.** Ten crashes, nine gaps, five rate shocks, 27 turbulent months in 2004–2026 (behind the turbulent-regime covariance) and 3 jump months in 2004–2026 (behind the jump betas) give wide intervals and weak tests.
8. **Proxies.** We rebuilt fund returns before December 2016 from indices and yields (Section 2). In the 1973–74 replay TIPS did not exist, and our CPI-plus-1% stand-in gains 21% over the episode; real TIPS would also have lost value as real yields rose, so that replay flatters every strategy that holds SCHP, LDCS included.
9. **The payout years.** The simulation ends at the January 2033 valuation; we argue the 2033–2042 payout phase rather than simulate it.
10. **One bucket design.** S0 refills only after up years; other bucket rules could rank differently.

---

## 17. References

Black, F., & Perold, A. F. (1992). Theory of constant proportion portfolio insurance. *Journal of Economic Dynamics and Control, 16*(3–4), 403–426.

Cochran, W. G. (1954). Some methods for strengthening the common χ² tests. *Biometrics, 10*(4), 417–451.

Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance, 1*(2), 223–236.

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological), 39*(1), 1–38.

Estrada, J. (2019). *The bucket approach for retirement: A suboptimal behavioral trick?* IESE Business School.

Fabozzi, F. J. (2016). *Bond markets, analysis, and strategies* (9th ed.). Pearson.

Federal Reserve Bank of St. Louis. (2026). *Federal Reserve Economic Data (FRED)* [Data set]. Retrieved November 20, 2026.

French, K. R. (2026). *Kenneth R. French data library* [Data set]. Tuck School of Business, Dartmouth College.

Garwood, F. (1936). Fiducial limits for the Poisson distribution. *Biometrika, 28*(3/4), 437–442.

Glasserman, P. (2003). *Monte Carlo methods in financial engineering*. Springer.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica, 57*(2), 357–384.

Jorion, P. (2007). *Value at risk: The new benchmark for managing financial risk* (3rd ed.). McGraw-Hill.

Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2012). *Loss models: From data to decisions* (4th ed.). Wiley.

Kotz, S., Balakrishnan, N., & Johnson, N. L. (2000). *Continuous multivariate distributions: Vol. 1. Models and applications* (2nd ed.). Wiley.

Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics, 17*(3), 1217–1241.

Leibowitz, M. L., & Weinberger, A. (1982). Contingent immunization—Part I: Risk control procedures. *Financial Analysts Journal, 38*(6), 17–31.

Lilliefors, H. W. (1969). On the Kolmogorov–Smirnov test for the exponential distribution with mean unknown. *Journal of the American Statistical Association, 64*(325), 387–389.

Macaulay, F. R. (1938). *Some theoretical problems suggested by the movements of interest rates, bond yields and stock prices in the United States since 1856*. National Bureau of Economic Research.

Maillard, S., Roncalli, T., & Teïletche, J. (2010). The properties of equally weighted risk contribution portfolios. *The Journal of Portfolio Management, 36*(4), 60–70.

Massey, F. J., Jr. (1951). The Kolmogorov–Smirnov test for goodness of fit. *Journal of the American Statistical Association, 46*(253), 68–78.

McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157.

Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics, 3*(1–2), 125–144.

Redington, F. M. (1952). Review of the principles of life-office valuations. *Journal of the Institute of Actuaries, 78*(3), 286–340.

Ross, S. M. (2014). *Introduction to probability models* (11th ed.). Academic Press.

Shiller, R. J. (2026). *U.S. stock markets 1871–present and CAPE ratio* [Data set]. Yale University.

Wald, A. (1950). *Statistical decision functions*. Wiley.
