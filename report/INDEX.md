# Index of figures and tables

Every chart in `figures/`, the table behind it in `results/tables/`, and the file the numbers come from. Regenerate everything with `python src/run_all.py`.

| # | Figure | Shows | Table | Source data |
|---|---|---|---|---|
| 01 | [`01_poisson_pmf_windows.png`](../figures/01_poisson_pmf_windows.png) | Poisson probability of 0, 1, 2, 3+ crashes in three windows | `poisson_by_window.csv` | `crash_events.json` |
| 02 | [`02_fund_level_lambda.png`](../figures/02_fund_level_lambda.png) | Each fund's own crash rate and dollars held | `fund_level_poisson.csv` | `laura_poisson.json` |
| 03 | [`03_crash_damage_exceedance.png`](../figures/03_crash_damage_exceedance.png) | Chance crashes cost Laura at least $X (compound Poisson) | `compound_poisson_damage.csv` | `laura_poisson.json` |
| 04 | [`04_dollar_change_by_fund_and_type.png`](../figures/04_dollar_change_by_fund_and_type.png) | Dollar change in each holding, one crash of each type | `dollar_loss_by_fund_and_type.csv` | `laura_poisson.json` |
| 05 | [`05_markov_modulated_horizon.png`](../figures/05_markov_modulated_horizon.png) | Plain Poisson against Markov-modulated Poisson | `markov_modulated_curves.csv` | `crash_events.json` |
| 06 | [`06_probability_crash_affects_laura.png`](../figures/06_probability_crash_affects_laura.png) | Probability each crash type affects Laura in 2031–32 | `probability_crash_affects_laura.csv` | `laura_poisson.json` |
| 07 | [`07_drawdown_history_by_type.png`](../figures/07_drawdown_history_by_type.png) | The ten falls of 20%+ in 1926–2026, by type | `crash_episodes.csv` | `crash_events.json` |
| 08 | [`08_poisson_goodness_of_fit.png`](../figures/08_poisson_goodness_of_fit.png) | Waiting times and yearly counts against Poisson/exponential | `gaps_between_crashes.csv` | `crash_events.json` |
| 09 | [`09_fat_tails_normal_vs_observed.png`](../figures/09_fat_tails_normal_vs_observed.png) | Observed monthly falls against a normal model | (normal check in SENSITIVITY.md) | `crash_events.json` |
| 10 | [`10_markov_regime_probability.png`](../figures/10_markov_regime_probability.png) | Probability of the turbulent state through time | `markov_states.csv` | `regime_probability.csv` |
| 11 | [`11_markov_transition_matrix.png`](../figures/11_markov_transition_matrix.png) | Switching probabilities, one month and twelve months | `markov_states.csv` | `crash_events.json` |
| 12 | [`12_mc_funding_certainty.png`](../figures/12_mc_funding_certainty.png) | Funding certainty by strategy and return case | `strategy_summary.csv` | `strategy_results.json` |
| 13 | [`13_mc_contribution_range.png`](../figures/13_mc_contribution_range.png) | Facility contribution range by strategy (H and S) | `strategy_summary.csv` | `strategy_results.json` |
| 14 | [`14_mc_crash_replay_certainty.png`](../figures/14_mc_crash_replay_certainty.png) | Funding certainty when each crash type is replayed at the worst timing | `crash_replay_summary.csv` | `strategy_results.json` |
| 15 | [`15_model_interconnections.png`](../figures/15_model_interconnections.png) | How the Poisson, Markov and Monte Carlo layers connect | (diagram) | (none) |
| 16 | [`16_crash_type_promise_by_strategy.png`](../figures/16_crash_type_promise_by_strategy.png) | Promise kept under each crash type, LDCS against six alternatives, and the joint Poisson risk | `crash_type_promise.csv` | `robustness.json` |
| 17 | [`17_contribution_ogive.png`](../figures/17_contribution_ogive.png) | Ogive (cumulative distribution) of the 2033 contribution, nine strategies, cases H and J | (simulated paths) | strategy_sim.py |
| 18 | [`18_rank_robustness.png`](../figures/18_rank_robustness.png) | How often each strategy ranks 1st or top 3 under 20,000 random criterion weightings | `weight_robustness.csv` | `robustness.json` |
| 19 | [`19_markov_conditional_shortfall.png`](../figures/19_markov_conditional_shortfall.png) | Shortfall and broken promise by months spent in the turbulent regime in 2031-32 | `markov_conditional_shortfall.csv` | `robustness.json` |

## All tables in `results/tables/`

| Table | Rows | Columns |
|---|---|---|
| [`compound_poisson_damage.csv`](../results/tables/compound_poisson_damage.csv) | 4 | window, lam_t, expected, expected_formula, sd, p_ge_50k … |
| [`crash_episodes.csv`](../results/tables/crash_episodes.csv) | 10 | peak, trough, recovered, depth, type, d10y … |
| [`crash_impact_by_strategy.csv`](../results/tables/crash_impact_by_strategy.csv) | 54 | type, name, lam, p_2031_32, strategy, strategy_name … |
| [`crash_replay_summary.csv`](../results/tables/crash_replay_summary.csv) | 120 | crash, timing, strategy, funding_certainty_4pct, contribution_p50 |
| [`crash_type_promise.csv`](../results/tables/crash_type_promise.csv) | 6 | type, name, events, years, lambda, lambda_lo90 … |
| [`dollar_loss_by_fund_and_type.csv`](../results/tables/dollar_loss_by_fund_and_type.csv) | 8 | fund, T1 Recession bear, T2 Exogenous V-shock, T3 Inflation / rate shock, T4 Valuation bust |
| [`fund_level_poisson.csv`](../results/tables/fund_level_poisson.csv) | 8 | fund, dollars, weight, max_drawdown, n10, lam10 … |
| [`gaps_between_crashes.csv`](../results/tables/gaps_between_crashes.csv) | 9 | gap_years |
| [`joint_poisson_shortfall.csv`](../results/tables/joint_poisson_shortfall.csv) | 9 | strategy, name, shortfall_sum, shortfall_union, shortfall_sum_lo, shortfall_sum_hi … |
| [`markov_conditional_shortfall.csv`](../results/tables/markov_conditional_shortfall.csv) | 36 | turbulent_months_2031_32, share_of_paths, strategy, name, p_shortfall, p_promise_broken … |
| [`markov_modulated_curves.csv`](../results/tables/markov_modulated_curves.csv) | 16 | years, plain_poisson, from today's state, from the long-run mix, just after turbulence starts |
| [`markov_states.csv`](../results/tables/markov_states.csv) | 2 | state, stationary, expected_months, lam, to_calm, to_turbulent |
| [`paired_tests.csv`](../results/tables/paired_tests.csv) | 32 | case, rival, rival_name, f4_ldcs, f4_rival, se_f4_ldcs … |
| [`poisson_by_type.csv`](../results/tables/poisson_by_type.csv) | 6 | type, n, lam, lam_lo90, lam_hi90, mean_wait_years … |
| [`poisson_by_window.csv`](../results/tables/poisson_by_window.csv) | 8 | threshold, window, lam, p_at_least_one, lo90, hi90 |
| [`probability_crash_affects_laura.csv`](../results/tables/probability_crash_affects_laura.csv) | 6 | type, scenario, lam, p_occurs_2031_32, S9_p_shortfall, S9_p_promise_broken … |
| [`scoreboard_ranked.csv`](../results/tables/scoreboard_ranked.csv) | 9 | strategy, name, Minimum funding certainty, Average funding certainty, Median contribution, Bad-case contribution (P5, J) … |
| [`seed_stability.csv`](../results/tables/seed_stability.csv) | 9 | strategy, name, equal: share of seeds ranked 1st, equal: best-worst place, safety: share of seeds ranked 1st, safety: best-worst place … |
| [`strategy_summary.csv`](../results/tables/strategy_summary.csv) | 40 | case, strategy, funding_certainty_4pct, funding_certainty_3pct, funding_certainty_ladder, contribution_p5 … |
| [`verification_checks.csv`](../results/tables/verification_checks.csv) | 1044 | group, check, computed, reference, tolerance, ok … |
| [`weight_robustness.csv`](../results/tables/weight_robustness.csv) | 9 | strategy, name, p_rank_1, p_top_3 |

## Documents

| Document | What it is |
|---|---|
| [REPORT.md](REPORT.md) | The final report: every figure with its data, how each number is calculated |
| [INTERCONNECTIONS.md](INTERCONNECTIONS.md) | How the Poisson, Markov and Monte Carlo layers connect, and how this repository relates to the two earlier ones |
| [DATA.md](DATA.md) | Every input and output file, with sources, row counts and proxy validation |
| [SENSITIVITY.md](SENSITIVITY.md) | How λ changes with the crash threshold and the length of history |
| [MATHEMATICS.md](MATHEMATICS.md) | Every formula, case and statistical tool derived from first principles, with worked examples from the results and APA references |
| [ROBUSTNESS.md](ROBUSTNESS.md) | Nine-strategy scoreboard with one-line criterion definitions, random-weight and seed-stability tests, paired tests, joint Poisson shortfall, Poisson assumption tests, Markov conditional shortfall, which crash hurts Laura most under each strategy, LDCS under the IPS tolerance band |
| [../docs/Poisson_Crash_Model_Final_Report.pdf](../docs/Poisson_Crash_Model_Final_Report.pdf) | The same material as one printable PDF |