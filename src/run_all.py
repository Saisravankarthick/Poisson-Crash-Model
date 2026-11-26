"""Run the whole pipeline in order:  python src/run_all.py   (strategy_sim takes about 10 minutes)."""
import build_data, crash_events, laura_poisson, figures, figures_more, runpy
build_data.main()
crash_events.main()
runpy.run_module("strategy_sim", run_name="__main__")
laura_poisson.main()
import robustness
robustness.main()                                          # step 4: ranking robustness (about 3 minutes)
figures.main()
try:
    runpy.run_module("verify", run_name="__main__")      # independent checks; raises SystemExit
except SystemExit as e:
    if e.code:
        raise
figures_more.main()                                        # charts 07-15 and the CSV tables (after verify so verification_checks.csv exists)
runpy.run_module("make_report", run_name="__main__")     # writes report/*.md from the results
