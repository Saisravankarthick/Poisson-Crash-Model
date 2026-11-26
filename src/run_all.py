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
    runpy.run_module("verify", run_name="__main__")      # first pass writes verification.json (needed by figures_more)
except SystemExit:
    pass                                                   # table-on-disk checks can fail here: figures_more rewrites those tables next
figures_more.main()                                        # charts 07-15 and the CSV tables
try:
    runpy.run_module("verify", run_name="__main__")      # final pass on the finished tables; stops the run if any check fails
except SystemExit as e:
    if e.code:
        raise
runpy.run_module("final_report_figures", run_name="__main__")  # figures quoted in the Final Report
runpy.run_module("make_report", run_name="__main__")     # writes report/*.md from the results
