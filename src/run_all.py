"""Run the whole pipeline in order:  python src/run_all.py   (strategy_sim takes about 10 minutes)."""
import build_data, crash_events, laura_poisson, figures, runpy
build_data.main()
crash_events.main()
runpy.run_module("strategy_sim", run_name="__main__")
laura_poisson.main()
figures.main()
