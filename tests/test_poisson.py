"""Checks on the mathematics (run: python -m pytest tests -q)."""
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import poisson_core as pc


def test_pmf_sums_to_one():
    assert abs(sum(pc.pmf(k, 0.6) for k in range(40)) - 1) < 1e-12


def test_binomial_limit():
    lam = 0.0997; p_binom = 1 - (1 - lam / 12) ** 24
    assert abs(p_binom - pc.p_at_least_one(lam, 2)) < 0.002


def test_garwood_matches_known_values():
    lo, hi = pc.garwood_ci(10, 100.3)          # 10 crashes in 100.3 years
    assert abs(lo - 0.0541) < 5e-4 and abs(hi - 0.1691) < 5e-4
    lo0, hi0 = pc.garwood_ci(0, 36.25)         # zero events: upper bound = -ln(0.05)/T
    assert lo0 == 0 and abs(hi0 - (-math.log(0.05) / 36.25)) < 1e-9


def test_episodes_counts_peak_to_recovery_once():
    lvl = pd.Series([100, 90, 75, 85, 101, 120, 80, 70, 125], index=pd.date_range("2000-01-31", periods=9, freq="ME"))
    ev = pc.episodes(lvl, 0.20)
    assert len(ev) == 2 and abs(ev[1]["depth"] - (70 / 120 - 1)) < 1e-12


def test_compound_poisson_mean_formula():
    rng = np.random.default_rng(0); L = rng.uniform(10, 50, 100_000); lam_t = 0.6
    N = rng.poisson(lam_t, 200_000); S = np.array([rng.choice(L, k).sum() for k in N[:20_000]])
    m, v = pc.compound_moments(lam_t, L)
    assert abs(S.mean() - m) / m < 0.05


def test_dispersion_equal_for_poisson_sample():
    c = np.random.default_rng(1).poisson(0.1, 5000)
    assert 0.9 < pc.dispersion_test(c)["D"] < 1.1
