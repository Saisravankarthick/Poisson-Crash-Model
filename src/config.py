"""Shared settings: Laura Gao's cash flows (Client Case Study only) and the team's final holdings at cost basis."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, RES, FIG = ROOT / "data/raw", ROOT / "data/processed", ROOT / "results", ROOT / "figures"
for p in (PROC, RES, FIG):
    p.mkdir(parents=True, exist_ok=True)

# ---- Case Study cash flows ----
INITIAL = 300_000          # start of 2027
TOP_UP = 150_000           # start of 2028
PAYMENT, N_PAYMENTS = 50_000, 10   # start of each year 2033-2042 (annuity-due)
IPS_RATE, STRESS_RATE = 0.04, 0.03

# ---- Final portfolio at cost basis, forced to $300,000 with a cash row (team decision) ----
HOLDINGS = {"VTI": 64_962.10, "VXUS": 39_786.00, "EQAL": 14_303.35,
            "IEF": 35_736.00, "SCHP": 27_698.00, "SGOV": 30_159.00, "SHY": 40_590.00, "VGIT": 40_446.00}
CASH = INITIAL - sum(HOLDINGS.values())            # 6,319.55, held like SGOV in the models
FUNDS = list(HOLDINGS)
EQUITY = ["VTI", "VXUS", "EQAL"]
BONDS = ["IEF", "SCHP", "SGOV", "SHY", "VGIT"]
DURATION = {"VTI": 0, "VXUS": 0, "EQAL": 0, "IEF": 7.4, "SCHP": 6.6, "SGOV": 0.1, "SHY": 1.9, "VGIT": 5.1}

# ---- Crash definitions agreed with the team ----
THRESHOLDS = (0.20, 0.30)  # peak-to-trough falls, measured on month-end closes
WINDOWS = {"2031-32 (critical years)": 2, "2027-32 (build-up)": 6, "2027-42 (whole plan)": 16, "2027 (first year)": 1}

# ---- Muted categorical palette (validated: adjacent CVD and normal-vision separation pass) ----
PALETTE = ["#3b6ea8", "#c8733f", "#8a64b0", "#2e9474"]
INK, MUTED_INK, GRID = "#222222", "#5a5a5a", "#e4e4e4"
SEED = 42
