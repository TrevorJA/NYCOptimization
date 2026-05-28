"""
scripts/demo_ffmp_policy_plots.py - Demo FFMP policy visualizations.

Generates example figures for:
  1. Standard FFMP - baseline (default 2017 FFMP)
  2. Standard FFMP - storage-priority archetype
  3. Standard FFMP - flow-priority archetype
  4. Standard FFMP - 30 random solutions (Pareto overlay)
  5-8. VR-FFMP N=8 versions of (1)-(4)

The two archetypes are constructed by pushing each parameter to opposite
ends of its bounds depending on whether that parameter favors reservoir
storage or downstream flow. They illustrate the operational range the
parameterization can express within the same FFMP rule structure.

Usage:
    python scripts/demo_ffmp_policy_plots.py
"""

import fnmatch
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.formulations import get_baseline_values, get_bounds, get_var_names
from src.plotting.ffmp_policy_plot import plot_ffmp_policy

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DPI = 150


# ---------------------------------------------------------------------------
# Archetype rules (param-name pattern → t-fraction in [0, 1] of bounds)
# ---------------------------------------------------------------------------
#
# Patterns are matched in order; first match wins. Default 0.5 (midpoint).
# t = 0 → lower bound, t = 1 → upper bound.
#
# STORAGE_PRIORITY: aggressive drought triggers + minimal downstream flows.
#   - zone_shift_*  → +0.10 (curves up; drought levels fire at higher storage)
#   - mrf_*         → near minimum  (low minimum releases)
#   - drought factors → near minimum (strong delivery cuts when drought)
#   - flood_max_*   → near minimum (conservative flood releases)
#   - mrf_profile_scale_* → low (~0.6) (compress seasonal min-flow profile)
#
# FLOW_PRIORITY: late drought triggers + maximum downstream flows.
#   - zone_shift_*  → −0.10 (curves down; drought levels fire late)
#   - mrf_*         → near maximum
#   - drought factors → near maximum (≈1.0; minimal cuts)
#   - flood_max_*   → near maximum (large flood releases allowed)
#   - mrf_profile_scale_* → high (~1.7)

STORAGE_PRIORITY_RULES = [
    ("mrf_profile_scale_*",   0.10),   # 0.5 + 0.10 * 1.5 ≈ 0.65 multiplier
    ("mrf_*",                 0.05),   # near min for the 5 MRF baselines
    ("zone_shift_*",          0.95),   # +0.08 (curves shift up)
    ("nyc_drought_factor_*",  0.05),
    ("nj_drought_factor_*",   0.05),
    ("flood_max_*",           0.10),
    ("max_nyc_delivery",      0.05),
    ("*",                     0.5),
]

FLOW_PRIORITY_RULES = [
    ("mrf_profile_scale_*",   0.85),   # 0.5 + 0.85 * 1.5 ≈ 1.78
    ("mrf_*",                 0.95),
    ("zone_shift_*",          0.05),   # −0.08 (curves shift down)
    ("nyc_drought_factor_*",  0.95),
    ("nj_drought_factor_*",   0.95),
    ("flood_max_*",           0.95),
    ("max_nyc_delivery",      0.95),
    ("*",                     0.5),
]


def _t_for(name: str, rules) -> float:
    for pat, val in rules:
        if fnmatch.fnmatch(name, pat):
            return val
    return 0.5


def _build_archetype(formulation: str, rules) -> np.ndarray:
    """Build a DV vector by interpolating each parameter at the
    rule-specified t-fraction of its bounds."""
    names = get_var_names(formulation)
    lo, hi = get_bounds(formulation)
    out = np.empty(len(names))
    for i, name in enumerate(names):
        t = _t_for(name, rules)
        out[i] = lo[i] + t * (hi[i] - lo[i])
    return out


def _random_dvs(formulation: str, n: int, seed: int = 42) -> np.ndarray:
    lo, hi = get_bounds(formulation)
    rng = np.random.default_rng(seed)
    return lo + (hi - lo) * rng.random((n, len(lo)))


# ---------------------------------------------------------------------------
# Demo figure builders
# ---------------------------------------------------------------------------

def demo_single(
    formulation: str,
    dvs: np.ndarray,
    title: str,
    outname: str,
    show_baseline: bool = False,
):
    fig = plot_ffmp_policy(
        dvs,
        formulation_name=formulation,
        show_baseline=show_baseline,
        title=title,
    )
    out = FIGURES_DIR / outname
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"    → {out}")
    return fig


def demo_pareto(formulation: str, n_sols: int, outname: str):
    print(f"  [{formulation}] {n_sols} random solutions (Pareto overlay)...")
    dvs = _random_dvs(formulation, n_sols)
    dvs[0] = get_baseline_values(formulation)

    fig = plot_ffmp_policy(
        dvs,
        formulation_name=formulation,
        highlight_idx=0,
        show_baseline=True,
        title=f"{formulation.upper()} — {n_sols} random solutions (operational range)",
    )
    out = FIGURES_DIR / outname
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"    → {out}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("FFMP POLICY VISUALIZATION DEMO")
    print("=" * 60)

    for form, n_pareto, suffix in [("ffmp", 30, ""), ("ffmp_8", 20, "_vr8")]:
        label = form.upper()
        print(f"\n[{label}]")

        baseline = get_baseline_values(form)
        storage  = _build_archetype(form, STORAGE_PRIORITY_RULES)
        flow     = _build_archetype(form, FLOW_PRIORITY_RULES)

        print(f"  baseline  → demo_ffmp{suffix}_baseline.png")
        demo_single(form, baseline,
                    f"{label} — baseline (default 2017 FFMP)",
                    f"demo_ffmp{suffix}_baseline.png")

        print(f"  storage-priority archetype  → demo_ffmp{suffix}_storage.png")
        demo_single(form, storage,
                    f"{label} — storage-priority archetype",
                    f"demo_ffmp{suffix}_storage.png")

        print(f"  flow-priority archetype  → demo_ffmp{suffix}_flow.png")
        demo_single(form, flow,
                    f"{label} — flow-priority archetype",
                    f"demo_ffmp{suffix}_flow.png")

        demo_pareto(form, n_pareto, f"demo_ffmp{suffix}_pareto.png")

    print("\nDone. Figures in:", FIGURES_DIR)
