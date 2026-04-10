"""
figures/make_all_figures.py - Regenerate all manuscript figures.

Runs each figure script in sequence.  Set SKIP_MISSING=True to skip figures
whose data prerequisites are not yet available.

Usage (from project root):
    python figures/make_all_figures.py
    python figures/make_all_figures.py --skip-missing
"""

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

FIGURE_SCRIPTS = [
    "fig01_system_map.py",
    "fig02_architecture_schematic.py",
    "fig03_pareto_comparison.py",
    "fig04_resolution_curve.py",
    "fig05_robustness_degradation.py",
    "fig06_vulnerability_maps.py",
    "figSI_lhs_diagnostics.py",
    "figSI_convergence.py",
]

FIGURES_DIR = Path(__file__).parent


def run_figure(script_name: str, skip_missing: bool) -> bool:
    """Import and run make_figure() from the given script. Returns True on success."""
    script_path = FIGURES_DIR / script_name
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.make_figure()
        return True
    except NotImplementedError:
        print(f"  [SKIP] {script_name}: not yet implemented")
        return False
    except FileNotFoundError as exc:
        if skip_missing:
            print(f"  [SKIP] {script_name}: missing data — {exc}")
            return False
        raise
    except Exception:
        print(f"  [FAIL] {script_name}:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate all manuscript figures.")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip figures whose data files are not yet available")
    args = parser.parse_args()

    results = {}
    for script in FIGURE_SCRIPTS:
        print(f"\n--- {script} ---")
        results[script] = run_figure(script, args.skip_missing)

    n_ok = sum(results.values())
    print(f"\n{'='*40}")
    print(f"Done: {n_ok}/{len(FIGURE_SCRIPTS)} figures succeeded.")
