"""
scripts/demo_policy_plots.py - Produce one operational-policy plot per architecture.

Verifies the new per-policy plotting modules under `src/plotting/` end-to-end.
Spline uses real parameters from the smoke-test refset when available;
RBF / Tree / ANN use uniform-random feasible parameters drawn from
`policy.get_bounds()` (no refsets exist for those yet).

Usage:
    python scripts/demo_policy_plots.py

Outputs:
    outputs/figures/policy_demo/{spline,tree,rbf,ann}_demo.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.formulations.external import get_architecture  # noqa: E402
from src.plotting.style import apply_style                # noqa: E402
from src.plotting.spline_policy_plot import plot_spline_policy  # noqa: E402
from src.plotting.tree_policy_plot import plot_tree_policy      # noqa: E402
from src.plotting.rbf_policy_plot import plot_rbf_policy        # noqa: E402
from src.plotting.ann_policy_plot import plot_ann_policy        # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "policy_demo"


def _smoke_refset_path(arch: str) -> Path:
    """Path to the seed-1 refset from the smoke test for `arch`, if it exists."""
    return (
        PROJECT_ROOT / "outputs" / "optimization" / f"smoke_{arch}"
        / "sets" / f"seed_01_smoke_{arch}.set"
    )


def _load_first_theta(refset_path: Path, n_params: int) -> np.ndarray:
    """Load the first Pareto row from a Borg .set file; return its DV slice."""
    rows = np.loadtxt(refset_path, comments="#")
    if rows.ndim == 1:
        rows = rows[None, :]
    return rows[0, :n_params].copy()


def _random_theta(policy, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lb, ub = policy.get_bounds()
    return rng.uniform(lb, ub)


def main() -> int:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    arch_plot = (
        ("spline", plot_spline_policy, "Spline Additive Policy"),
        ("tree",   plot_tree_policy,   "Soft Oblique Tree Policy"),
        ("rbf",    plot_rbf_policy,    "RBF Policy"),
        ("ann",    plot_ann_policy,    "ANN Policy"),
    )

    for arch, plot_fn, label in arch_plot:
        policy = get_architecture(arch)
        refset = _smoke_refset_path(arch)
        if refset.exists():
            theta = _load_first_theta(refset, policy.n_params)
            source = f"refset row 0  ({refset.name})"
        else:
            theta = _random_theta(policy, seed=42)
            source = "random feasible (seed=42)"
        policy.set_params(theta)
        print(f"[{arch}] n_inputs={policy.n_inputs}  n_params={policy.n_params}  source={source}")
        fig = plot_fn(policy, title=f"{label} — params: {source}")
        out = OUT_DIR / f"{arch}_demo.png"
        fig.savefig(out)
        print(f"  saved {out}")

    print(f"\nAll plots saved under {OUT_DIR.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
