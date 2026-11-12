"""demand_utilization_calibration.py - Tabulate ``u(k)`` for the NYC utilization axis.

Generates the uncapped historic-flow NYC demand extrapolation once (Pywr-DRB's
``ExtrapolatedDiversionPreprocessor``, deterministic seeded draws), tabulates
the nominal utilization ``u(k) = mean(cap_800(k * D_ref)) / 800`` of every
scale ``k`` on the dense grid of ``src.demand_utilization.CALIBRATION_K_GRID``,
and persists, under ``outputs/reference/demand_utilization/`` (gitignored,
regenerable):

    demand_utilization_calibration.csv     columns k, u (the table ``k_from_u`` inverts)
    demand_reference_uncapped_mgd.csv      columns datetime, aggregate (the reference series)

Then renders the SI figure ``demand_utilization_calibration`` of
``src/figures/registry.py`` (``u(k)`` and the implied monthly climatology at
``u`` in {0.45, 0.7, 0.9, 1.0}). Seconds on a login node. No settings: every
constant lives in ``src/demand_utilization.py``.

Usage:
    python scripts/supplemental/demand_utilization_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src import demand_utilization as du  # noqa: E402

#: Scales whose tabulated utilization is printed as a sanity line.
REFERENCE_SCALES: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 2.0, 3.0, du.DEMAND_SCALE_MAX)


def write_calibration() -> Path:
    """Build and persist the table and the reference series; return the table path."""
    du.DEMAND_UTILIZATION_DIR.mkdir(parents=True, exist_ok=True)
    table, reference = du.build_calibration_table()
    if float(table["k"].iloc[-1]) != float(du.DEMAND_SCALE_MAX):
        raise RuntimeError(
            f"u(k) saturates at k = {table['k'].iloc[-1]} on the reference record, below "
            f"DEMAND_SCALE_MAX = {du.DEMAND_SCALE_MAX}; the documented clip point no longer "
            f"holds. Lower DEMAND_SCALE_MAX in src/demand_utilization.py."
        )
    table.to_csv(du.CALIBRATION_CSV, index=False)
    reference.rename_axis("datetime").to_frame().to_csv(du.REFERENCE_SERIES_CSV)
    print(f"[demand-calibration] reference {reference.index.min().date()} to "
          f"{reference.index.max().date()}, uncapped mean {reference.mean():.1f} MGD")
    us = du.u_from_k(np.asarray(REFERENCE_SCALES), table)
    print("[demand-calibration] u(k): "
          + ", ".join(f"{k:g} -> {u:.3f}" for k, u in zip(REFERENCE_SCALES, us)))
    k_cap, clipped = du.k_from_u(np.asarray([du.DEMAND_UTILIZATION_BOUNDS[0], 0.7, 0.9, 1.0]), table)
    print(f"[demand-calibration] k(u): 0.45 -> {k_cap[0]:.3f}, 0.7 -> {k_cap[1]:.3f}, "
          f"0.9 -> {k_cap[2]:.3f}, 1.0 -> {k_cap[3]:.3f} (clipped={bool(clipped[3])})")
    print(f"[demand-calibration] wrote {du.CALIBRATION_CSV} "
          f"(sha256 {du.calibration_sha256()[:12]}...) and {du.REFERENCE_SERIES_CSV.name}")
    return du.CALIBRATION_CSV


def render_figure() -> None:
    """Render the registered SI calibration figure from the persisted artifacts."""
    from src.figures.registry import FigureContext, by_name
    from src.plotting import style

    spec = by_name("demand_utilization_calibration")
    ctx = FigureContext()
    style.apply_style()
    out_stub = spec.out_dir() / spec.stem
    spec.builder(ctx, out_stub, ctx.tables(spec.kind))
    print(f"[demand-calibration] figure -> {out_stub}.png")


def main() -> None:
    write_calibration()
    render_figure()


if __name__ == "__main__":
    main()
