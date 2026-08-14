"""
search_outcomes.py - SI figure: per-design MOEA search convergence.

One panel per scenario design: the runtime hypervolume indicator vs runtime
snapshot, one thin line per seed x island. This is a PER-DESIGN convergence
diagnostic only -- "each search ran to convergence before its archive was
re-evaluated" -- and deliberately nothing more.

**Search-time values are never compared across designs.** Each design's
search objectives (and therefore its hypervolume) are computed under its OWN
search ensemble; the designs evaluate different measures, so cross-design
magnitudes are not commensurable and every panel here carries its own
y-scale. The one common basis of comparison is the held-out E_test
re-evaluation, where the cross-design figures live (criteria robustness,
regret, factor maps).

Data contract (per design, under ``outputs/{design}/{slug}/``): runtime
metrics ``metrics/seed_XX_{formulation}_{island}.metrics`` (MOEAFramework
MetricsEvaluator; NFE via the sibling ``runtime/`` files when present).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.plotting import style
from src.plotting.hypervolume_convergence import (_load_metrics_file,
                                                  _read_runtime_nfe)
from src.plotting.layout import panel_grid, panel_label
from src.plotting.style import DESIGN_ORDER, design_color, design_label

#: Runtime indicator preferred; the first indicator column actually present
#: in the metrics files is used when it is absent.
PREFERRED_INDICATOR = "Hypervolume"

#: Metrics filename grammar: ``seed_XX_{formulation}_{island}.metrics``.
_METRICS_STEM = re.compile(r"seed_(\d+)_(.+)_(\d+)$")


def _design_metrics(run_dir: Path) -> list[tuple[int, int, pd.DataFrame, list]]:
    """Every (seed, island, metrics frame, NFE marks) under one run dir."""
    out = []
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        return out
    for path in sorted(metrics_dir.glob("*.metrics")):
        m = _METRICS_STEM.match(path.stem)
        if not m:
            continue
        try:
            frame = _load_metrics_file(path)
        except Exception as exc:  # noqa: BLE001 - a bad file is a warning
            warnings.warn(f"unreadable metrics file {path}: {exc}")
            continue
        nfe = _read_runtime_nfe(path, len(frame))
        out.append((int(m.group(1)), int(m.group(3)), frame, nfe))
    return out


def fig_search_outcomes(ctx, out_stub: Path, table_dir: Path) -> list[Path]:
    """Per-design convergence panels (one y-scale each; never compared).

    Args:
        ctx: Figure context; only ``ctx.slug`` is consumed (no cubes needed).
        out_stub: Output path without extension.
        table_dir: Directory for the companion CSV
            (``search_convergence.csv``: design, seed, island, final NFE,
            final indicator per metrics file).

    Returns:
        The written figure paths.

    Raises:
        FileNotFoundError: If no design has metrics files (search outputs are
            Anvil-side). A design without metrics is tolerated with a warning.
    """
    per_design = {}
    for design in DESIGN_ORDER:
        rows = _design_metrics(config.OUTPUTS_DIR / design / ctx.slug)
        if rows:
            per_design[design] = rows
        else:
            warnings.warn(f"[search_outcomes] no metrics for '{design}' under "
                          f"outputs/{design}/{ctx.slug}/metrics -- skipped.")
    if not per_design:
        raise FileNotFoundError(
            f"no runtime metrics found for any design under "
            f"outputs/*/{ctx.slug}/metrics (Anvil-side artifacts)."
        )

    designs = list(per_design)
    fig, axes = panel_grid(1, len(designs), panel_aspect=0.85)
    axes = np.atleast_1d(axes)

    csv_rows = []
    for i, (ax, design) in enumerate(zip(axes, designs)):
        color = design_color(design)
        indicator = None
        for seed, island, frame, nfe in per_design[design]:
            if indicator is None:
                indicator = (PREFERRED_INDICATOR
                             if PREFERRED_INDICATOR in frame.columns
                             else frame.columns[0])
            y = frame[indicator].to_numpy(dtype=float)
            x = (np.asarray(nfe, dtype=float)[:len(y)]
                 if nfe is not None and len(nfe) >= len(y)
                 else np.arange(len(y)))
            ax.plot(x, y, color=color, lw=0.9, alpha=0.8)
            csv_rows.append({
                "design": design, "seed": seed, "island": island,
                "indicator": indicator,
                "final_nfe": float(x[-1]) if len(x) else np.nan,
                "final_value": float(y[-1]) if len(y) else np.nan,
            })
        ax.set_title(design_label(design), fontsize=9)
        panel_label(ax, chr(ord("a") + i))
        ax.set_xlabel("NFE" if any(n is not None and len(n)
                                   for *_, n in per_design[design])
                      else "Runtime snapshot")
        if i == 0:
            ax.set_ylabel(f"{indicator} (per-design scale;\n"
                          f"not comparable across panels)")
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)

    written = style.save_manuscript_figure(fig, out_stub)
    import matplotlib.pyplot as plt
    plt.close(fig)
    pd.DataFrame(csv_rows).to_csv(table_dir / "search_convergence.csv",
                                  index=False)
    return written
