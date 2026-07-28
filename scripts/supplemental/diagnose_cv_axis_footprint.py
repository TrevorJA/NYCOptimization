"""diagnose_cv_axis_footprint.py - Hazard-space footprint of the CV forcing axis.

Supplemental (SI) diagnostic for the ``ENSEMBLE_FORCING_VARIANCE_AXIS`` decision:
does adding the independent CMIP6-derived CV axis (``c = a * v``) materially widen the
drought/flood tail stress the forced ensembles present, beyond what the 3-D mean box
``[m, r1, r2]`` already spans?

Design: two PAIRED generations over the E_test forcing box (full CMIP6 range widened by
``E_TEST_MARGIN``), sharing one root seed:

    OFF  variance_axis=False  ->  c = a  (CV-preserving; the v = 1 slice)
    ON   variance_axis=True   ->  c = a * v, v drawn from the CMIP6 CV envelope

The mean-axis draws AND the per-realization bootstrap streams are identical between the
two runs (the CV draw uses its own namespaced seed stream), so realization k differs
only through its CV profile v_k: per-axis paired hazard deltas are attributable to the
CV axis alone. Each theta gets one realization (R = 1) at the search-side L = 10 yr.

Hazard descriptors: the full 8-axis candidate image (SSI-6 controlling-event run theory
dry + POT wet) on aggregate NYC inflow, shared metric window (first 6 months excluded).

Configuration (no CLI value flags):

    NYCOPT_CVDIAG_N_THETA   forcing profiles per run   (default 48)
    NYCOPT_CVDIAG_YEARS     realization length, years  (default 10)
    NYCOPT_CVDIAG_SEED      root seed                  (default 20260728)

Outputs -> ``outputs/supplemental/cv_axis_footprint/``:
    hazard_images.npz      H_off, H_on (N x 8), axis names, theta matrices
    footprint_summary.csv  per-axis footprint stats (both runs) + widening ratios
    paired_deltas.csv      per-realization per-axis deltas (ON - OFF)
    cv_axis_footprint.png  per-axis normalized range/tail comparison
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from scengen.forcing_ensemble import ForcingEnsembleConfig
from scengen.hazard_filling import daily_to_monthly
from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES
from synhydro import Ensemble

from src.ensemble_generation import (
    _disaggregate_fill_inflow,
    _generate_profile_monthly,
    _hazard_block,
    _prepare_generators,
)
from src.etest import E_TEST_BOUND_PCT, E_TEST_MARGIN
from src.load.historical_flows import load_historical_flows

N_THETA = int(os.environ.get("NYCOPT_CVDIAG_N_THETA", "48"))
YEARS = int(os.environ.get("NYCOPT_CVDIAG_YEARS", "10"))
SEED = int(os.environ.get("NYCOPT_CVDIAG_SEED", "20260728"))
BLOCK = 16  # profiles per generation block (bounds peak memory)

OUT_DIR = config.OUTPUTS_DIR / "supplemental" / "cv_axis_footprint"


def _diag_config(*, variance_axis: bool) -> ForcingEnsembleConfig:
    """One run's generation config: E_test forcing box, R = 1, shared root seed."""
    return ForcingEnsembleConfig(
        root_seed=SEED,
        n_forcing_profiles=N_THETA,
        realizations_per_profile=1,
        realization_years=YEARS,
        population="du_forced",
        theta_sampler="lhs",
        mean_frac_csv=config.ENSEMBLE_FORCING_MEAN_FRAC_CSV,
        variance_axis=variance_axis,
        mean_abs_csv=config.ENSEMBLE_FORCING_MEAN_ABS_CSV,
        std_csv=config.ENSEMBLE_FORCING_STD_CSV,
        bound_pct=E_TEST_BOUND_PCT,
        margin=E_TEST_MARGIN,
        output_dir=OUT_DIR,
    )


def _hazard_image(cfg: ForcingEnsembleConfig) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Generate the run's realizations and return (H, hazard_axes, theta, theta_names)."""
    setup = _prepare_generators(cfg)

    ref = load_historical_flows(gage=False, period="full", flowtype=cfg.flowtype)
    ref_daily_series = ref.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    reference_monthly = daily_to_monthly(ref_daily_series, agg="mean")
    reference_daily = ref_daily_series.to_numpy(dtype=float)

    H_blocks: list[np.ndarray] = []
    axes: list[str] = []
    for b0 in range(0, cfg.n_forcing_profiles, BLOCK):
        monthly: dict[int, pd.DataFrame] = {}
        metadata = None
        for p in range(b0, min(b0 + BLOCK, cfg.n_forcing_profiles)):
            frames, md = _generate_profile_monthly(setup, cfg, p)
            monthly.update(frames)
            if metadata is None:
                metadata = md
        _, inflow, _ = _disaggregate_fill_inflow(
            Ensemble(monthly, metadata=metadata),
            nowak=setup.nowak, kdes=setup.kdes,
            root_seed=cfg.root_seed, start_date=cfg.start_date,
        )
        H_block, axes = _hazard_block(
            inflow, sorted(inflow), DEFAULT_NYC_INFLOW_NODES,
            reference_monthly, reference_daily,
        )
        H_blocks.append(H_block)
        print(f"[cvdiag]   profiles [{b0}, {min(b0 + BLOCK, cfg.n_forcing_profiles)}) done")
    return np.vstack(H_blocks), axes, setup.theta_params, setup.theta_names


def _footprint_stats(H: np.ndarray, axes: list[str], label: str) -> pd.DataFrame:
    """Per-axis footprint statistics of one run's hazard image."""
    q = np.nanpercentile(H, [5, 50, 95], axis=0)
    return pd.DataFrame({
        "axis": axes,
        "run": label,
        "min": np.nanmin(H, axis=0),
        "p5": q[0],
        "median": q[1],
        "p95": q[2],
        "max": np.nanmax(H, axis=0),
        "span_p5_p95": q[2] - q[0],
    })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[cvdiag] OFF run (c = a): N_theta={N_THETA}, L={YEARS} yr, seed={SEED}")
    H_off, axes, theta_off, names_off = _hazard_image(_diag_config(variance_axis=False))
    print(f"[cvdiag] ON run (c = a*v): same mean draws + bootstrap streams")
    H_on, axes_on, theta_on, names_on = _hazard_image(_diag_config(variance_axis=True))
    assert axes == axes_on
    # Paired-design invariant: the mean-axis draws must be identical between runs.
    np.testing.assert_allclose(theta_off, theta_on[:, : theta_off.shape[1]])

    np.savez(
        OUT_DIR / "hazard_images.npz",
        H_off=H_off, H_on=H_on, hazard_axes=np.array(axes),
        theta_off=theta_off, theta_on=theta_on,
        theta_names_off=np.array(names_off), theta_names_on=np.array(names_on),
        n_theta=N_THETA, years=YEARS, seed=SEED,
        bound_pct=np.array(E_TEST_BOUND_PCT), margin=E_TEST_MARGIN,
    )

    stats = pd.concat([
        _footprint_stats(H_off, axes, "off"),
        _footprint_stats(H_on, axes, "on"),
    ])
    wide = stats.pivot(index="axis", columns="run", values=["p95", "max", "span_p5_p95"])
    summary = pd.DataFrame({
        "axis": wide.index,
        "p95_off": wide[("p95", "off")],
        "p95_on": wide[("p95", "on")],
        "max_off": wide[("max", "off")],
        "max_on": wide[("max", "on")],
        "span_off": wide[("span_p5_p95", "off")],
        "span_on": wide[("span_p5_p95", "on")],
    }).reset_index(drop=True)
    summary["span_ratio_on_off"] = summary["span_on"] / summary["span_off"]
    summary["p95_shift_in_off_spans"] = (
        (summary["p95_on"] - summary["p95_off"]) / summary["span_off"]
    )
    summary.to_csv(OUT_DIR / "footprint_summary.csv", index=False)

    deltas = pd.DataFrame(H_on - H_off, columns=axes)
    deltas.insert(0, "realization", np.arange(N_THETA))
    deltas.to_csv(OUT_DIR / "paired_deltas.csv", index=False)

    # Figure: per-axis footprint (min-max whisker + p5-p95 bar), normalized by OFF span.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, name in enumerate(axes):
        off_col, on_col = H_off[:, i], H_on[:, i]
        lo, span = np.nanpercentile(off_col, 5), summary.loc[i, "span_off"]
        for dx, col, color, label in ((-0.16, off_col, "0.45", "off (c=a)"),
                                      (0.16, on_col, "#c1272d", "on (c=a*v)")):
            z = (col - lo) / span
            ax.plot([i + dx] * 2, [np.nanmin(z), np.nanmax(z)], color=color, lw=1.0,
                    label=label if i == 0 else None)
            ax.plot([i + dx] * 2, np.nanpercentile(z, [5, 95]), color=color, lw=4.0)
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels([a.replace("_", "\n") for a in axes], fontsize=8)
    ax.set_ylabel("normalized to OFF p5-p95 span")
    ax.set_title(
        f"CV-axis hazard footprint, paired runs (N={N_THETA}, L={YEARS} yr, E_test box)"
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cv_axis_footprint.png", dpi=200)

    findings = {
        "n_theta": N_THETA, "years": YEARS, "seed": SEED,
        "bound_pct": list(E_TEST_BOUND_PCT), "margin": E_TEST_MARGIN,
        "median_span_ratio": float(summary["span_ratio_on_off"].median()),
        "max_span_ratio": float(summary["span_ratio_on_off"].max()),
        "per_axis_span_ratio": dict(zip(summary["axis"], summary["span_ratio_on_off"])),
        "per_axis_p95_shift_in_off_spans": dict(
            zip(summary["axis"], summary["p95_shift_in_off_spans"])
        ),
    }
    (OUT_DIR / "findings.json").write_text(json.dumps(findings, indent=2))
    print(summary.round(3).to_string(index=False))
    print(f"[cvdiag] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
