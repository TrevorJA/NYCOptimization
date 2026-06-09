"""
bench_ensemble_eval.py - Wall-clock benchmark for ensemble simulation eval.

Times one or more invocations of ``run_simulation_ensemble_inmemory()`` for the
ensemble preset selected by ``NYCOPT_ENSEMBLE_PRESET`` (resolved at config
import time as ``SEARCH_ENSEMBLE_SPEC``). Reports cold-cache and warm-cache
per-eval wall time, and writes a CSV row per eval to
``outputs/diagnostics/ensemble_bench/{preset}_{slug}_{timestamp}.csv`` for
later regression tracking.

Use this from a SLURM batch job (see ``slurm/bench_ensemble.sh``); do not run
on a login node — a single full-window N=5 eval is in the 10–20 minute range.

Usage:
    python scripts/supplemental/bench_ensemble_eval.py
    python scripts/supplemental/bench_ensemble_eval.py --n-evals 3
    python scripts/supplemental/bench_ensemble_eval.py --formulation ffmp --n-evals 1

The selected preset is read from ``config.SEARCH_ENSEMBLE_SPEC`` at import.
Override via ``NYCOPT_ENSEMBLE_PRESET`` in the calling environment (the
SLURM wrapper sources an env file).
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402

from config import (  # noqa: E402
    OUTPUT_DIAGNOSTICS_DIR,
    SEARCH_ENSEMBLE_SPEC,
    derive_slug,
)
from src.formulations import get_baseline_values  # noqa: E402
from src.simulation import (  # noqa: E402
    _ensemble_window,
    dvs_to_config,
    run_simulation_ensemble_inmemory,
)


def _check_data_per_real(data_per_real, expected_n: int) -> None:
    """Sanity-check the ensemble result before recording timings."""
    assert len(data_per_real) == expected_n, (
        f"expected {expected_n} realizations, got {len(data_per_real)}"
    )
    required = {
        "res_storage", "major_flow", "ibt_demands",
        "ibt_diversions", "mrf_target", "flood_stage",
    }
    for i, d in enumerate(data_per_real):
        missing = required - set(d.keys())
        assert not missing, f"realization {i} missing keys: {missing}"


def _bench_one_eval(nyc_config, spec) -> float:
    t0 = time.time()
    data_per_real = run_simulation_ensemble_inmemory(nyc_config, spec)
    elapsed = time.time() - t0
    _check_data_per_real(data_per_real, spec.n_realizations)
    return elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--formulation", default="ffmp",
        help="Formulation name (default: ffmp).",
    )
    parser.add_argument(
        "--n-evals", type=int, default=2,
        help="Number of consecutive evaluations to time (default: 2; first is "
             "cold-cache, subsequent are warm-cache).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    spec = SEARCH_ENSEMBLE_SPEC
    if not spec.is_ensemble:
        logging.error(
            f"SEARCH_ENSEMBLE_SPEC.preset_name={spec.preset_name!r} is not an "
            f"ensemble. Set NYCOPT_ENSEMBLE_PRESET to an ensemble preset (e.g., "
            f"wcu_kirsch_n5) before running this benchmark."
        )
        return 2

    bench_start, bench_end = _ensemble_window(spec)
    logging.info(
        f"[bench] preset={spec.preset_name} N={spec.n_realizations} "
        f"window={bench_start}..{bench_end} "
        f"realization_years={spec.realization_years or 'full'} "
        f"formulation={args.formulation} n_evals={args.n_evals}"
    )

    dv = np.array(get_baseline_values(args.formulation))
    nyc_config = dvs_to_config(dv, formulation_name=args.formulation)

    timings: list[float] = []
    for i in range(args.n_evals):
        kind = "cold-cache" if i == 0 else f"warm-cache (#{i+1})"
        logging.info(f"[bench] eval {i+1}/{args.n_evals} ({kind}) running ...")
        elapsed = _bench_one_eval(nyc_config, spec)
        timings.append(elapsed)
        logging.info(f"[bench] eval {i+1}: {elapsed:.1f}s ({kind})")

    # Persist results to a per-run CSV under outputs/diagnostics/ensemble_bench/
    out_dir = Path(OUTPUT_DIAGNOSTICS_DIR) / "ensemble_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = derive_slug(args.formulation)
    out_csv = out_dir / f"{spec.preset_name}_{slug}_{ts}.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "preset", "formulation", "n_realizations", "realization_years",
            "sim_start", "sim_end", "eval_index", "kind", "wall_seconds",
        ])
        for i, t in enumerate(timings):
            writer.writerow([
                spec.preset_name, args.formulation, spec.n_realizations,
                spec.realization_years or "full",
                bench_start, bench_end, i,
                "cold" if i == 0 else "warm",
                f"{t:.3f}",
            ])
    logging.info(f"[bench] wrote {out_csv}")

    # Concise summary at exit.
    cold = timings[0]
    warm_avg = sum(timings[1:]) / max(len(timings) - 1, 1) if len(timings) > 1 else float("nan")
    logging.info(
        f"[bench] summary: cold={cold:.1f}s "
        f"warm_avg={warm_avg:.1f}s (n_warm={len(timings) - 1})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
