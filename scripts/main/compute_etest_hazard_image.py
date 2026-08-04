"""compute_etest_hazard_image.py - E_test hazard coordinates on disjoint 10-yr sub-windows.

Computes the hazard image of the staged E_test so it can be overlaid on the candidate
pool and the realized search ensembles (``scripts/main/plot_etest_hazard_overlay.py``).
E_test realizations are ``L_test`` (50) years long while the pool convention scores
``SCENARIO_YEARS`` (10) year scenarios, so each realization is split into
``L_test // SCENARIO_YEARS`` DISJOINT 10-yr sub-windows and every sub-window is scored
exactly as a pool scenario would be: SSI-6 controlling-event run-theory dry axes on the
window's monthly aggregate NYC inflow (the leading 6 months excluded implicitly by the
SSI accumulation spin-up) and POT wet axes on its daily series with the leading
``METRIC_EXCLUSION_MONTHS`` cut by date. The SSI fit, POT threshold, and reference mean
stay fitted once on the full historical record — identical to the pool's convention —
so E_test sub-window coordinates are commensurable with pool coordinates.

Writes ``hazard_image_subwindows.npz`` into the staged E_test directory with row keys
``(realization_id, window_index)`` plus ``theta_index = realization_id // R_test``.
Chunks are processed independently and cached as shard files, so an interrupted run
resumes where it stopped (delete the shards to force recomputation).

Run after E_test is staged (workflow step 12 / ``generate_test_ensemble.py``)::

    python scripts/main/compute_etest_hazard_image.py

Configuration is via environment variables (no CLI value flags):

    NYCOPT_ETEST_VARIANT             E_test variant whose staged slug is scored (default "kn")
    NYCOPT_ETEST_HAZARD_SHARD_INDEX  When set: score ONLY this chunk index, write its
                                     shard file, and exit (one SLURM array task per
                                     chunk; workflow/supplemental/etest_hazard_image_shards.sh).
                                     Unset: the original serial loop over all chunks
                                     (the reference path).
    NYCOPT_ETEST_HAZARD_MERGE        "1": verify every shard file exists, then merge and
                                     write the final artifact without recomputing anything
                                     (workflow/supplemental/etest_hazard_image_merge.sh).

The merge is a pure function of the shard contents: rows are lexsorted by
(realization_id, window_index) and every key is unique, so shard-then-merge is
byte-identical to the serial loop regardless of completion order.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import METRIC_EXCLUSION_MONTHS, SCENARIO_YEARS  # noqa: E402
from src.ensembles import pool_chunk_specs, staged_ensemble_dir  # noqa: E402
from src.etest import E_TEST_VARIANT, assert_staged_etest_contract, get_etest_variant  # noqa: E402
from src.load.historical_flows import load_historical_flows  # noqa: E402

#: Realizations loaded per HDF5 read, bounding peak memory to ~batch x L_test daily frames.
_READ_BATCH = 100


def _reference_series(flowtype: str) -> tuple[np.ndarray, np.ndarray]:
    """Historical aggregate-NYC-inflow reference for the SSI/POT fits (pool convention).

    Args:
        flowtype: pywrdrb inflow-dataset key recorded in the staged ``_meta.json``.

    Returns:
        ``(reference_monthly, reference_daily)`` as float arrays.
    """
    from scengen.hazard_filling import daily_to_monthly
    from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES

    ref = load_historical_flows(gage=False, period="full", flowtype=flowtype)
    ref_daily = ref.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    return (
        np.asarray(daily_to_monthly(ref_daily, agg="mean"), dtype=float),
        ref_daily.to_numpy(dtype=float),
    )


def _window_bounds(index: pd.DatetimeIndex, n_windows: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Disjoint ``SCENARIO_YEARS``-year sub-window date bounds over a realization's index.

    Args:
        index: A realization's daily DatetimeIndex (October-aligned start).
        n_windows: Number of disjoint sub-windows (``L_test // SCENARIO_YEARS``).

    Returns:
        ``[(start, end), ...]`` half-open ``[start, end)`` bounds.
    """
    starts = [index[0] + pd.DateOffset(years=SCENARIO_YEARS * k) for k in range(n_windows)]
    ends = starts[1:] + [index[0] + pd.DateOffset(years=SCENARIO_YEARS * n_windows)]
    return list(zip(starts, ends))


def _score_chunk(
    chunk_path: Path,
    local_ids: list[int],
    global_ids: list[int],
    n_windows: int,
    reference_monthly: np.ndarray,
    reference_daily: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Score one staged chunk's realizations on all sub-windows.

    Rows are grouped by window index (sub-window k shares one calendar span across
    realizations, so its daily arrays have equal length; spans differ across k by
    leap days), then re-sorted to (realization, window) order.

    Args:
        chunk_path: Staged chunk directory holding ``catchment_inflow_mgd.hdf5``.
        local_ids: Chunk-local realization keys ``0..S-1``.
        global_ids: Global realization ids aligned with ``local_ids``.
        n_windows: Disjoint sub-windows per realization.
        reference_monthly: Historical monthly reference (SSI fit).
        reference_daily: Historical daily reference (POT threshold + mean).

    Returns:
        ``(H, realization_ids, window_index, hazard_axes)`` with one row per
        (realization, window), ordered by realization then window.
    """
    from synhydro import Ensemble

    from scengen.hazard_filling import daily_to_monthly
    from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES, compute_candidate_hazard_image

    rows_by_key: dict[tuple[int, int], np.ndarray] = {}
    axes: list[str] = []
    t_read = t_score = 0.0
    for b0 in range(0, len(local_ids), _READ_BATCH):
        batch_local = local_ids[b0:b0 + _READ_BATCH]
        batch_global = global_ids[b0:b0 + _READ_BATCH]
        _t = time.perf_counter()
        ens = Ensemble.from_hdf5(
            str(chunk_path / "catchment_inflow_mgd.hdf5"), realization_subset=batch_local
        )
        t_read += time.perf_counter() - _t
        _t = time.perf_counter()
        # from_hdf5 re-keys the subset 0..len-1 in `batch_local` order.
        agg = {
            g: ens.data_by_realization[i].loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
            for i, g in enumerate(batch_global)
        }
        idx = pd.DatetimeIndex(agg[batch_global[0]].index)
        for k, (w0, w1) in enumerate(_window_bounds(idx, n_windows)):
            in_win = (idx >= w0) & (idx < w1)
            cutoff = w0 + pd.DateOffset(months=METRIC_EXCLUSION_MONTHS)
            wet_cut = int(((idx >= w0) & (idx < cutoff)).sum())
            daily_rows, monthly_rows = [], []
            for g in batch_global:
                w_daily = agg[g].loc[in_win]
                daily_rows.append(w_daily.to_numpy(dtype=float))
                monthly_rows.append(daily_to_monthly(w_daily, agg="mean"))
            H_win, axes = compute_candidate_hazard_image(
                np.vstack(monthly_rows), np.vstack(daily_rows),
                reference_monthly, reference_daily, wet_exclusion_days=wet_cut,
            )
            for r, g in enumerate(batch_global):
                rows_by_key[(g, k)] = H_win[r]
        t_score += time.perf_counter() - _t

    print(f"[etest-hazard]   {chunk_path.name}: read {t_read:.1f}s, "
          f"score {t_score:.1f}s", flush=True)
    keys = sorted(rows_by_key)
    H = np.vstack([rows_by_key[key] for key in keys])
    rid = np.asarray([key[0] for key in keys], dtype=int)
    win = np.asarray([key[1] for key in keys], dtype=int)
    return H, rid, win, list(axes)


def _merge_shards(shard_paths: list[Path], out_path: Path, R: int) -> None:
    """Merge per-chunk shard files into the canonical sub-window image.

    A pure function of the shard contents: rows are lexsorted by
    ``(realization_id, window_index)`` and every key is unique with disjoint
    rid ranges per chunk, so the output is independent of shard completion
    order and of row order within a shard. Shards are unlinked after the write.
    """
    parts = [np.load(p, allow_pickle=True) for p in shard_paths]
    H = np.vstack([p["H"] for p in parts])
    rid = np.concatenate([p["realization_ids"] for p in parts])
    win = np.concatenate([p["window_index"] for p in parts])
    axes = [str(a) for a in parts[0]["hazard_axes"]]
    order = np.lexsort((win, rid))
    H, rid, win = H[order], rid[order], win[order]
    np.savez(
        out_path,
        H=H, hazard_axes=np.asarray(axes, dtype=object),
        realization_ids=rid, window_index=win, theta_index=rid // R,
        window_years=np.asarray(SCENARIO_YEARS), exclusion_months=np.asarray(METRIC_EXCLUSION_MONTHS),
    )
    for p in shard_paths:
        p.unlink()
    print(f"[etest-hazard] wrote {out_path} ({H.shape[0]} rows x {H.shape[1]} axes); "
          f"removed {len(shard_paths)} shards.")


def main() -> None:
    """Score the staged E_test on disjoint sub-windows and persist the hazard image."""
    variant = get_etest_variant(E_TEST_VARIANT)
    slug = variant.slug
    meta = assert_staged_etest_contract(slug)
    out_dir = staged_ensemble_dir(slug)
    out_path = out_dir / "hazard_image_subwindows.npz"
    if out_path.exists():
        print(f"[etest-hazard] already computed: {out_path}. Delete it to recompute.")
        return

    L = int(meta.get("realization_years") or meta.get("n_years"))
    if L % SCENARIO_YEARS != 0:
        raise ValueError(
            f"L_test={L} is not a multiple of the pool window SCENARIO_YEARS="
            f"{SCENARIO_YEARS}; disjoint sub-windows would not tile the realization."
        )
    n_windows = L // SCENARIO_YEARS
    R = int(meta["realizations_per_profile"])
    flowtype = meta.get("flowtype", "pub_nhmv10_BC_withObsScaled")
    reference_monthly, reference_daily = _reference_series(flowtype)

    chunks = pool_chunk_specs(slug)
    print(f"[etest-hazard] '{slug}': {len(chunks)} chunk(s), {n_windows} disjoint "
          f"{SCENARIO_YEARS}-yr sub-windows per realization.")

    shard_env = os.environ.get("NYCOPT_ETEST_HAZARD_SHARD_INDEX", "")
    shard_index = int(shard_env) if shard_env != "" else None
    merge_only = os.environ.get("NYCOPT_ETEST_HAZARD_MERGE", "0") == "1"
    if shard_index is not None and not (0 <= shard_index < len(chunks)):
        raise ValueError(
            f"NYCOPT_ETEST_HAZARD_SHARD_INDEX={shard_index} out of range for "
            f"{len(chunks)} chunks."
        )

    shard_paths: list[Path] = []
    for i, (spec, gids) in enumerate(chunks):
        shard = out_dir / f"hazard_image_subwindows_shard_{i:03d}.npz"
        shard_paths.append(shard)
        if merge_only or (shard_index is not None and i != shard_index):
            continue
        if shard.exists():
            print(f"[etest-hazard] chunk {i}: shard already staged, skipping.")
            continue
        chunk_dir = staged_ensemble_dir(spec.inflow_type)
        H, rid, win, axes = _score_chunk(
            chunk_dir, list(range(len(gids))), [int(g) for g in gids],
            n_windows, reference_monthly, reference_daily,
        )
        np.savez(shard, H=H, realization_ids=rid, window_index=win,
                 hazard_axes=np.asarray(axes, dtype=object))
        print(f"[etest-hazard] chunk {i}: {H.shape[0]} rows -> {shard.name}")

    if shard_index is not None:
        print(f"[etest-hazard] shard mode: chunk {shard_index} done; merge "
              f"separately with NYCOPT_ETEST_HAZARD_MERGE=1.")
        return

    if merge_only:
        missing = [p.name for p in shard_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"[etest-hazard] merge: {len(missing)} shard(s) missing: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

    _merge_shards(shard_paths, out_path, R)


if __name__ == "__main__":
    main()
