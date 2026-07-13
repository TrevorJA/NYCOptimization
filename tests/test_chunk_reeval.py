"""tests/test_chunk_reeval.py - Chunk-and-aggregate re-evaluation correctness (no pywrdrb sim).

The new machinery in ``src.chunk_reeval`` is the chunking bookkeeping: distribute (solution, chunk)
units, re-key each chunk's local rows to master-global realization ids, reassemble per-solution
matrices, and persist via the re-eval path. This test verifies that bookkeeping deterministically by
stubbing ``src.simulation.evaluate_raw`` (so no Pywr-DRB run is needed): a chunked run over a fake
2-chunk master must reproduce the same per-(solution, global-realization) metrics as a direct
unchunked assembly, and yield a valid robustness scorecard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import pytest

import config
from src import ensembles, reeval_core, chunk_reeval


N_M, CHUNK, L = 16, 8, 2  # 2 chunks of 8 realizations


def _stage_fake_master(root: Path) -> str:
    """Create a JSON-only chunked master (no HDF5 — evaluate_raw is stubbed)."""
    master = "master_2yr_n16"
    (root / master).mkdir(parents=True)
    chunks = []
    for j, start in enumerate(range(0, N_M, CHUNK)):
        gids = list(range(start, start + CHUNK))
        slug = f"{master}__chunk{j:03d}"
        (root / slug).mkdir()
        (root / slug / "_meta.json").write_text(json.dumps({
            "slug": slug, "n_realizations": CHUNK, "realization_years": L,
            "global_realization_ids": gids, "source_kind": "synhydro_kn",
        }))
        chunks.append({"chunk_index": j, "slug": slug,
                       "global_start": start, "global_end": start + CHUNK,
                       "n_realizations": CHUNK})
    (root / master / "chunk_index.json").write_text(json.dumps(
        {"master_slug": master, "n_realizations": N_M, "chunk_size": CHUNK,
         "n_chunks": len(chunks), "chunks": chunks}))
    (root / master / "_meta.json").write_text(json.dumps(
        {"slug": master, "n_realizations": N_M, "realization_years": L,
         "source_kind": "synhydro_kn"}))
    return master


def test_chunk_reeval_matches_unchunked(tmp_path, monkeypatch):
    staged = tmp_path / "synthetic_ensembles"
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", staged)
    master = _stage_fake_master(staged)

    master_spec = ensembles.get_ensemble_spec(master)
    monkeypatch.setattr(config, "REEVAL_ENSEMBLE_SPEC", master_spec)
    # The ensemble path resolves the annual-unit objective names, not the single-trace set.
    # Use objectives with DISTINCT §1 bases: the persisted re-eval matrix is keyed by
    # base name, so the mean/P99 flood variants (which share a base) cannot coexist here.
    monkeypatch.setattr(config, "ACTIVE_OBJECTIVES", [
        "downstream_flood_days_annual",
        "montague_flow_deficit_p99_pct",
        "nyc_delivery_reliability_annual",
    ])
    monkeypatch.setattr(reeval_core, "_REEVAL_CACHE", None)

    obj_set, _, is_ens = reeval_core.resolve_reeval()
    assert is_ens
    base_names = [o.base.name for o in obj_set]
    M = len(base_names)

    # Ground-truth per-(solution, global-realization) metric = gid + 100*k + 1000*sol_offset.
    dvs = np.array([[0.10], [0.90]], dtype=float)  # 2 solutions (distinct offsets; DVs unused by stub)

    def _truth(dv, gid):
        return np.array([gid + 100.0 * k + 1000.0 * float(dv[0]) for k in range(M)])

    def _fake_evaluate_raw(dv_vector, *, formulation_name, objective_set,
                           ensemble_spec, realization_batch=None):
        # Return this chunk's rows in local order, keyed by its global ids.
        meta = json.loads((staged / ensemble_spec.inflow_type / "_meta.json").read_text())
        gids = meta["global_realization_ids"]
        mat = np.vstack([_truth(dv_vector, g) for g in gids])
        return mat, base_names

    monkeypatch.setattr("src.simulation.evaluate_raw", _fake_evaluate_raw)

    reeval_dir = tmp_path / "out"
    reeval_dir.mkdir()
    out = chunk_reeval.simulate_master_chunks(
        "ffmp", dvs, solution_ids=[0, 1], seed=0, reeval_dir=reeval_dir,
    )
    assert out == reeval_dir

    # The persisted long table must carry every (solution, GLOBAL realization, objective) with the
    # exact ground-truth value — proving chunk rows were re-keyed to global ids and reassembled.
    raw_pq = reeval_dir / "reeval_raw.parquet"
    long_df = pd.read_parquet(raw_pq) if raw_pq.exists() else pd.read_csv(
        reeval_dir / "reeval_raw.csv.gz")
    assert set(long_df["realization_id"]) == set(range(N_M))
    assert len(long_df) == 2 * N_M * M
    for sol, dv in zip((0, 1), dvs):
        for gid in range(N_M):
            for k, name in enumerate(base_names):
                cell = long_df[(long_df.solution_id == sol) &
                               (long_df.realization_id == gid) &
                               (long_df.objective == name)]["value"]
                assert cell.iloc[0] == pytest.approx(_truth(dv, gid)[k])

    # Robustness ran end-to-end on the assembled cube.
    assert (reeval_dir / "reeval_raw_meta.json").exists()
    assert (reeval_dir / "objectives_summary.csv").exists()
    assert (reeval_dir / "robustness_scorecard.csv").exists()
    meta = json.loads((reeval_dir / "reeval_raw_meta.json").read_text())
    assert meta["n_realizations"] == N_M  # keyed to the global master, not a chunk
