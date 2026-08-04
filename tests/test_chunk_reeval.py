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
        "downstream_flood_exceedance_annual",
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
    out = chunk_reeval.simulate_test_chunks(
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


###############################################################################
# Persistence-layout / scheduling / merge equivalence (all output-invariant)
###############################################################################

_TEST_OBJECTIVES = [
    "downstream_flood_exceedance_annual",
    "montague_flow_deficit_p99_pct",
    "nyc_delivery_reliability_annual",
]


def _setup_campaign(tmp_path, monkeypatch):
    """Fake 2-chunk master + resolved objective set + analytic ground truth."""
    staged = tmp_path / "synthetic_ensembles"
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", staged)
    master = _stage_fake_master(staged)
    master_spec = ensembles.get_ensemble_spec(master)
    monkeypatch.setattr(config, "REEVAL_ENSEMBLE_SPEC", master_spec)
    monkeypatch.setattr(config, "ACTIVE_OBJECTIVES", list(_TEST_OBJECTIVES))
    monkeypatch.setattr(reeval_core, "_REEVAL_CACHE", None)

    obj_set, _, is_ens = reeval_core.resolve_reeval()
    assert is_ens
    base_names = [o.base.name for o in obj_set]
    dvs = np.array([[0.10], [0.90]], dtype=float)

    def truth(dv, gid):
        return np.array([gid + 100.0 * k + 1000.0 * float(dv[0])
                         for k in range(len(base_names))])

    return staged, dvs, base_names, truth


def _install_stub(monkeypatch, staged, base_names, truth, *,
                  failing=frozenset(), calls=None):
    """Stub evaluate_raw; optionally raise on chosen (sol_idx, chunk_idx) units."""
    def _fake(dv_vector, *, formulation_name, objective_set,
              ensemble_spec, realization_batch=None):
        sol = 0 if abs(float(dv_vector[0]) - 0.10) < 1e-9 else 1
        j = int(ensemble_spec.inflow_type.rsplit("chunk", 1)[1])
        if calls is not None:
            calls.append((sol, j))
        if (sol, j) in failing:
            raise RuntimeError("transient failure (test)")
        meta = json.loads(
            (staged / ensemble_spec.inflow_type / "_meta.json").read_text())
        gids = meta["global_realization_ids"]
        return np.vstack([truth(dv_vector, g) for g in gids]), base_names

    monkeypatch.setattr("src.simulation.evaluate_raw", _fake)


def _set_knobs(monkeypatch, *, incremental=1, schedule="claim", merge="job",
               retry_failed=0, allow_partial=0, job_id="testjob"):
    monkeypatch.setattr(config, "CHUNK_INCREMENTAL", incremental)
    monkeypatch.setattr(config, "CHUNK_SCHEDULE", schedule)
    monkeypatch.setattr(config, "CHUNK_MERGE", merge)
    monkeypatch.setattr(config, "CHUNK_RETRY_FAILED", retry_failed)
    monkeypatch.setattr(config, "CHUNK_MERGE_ALLOW_PARTIAL", allow_partial)
    monkeypatch.setenv("SLURM_JOB_ID", job_id)  # scopes the claim dir per "job"


def _read_long(reeval_dir: Path) -> pd.DataFrame:
    raw_pq = reeval_dir / "reeval_raw.parquet"
    df = pd.read_parquet(raw_pq) if raw_pq.exists() else pd.read_csv(
        reeval_dir / "reeval_raw.csv.gz")
    return df.sort_values(["solution_id", "realization_id", "objective"]
                          ).reset_index(drop=True)


def _run(reeval_dir: Path, dvs) -> None:
    reeval_dir.mkdir(exist_ok=True)
    chunk_reeval.simulate_test_chunks("ffmp", dvs, solution_ids=[0, 1],
                                      seed=0, reeval_dir=reeval_dir)


def test_incremental_matches_legacy(tmp_path, monkeypatch):
    """Per-unit persistence must reproduce the legacy one-shot long table."""
    staged, dvs, base_names, truth = _setup_campaign(tmp_path, monkeypatch)
    _install_stub(monkeypatch, staged, base_names, truth)

    _set_knobs(monkeypatch, incremental=0)
    _run(tmp_path / "legacy", dvs)
    _set_knobs(monkeypatch, incremental=1, schedule="claim")
    _run(tmp_path / "incremental", dvs)

    pd.testing.assert_frame_equal(
        _read_long(tmp_path / "legacy"), _read_long(tmp_path / "incremental"))


def test_schedule_invariance(tmp_path, monkeypatch):
    """Assignment order provably cannot change the merged output."""
    staged, dvs, base_names, truth = _setup_campaign(tmp_path, monkeypatch)
    _install_stub(monkeypatch, staged, base_names, truth)

    frames = {}
    for schedule in ("contiguous", "interleave", "claim"):
        _set_knobs(monkeypatch, schedule=schedule, job_id=f"job_{schedule}")
        _run(tmp_path / schedule, dvs)
        frames[schedule] = _read_long(tmp_path / schedule)

    pd.testing.assert_frame_equal(frames["contiguous"], frames["interleave"])
    pd.testing.assert_frame_equal(frames["contiguous"], frames["claim"])


def test_resume_matches_oneshot(tmp_path, monkeypatch):
    """A transient-failure run + retry resume equals a clean one-shot run."""
    staged, dvs, base_names, truth = _setup_campaign(tmp_path, monkeypatch)

    # Reference: clean one-shot.
    _install_stub(monkeypatch, staged, base_names, truth)
    _set_knobs(monkeypatch)
    _run(tmp_path / "oneshot", dvs)

    # Run A: two units fail transiently; merge deferred.
    failing = {(0, 1), (1, 0)}
    _install_stub(monkeypatch, staged, base_names, truth, failing=failing)
    _set_knobs(monkeypatch, merge="off", job_id="job_a")
    resumed = tmp_path / "resumed"
    _run(resumed, dvs)
    units_dir = resumed / "partial" / "units"
    assert (units_dir / "chunk001" / "sol00000.failed").exists()
    assert (units_dir / "chunk000" / "sol00001.failed").exists()

    # Run B: stub healthy again; resume re-attempts ONLY the failed units.
    calls: list = []
    _install_stub(monkeypatch, staged, base_names, truth, calls=calls)
    _set_knobs(monkeypatch, retry_failed=1, job_id="job_b")
    _run(resumed, dvs)
    assert sorted(calls) == sorted(failing)  # skip-idempotency
    assert not (units_dir / "chunk001" / "sol00000.failed").exists()

    pd.testing.assert_frame_equal(
        _read_long(tmp_path / "oneshot"), _read_long(resumed))


def test_standalone_merge_matches_injob(tmp_path, monkeypatch):
    """merge=off + merge_test_chunks == in-job merge; partial-merge semantics."""
    staged, dvs, base_names, truth = _setup_campaign(tmp_path, monkeypatch)
    _install_stub(monkeypatch, staged, base_names, truth)

    _set_knobs(monkeypatch)
    _run(tmp_path / "injob", dvs)

    _set_knobs(monkeypatch, merge="off", job_id="job_off")
    deferred = tmp_path / "deferred"
    _run(deferred, dvs)
    assert not (deferred / "reeval_raw_meta.json").exists()  # merge really deferred
    chunk_reeval.merge_test_chunks("ffmp", [0, 1], seed=0, reeval_dir=deferred)
    pd.testing.assert_frame_equal(
        _read_long(tmp_path / "injob"), _read_long(deferred))

    # Partial merge: a deleted unit refuses by default, NaN-fills when allowed.
    victim = deferred / "partial" / "units" / "chunk001" / "sol00001.parquet"
    victim.unlink()
    with pytest.raises(FileNotFoundError, match="unit\\(s\\) missing"):
        chunk_reeval.merge_test_chunks("ffmp", [0, 1], seed=0, reeval_dir=deferred)
    monkeypatch.setattr(config, "CHUNK_MERGE_ALLOW_PARTIAL", 1)
    chunk_reeval.merge_test_chunks("ffmp", [0, 1], seed=0, reeval_dir=deferred)
    long_df = _read_long(deferred)
    gone = long_df[(long_df.solution_id == 1)
                   & (long_df.realization_id >= CHUNK)]["value"]
    assert gone.isna().all()
    kept = long_df[(long_df.solution_id == 1)
                   & (long_df.realization_id < CHUNK)]["value"]
    assert kept.notna().all()
