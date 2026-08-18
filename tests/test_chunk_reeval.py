"""tests/test_chunk_reeval.py - Chunk-and-aggregate re-evaluation correctness (no pywrdrb sim).

The machinery in ``src.chunk_reeval`` is the chunking bookkeeping: distribute (solution, chunk)
units, pool each chunk's realizations into per-SOW annual-unit objective rows keyed by GLOBAL SOW
id, reassemble per-solution ``(n_sow, M)`` matrices, and persist via the re-eval path. This test
verifies that bookkeeping deterministically by stubbing ``src.simulation.evaluate_annual_units``
(so no Pywr-DRB run is needed): the stub returns a stage-(i) ``(R, M, U)`` unit tensor that is a
pure function of (global realization id, objective index, unit index), so every merged per-SOW
value is EXACTLY computable by an inline re-implementation of the §2 unit operators — and a
chunked run over a fake 2-chunk master must reproduce it cell for cell, whatever the persistence
layout, scheduling, or merge placement.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# src.chunk_reeval imports the Unix-only ``resource`` module at import time (per-rank
# RSS telemetry on the HPC). Shim it on Windows so the bookkeeping under test stays
# importable; the shim only feeds the progress print.
if sys.platform == "win32" and "resource" not in sys.modules:
    _fake_resource = types.ModuleType("resource")
    _fake_resource.RUSAGE_SELF = 0
    _fake_resource.getrusage = lambda *_: types.SimpleNamespace(ru_maxrss=0)
    sys.modules["resource"] = _fake_resource

import numpy as np
import pandas as pd
import pytest

import config
from src import ensembles, reeval_core, chunk_reeval


N_M, CHUNK, L = 16, 8, 2          # 2 chunks of 8 realizations
R_PER_SOW = 2                     # -> 8 global SOWs; each chunk holds 4 WHOLE SOWs
N_SOW = N_M // R_PER_SOW
U = 3                             # stage-(i) unit-years per realization

#: Annual-unit objectives with three DISTINCT §2 unit operators, so a misplaced
#: row cannot cancel out: pooled mean, pooled P99, failure frequency (k = 3).
_TEST_OBJECTIVES = [
    "downstream_flood_exceedance_annual",   # PooledMeanOp
    "montague_flow_deficit_p99_pct",        # PooledPercentileOp(99)
    "nyc_delivery_reliability_annual",      # FailureFrequencyOp(k=3)
]
M = len(_TEST_OBJECTIVES)


def _stage_fake_master(root: Path) -> str:
    """Create a JSON-only chunked, SOW-grouped master (evaluate_annual_units is stubbed)."""
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
            "start_date": config.ENSEMBLE_START_DATE,
        }))
        chunks.append({"chunk_index": j, "slug": slug,
                       "global_start": start, "global_end": start + CHUNK,
                       "n_realizations": CHUNK})
    (root / master / "chunk_index.json").write_text(json.dumps(
        {"master_slug": master, "n_realizations": N_M, "chunk_size": CHUNK,
         "n_chunks": len(chunks), "chunks": chunks}))
    # The master meta carries the SOW grouping (DU-forced, R realizations per
    # theta) that sow_grouping() recovers; a chunk never splits a SOW.
    (root / master / "_meta.json").write_text(json.dumps(
        {"slug": master, "n_realizations": N_M, "realization_years": L,
         "source_kind": "synhydro_kn", "population": "du_forced",
         "theta_sampler": "lhs", "n_forcing_profiles": N_SOW,
         "realizations_per_profile": R_PER_SOW,
         "start_date": config.ENSEMBLE_START_DATE}))
    return master


###############################################################################
# Analytic ground truth: units as a function of (global id, objective, unit)
###############################################################################

def _truth_units(dv, gids) -> np.ndarray:
    """Stage-(i) ``(R, M, U)`` tensor, deterministic in (gid, objective, unit).

    Objective 0 feeds a pooled MEAN, 1 a pooled P99, and 2 a failure-frequency
    count (k = 3), so each per-SOW value depends on WHICH global realizations
    were pooled — any re-keying or placement error changes the answer.
    """
    sol = 1000.0 * float(dv[0])
    shift = int(float(dv[0]) > 0.5)
    out = np.empty((len(gids), M, U), dtype=float)
    for r, g in enumerate(gids):
        for u in range(U):
            out[r, 0, u] = sol + 2.0 * g + 10.0 * u
            out[r, 1, u] = sol + 3.0 * g + 7.0 * u
            out[r, 2, u] = float((int(g) + u + shift) % 5)  # failing-week count
    return out


def _expected_sow_matrix(dv) -> np.ndarray:
    """Inline re-implementation of the §2 unit operators per global SOW.

    Independent of the project's operator classes, so the equality below is a
    check of the whole chunk->SOW pipeline, not a tautology.
    """
    J = np.empty((N_SOW, M), dtype=float)
    for g in range(N_SOW):
        gids = [g * R_PER_SOW + i for i in range(R_PER_SOW)]
        units = _truth_units(dv, gids)
        J[g, 0] = units[:, 0, :].mean()                          # pooled mean
        J[g, 1] = np.percentile(units[:, 1, :].ravel(), 99.0)    # pooled P99
        J[g, 2] = float(np.mean(units[:, 2, :].ravel() < 3))     # freq, k = 3
    return J


###############################################################################
# Campaign setup + stub
###############################################################################

def _setup_campaign(tmp_path, monkeypatch):
    """Fake 2-chunk master + resolved annual-unit objective set."""
    staged = tmp_path / "synthetic_ensembles"
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", staged)
    master = _stage_fake_master(staged)
    master_spec = ensembles.get_ensemble_spec(master)
    monkeypatch.setattr(config, "REEVAL_ENSEMBLE_SPEC", master_spec)
    monkeypatch.setattr(config, "ACTIVE_OBJECTIVES", list(_TEST_OBJECTIVES))
    monkeypatch.setattr(reeval_core, "_REEVAL_CACHE", None)

    obj_set, _, is_ens = reeval_core.resolve_reeval()
    assert is_ens
    assert [o.name for o in obj_set] == _TEST_OBJECTIVES

    dvs = np.array([[0.10], [0.90]], dtype=float)  # 2 solutions (DVs unused by stub)
    return staged, dvs


def _install_stub(monkeypatch, staged, *, failing=frozenset(), calls=None):
    """Stub evaluate_annual_units; optionally raise on chosen (sol, chunk) units."""
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
        # (R, M, U) in the chunk's LOCAL row order + the ANNUAL objective names.
        return _truth_units(dv_vector, gids), [o.name for o in objective_set]

    monkeypatch.setattr("src.simulation.evaluate_annual_units", _fake)


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
    return df.sort_values(["solution_id", "sow_id", "objective"]
                          ).reset_index(drop=True)


def _run(reeval_dir: Path, dvs) -> None:
    reeval_dir.mkdir(exist_ok=True)
    chunk_reeval.simulate_test_chunks("ffmp", dvs, solution_ids=[0, 1],
                                      seed=0, reeval_dir=reeval_dir)


###############################################################################
# Merged cube == the analytic per-SOW ground truth
###############################################################################

def test_chunk_reeval_reproduces_the_per_sow_ground_truth(tmp_path, monkeypatch):
    """Every merged (solution, GLOBAL SOW, objective) cell must equal the inline
    §2-operator reference on that SOW's own global realizations — proving chunk
    rows were keyed to global SOW ids and reassembled without displacement."""
    staged, dvs = _setup_campaign(tmp_path, monkeypatch)
    _install_stub(monkeypatch, staged)
    _set_knobs(monkeypatch)

    reeval_dir = tmp_path / "out"
    out = None
    reeval_dir.mkdir()
    out = chunk_reeval.simulate_test_chunks(
        "ffmp", dvs, solution_ids=[0, 1], seed=0, reeval_dir=reeval_dir)
    assert out == reeval_dir

    long_df = _read_long(reeval_dir)
    assert list(long_df.columns) == ["solution_id", "sow_id", "objective", "value"]
    assert set(long_df["sow_id"]) == set(range(N_SOW))
    assert len(long_df) == 2 * N_SOW * M

    for sol, dv in zip((0, 1), dvs):
        expected = _expected_sow_matrix(dv)                     # (n_sow, M)
        piv = (long_df[long_df.solution_id == sol]
               .pivot_table(index="sow_id", columns="objective", values="value")
               .reindex(index=range(N_SOW), columns=_TEST_OBJECTIVES))
        assert np.allclose(piv.to_numpy(dtype=float), expected), f"sol {sol}"

    # Meta keys the offline scorer relies on, snapshotted at persist time.
    meta = json.loads((reeval_dir / "reeval_raw_meta.json").read_text())
    assert meta["obj_names"] == _TEST_OBJECTIVES
    assert meta["substrate"] == "sow_annual_unit"
    assert meta["sow_labels"] == list(range(N_SOW))
    assert meta["n_sow"] == N_SOW
    assert meta["realizations_per_sow"] == R_PER_SOW
    assert meta["n_realizations"] == N_M      # keyed to the global master, not a chunk
    assert meta["solution_ids"] == [0, 1]
    assert all(n in meta["thresholds"] for n in _TEST_OBJECTIVES)

    # Robustness ran end-to-end on the assembled per-SOW cube, and the derived
    # summary is the mean over SOWs under sowmean__ columns.
    assert (reeval_dir / "robustness_scorecard.csv").exists()
    summary = pd.read_csv(reeval_dir / "objectives_summary.csv",
                          index_col="solution_id")
    assert list(summary.columns) == [f"sowmean__{n}" for n in _TEST_OBJECTIVES]
    for sol, dv in zip((0, 1), dvs):
        assert np.allclose(summary.loc[sol].to_numpy(dtype=float),
                           _expected_sow_matrix(dv).mean(axis=0))


###############################################################################
# Persistence-layout / scheduling / merge equivalence (all output-invariant)
###############################################################################

def test_incremental_matches_legacy(tmp_path, monkeypatch):
    """Per-unit persistence must reproduce the legacy one-shot long table."""
    staged, dvs = _setup_campaign(tmp_path, monkeypatch)
    _install_stub(monkeypatch, staged)

    _set_knobs(monkeypatch, incremental=0)
    _run(tmp_path / "legacy", dvs)
    _set_knobs(monkeypatch, incremental=1, schedule="claim")
    _run(tmp_path / "incremental", dvs)

    pd.testing.assert_frame_equal(
        _read_long(tmp_path / "legacy"), _read_long(tmp_path / "incremental"))


def test_schedule_invariance(tmp_path, monkeypatch):
    """Assignment order provably cannot change the merged (n_sow, M) cube."""
    staged, dvs = _setup_campaign(tmp_path, monkeypatch)
    _install_stub(monkeypatch, staged)

    frames = {}
    for schedule in ("contiguous", "interleave", "claim"):
        _set_knobs(monkeypatch, schedule=schedule, job_id=f"job_{schedule}")
        _run(tmp_path / schedule, dvs)
        frames[schedule] = _read_long(tmp_path / schedule)

    pd.testing.assert_frame_equal(frames["contiguous"], frames["interleave"])
    pd.testing.assert_frame_equal(frames["contiguous"], frames["claim"])
    # ...and not merely self-consistent: equal to the analytic ground truth.
    ref = frames["contiguous"]
    for sol, dv in zip((0, 1), dvs):
        piv = (ref[ref.solution_id == sol]
               .pivot_table(index="sow_id", columns="objective", values="value")
               .reindex(index=range(N_SOW), columns=_TEST_OBJECTIVES))
        assert np.allclose(piv.to_numpy(dtype=float), _expected_sow_matrix(dv))


def test_resume_matches_oneshot(tmp_path, monkeypatch):
    """A transient-failure run + retry resume equals a clean one-shot run."""
    staged, dvs = _setup_campaign(tmp_path, monkeypatch)

    # Reference: clean one-shot.
    _install_stub(monkeypatch, staged)
    _set_knobs(monkeypatch)
    _run(tmp_path / "oneshot", dvs)

    # Run A: two units fail transiently; merge deferred.
    failing = {(0, 1), (1, 0)}
    _install_stub(monkeypatch, staged, failing=failing)
    _set_knobs(monkeypatch, merge="off", job_id="job_a")
    resumed = tmp_path / "resumed"
    _run(resumed, dvs)
    units_dir = resumed / "partial" / "units"
    assert (units_dir / "chunk001" / "sol00000.failed").exists()
    assert (units_dir / "chunk000" / "sol00001.failed").exists()

    # Run B: stub healthy again; resume re-attempts ONLY the failed units.
    calls: list = []
    _install_stub(monkeypatch, staged, calls=calls)
    _set_knobs(monkeypatch, retry_failed=1, job_id="job_b")
    _run(resumed, dvs)
    assert sorted(calls) == sorted(failing)  # skip-idempotency
    assert not (units_dir / "chunk001" / "sol00000.failed").exists()

    pd.testing.assert_frame_equal(
        _read_long(tmp_path / "oneshot"), _read_long(resumed))


def test_standalone_merge_matches_injob(tmp_path, monkeypatch):
    """merge=off + merge_test_chunks == in-job merge; partial-merge semantics."""
    staged, dvs = _setup_campaign(tmp_path, monkeypatch)
    _install_stub(monkeypatch, staged)

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
    # chunk 001 holds global SOWs 4..7, so exactly sol 1's rows there go NaN.
    stem = deferred / "partial" / "units" / "chunk001" / "sol00001"
    victims = [p for p in (stem.with_suffix(".parquet"), stem.with_suffix(".csv.gz"))
               if p.exists()]
    assert victims, "expected a persisted unit file to delete"
    for v in victims:
        v.unlink()
    with pytest.raises(FileNotFoundError, match="unit\\(s\\) missing"):
        chunk_reeval.merge_test_chunks("ffmp", [0, 1], seed=0, reeval_dir=deferred)
    monkeypatch.setattr(config, "CHUNK_MERGE_ALLOW_PARTIAL", 1)
    chunk_reeval.merge_test_chunks("ffmp", [0, 1], seed=0, reeval_dir=deferred)
    long_df = _read_long(deferred)
    gone = long_df[(long_df.solution_id == 1)
                   & (long_df.sow_id >= CHUNK // R_PER_SOW)]["value"]
    assert gone.isna().all()
    kept = long_df[(long_df.solution_id == 1)
                   & (long_df.sow_id < CHUNK // R_PER_SOW)]["value"]
    assert kept.notna().all()
    # The untouched solution still matches the ground truth exactly.
    piv = (long_df[long_df.solution_id == 0]
           .pivot_table(index="sow_id", columns="objective", values="value")
           .reindex(index=range(N_SOW), columns=_TEST_OBJECTIVES))
    assert np.allclose(piv.to_numpy(dtype=float), _expected_sow_matrix(dvs[0]))
