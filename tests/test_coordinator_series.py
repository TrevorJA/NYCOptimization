"""
tests/test_coordinator_series.py - The per-seed coordinator-archive series.

Convergence diagnostics are read from the MM Borg coordinator archive, one
trace per seed, reconstructed at every snapshot as the epsilon-box merge of
the island runtime files (src/diagnostics.py, src/runtime_archive.py) with
the project's one epsilon-archive implementation. These tests pin the pure
pieces on synthetic text and, when a Java runtime and the MOEAFramework
problem JAR are available, that MetricsEvaluator scores the files the
pipeline writes.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.runtime_archive import (  # noqa: E402
    compare_archives,
    coordinator_series,
    epsilon_boxes,
    epsilon_merge,
    equal_nfe_seed_set,
    island_nfe_grid,
    read_set_epsilons,
    moea_result_header,
    nfe_markers,
    parse_result_file,
    parse_runtime_snapshots,
    snapshot_at,
    write_result_file,
)

_ISLAND_A = """//NFE=50
//ElapsedTime=1.0
//Improvements=3
1.0 2.0 0.10 0.20
1.5 2.5 0.30 0.10
#
//NFE=100
//ElapsedTime=2.0
1.0 2.0 0.05 0.20
0.5 0.5 0.20 0.05
#
"""

_ISLAND_B = """//NFE=50
//Empty
#
//NFE=100
//ElapsedTime=2.5
0.7 0.7 0.06 0.21
0.9 0.9 0.50 0.50
#
"""

_MOEA_RESULT = """# Version=5
# Problem=drb_x
# NumberOfVariables=2
# NumberOfObjectives=2
1.0 2.0 0.10 0.20
#
#
1.0 2.0 0.05 0.20
0.5 0.5 0.20 0.05
#
"""


# ---------------------------------------------------------------------------
# Parsing both layouts
# ---------------------------------------------------------------------------

def test_parse_result_file_reads_borg_runtime_entries_with_metadata():
    entries = parse_result_file(_ISLAND_B)
    assert len(entries) == 2
    meta, rows = entries[0]
    assert meta == {"NFE": "50", "Empty": ""}
    assert rows.shape == (0, 0)
    assert entries[1][0]["ElapsedTime"] == "2.5"
    assert entries[1][1].shape == (2, 4)


def test_parse_result_file_reads_moeaframework_header_and_empty_entry():
    entries = parse_result_file(_MOEA_RESULT)
    assert [rows.shape for _, rows in entries] == [(1, 4), (0, 0), (2, 4)]
    assert all(meta == {} for meta, _ in entries)


def test_parse_result_file_rejects_header_comment_inside_entry():
    with pytest.raises(ValueError, match="header comment inside"):
        parse_result_file("//NFE=1\n1 2 3 4\n# stray\n#\n")


def test_parse_runtime_snapshots_still_keys_by_nfe():
    snaps = parse_runtime_snapshots(_ISLAND_A)
    assert sorted(snaps) == [50, 100]
    assert snaps[100][1].tolist() == [0.5, 0.5, 0.20, 0.05]
    assert nfe_markers(_ISLAND_B) == [50, 100]


# ---------------------------------------------------------------------------
# Island grid agreement
# ---------------------------------------------------------------------------

def test_island_nfe_grid_requires_identical_snapshot_sequences(tmp_path):
    a = tmp_path / "seed_01_x_0.runtime"
    b = tmp_path / "seed_01_x_1.runtime"
    a.write_text(_ISLAND_A)
    b.write_text(_ISLAND_B)
    assert island_nfe_grid([a, b]) == [50, 100]
    b.write_text(_ISLAND_B + "//NFE=150\n1 1 1 1\n#\n")
    with pytest.raises(ValueError, match="grids differ"):
        island_nfe_grid([a, b])
    with pytest.raises(ValueError, match="no island runtime files"):
        island_nfe_grid([])


# ---------------------------------------------------------------------------
# The series and its file
# ---------------------------------------------------------------------------

def test_coordinator_series_merges_each_snapshot_across_islands(tmp_path):
    a = tmp_path / "seed_01_x_0.runtime"
    b = tmp_path / "seed_01_x_1.runtime"
    a.write_text(_ISLAND_A)
    b.write_text(_ISLAND_B)
    series = coordinator_series([a, b], n_vars=2, epsilons=[0.1, 0.1])
    assert [nfe for nfe, _ in series] == [50, 100]
    # NFE 50: island B is empty, island A's two rows are mutually nondominated
    # (0.30 / 0.1 is 2.999... in floating point, hence box 2).
    assert epsilon_boxes(series[0][1], 2, [0.1, 0.1]) == {(1, 2), (2, 1)}
    # NFE 100: the union of both islands reduces to two boxes.
    assert epsilon_boxes(series[1][1], 2, [0.1, 0.1]) == {(0, 2), (2, 0)}
    assert series[1][1].tolist() == epsilon_merge(
        snapshot_at([a, b], 100), 2, [0.1, 0.1]).tolist()


def test_write_result_file_round_trips_header_metadata_and_empty_entries(tmp_path):
    f = tmp_path / "seed_01_x.runtime"
    rows = np.array([[1.0, 2.0, 0.1, 0.2], [0.5, 0.5, 0.123456789012345678, 0.05]])
    header = moea_result_header("drb_x", 2, 2)
    write_result_file(f, [({"NFE": 200, "IslandNFE": 50}, np.empty((0, 0))),
                          ({"NFE": 400, "IslandNFE": 100}, rows)], header_lines=header)
    text = f.read_text()
    assert text.startswith("# Version=5\n# Problem=drb_x\n# NumberOfVariables=2\n")
    assert "# Variable.2.Definition=RealVariable(-1000000.0,1000000.0)" in header
    assert "# Objective.2.Definition=Minimize" in header
    assert nfe_markers(text) == [200, 400]
    entries = parse_result_file(text)
    assert entries[0][0] == {"NFE": "200", "IslandNFE": "50"}
    assert entries[0][1].shape == (0, 0)
    assert np.array_equal(entries[1][1], rows)   # full double precision


# ---------------------------------------------------------------------------
# Box-by-box comparison of two archives
# ---------------------------------------------------------------------------

def test_compare_archives_rounds_both_sides_to_set_precision():
    from src.runtime_archive import round_to_set_precision
    eps = [0.1, 0.1]
    full = np.array([[1.0, 2.0, 0.1 + 1e-9, 0.2]])      # a hair above the boundary
    rounded = round_to_set_precision(full)               # the .set writer's view
    assert rounded.tolist() == [[1.0, 2.0, 0.1, 0.2]]
    cmp = compare_archives(full, rounded, n_vars=2, epsilons=eps)
    assert cmp["boxes_only_a"] == [] and cmp["boxes_only_b"] == []


def test_compare_archives_reports_box_differences_and_tied_representatives():
    eps = [0.1, 0.1]
    a = np.array([[1.0, 2.0, 0.05, 0.20], [0.5, 0.5, 0.20, 0.05]])
    b = np.array([[0.7, 0.7, 0.06, 0.21], [0.5, 0.5, 0.20, 0.05]])  # same boxes
    cmp = compare_archives(a, b, n_vars=2, epsilons=eps)
    assert cmp["boxes_only_a"] == [] and cmp["boxes_only_b"] == []
    assert cmp["representatives_differ"] == 1
    c = np.array([[0.5, 0.5, 0.20, 0.05], [0.9, 0.9, 0.50, 0.50]])
    cmp = compare_archives(a, c, n_vars=2, epsilons=eps)
    assert cmp["boxes_only_a"] == [(0, 2)] and cmp["boxes_only_b"] == [(5, 5)]
    assert epsilon_boxes(np.empty((0, 0)), 2, eps) == set()


def test_python_epsilon_merge_of_island_snapshots_is_the_coordinator_archive(tmp_path):
    """The series entry at NFE k must equal the epsilon merge of the islands at k."""
    a = tmp_path / "seed_01_x_0.runtime"
    b = tmp_path / "seed_01_x_1.runtime"
    a.write_text(_ISLAND_A)
    b.write_text(_ISLAND_B)
    merged = epsilon_merge(snapshot_at([a, b], 100), n_vars=2, epsilons=[0.1, 0.1])
    assert epsilon_boxes(merged, 2, [0.1, 0.1]) == {(0, 2), (2, 0)}


# ---------------------------------------------------------------------------
# Per-seed equal-NFE set resolver
# ---------------------------------------------------------------------------

def test_equal_nfe_seed_set_picks_final_set_or_snapshot_extraction(tmp_path):
    assert equal_nfe_seed_set(tmp_path, "s", 2, 125_000, 125_000).name == "seed_02_s.set"
    assert equal_nfe_seed_set(tmp_path, "s", 1, 187_500, 125_000).name == "seed_01_s_nfe125000.set"
    with pytest.raises(ValueError, match="below the reporting NFE"):
        equal_nfe_seed_set(tmp_path, "s", 1, 100_000, 125_000)


def test_set_file_epsilon_header_round_trips_and_legacy_sets_have_none(tmp_path):
    from src.runtime_archive import write_set_file
    rows = np.array([[1.0, 2.0, 0.1, 0.2]])
    new = tmp_path / "seed_01_s.set"
    write_set_file(new, rows, ["v1", "v2"], ["o1", "o2"],
                   header_lines=["Formulation: ffmp, Seed: 1", "Epsilons: 0.05,10.0"])
    assert read_set_epsilons(new) == [0.05, 10.0]
    legacy = tmp_path / "seed_02_s.set"
    write_set_file(legacy, rows, ["v1", "v2"], ["o1", "o2"])
    assert read_set_epsilons(legacy) is None


def test_mmborg_set_writer_records_the_search_epsilons(tmp_path, monkeypatch):
    import src.mmborg as mm
    from config import get_epsilons

    class _Sol:
        def __init__(self, v, o): self.v, self.o = v, o
        def violatesConstraints(self): return False
        def getVariables(self): return self.v
        def getObjectives(self): return self.o

    n_vars = len(mm.get_var_names("ffmp"))
    n_objs = len(mm.get_obj_names())
    out = tmp_path / "seed_01_ffmp_obj8.set"
    mm._write_set_file([_Sol([0.5] * n_vars, [1.0] * n_objs)], out, "ffmp", 1)
    assert read_set_epsilons(out) == [float(e) for e in get_epsilons()]
    from src.load.reference_set import load_set_file
    assert load_set_file(out).shape == (1, n_vars + n_objs)


def test_check_series_against_set_uses_the_sets_own_epsilons(tmp_path):
    from src.diagnostics import check_series_against_set
    from src.runtime_archive import write_set_file
    a = tmp_path / "seed_01_x_0.runtime"
    b = tmp_path / "seed_01_x_1.runtime"
    a.write_text(_ISLAND_A)
    b.write_text(_ISLAND_B)
    # Controller archive under a finer vector than the one in use today.
    fine = [0.01, 0.01]
    final = epsilon_merge(snapshot_at([a, b], 100), 2, fine)
    set_file = tmp_path / "seed_01_x.set"
    write_set_file(set_file, final, ["v1", "v2"], ["o1", "o2"],
                   header_lines=["Epsilons: 0.01,0.01"])
    cmp = check_series_against_set([a, b], set_file, n_vars=2)
    assert cmp["boxes_only_a"] == [] and cmp["boxes_only_b"] == []
    # A legacy archive without the header cannot be checked.
    write_set_file(set_file, final, ["v1", "v2"], ["o1", "o2"])
    assert check_series_against_set([a, b], set_file, n_vars=2) is None
    # A wrong vector in the header is caught: under (1.0, 0.01) the two
    # archived rows fall into boxes (0, 20) and (0, 5), and the merge of the
    # islands keeps only the dominating box.
    write_set_file(set_file, final, ["v1", "v2"], ["o1", "o2"],
                   header_lines=["Epsilons: 1.0,0.01"])
    with pytest.raises(RuntimeError, match="different epsilon boxes"):
        check_series_against_set([a, b], set_file, n_vars=2)


def test_step07_reference_set_is_never_the_campaign_reference_set():
    from src.diagnostics import COORDINATOR_ARTIFACT, diagnostics_reference_set_path
    out = diagnostics_reference_set_path("ffmp_obj8", scenario="historic")
    assert out.name == "ffmp_obj8_diagnostics_reference.set"
    assert out.name != "ffmp_obj8_merged.set"
    assert COORDINATOR_ARTIFACT != "runtime"


# ---------------------------------------------------------------------------
# The step-07 driver end to end (MOEAFramework scoring stubbed)
# ---------------------------------------------------------------------------

def test_run_full_diagnostics_drives_series_reference_and_metrics(tmp_path, monkeypatch):
    import config
    import src.diagnostics as diag
    from src.formulations import get_n_objs, get_n_vars

    n_vars, n_objs = get_n_vars("ffmp"), get_n_objs()
    eps = np.asarray(config.get_epsilons(), dtype=float)
    rng = np.random.default_rng(3)
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(diag, "active_scenario_name", lambda: "synthetic")
    runtime_dir = tmp_path / "synthetic" / "ffmp_obj8" / "runtime"
    runtime_dir.mkdir(parents=True)
    for seed in (1, 2):
        for island in range(2):
            lines = []
            for nfe in (100, 200):
                lines.append(f"//NFE={nfe}")
                for _ in range(20):
                    row = np.concatenate([rng.uniform(0, 1, n_vars),
                                          rng.uniform(0, 4, n_objs) * eps * 5])
                    lines.append(" ".join(f"{v:.6e}" for v in row))
                lines.append("#")
            (runtime_dir / f"seed_{seed:02d}_ffmp_obj8_{island}.runtime").write_text(
                "\n".join(lines) + "\n")

    scored = []

    def fake_metrics(series_file, reference_file, slug, output_file=None):
        out = Path(series_file).parent.parent / "metrics" / f"{Path(series_file).stem}.metrics"
        out.parent.mkdir(exist_ok=True)
        n = len(parse_result_file(Path(series_file).read_text()))
        out.write_text("#Hypervolume GenerationalDistance\n" + "0.5 0.1\n" * n)
        scored.append((Path(series_file).name, Path(reference_file).name))
        return out

    monkeypatch.setattr(diag, "compute_metrics", fake_metrics)
    results = diag.run_full_diagnostics("ffmp_obj8")

    run_dir = tmp_path / "synthetic" / "ffmp_obj8"
    assert sorted(results) == [1, 2]
    assert sorted(p.name for p in (run_dir / "runtime_coordinator").iterdir()) == [
        "seed_01_ffmp_obj8.runtime", "seed_02_ffmp_obj8.runtime"]
    assert nfe_markers((run_dir / "runtime_coordinator" / "seed_01_ffmp_obj8.runtime").read_text()) == [200, 400]
    assert scored == [("seed_01_ffmp_obj8.runtime", "ffmp_obj8_diagnostics_reference.set"),
                      ("seed_02_ffmp_obj8.runtime", "ffmp_obj8_diagnostics_reference.set")]
    ref = parse_result_file((run_dir / "sets" / "ffmp_obj8_diagnostics_reference.set").read_text())
    assert len(ref) == 1 and ref[0][1].shape[1] == n_vars + n_objs
    assert not (run_dir / "sets" / "ffmp_obj8_merged.set").exists()


# ---------------------------------------------------------------------------
# MOEAFramework integration (Java-gated)
# ---------------------------------------------------------------------------

def _moea_available() -> bool:
    cli = PROJECT_DIR / "MOEAFramework-5.0" / "cli"
    jar = PROJECT_DIR / "MOEAFramework-5.0" / "lib" / "drb_ffmp.jar"
    return shutil.which("java") is not None and cli.exists() and jar.exists()


@pytest.mark.skipif(not _moea_available(),
                    reason="needs java on PATH and the drb_ffmp MOEAFramework JAR")
def test_metrics_evaluator_scores_the_python_written_series(tmp_path):
    """MetricsEvaluator reads the series and reference set this pipeline
    writes and returns one finite indicator row per snapshot; the final
    series entry matches a controller archive written in the .set layout."""
    from config import get_epsilons
    from src.diagnostics import (build_coordinator_series, compute_metrics,
                                 merge_reference_set)
    from src.formulations import get_n_objs, get_n_vars
    from src.plotting.hypervolume_convergence import _load_metrics_file
    from src.runtime_archive import write_set_file

    n_vars, n_objs = get_n_vars("ffmp"), get_n_objs()
    eps = np.asarray(get_epsilons(), dtype=float)
    rng = np.random.default_rng(7)
    grid = [100, 200, 300]
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    files = []
    for island in range(3):
        lines = []
        for nfe in grid:
            lines.append(f"//NFE={nfe}")
            n = 0 if (island == 2 and nfe == 100) else 40
            if n == 0:
                lines.append("//Empty")
            for _ in range(n):
                dv = rng.uniform(0, 1, n_vars)
                objs = rng.uniform(0, 4, n_objs) * eps * 5   # a few boxes per axis
                lines.append(" ".join(f"{v:.6e}" for v in np.concatenate([dv, objs])))
            lines.append("#")
        f = runtime_dir / f"seed_01_ffmp_obj8_{island}.runtime"
        f.write_text("\n".join(lines) + "\n")
        files.append(f)

    # The controller archive at shutdown is the merge of the final island
    # archives; write it in the .set layout the search driver uses.
    final = epsilon_merge(snapshot_at(files, 300), n_vars, eps)
    set_file = tmp_path / "seed_01_ffmp_obj8.set"
    write_set_file(set_file, final, [f"v{i}" for i in range(n_vars)],
                   [f"o{i}" for i in range(n_objs)],
                   header_lines=["Epsilons: " + ",".join(repr(float(e)) for e in eps)])

    series = build_coordinator_series(
        "ffmp_obj8", 1, runtime_dir=runtime_dir,
        output_file=tmp_path / "seed_01_ffmp_obj8.runtime", set_file=set_file)
    assert nfe_markers(series.read_text()) == [300, 600, 900]
    ref = merge_reference_set("ffmp_obj8", series_files=[series],
                              output_file=tmp_path / "ffmp_obj8_diagnostics_reference.set")
    metrics = compute_metrics(series, ref, "ffmp_obj8",
                              output_file=tmp_path / "seed_01_ffmp_obj8.metrics")
    df = _load_metrics_file(metrics)
    assert len(df) == len(grid)
    assert "Hypervolume" in df.columns
    assert np.isfinite(df.to_numpy(dtype=float)).all()
    assert (df["Hypervolume"] > 0).any()
