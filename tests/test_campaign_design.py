"""
tests/test_campaign_design.py - Pins of the adopted campaign design.

The campaign (docs/notes/methods/campaign_design.md) is N = 300, one search
draw per design, two seeds with seed 1 at 187,500 NFE/island and seed 2 at
125,000, reported at equal NFE from seed 1's runtime snapshot, on 12 Anvil
nodes at 128 ranks/node with a 150-realization batch. These tests pin the
code-side pieces that the note's numbers depend on: the per-seed NFE
resolution, the node-memory envelope the pre-flight enforces, the
runtime-snapshot extraction, and the K = 1 fallback of the variance
decomposition.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.moea_config import MOEAConfig, get_moea_config  # noqa: E402
from src.runtime_archive import (  # noqa: E402
    epsilon_merge,
    parse_runtime_snapshots,
    snapshot_at,
    write_set_file,
)


# ---------------------------------------------------------------------------
# Per-seed NFE
# ---------------------------------------------------------------------------

def test_max_evaluations_for_seed_defaults_and_overrides():
    c = MOEAConfig(name="x", max_evaluations=10, max_evaluations_by_seed=(30,))
    assert c.max_evaluations_for_seed(1) == 30
    assert c.max_evaluations_for_seed(2) == 10
    assert c.max_evaluations_for_seed(7) == 10
    with pytest.raises(ValueError):
        c.max_evaluations_for_seed(0)


def test_production_seed_scheme_is_the_campaign():
    """Seed 1 runs 750k total, every other seed 500k; snapshot cadence hits both."""
    c = get_moea_config("production")
    assert c.n_seeds == 2
    assert c.n_islands * c.max_evaluations_for_seed(1) == 750_000
    assert c.n_islands * c.max_evaluations_for_seed(2) == 500_000
    # The reporting NFE (the default) is a recorded snapshot of the long seed.
    assert c.max_evaluations % c.runtime_frequency == 0
    assert c.max_evaluations_for_seed(1) % c.runtime_frequency == 0
    assert c.max_evaluations_for_seed(1) > c.max_evaluations


# ---------------------------------------------------------------------------
# Memory envelope enforced by the pre-flight
# ---------------------------------------------------------------------------

def test_memory_envelope_reproduces_the_measured_production_run():
    """N=100, L=10 at 128 ranks measured 139-140 GB/node on both production runs."""
    import config
    gb = config.search_node_rss_gb(128, 100, 10, 0)
    assert 135 <= gb <= 145


def test_campaign_n_requires_the_batch_at_full_packing():
    import config
    line = config.NODE_MEMORY_GB * config.NODE_MEMORY_SAFETY_FRACTION
    assert config.search_node_rss_gb(128, 300, 10, 0) > line
    assert config.search_node_rss_gb(128, 300, 10, 150) < line
    # A batch larger than N is the unbatched case.
    assert config.search_rank_rss_mb(100, 10, 500) == config.search_rank_rss_mb(100, 10, 0)


# ---------------------------------------------------------------------------
# Runtime-snapshot extraction
# ---------------------------------------------------------------------------

_RUNTIME = """//NFE=50
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


def test_parse_runtime_snapshots_reads_blocks_by_nfe():
    snaps = parse_runtime_snapshots(_RUNTIME)
    assert sorted(snaps) == [50, 100]
    assert snaps[50].shape == (2, 4)
    assert snaps[100][1].tolist() == [0.5, 0.5, 0.20, 0.05]


def test_parse_runtime_snapshots_tolerates_missing_terminator():
    snaps = parse_runtime_snapshots("//NFE=10\n1 2 3 4\n")
    assert snaps[10].shape == (1, 4)


def test_snapshot_at_unions_islands_and_reports_missing_nfe(tmp_path):
    a = tmp_path / "seed_01_x_0.runtime"
    b = tmp_path / "seed_01_x_1.runtime"
    a.write_text(_RUNTIME)
    b.write_text(_RUNTIME.replace("0.5 0.5 0.20 0.05", "0.7 0.7 0.01 0.30"))
    rows = snapshot_at([a, b], 100)
    assert rows.shape == (4, 4)
    with pytest.raises(KeyError, match="NFE=75"):
        snapshot_at([a, b], 75)


def test_epsilon_merge_keeps_one_per_box_and_drops_dominated():
    rows = np.array([
        [1.0, 2.0, 0.05, 0.20],   # box (0, 2)
        [0.5, 0.5, 0.20, 0.05],   # box (2, 0)
        [0.7, 0.7, 0.06, 0.21],   # same box as row 0, farther from its corner
        [0.9, 0.9, 0.50, 0.50],   # dominated
    ])
    kept = epsilon_merge(rows, n_vars=2, epsilons=[0.1, 0.1])
    assert kept.shape == (2, 4)
    assert [1.0, 2.0] in kept[:, :2].tolist()
    assert [0.5, 0.5] in kept[:, :2].tolist()


def test_write_set_file_round_trips_through_the_set_loader(tmp_path):
    from src.load.reference_set import load_set_file
    rows = np.array([[1.0, 2.0, 0.1, 0.2], [0.5, 0.5, 0.2, 0.1]])
    out = tmp_path / "x.set"
    write_set_file(out, rows, ["v1", "v2"], ["o1", "o2"], header_lines=["Seed: 1"])
    back = load_set_file(out)
    assert np.allclose(back, rows)
    assert out.read_text().startswith("# Seed: 1\n# Variables: v1,v2\n# Objectives: o1,o2\n")


# ---------------------------------------------------------------------------
# One draw per design: the seed is the unit of analysis
# ---------------------------------------------------------------------------

def test_variance_components_fall_back_to_seed_unit_at_one_draw():
    from scripts.main.compare_designs import variance_components

    summary = pd.DataFrame({
        "design": ["A", "A", "B", "B"],
        "draw": [0, 0, 0, 0],
        "seed": [1, 2, 1, 2],
        "metric": ["m"] * 4,
        "best": [0.60, 0.62, 0.40, 0.44],
    })
    out = variance_components(summary, metric="m", statistic="best")
    design = out[out["component"] == "design (fixed)"].iloc[0]
    draw = out[out["component"].str.startswith("draw")].iloc[0]
    assert np.isfinite(design["f_stat"]) and design["f_stat"] > 1
    assert "SEED is the unit of analysis" in design["note"]
    assert draw["df"] == 0 and "not identified" in draw["note"]
