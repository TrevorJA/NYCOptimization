"""
tests/test_scenario_discovery.py - Unit tests for scripts/main/scenario_discovery.py.

The script's whole purpose is to DETECT A SIGNAL — a shape test would be
worthless here. So every test plants a KNOWN relationship in a synthetic hazard
image and asserts the machinery recovers it:

  1. The Olden & Poff redundancy screen drops the planted duplicate axis and
     keeps the operationally-preferred representative of the cluster.
  2. The gradient-boosted classifier recovers the PLANTED axis as the top factor
     importance (failures planted in one corner of hazard space).
  3. The KS shift statistic independently ranks the same axis first, and gets the
     shift DIRECTION right.
  4. The mechanism test's sign: an UNDER-COVERING search ensemble (one that never
     samples the failure corner) gives a strongly POSITIVE coverage-deficit ->
     failure association (AUC >> 0.5), while a WELL-COVERING (uniform) search
     ensemble gives a null (AUC ~ 0.5). A test that could not distinguish these
     two would not be a test of the mechanism.
  5. cdf_transform maps a second point set into the REFERENCE's rank space (not
     its own) — the join that makes the two ensembles comparable at all.
  6. End-to-end: discover_for_design on a synthetic per-SOW re-eval cube picks
     the compromise policy, labels each SOW by the all-criteria conjunction on
     its per-SOW annual-unit objective values, recovers the planted axis, and
     returns a supported-mechanism verdict.
  7. align_hazard_to_cube joins the per-REALIZATION hazard image onto the
     cube's SOW axis by label (realization_id // R == sow_id) and collapses
     each SOW's realizations to their hazard-coordinate MEAN.

Run:
    venv/bin/python -m pytest tests/test_scenario_discovery.py -v
"""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import src.robustness as rob  # noqa: E402


def _load_script():
    """Import scripts/main/scenario_discovery.py (not an importable package)."""
    path = PROJECT_DIR / "scripts" / "main" / "scenario_discovery.py"
    spec = importlib.util.spec_from_file_location("scenario_discovery", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scenario_discovery"] = mod
    spec.loader.exec_module(mod)
    return mod


sd = _load_script()


# ---------------------------------------------------------------------------
# Synthetic hazard image with a PLANTED failure corner
# ---------------------------------------------------------------------------
# 4 candidate axes, one of which (drought_duration) is a near-duplicate of the
# planted axis (drought_magnitude) — so the redundancy screen has something
# to remove and the importance ranking has something to be destabilized by.
PLANTED_AXIS = "drought_magnitude"
REDUNDANT_AXIS = "drought_duration"
AXES = [PLANTED_AXIS, REDUNDANT_AXIS, "drought_onset_rate", "flood_peak_discharge"]
N_TEST = 400
FAIL_CUT = 0.70          # failures planted where the planted axis exceeds this
N_SEARCH = 60


@pytest.fixture(scope="module")
def image():
    """(H, y): E_test hazard image + planted failure labels."""
    rng = np.random.default_rng(7)
    v = rng.uniform(0.0, 1.0, N_TEST)                       # planted axis
    dup = 0.95 * v + 0.05 * rng.uniform(0.0, 1.0, N_TEST)   # redundant with v
    o = rng.uniform(0.0, 1.0, N_TEST)
    f = rng.uniform(0.0, 1.0, N_TEST)
    H = np.column_stack([v, dup, o, f])
    y = v > FAIL_CUT                                        # the planted corner
    return H, y


@pytest.fixture(scope="module")
def screen(image):
    H, _ = image
    return sd.screen_hazard_axes(H, AXES)


def _search_points(kind: str, seed: int = 11) -> np.ndarray:
    """Search-ensemble hazard coords on the RETAINED axes (v, onset, flood)."""
    rng = np.random.default_rng(seed)
    if kind == "undercovering":
        # Never samples the failure corner: the planted axis is capped below the
        # failure cut, exactly the "this design left that hazard region unsampled"
        # situation the mechanism test is meant to catch.
        v = rng.uniform(0.0, FAIL_CUT - 0.10, N_SEARCH)
    else:  # "wellcovering"
        v = rng.uniform(0.0, 1.0, N_SEARCH)
    return np.column_stack([v, rng.uniform(0, 1, N_SEARCH), rng.uniform(0, 1, N_SEARCH)])


# ---------------------------------------------------------------------------
# 1. Redundancy screen (the Quinn et al. 2020 caveat, implemented)
# ---------------------------------------------------------------------------

def test_screen_drops_the_redundant_axis(screen):
    retained = screen["retained"]
    assert PLANTED_AXIS in retained, "screen dropped the operationally-preferred axis"
    assert REDUNDANT_AXIS not in retained, "screen kept a |rho_S| ~ 1 duplicate axis"
    assert set(retained) == {PLANTED_AXIS, "drought_onset_rate", "flood_peak_discharge"}
    # The planted duplicate must be clustered WITH the axis it duplicates.
    cluster = next(c for c in screen["clusters"] if PLANTED_AXIS in c)
    assert REDUNDANT_AXIS in cluster
    # And no correlated pair may survive into the fit.
    assert screen["residual_max_rho"] < sd.fm.FM_RHO_THRESHOLD


# ---------------------------------------------------------------------------
# 2-3. The classifier and the KS screen both recover the planted axis
# ---------------------------------------------------------------------------

def test_classifier_recovers_planted_axis(image, screen):
    H, y = image
    axes = screen["retained"]
    Hs = H[:, screen["retained_idx"]]
    X = sd.cdf_transform(Hs, Hs)
    model = sd.fit_classifier(X, y, axes, space="hazard")

    top = model.axes[int(np.argmax(model.importances))]
    assert top == PLANTED_AXIS, f"top importance was {top!r}, not the planted axis"
    # The signal is deterministic, so the planted axis should dominate, not merely lead.
    assert model.importances[model.axes.index(PLANTED_AXIS)] > 0.8
    assert np.isclose(model.importances.sum(), 1.0)
    assert model.train_accuracy > 0.95


def test_ks_shift_ranks_planted_axis_first_with_correct_direction(image, screen):
    H, y = image
    axes = screen["retained"]
    shifts = sd.hazard_shift_stats(H[:, screen["retained_idx"]], y, axes)
    top = shifts.sort_values("ks_stat", ascending=False).iloc[0]
    assert top["axis"] == PLANTED_AXIS
    assert top["shift_direction"] == "higher"   # failures sit at HIGH drought volume
    assert top["fail_mean"] > top["success_mean"]
    # The nuisance axes must not manufacture a shift.
    others = shifts[shifts["axis"] != PLANTED_AXIS]
    assert (others["ks_stat"] < top["ks_stat"] / 2).all()


# ---------------------------------------------------------------------------
# 4. THE MECHANISM TEST — sign, magnitude, and the null
# ---------------------------------------------------------------------------

def _association(image, screen, kind: str, seed: int = 11, n_boot: int = 100):
    H, y = image
    Hs = H[:, screen["retained_idx"]]
    X = sd.cdf_transform(Hs, Hs)
    deficit = sd.coverage_deficit(X, sd.cdf_transform(_search_points(kind, seed), Hs))
    stats, bins = sd.deficit_association(deficit, y, X_test=X, n_search=N_SEARCH,
                                         n_boot=n_boot)
    return stats, bins


def test_undercovering_search_ensemble_gives_positive_association(image, screen):
    stats, bins = _association(image, screen, "undercovering")

    assert stats["auc"] > 0.75, "failed to detect a planted coverage gap"
    assert stats["mannwhitney_p"] < 1e-6
    assert stats["deficit_mean_fail"] > stats["deficit_mean_success"]
    assert stats["logistic_slope"] > 0
    # The association must survive the random-coverage null, not just beat 0.5.
    assert stats["auc_excess"] > 0.10
    assert stats["p_vs_null"] <= 0.05
    assert "mechanism supported" in stats["verdict"]
    # The binned picture must rise, not just the scalar.
    assert bins.iloc[-1]["failure_rate"] > bins.iloc[0]["failure_rate"]


def test_wellcovering_search_ensemble_is_not_flagged(image, screen):
    """Calibration, not one draw: the statistic must not systematically flag a
    design that DID cover hazard space.

    A single search-ensemble draw at n=60 has an AUC standard error of ~0.07, so a
    single-draw assertion would be testing the draw, not the statistic. What must
    hold is calibration ACROSS draws: a near-zero MEDIAN excess over the
    random-coverage null, and a bounded false-positive rate (measured ~10% at the
    nominal 5% — the statistic is mildly anti-conservative and is documented as
    such in ``random_coverage_null``).
    """
    excess, raw_auc, flagged = [], [], 0
    for s in range(20):
        stats, _ = _association(image, screen, "wellcovering", seed=2000 + s,
                                n_boot=sd.N_NULL_BOOT)
        excess.append(stats["auc_excess"])
        raw_auc.append(stats["auc"])
        flagged += "mechanism supported" in stats["verdict"]

    # THE ARTIFACT the null exists to remove: even uniform coverage scores a RAW
    # AUC above 0.5, because nearest-neighbor distance grows toward the manifold
    # boundary and the planted failures live in a tail. An absolute-AUC rule would
    # "support the mechanism" for a design with no coverage gap at all.
    assert np.median(raw_auc) > 0.5

    # After the null correction, the same design is correctly scored as a null.
    assert abs(np.median(excess)) < 0.05
    assert flagged <= 5, f"false-positive rate {flagged}/20 — the null is miscalibrated"


def test_null_baseline_is_what_separates_the_two_designs(image, screen):
    """The excess-AUC statistic must discriminate where the raw AUC does not."""
    under, _ = _association(image, screen, "undercovering")
    well = [_association(image, screen, "wellcovering", seed=2000 + s,
                         n_boot=sd.N_NULL_BOOT)[0] for s in range(10)]
    assert under["auc_excess"] - np.median([w["auc_excess"] for w in well]) > 0.10
    # Same manifold, same n_search => every design shares (approximately) the same
    # null, so the separation above is a COVERAGE difference, not a scale one.
    assert abs(under["auc_null_mean"] - np.median([w["auc_null_mean"] for w in well])) < 0.05


def test_auc_is_nan_without_both_classes(image, screen):
    H, _ = image
    Hs = H[:, screen["retained_idx"]]
    X = sd.cdf_transform(Hs, Hs)
    deficit = sd.coverage_deficit(X, sd.cdf_transform(_search_points("wellcovering"), Hs))
    stats, bins = sd.deficit_association(deficit, np.zeros(len(X), dtype=bool),
                                         X_test=X, n_search=N_SEARCH, n_boot=20)
    assert np.isnan(stats["auc"]) and bins.empty
    assert "no discrimination" in stats["verdict"]


# ---------------------------------------------------------------------------
# 5. Reference-anchored normalization
# ---------------------------------------------------------------------------

def test_cdf_transform_uses_the_reference_not_the_input():
    ref = np.linspace(0.0, 100.0, 101).reshape(-1, 1)
    # A point set confined to the reference's lower tail must STAY in the lower
    # tail after transform; self-ranking would spread it over [0, 1] and erase
    # exactly the coverage gap the mechanism test looks for.
    pts = np.array([[0.0], [5.0], [10.0]])
    out = sd.cdf_transform(pts, ref)
    assert out.max() < 0.15
    assert np.all(np.diff(out.ravel()) > 0)
    assert sd.cdf_transform(ref, ref).max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. End-to-end on a synthetic re-eval cube
# ---------------------------------------------------------------------------

def _write_reeval(tmp_path: Path, cube: np.ndarray, sow_ids, obj_names,
                  thresholds, kinds, directions,
                  realizations_per_sow: int = 1) -> Path:
    """Persist an (S, G, M) per-SOW cube in the reeval_raw long format."""
    rows = []
    for si in range(cube.shape[0]):
        for gi, g in enumerate(sow_ids):
            for k, name in enumerate(obj_names):
                rows.append((si, int(g), name, float(cube[si, gi, k])))
    df = pd.DataFrame(rows, columns=["solution_id", "sow_id",
                                     "objective", "value"])
    out = tmp_path / "reeval"
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "reeval_raw.csv.gz", "wt") as fh:
        df.to_csv(fh, index=False)
    (out / "reeval_raw_meta.json").write_text(json.dumps({
        "obj_names": obj_names, "thresholds": thresholds, "kinds": kinds,
        "directions": directions, "is_ensemble": True,
        "sow_labels": sorted(int(g) for g in sow_ids),
        "realizations_per_sow": realizations_per_sow,
        "substrate": "sow_annual_unit",
        "solution_ids": list(range(cube.shape[0])),
    }))
    return out


def test_end_to_end_recovers_planted_axis_and_mechanism(tmp_path, image, screen,
                                                        monkeypatch):
    H, y_planted = image
    # One realization per SOW here, so SOW g's hazard coordinate is exactly
    # H[g] and the planted per-realization labels ARE the per-SOW labels.
    sow_ids = np.arange(N_TEST)

    # Two policies, one objective ("deficit", minimize, satisfice at <= 1.0);
    # the cells are per-SOW annual-unit values:
    #   solution 0 — the compromise: fails EXACTLY on the planted corner.
    #   solution 1 — a strictly worse policy that fails everywhere (so the
    #                compromise rule has a real choice to make).
    cube = np.zeros((2, N_TEST, 1))
    cube[0, :, 0] = np.where(y_planted, 2.0, 0.5)
    cube[1, :, 0] = 3.0
    reeval_dir = _write_reeval(
        tmp_path, cube, sow_ids, ["deficit"],
        thresholds={"deficit": 1.0}, kinds={"deficit": "le"},
        directions={"deficit": "minimize"},
    )
    raw = rob.load_raw(reeval_dir)

    etest = {"H": H, "hazard_axes": AXES, "chosen_axes": AXES,
             "realization_ids": np.arange(N_TEST),
             "selected_rows": np.arange(N_TEST)}

    # The design's SEARCH ensemble never sampled the failure corner.
    search_H = np.zeros((N_SEARCH, len(AXES)))
    pts = _search_points("undercovering")          # (n, 3) on the retained axes
    for j, a in enumerate(screen["retained"]):
        search_H[:, AXES.index(a)] = pts[:, j]
    search_H[:, AXES.index(REDUNDANT_AXIS)] = search_H[:, AXES.index(PLANTED_AXIS)]
    monkeypatch.setattr(sd, "search_hazard_image", lambda design, draw: {
        "H": search_H, "hazard_axes": AXES, "chosen_axes": AXES,
        "realization_ids": np.arange(N_SEARCH),
        "selected_rows": np.arange(N_SEARCH),
    })

    res = sd.discover_for_design("monte_carlo", raw, etest, screen,
                                 label="criterion:reference_all8")

    # The compromise rule must pick the policy that actually satisfices somewhere.
    assert res.solution_id == 0
    assert res.compromise["satisficing"] == pytest.approx(1.0 - y_planted.mean())
    # Failure labels are per SOW: the all-criteria conjunction on the per-SOW
    # values, one label per state of the world, joined on the SOW label.
    assert len(res.y) == raw.n_sow
    assert np.array_equal(res.y, y_planted)
    # The classifier recovers the planted axis...
    assert res.model.axes[int(np.argmax(res.model.importances))] == PLANTED_AXIS
    # ...and the mechanism test fires in the predicted direction.
    assert res.stats["auc"] > 0.75
    assert "mechanism supported" in res.stats["verdict"]
    assert res.n_search == N_SEARCH


def test_align_hazard_to_cube_aggregates_realizations_to_sow_means(tmp_path,
                                                                   image, screen):
    """The hazard image stays per-REALIZATION; the cube's unit is the SOW.

    Each SOW's coordinate must be the MEAN over its realizations' hazard rows,
    joined on realization_id // realizations_per_sow == sow_id — a positional
    join would silently attach the WRONG hazard coordinates to every label.
    """
    H, _ = image
    rps = 2                                        # 200 SOWs of 2 realizations
    # Re-eval covers only a shuffled subset of the SOWs (as a partial re-eval would).
    subset = [151, 4, 97]
    cube = np.ones((1, len(subset), 1))
    reeval_dir = _write_reeval(
        tmp_path, cube, subset, ["deficit"], thresholds={"deficit": 1.0},
        kinds={"deficit": "le"}, directions={"deficit": "minimize"},
        realizations_per_sow=rps,
    )
    raw = rob.load_raw(reeval_dir)
    aligned = sd.align_hazard_to_cube(
        raw, {"H": H, "realization_ids": np.arange(N_TEST)},
        screen["retained_idx"])
    idx = screen["retained_idx"]
    expected = np.vstack([
        H[np.ix_([g * rps, g * rps + 1], idx)].mean(axis=0)
        for g in sorted(subset)                    # rows keyed by ascending sow_id
    ])
    assert aligned.shape == (len(subset), len(idx))
    assert np.allclose(aligned, expected)

    # A re-evaluated SOW with no hazard coordinates is a hard error, never a
    # positional guess.
    with pytest.raises(KeyError):
        sd.align_hazard_to_cube(raw, {"H": H[:10],
                                      "realization_ids": np.arange(10)},
                                screen["retained_idx"])
