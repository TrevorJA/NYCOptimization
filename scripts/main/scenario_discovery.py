"""scenario_discovery.py - Scenario discovery on E_test failures, IN HAZARD SPACE.

This is the **mechanism test** for the study's central claim, not a decorative
post-processing step. The claim is that covering *hazard space* during MOEA
search produces policies that are more robust on the held-out test ensemble
``E_test``. The falsifiable prediction it implies is::

    A design's policies should FAIL on E_test in the hazard region that design
    UNDER-COVERED during search.

so the script does two things, per scenario design:

1. **Scenario discovery.** Label each E_test realization success/failure for the
   design's compromise policy (the multivariate Starr domain criterion — the
   same all-criteria conjunction that defines the primary robustness metric, so
   discovery inherits exactly the robustness criteria, as is standard), then fit
   a gradient-boosted classifier of failure on the realization's HAZARD
   coordinates. Reports factor importances, a 2-D factor map, and the
   failure/success distributional shift per axis (two-sample KS).

2. **The mechanism test.** For each E_test realization, compute the
   **coverage deficit** — the distance, in the E_test hazard image's
   empirical-CDF/rank space, to the nearest member of that design's SEARCH
   ensemble — and test whether failure probability is POSITIVELY associated with
   it (AUC of deficit as a failure predictor, its excess over a RANDOM-COVERAGE
   null, an empirical p-value, the logistic slope, and failure rate by deficit
   decile). Hazard-filling designs, having filled the space, should show NO excess
   association; ``historic`` / ``fixed_probabilistic`` should show failures
   concentrating where their ensembles left hazard space unsampled. A null is a
   real, reportable result and is written as such.

   The random-coverage null is load-bearing, not a nicety: nearest-neighbor
   distance is systematically larger near the boundary of the hazard manifold, and
   failures sit in a tail, so a uniformly-covering ensemble still scores AUC ~ 0.62
   from geometry alone. The verdict is therefore read off AUC MINUS its null, never
   off the raw AUC (see :func:`random_coverage_null`).

What is novel here. Standard scenario discovery — Kasprzyk et al. (2013) with
PRIM, Gold et al. (2022, 2023) with boosted trees — runs in the INPUT / forcing
parameter space. This runs in HAZARD space (drought/flood event descriptors of
the realized sequence), which is the space in which the coverage hypothesis is
stated and the only space in which "the design under-covered here" is even
definable.

Classifier. Gradient boosting with the Gold et al. (2023) settings: 250 trees,
``max_depth=2``, ``learning_rate=0.1``. Trees are monotone-invariant, so the fit
is done in rank space (where the factor map is plotted); importances are
unchanged by that choice.

Correlated-axis caveat, IMPLEMENTED not just documented. Factor importances are
unstable under correlated factors — Quinn et al. (2020) show Sobol first-order
indices going NEGATIVE (a negative interaction term = redundancy) exactly in
this situation. So the hazard axes are SCREENED before fitting with the same
Olden & Poff (2003) redundancy screen the hazard-filling selector uses
(``scengen.diagnostics.spearman_clusters``, ``|rho_S| >= 0.7``, one
representative per cluster; degenerate axes dropped by ``per_metric_spread``).
Retained axes and the clusters are written out, and a warning is printed if any
residual pair still exceeds the threshold.

Inputs (all pre-existing):
  * ``outputs/{design}/{moea_slug}/reeval/{reeval_tag}[/seed_NN]/reeval_raw.parquet``
  * ``{STAGED_ENSEMBLE_DIR}/{etest_slug}/hazard_image.npz``  (E_test hazard image;
    generate E_test with ``compute_hazard_image=True`` — workflow step 02)
  * each design's SEARCH ensemble hazard image (loaded, or computed and cached
    from the staged daily inflows with the identical generation-time code path).

Outputs:
  * tables  -> ``outputs/comparison/scenario_discovery/*.csv``
  * figures -> ``figures/_exploratory/{design}/{slug}/scenario_discovery/`` and
    ``figures/_exploratory/comparison/{slug}/scenario_discovery/``

Settings are module constants (env-overridable), never CLI value flags; only
identifiers (``--formulation``, ``--reeval-tag``, ``--seed``, ``--designs``,
``--draw``) are accepted on the command line.

Run::

    python scripts/main/scenario_discovery.py --formulation ffmp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, mannwhitneyu

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from src import robustness as rob  # noqa: E402
from src.plotting.style import apply_style, save_figure  # noqa: E402
from src.scenario_designs import campaign_designs, get_scenario_design  # noqa: E402

try:  # sklearn is the intended backend; the fallback is declared, never silent.
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only on a stripped env
    _HAS_SKLEARN = False


###############################################################################
# Settings (module constants; env-overridable. NO CLI value flags — repo rule)
###############################################################################

#: Gradient-boosted classifier hyperparameters (Gold et al. 2023).
GBC_N_ESTIMATORS: int = int(os.environ.get("NYCOPT_SD_N_TREES", "250"))
GBC_MAX_DEPTH: int = int(os.environ.get("NYCOPT_SD_MAX_DEPTH", "2"))
GBC_LEARNING_RATE: float = float(os.environ.get("NYCOPT_SD_LEARNING_RATE", "0.1"))

#: Spearman |rho| above which two hazard axes are redundant (Olden & Poff 2003).
REDUNDANCY_THRESHOLD: float = float(os.environ.get("NYCOPT_SD_RHO_THRESHOLD", "0.7"))

#: Rule selecting the per-design analysis policy. Scenario discovery is run on a
#: small number of COMPROMISE solutions, not the whole front (Kasprzyk et al.
#: 2013). Both rules are always computed and written to
#: ``compromise_solutions.csv``; this one names the policy actually analyzed.
#:   "best_satisficing" -- highest multivariate (Starr) satisficing fraction on
#:                         E_test; ties broken by min-distance-to-ideal.
#:   "min_dist_ideal"   -- minimum Euclidean distance to the ideal point in the
#:                         min-max-normalized, direction-oriented mean re-evaluated
#:                         objective space.
COMPROMISE_RULE: str = os.environ.get("NYCOPT_SD_COMPROMISE_RULE", "best_satisficing")

#: Bins for the failure-rate-vs-coverage-deficit table (deciles by default).
N_DEFICIT_BINS: int = int(os.environ.get("NYCOPT_SD_DEFICIT_BINS", "10"))

#: Bootstrap replicates for the random-coverage null of the mechanism test.
N_NULL_BOOT: int = int(os.environ.get("NYCOPT_SD_NULL_BOOT", "200"))

#: Factor-map grid resolution per axis.
GRID_RES: int = int(os.environ.get("NYCOPT_SD_GRID_RES", "60"))

#: Table output root.
TABLE_DIR = config.OUTPUTS_DIR / "comparison" / "scenario_discovery"

_FIG_KIND = "scenario_discovery"


###############################################################################
# Hazard-space normalization
###############################################################################

def cdf_transform(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map columns of ``values`` into the empirical-CDF/rank space of ``reference``.

    The reference-anchored generalization of
    ``scengen.subsample.empirical_cdf_normalize`` (which ranks a matrix against
    itself). Anchoring is REQUIRED here: a search ensemble and E_test must land
    in one common, comparable coordinate system before their distances mean
    anything, and self-ranking each set separately would map each set's own
    extremes to 0 and 1 — destroying exactly the gaps the mechanism test looks
    for. Applied to the reference itself this reproduces
    ``empirical_cdf_normalize`` up to tie handling (max rank vs average rank).

    Args:
        values: ``(n, m)`` raw hazard coordinates to transform.
        reference: ``(R, m)`` raw hazard coordinates defining the CDF (E_test).

    Returns:
        ``(n, m)`` array in ``[0, 1]``.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    reference = np.asarray(reference, dtype=float)
    out = np.empty_like(values, dtype=float)
    n_ref = reference.shape[0]
    for a in range(values.shape[1]):
        ref_sorted = np.sort(reference[:, a])
        out[:, a] = np.searchsorted(ref_sorted, values[:, a], side="right") / n_ref
    return np.clip(out, 0.0, 1.0)


###############################################################################
# Axis screening (Olden & Poff 2003; the Quinn et al. 2020 caveat, implemented)
###############################################################################

def screen_hazard_axes(H: np.ndarray, axes: list[str],
                       threshold: float = REDUNDANCY_THRESHOLD) -> dict:
    """Drop degenerate axes and keep one representative per redundant cluster.

    Reuses the screen the hazard-filling selector itself uses
    (``scengen.hazard_filling.select_from_candidate_image``), so discovery is run
    on the same low-redundancy axis basis the design was built in. Importances
    reported over correlated axes are not interpretable (Quinn et al. 2020), which
    is why this runs BEFORE the fit rather than as a footnote after it.

    Args:
        H: ``(R, m)`` hazard image (raw metric values).
        axes: Length-``m`` axis names.
        threshold: Spearman ``|rho|`` redundancy cut.

    Returns:
        Dict with ``retained`` (axis names), ``retained_idx``, ``clusters``,
        ``degenerate``, ``rho`` (over the non-degenerate axes), and
        ``residual_max_rho`` (largest ``|rho|`` still present among retained axes).
    """
    from scengen.diagnostics import per_metric_spread, spearman_clusters
    from scengen.hazard_filling import DEFAULT_AXIS_PRIORITY

    H = np.asarray(H, dtype=float)
    spread = per_metric_spread(H, axes)
    degenerate = [a for a in axes if spread[a]["degenerate"]]
    kept = [a for a in axes if a not in degenerate]
    if not kept:  # every axis degenerate: fall back to the full set, loudly
        warnings.warn(
            "Every hazard axis was flagged degenerate by the spread screen; "
            "falling back to the unscreened axis set. Importances over these "
            "axes are not interpretable."
        )
        kept = list(axes)
        degenerate = []

    keep_idx = [axes.index(a) for a in kept]
    clusters = spearman_clusters(
        H[:, keep_idx], kept, threshold=threshold, priority=DEFAULT_AXIS_PRIORITY,
    )
    retained = list(clusters["representatives"])
    retained_idx = [axes.index(a) for a in retained]

    rho = np.atleast_2d(clusters["rho"])
    sub = [kept.index(a) for a in retained]
    resid = rho[np.ix_(sub, sub)].copy() if len(sub) > 1 else np.ones((1, 1))
    np.fill_diagonal(resid, 0.0)
    residual_max = float(np.abs(resid).max()) if len(sub) > 1 else 0.0
    if residual_max >= threshold:
        warnings.warn(
            f"Retained hazard axes still contain a pair with |rho_S| = "
            f"{residual_max:.2f} >= {threshold}. Factor importances over "
            f"correlated axes are unstable (Quinn et al. 2020) -- read the "
            f"importance ranking as indicative only."
        )
    return {
        "retained": retained,
        "retained_idx": retained_idx,
        "clusters": clusters["clusters"],
        "degenerate": degenerate,
        "rho": rho,
        "screened_axes": kept,
        "residual_max_rho": residual_max,
    }


###############################################################################
# Failure labels + compromise-policy selection
###############################################################################

def failure_matrix(raw: rob.RawCube) -> np.ndarray:
    """Boolean ``(S, R)`` failure matrix: the all-criteria conjunction, negated.

    A realization FAILS for a solution when the joint (all-objective) satisficing
    conjunction is False — the multivariate Starr (1962) domain criterion, which
    is also the primary robustness metric (``robustness.satisficing_multivariate``).
    Discovery therefore inherits exactly the robustness criteria; that is standard
    practice and must be stated when the result is reported.
    """
    return ~rob._satisfaction_cube(raw).all(axis=2)


def _normalized_mean_objectives(raw: rob.RawCube) -> np.ndarray:
    """``(S, M)`` mean re-evaluated objectives, oriented so 0 = ideal, 1 = worst.

    Direction-oriented (maximize objectives flipped) and min-max normalized over
    the solution set, so the ideal point is the origin.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(raw.cube, axis=1)  # (S, M)
    signs = raw.direction_signs()
    loss = means * -signs[None, :]  # lower = better, for every objective
    lo = np.nanmin(loss, axis=0)
    hi = np.nanmax(loss, axis=0)
    span = np.where(np.abs(hi - lo) > 0, hi - lo, 1.0)
    return (loss - lo[None, :]) / span[None, :]


def select_compromise(raw: rob.RawCube, rule: str = COMPROMISE_RULE) -> dict:
    """Choose the per-design analysis policy, and report both candidate rules.

    Scenario discovery is run on a small number of compromise solutions, not the
    whole Pareto front (Kasprzyk et al. 2013; Gold et al. 2022) — a factor map of
    "the front" is not a statement about any policy anyone would adopt.

    Args:
        raw: The design's re-eval cube.
        rule: ``"best_satisficing"`` or ``"min_dist_ideal"``.

    Returns:
        Dict with ``solution_id`` (the analyzed policy), ``rule``, the two
        candidate ids (``best_satisficing_id``, ``min_dist_ideal_id``), the
        chosen policy's ``satisficing`` fraction and ``distance_to_ideal``, and
        ``index`` (row into the cube).

    Raises:
        ValueError: For an unknown ``rule``, or when every solution is all-NaN.
    """
    if rule not in ("best_satisficing", "min_dist_ideal"):
        raise ValueError(
            f"unknown compromise rule {rule!r}; expected 'best_satisficing' or "
            f"'min_dist_ideal' (set NYCOPT_SD_COMPROMISE_RULE)."
        )
    sat = rob.satisficing_multivariate(raw).to_numpy(dtype=float)   # (S,)
    dist = np.linalg.norm(_normalized_mean_objectives(raw), axis=1)  # (S,)

    alive = np.any(np.isfinite(raw.cube), axis=(1, 2))
    if not alive.any():
        raise ValueError("every solution in this re-eval cube is all-NaN (failed).")
    masked_dist = np.where(alive, dist, np.inf)
    masked_sat = np.where(alive, sat, -np.inf)

    # Ties on satisficing are common (a saturated criterion ties everything —
    # Bonham et al. 2024), so break them on distance-to-ideal rather than on
    # solution-id order, which would be an arbitrary, ordering-dependent choice.
    best_sat = int(np.lexsort((masked_dist, -masked_sat))[0])
    best_dist = int(np.argmin(masked_dist))
    idx = best_sat if rule == "best_satisficing" else best_dist
    return {
        "rule": rule,
        "index": idx,
        "solution_id": int(raw.solution_ids[idx]),
        "best_satisficing_id": int(raw.solution_ids[best_sat]),
        "min_dist_ideal_id": int(raw.solution_ids[best_dist]),
        "satisficing": float(sat[idx]),
        "distance_to_ideal": float(dist[idx]),
    }


###############################################################################
# Failure classifier (Gold et al. 2022, 2023)
###############################################################################

@dataclass
class FailureModel:
    """A fitted failure classifier over the retained hazard axes.

    Attributes:
        axes: Retained axis names (feature order).
        importances: Per-axis factor importance, summing to 1.
        backend: ``"gradient_boosting"`` or the declared fallback.
        predict_proba: Maps ``(n, m)`` rank-space points to failure probability.
        train_accuracy: Resubstitution accuracy (a fit diagnostic, not a claim).
    """

    axes: list[str]
    importances: np.ndarray
    backend: str
    predict_proba: object = field(repr=False, default=None)
    train_accuracy: float = float("nan")


def _knn_failure_prob(X: np.ndarray, y: np.ndarray, k: int = 25):
    """k-NN failure-probability surface — the declared no-sklearn fallback."""
    tree = cKDTree(X)
    k = int(min(max(3, k), len(X)))

    def _predict(grid: np.ndarray) -> np.ndarray:
        _, idx = tree.query(np.atleast_2d(grid), k=k)
        return y[np.atleast_2d(idx)].mean(axis=1)

    return _predict


def fit_failure_classifier(X: np.ndarray, y: np.ndarray,
                           axes: list[str]) -> FailureModel:
    """Fit failure ~ hazard coordinates with the Gold et al. (2023) settings.

    Gradient boosting, 250 trees, ``max_depth=2``, ``learning_rate=0.1`` — the
    boosted-tree scenario-discovery configuration of Gold et al. (2022, 2023),
    used unchanged so the method is the literature's and only the SPACE (hazard,
    not input) is ours.

    If sklearn is unavailable the model degrades — loudly, never silently — to
    KS-statistic importances with a k-NN failure-probability surface, which
    preserves the factor map and the ranking but not the interaction structure.

    Args:
        X: ``(R, m)`` hazard coordinates (rank space; trees are monotone-invariant,
            so this is equivalent to fitting on raw values).
        y: Length-``R`` boolean/int failure labels.
        axes: Length-``m`` feature names.

    Returns:
        A :class:`FailureModel`.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    if not _HAS_SKLEARN:
        warnings.warn(
            "scikit-learn is not installed: falling back to KS-statistic "
            "importances + a k-NN failure surface. The reported importances are "
            "NOT gradient-boosted factor importances -- say so when reporting."
        )
        ks = np.array([
            ks_2samp(X[y == 1, a], X[y == 0, a]).statistic
            if (y == 1).any() and (y == 0).any() else 0.0
            for a in range(X.shape[1])
        ])
        imp = ks / ks.sum() if ks.sum() > 0 else np.full(len(ks), 1.0 / len(ks))
        return FailureModel(
            axes=list(axes), importances=imp, backend="ks_knn_fallback",
            predict_proba=_knn_failure_prob(X, y),
        )

    clf = GradientBoostingClassifier(
        n_estimators=GBC_N_ESTIMATORS,
        max_depth=GBC_MAX_DEPTH,
        learning_rate=GBC_LEARNING_RATE,
        random_state=0,
    ).fit(X, y)
    return FailureModel(
        axes=list(axes),
        importances=np.asarray(clf.feature_importances_, dtype=float),
        backend="gradient_boosting",
        predict_proba=lambda g: clf.predict_proba(np.atleast_2d(g))[:, 1],
        train_accuracy=float(clf.score(X, y)),
    )


def hazard_shift_stats(H: np.ndarray, y: np.ndarray, axes: list[str]) -> pd.DataFrame:
    """Where do failures live? Per-axis failure/success distributional shift.

    The cheap, interpretable companion to the classifier: a two-sample KS
    statistic per axis (how far the failing realizations' marginal is displaced
    from the succeeding ones'), plus the failure-weighted mean and quartiles in
    RAW hazard units, so the failure region can be described in the units the
    metric is defined in.

    Args:
        H: ``(R, m)`` raw hazard image restricted to the retained axes.
        y: Length-``R`` boolean failure labels.
        axes: Length-``m`` axis names.

    Returns:
        Tidy frame: axis, ks_stat, ks_pvalue, fail_mean, fail_q25/50/75,
        success_mean, shift_direction.
    """
    y = np.asarray(y).astype(bool)
    rows = []
    for a, name in enumerate(axes):
        f, s = H[y, a], H[~y, a]
        if f.size and s.size:
            ks = ks_2samp(f, s)
            stat, pval = float(ks.statistic), float(ks.pvalue)
        else:
            stat, pval = float("nan"), float("nan")
        fm = float(f.mean()) if f.size else float("nan")
        sm = float(s.mean()) if s.size else float("nan")
        rows.append({
            "axis": name,
            "ks_stat": stat,
            "ks_pvalue": pval,
            "fail_mean": fm,
            "fail_q25": float(np.quantile(f, 0.25)) if f.size else float("nan"),
            "fail_q50": float(np.quantile(f, 0.50)) if f.size else float("nan"),
            "fail_q75": float(np.quantile(f, 0.75)) if f.size else float("nan"),
            "success_mean": sm,
            "shift_direction": ("higher" if fm > sm else "lower") if f.size and s.size else "",
        })
    return pd.DataFrame(rows)


###############################################################################
# THE MECHANISM TEST: coverage deficit -> failure
###############################################################################

def coverage_deficit(X_test: np.ndarray, X_search: np.ndarray) -> np.ndarray:
    """Distance from each E_test realization to the nearest SEARCH-ensemble member.

    Both point sets must already be in the SAME normalized hazard space (E_test's
    empirical-CDF/rank space — see :func:`cdf_transform`). This is the
    per-realization operationalization of "the design under-covered this part of
    hazard space": large deficit = no search scenario resembled this test
    realization, hazard-wise.

    Args:
        X_test: ``(R, m)`` E_test hazard coordinates, normalized.
        X_search: ``(n, m)`` search-ensemble hazard coordinates, normalized.

    Returns:
        Length-``R`` array of nearest-neighbor distances.
    """
    dist, _ = cKDTree(np.atleast_2d(X_search)).query(np.atleast_2d(X_test), k=1)
    return np.asarray(dist, dtype=float).ravel()


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    """AUC of ``scores`` as a predictor of ``y`` (rank formula; ties = 0.5)."""
    y = np.asarray(y).astype(bool)
    pos, neg = scores[y], scores[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def random_coverage_null(X_test: np.ndarray, y: np.ndarray, n_search: int,
                         n_boot: int = N_NULL_BOOT, seed: int = 0) -> dict:
    """Null distribution of the deficit->failure AUC under RANDOM hazard coverage.

    **This baseline is not optional, and the test is invalid without it.** The
    nearest-neighbor deficit is systematically LARGER near the boundary of the
    hazard manifold (fewer neighbors on one side), so any failure region that sits
    in a tail — which is precisely where failures sit — inherits a positive
    deficit-failure association from pure geometry, with no coverage gap at all.
    Measured on this project's own synthetic fixture, a search ensemble covering
    hazard space UNIFORMLY still scores AUC ~ 0.62 against a tail failure region.
    An absolute AUC threshold would therefore "support the mechanism" for every
    design, including the ones that have no gap.

    So the observed AUC is compared against the AUC obtained when the search
    ensemble is a RANDOM sample of the same hazard manifold at the same size — the
    same logic ``scengen.diagnostics.expected_random_discrepancy`` uses to judge
    coverage relative to chance rather than asserting it. Random subsets are drawn
    from the E_test manifold itself (its empirical joint law, correlations
    included, rather than an idealized uniform cube); exact self-matches are
    excluded from the nearest-neighbor query, since a realization coinciding with
    a search member is an artifact of resampling one finite point set, not a
    property a real search ensemble has.

    Power and calibration (measured on ``tests/test_scenario_discovery.py``'s
    fixture: R = 400, m = 3, n_search = 60). The null AUC is 0.53 +/- 0.07, so a
    SINGLE search-ensemble draw carries an AUC standard error of ~0.07: this test
    detects a gross coverage gap (planted gap scores +0.45 excess) but is not
    powered to resolve small differences between two well-covering designs. The
    false-positive rate at the nominal 5% level measures ~8-10%, slightly
    anti-conservative. Report ``auc_null_std`` alongside ``auc_excess``, and
    compare designs on the SIGN and MAGNITUDE of the excess, not on p-values alone.

    Args:
        X_test: ``(R, m)`` normalized E_test hazard coordinates.
        y: Length-``R`` boolean failure labels.
        n_search: Size of the design's search ensemble (the null matches it).
        n_boot: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        Dict with ``mean``, ``std`` and the raw ``samples`` of the null AUC.
    """
    rng = np.random.default_rng(seed)
    R = len(X_test)
    n = int(min(max(1, n_search), R))
    tree_pts = np.atleast_2d(X_test)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.choice(R, size=n, replace=False)
        dist, _ = cKDTree(tree_pts[idx]).query(tree_pts, k=min(2, n))
        dist = np.atleast_2d(dist.reshape(R, -1))
        # Drop the self-match (distance 0) for the sampled rows.
        d = dist[:, 0].copy()
        if dist.shape[1] > 1:
            self_hit = np.zeros(R, dtype=bool)
            self_hit[idx] = True
            d[self_hit] = dist[self_hit, 1]
        samples[b] = _auc(d, y)
    return {"mean": float(np.nanmean(samples)), "std": float(np.nanstd(samples)),
            "samples": samples}


def deficit_association(deficit: np.ndarray, y: np.ndarray,
                        *, X_test: np.ndarray | None = None,
                        n_search: int | None = None,
                        n_boot: int = N_NULL_BOOT, seed: int = 0,
                        n_bins: int = N_DEFICIT_BINS) -> tuple[dict, pd.DataFrame]:
    """Is failure POSITIVELY associated with the coverage deficit? (the test)

    Reported several ways, because one number would not be believed: the AUC of
    the deficit as a failure score (> 0.5 = failures sit at HIGHER deficit, the
    predicted direction); the same AUC MINUS its random-coverage null
    (:func:`random_coverage_null`), which is the quantity the verdict is actually
    read off, since the raw AUC is inflated by manifold-boundary geometry; a
    one-sided empirical p-value against that null; the one-sided Mann-Whitney p
    (descriptive: it tests AUC > 0.5, not AUC > null); the logistic slope on the
    standardized deficit; and the binned failure rate, which is the
    non-parametric picture the figure plots.

    A null (excess AUC ~ 0) is a real, reportable result: failures did NOT
    concentrate where that design under-covered, which is evidence AGAINST the
    coverage mechanism for that design.

    Args:
        deficit: Length-``R`` coverage deficits.
        y: Length-``R`` boolean failure labels.
        X_test: ``(R, m)`` normalized E_test coordinates; enables the null.
        n_search: Search-ensemble size the null is matched to.
        n_boot: Null bootstrap replicates.
        seed: Null RNG seed.
        n_bins: Quantile bins of the deficit (10 = deciles).

    Returns:
        ``(stats, bins)`` — a summary dict and the per-bin failure-rate frame.
    """
    y = np.asarray(y).astype(bool)
    n_fail, n_ok = int(y.sum()), int((~y).sum())
    stats: dict = {
        "n_realizations": int(y.size),
        "n_fail": n_fail,
        "failure_rate": float(y.mean()) if y.size else float("nan"),
        "deficit_mean": float(np.mean(deficit)),
        "deficit_mean_fail": float(deficit[y].mean()) if n_fail else float("nan"),
        "deficit_mean_success": float(deficit[~y].mean()) if n_ok else float("nan"),
        "auc": float("nan"),
        "auc_null_mean": float("nan"),
        "auc_null_std": float("nan"),
        "auc_excess": float("nan"),
        "p_vs_null": float("nan"),
        "mannwhitney_p": float("nan"),
        "logistic_slope": float("nan"),
        "verdict": "no discrimination (all realizations same class)",
    }
    if n_fail == 0 or n_ok == 0:
        return stats, pd.DataFrame()

    stats["auc"] = _auc(deficit, y)
    stats["mannwhitney_p"] = float(
        mannwhitneyu(deficit[y], deficit[~y], alternative="greater").pvalue
    )
    if _HAS_SKLEARN:
        s = deficit.std()
        z = (deficit - deficit.mean()) / (s if s > 0 else 1.0)
        lr = LogisticRegression(max_iter=1000).fit(z.reshape(-1, 1), y.astype(int))
        stats["logistic_slope"] = float(lr.coef_[0][0])

    if X_test is not None and n_search:
        null = random_coverage_null(X_test, y, n_search, n_boot=n_boot, seed=seed)
        stats["auc_null_mean"] = null["mean"]
        stats["auc_null_std"] = null["std"]
        stats["auc_excess"] = stats["auc"] - null["mean"]
        # One-sided empirical p: how often random coverage matches this AUC.
        stats["p_vs_null"] = float(
            (np.sum(null["samples"] >= stats["auc"]) + 1) / (len(null["samples"]) + 1)
        )
        excess, p = stats["auc_excess"], stats["p_vs_null"]
        if excess >= 0.05 and p <= 0.05:
            verdict = ("failures concentrate at HIGH coverage deficit, beyond "
                       "random coverage (mechanism supported)")
        elif excess <= -0.05:
            verdict = ("failures concentrate at LOW coverage deficit "
                       "(mechanism contradicted)")
        else:
            verdict = ("no coverage-deficit association beyond random coverage "
                       "(null)")
    else:  # no null available: report the raw association, and say so
        verdict = ("raw AUC only -- no random-coverage null; NOT interpretable as "
                   "mechanism evidence (boundary geometry inflates it)")
    stats["verdict"] = verdict

    # Quantile bins; duplicate edges collapse when the deficit is highly tied.
    try:
        codes, edges = pd.qcut(deficit, n_bins, labels=False, retbins=True,
                               duplicates="drop")
    except ValueError:  # pragma: no cover - degenerate (constant) deficit
        return stats, pd.DataFrame()
    frame = pd.DataFrame({"bin": codes, "deficit": deficit, "fail": y.astype(int)})
    bins = (frame.groupby("bin")
            .agg(n=("fail", "size"), failure_rate=("fail", "mean"),
                 deficit_mid=("deficit", "median"))
            .reset_index())
    bins["bin_lo"] = [edges[int(b)] for b in bins["bin"]]
    bins["bin_hi"] = [edges[int(b) + 1] for b in bins["bin"]]
    return stats, bins


###############################################################################
# Hazard images: E_test, and each design's SEARCH ensemble
###############################################################################

def _staged_dir(slug: str) -> Path:
    from src.ensembles import staged_ensemble_dir
    return Path(staged_ensemble_dir(slug))


def load_etest_hazard_image(spec) -> dict:
    """Load the hazard image staged next to the E_test ensemble.

    Raises:
        SystemExit: If the image is not staged. There is deliberately NO fallback
            to forcing parameters: the whole point is that the coverage hypothesis
            is stated in hazard space, and a silent input-space substitution would
            answer a different question while looking like this one.
    """
    from scengen.diagnostics import load_hazard_image

    path = _staged_dir(spec.inflow_type) / "hazard_image.npz"
    if not path.exists():
        sys.exit(
            f"[scenario_discovery] No hazard image for the re-eval (test) ensemble "
            f"'{spec.inflow_type}':\n    {path}\n"
            f"Scenario discovery is run in HAZARD space, so E_test must carry its "
            f"hazard coordinates. Regenerate E_test with compute_hazard_image=True "
            f"(workflow step 02) and re-run. There is no forcing-parameter fallback "
            f"-- that would silently answer a different question."
        )
    img = load_hazard_image(path)
    if len(img["realization_ids"]) != img["H"].shape[0]:
        sys.exit(f"[scenario_discovery] Corrupt hazard image (id/row mismatch): {path}")
    return img


def _compute_hazard_image(slug: str) -> dict | None:
    """Compute (and cache) the hazard image of a staged ensemble from its flows.

    Only the hazard-filling designs stage a hazard image at build time (the SSI-6 +
    POT pass is pure waste for the others), but the MECHANISM TEST needs the search
    hazard coordinates of EVERY design — the prediction for ``historic`` /
    ``fixed_probabilistic`` is precisely that their failures concentrate where their
    ensembles left hazard space unsampled. So the image is computed here, on demand,
    through the IDENTICAL generation-time code path
    (``src.ensemble_generation._hazard_block``), and cached next to the ensemble so
    the coordinates are commensurable with the staged ones by construction.

    Returns:
        The hazard-image dict, or ``None`` if the staged daily inflows are absent.
    """
    from scengen.diagnostics import load_hazard_image, save_hazard_image
    from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES
    from scengen.hazard_filling import daily_to_monthly
    from src.ensemble_generation import _hazard_block
    from src.ensembles import load_chunk_index, pool_chunk_specs
    from src.load.historical_flows import load_historical_flows
    from synhydro.core.ensemble import Ensemble

    out_dir = _staged_dir(slug)
    cached = out_dir / "hazard_image.npz"
    if cached.exists():
        return load_hazard_image(cached)

    index = load_chunk_index(slug)
    if index and index.get("n_chunks"):
        parts = [(_staged_dir(spec.inflow_type), gids)
                 for spec, gids in pool_chunk_specs(slug)]
    else:
        parts = [(out_dir, None)]

    inflow_by_real: dict[int, pd.DataFrame] = {}
    for part_dir, gids in parts:
        h5 = part_dir / "catchment_inflow_mgd.hdf5"
        if not h5.exists():
            return None
        local = Ensemble.from_hdf5(str(h5)).data_by_realization
        keys = sorted(local)
        ids = [int(g) for g in gids] if gids is not None else keys
        inflow_by_real.update({int(g): local[k] for k, g in zip(keys, ids)})

    ordered = sorted(inflow_by_real)
    ref = load_historical_flows(gage=False, period="full")
    ref_daily = ref.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    H, axes = _hazard_block(
        inflow_by_real, ordered, DEFAULT_NYC_INFLOW_NODES,
        daily_to_monthly(ref_daily, agg="mean"), ref_daily.to_numpy(dtype=float),
    )
    rows = np.arange(len(ordered))
    save_hazard_image(cached, H=H, hazard_axes=axes,
                      realization_ids=ordered, selected_rows=rows)
    return {"H": H, "hazard_axes": list(axes), "chosen_axes": list(axes),
            "realization_ids": np.asarray(ordered, dtype=int), "selected_rows": rows}


def _historic_hazard_points(n_years: int) -> dict:
    """Hazard image of the historical record, as its rolling ``L``-year windows.

    The ``historic`` design searches on ONE continuous trace, so it stages no
    ensemble and has no hazard image — yet the mechanism test needs its search
    coverage, and it is the design the prediction is sharpest for. The hazard
    content the search actually saw is the set of L-year windows the record
    contains, so the record is imaged as its water-year-aligned rolling windows
    (1-year step; windows truncated to a common length so the POT/SSI operators
    see rectangular input). This is a modeling choice and is reported as one.
    """
    from scengen.hazard_filling import daily_to_monthly
    from scengen.hazard_metrics import (
        DEFAULT_NYC_INFLOW_NODES, compute_candidate_hazard_image,
    )
    from src.load.historical_flows import load_historical_flows

    ref = load_historical_flows(gage=False, period="full")
    agg = ref.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    years = sorted({t.year for t in agg.index})
    windows = []
    for y0 in years:
        start = pd.Timestamp(year=y0, month=10, day=1)
        end = pd.Timestamp(year=y0 + n_years, month=10, day=1)
        if start < agg.index[0] or end > agg.index[-1]:
            continue
        windows.append(agg.loc[start:end - pd.Timedelta(days=1)])
    if not windows:
        return {}
    n_days = min(len(w) for w in windows)
    daily = np.vstack([w.to_numpy(dtype=float)[:n_days] for w in windows])
    monthly = np.vstack([daily_to_monthly(w.iloc[:n_days], agg="mean") for w in windows])
    ref_daily = agg.to_numpy(dtype=float)
    H, axes = compute_candidate_hazard_image(
        monthly, daily, daily_to_monthly(agg, agg="mean"), ref_daily,
    )
    rows = np.arange(len(windows))
    return {"H": H, "hazard_axes": list(axes), "chosen_axes": list(axes),
            "realization_ids": rows, "selected_rows": rows}


def search_hazard_image(design, draw: int) -> dict | None:
    """Hazard coordinates of a design's SEARCH ensemble (the coverage it achieved).

    Args:
        design: The ``ScenarioDesign``.
        draw: Ensemble-draw index.

    Returns:
        A hazard-image dict whose ``H[selected_rows]`` are the search ensemble's
        coordinates, or ``None`` when the design's ensemble is not staged.
    """
    if design.construction == "preset" and design.n_realizations == 1:
        return _historic_hazard_points(config.SCENARIO_YEARS)
    slug = (design.pool_slug(draw) if design.construction == "pool_resample"
            else design.search_ensemble_slug(draw))
    if slug is None or not _staged_dir(slug).exists():
        return None
    return _compute_hazard_image(slug)


###############################################################################
# Per-design analysis
###############################################################################

@dataclass
class DesignResult:
    """Everything scenario discovery learned about one scenario design."""

    design: str
    solution_id: int
    compromise: dict
    model: FailureModel
    shifts: pd.DataFrame
    stats: dict
    bins: pd.DataFrame
    X: np.ndarray          # (R, m) E_test hazard coords, rank space
    H: np.ndarray          # (R, m) E_test hazard coords, raw units
    y: np.ndarray          # (R,) failure labels
    deficit: np.ndarray | None
    n_search: int


def align_hazard_to_cube(raw: rob.RawCube, image: dict,
                         axis_idx: list[int]) -> np.ndarray:
    """Join the hazard image to the re-eval cube ON ``realization_id``.

    Never positionally: the cube's realization axis is the sorted union of ids
    actually simulated, which need not be the image's row order (and will not be
    if any re-eval batch failed). A positional join here would silently attach the
    wrong hazard coordinates to every failure label and quietly invalidate the
    entire mechanism test.

    Returns:
        ``(R_cube, len(axis_idx))`` raw hazard coordinates aligned to
        ``raw.realization_ids``.

    Raises:
        KeyError: If a re-evaluated realization has no hazard coordinates.
    """
    rid_to_row = {int(r): i for i, r in enumerate(image["realization_ids"])}
    missing = [r for r in raw.realization_ids if int(r) not in rid_to_row]
    if missing:
        raise KeyError(
            f"{len(missing)} re-evaluated realization(s) have no hazard "
            f"coordinates in the E_test hazard image (e.g. {missing[:5]}). The "
            f"hazard image and the re-eval ensemble are not the same ensemble."
        )
    rows = [rid_to_row[int(r)] for r in raw.realization_ids]
    return np.asarray(image["H"], dtype=float)[np.ix_(rows, axis_idx)]


def discover_for_design(design_name: str, raw: rob.RawCube, etest: dict,
                        screen: dict, draw: int = 0) -> DesignResult:
    """Run scenario discovery + the mechanism test for one design.

    Args:
        design_name: Registered scenario-design name.
        raw: The design's re-eval cube on E_test.
        etest: E_test's hazard image.
        screen: Output of :func:`screen_hazard_axes` on E_test's image.
        draw: Ensemble-draw index for resolving the search ensemble.

    Returns:
        A :class:`DesignResult`.
    """
    axes = screen["retained"]
    H = align_hazard_to_cube(raw, etest, screen["retained_idx"])       # raw units
    H_ref = np.asarray(etest["H"], dtype=float)[:, screen["retained_idx"]]
    X = cdf_transform(H, H_ref)                                        # rank space

    compromise = select_compromise(raw)
    y = failure_matrix(raw)[compromise["index"]]                       # (R,)

    model = fit_failure_classifier(X, y, axes)
    shifts = hazard_shift_stats(H, y, axes)

    # -- the mechanism test -------------------------------------------------
    deficit, n_search = None, 0
    design = get_scenario_design(design_name)
    img = search_hazard_image(design, draw)
    if img is None:
        warnings.warn(
            f"[{design_name}] search ensemble not staged; the coverage-deficit "
            f"mechanism test is skipped for this design (discovery still ran)."
        )
        stats, bins = {"verdict": "search ensemble unavailable"}, pd.DataFrame()
    else:
        img_axes = list(img["hazard_axes"])
        if any(a not in img_axes for a in axes):
            warnings.warn(
                f"[{design_name}] its search hazard image lacks axes "
                f"{[a for a in axes if a not in img_axes]}; mechanism test skipped."
            )
            stats, bins = {"verdict": "search hazard axes incompatible"}, pd.DataFrame()
        else:
            cols = [img_axes.index(a) for a in axes]
            H_search = np.asarray(img["H"], dtype=float)[
                np.ix_(np.asarray(img["selected_rows"], dtype=int), cols)]
            n_search = len(H_search)
            # Both sets mapped into E_TEST's CDF space: one common geometry.
            deficit = coverage_deficit(X, cdf_transform(H_search, H_ref))
            stats, bins = deficit_association(deficit, y, X_test=X, n_search=n_search)

    stats = {"design": design_name, "solution_id": compromise["solution_id"],
             "n_search": n_search, **stats}
    return DesignResult(
        design=design_name, solution_id=compromise["solution_id"],
        compromise=compromise, model=model, shifts=shifts, stats=stats, bins=bins,
        X=X, H=H, y=y, deficit=deficit, n_search=n_search,
    )


###############################################################################
# Figures
###############################################################################

def plot_factor_map(res: DesignResult, slug: str) -> Path | None:
    """Factor map on the top-2 hazard axes + the factor-importance bars.

    Left: the classifier's predicted failure-probability surface over the two most
    important retained hazard axes (remaining axes held at their E_test median,
    i.e. 0.5 in rank space), with the actual E_test realizations overlaid and
    colored by success/failure. Right: factor importances over the retained axes.
    """
    axes_n = res.model.axes
    if len(axes_n) < 2 or res.model.predict_proba is None:
        return None
    order = np.argsort(res.model.importances)[::-1]
    a1, a2 = int(order[0]), int(order[1])

    g = np.linspace(0.0, 1.0, GRID_RES)
    G1, G2 = np.meshgrid(g, g)
    grid = np.full((G1.size, len(axes_n)), 0.5)
    grid[:, a1] = G1.ravel()
    grid[:, a2] = G2.ravel()
    P = np.asarray(res.model.predict_proba(grid), dtype=float).reshape(G1.shape)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})
    cs = ax.contourf(G1, G2, P, levels=np.linspace(0, 1, 11),
                     cmap="RdYlBu_r", vmin=0, vmax=1, alpha=0.85)
    fail = res.y.astype(bool)
    ax.scatter(res.X[~fail, a1], res.X[~fail, a2], s=14, c="white",
               edgecolors="0.25", linewidths=0.5, label="E_test: satisficing")
    ax.scatter(res.X[fail, a1], res.X[fail, a2], s=18, c="black", marker="x",
               linewidths=0.9, label="E_test: failure")
    ax.set_xlabel(f"{axes_n[a1]}  (E_test CDF rank)")
    ax.set_ylabel(f"{axes_n[a2]}  (E_test CDF rank)")
    ax.set_title(f"{res.design}: predicted failure probability\n"
                 f"(solution {res.solution_id}; {res.model.backend})")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    fig.colorbar(cs, ax=ax, label="P(failure)")

    pos = np.arange(len(axes_n))
    axb.barh(pos, res.model.importances, color="steelblue")
    axb.set_yticks(pos)
    axb.set_yticklabels(axes_n, fontsize=8)
    axb.invert_yaxis()
    axb.set_xlabel("factor importance")
    axb.set_title("Hazard-axis importance (screened axes)")
    fig.tight_layout()

    out = config.figure_dir_for(res.design, slug, _FIG_KIND) / "factor_map"
    save_figure(fig, out)
    plt.close(fig)
    return out


def plot_mechanism(results: list[DesignResult], slug: str) -> Path | None:
    """THE mechanism figure: failure rate vs coverage-deficit decile, per design.

    A rising line = that design's policies failed where its search ensemble left
    hazard space uncovered (the prediction). A flat line = no association (a null,
    reported as such).
    """
    usable = [r for r in results if r.deficit is not None and not r.bins.empty]
    if not usable:
        return None
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(12.0, 4.8),
                                  gridspec_kw={"width_ratios": [1.3, 1.0]})
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(usable):
        auc = r.stats.get("auc", float("nan"))
        exc = r.stats.get("auc_excess", float("nan"))
        ax.plot(r.bins["deficit_mid"], r.bins["failure_rate"], marker="o", lw=1.6,
                color=cmap(i % 10),
                label=f"{r.design} (AUC={auc:.2f}, excess={exc:+.2f})")
    ax.set_xlabel("Coverage deficit: distance in hazard space to the nearest\n"
                  "SEARCH-ensemble member (E_test CDF rank space)")
    ax.set_ylabel("Failure rate on E_test")
    ax.set_title("Mechanism test: do policies fail where their design\n"
                 "under-covered hazard space? (rising = yes)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, frameon=False)

    # The verdict panel: AUC against its RANDOM-COVERAGE null, which is what the
    # claim is actually read off. The raw AUC alone is inflated by the geometry of
    # the hazard manifold's boundary (see random_coverage_null).
    pos = np.arange(len(usable))
    axb.barh(pos, [r.stats.get("auc_excess", np.nan) for r in usable],
             color=[cmap(i % 10) for i in range(len(usable))])
    for i, r in enumerate(usable):
        sd_null = r.stats.get("auc_null_std", np.nan)
        if np.isfinite(sd_null):
            axb.errorbar(r.stats.get("auc_excess", np.nan), i, xerr=2 * sd_null,
                         fmt="none", ecolor="0.3", capsize=3, lw=1.0)
    axb.axvline(0.0, color="black", lw=1.0)
    axb.set_yticks(pos)
    axb.set_yticklabels([r.design for r in usable], fontsize=8)
    axb.invert_yaxis()
    axb.set_xlabel("AUC - random-coverage null  (>0 = mechanism supported)")
    axb.set_title("Excess association over random coverage\n(bars: +/-2 SD of the null)")
    fig.tight_layout()

    out = config.figure_dir_for("comparison", slug, _FIG_KIND) / "coverage_deficit_vs_failure"
    save_figure(fig, out)
    plt.close(fig)
    return out


###############################################################################
# Orchestration
###############################################################################

def _resolve_reeval_dir(design_name: str, slug: str, tag: str,
                        seed: int | None) -> Path | None:
    """Locate a design's re-eval dir, tolerating the per-seed subdir layout."""
    base = config.OUTPUTS_DIR / design_name / slug / "reeval" / tag
    cands = [base / f"seed_{seed:02d}"] if seed is not None else [base]
    if seed is None:
        cands += sorted(base.glob("seed_*"))
    for c in cands:
        if any((c / f).exists() for f in ("reeval_raw.parquet", "reeval_raw.csv.gz")):
            return c
    return None


def run(formulation: str, designs: list[str], reeval_tag: str | None,
        seed: int | None, draw: int) -> dict:
    """Run scenario discovery + the mechanism test across scenario designs.

    Args:
        formulation: Formulation identifier (resolves the moea slug).
        designs: Scenario-design names to analyze; unrun designs are skipped.
        reeval_tag: Re-eval ensemble preset (defaults to the configured E_test).
        seed: Optional MOEA seed subdir.
        draw: Ensemble-draw index used to resolve each design's search ensemble.

    Returns:
        Dict summarizing what ran and what was written.
    """
    from src.ensembles import get_ensemble_spec
    from src.reeval_core import reeval_tag as tag_of

    apply_style()
    spec = get_ensemble_spec(reeval_tag) if reeval_tag else config.REEVAL_ENSEMBLE_SPEC
    tag = tag_of(spec)
    slug = config.derive_slug(formulation)

    # After the E_test guard, so a failed pre-flight leaves no empty output dir.
    etest = load_etest_hazard_image(spec)
    screen = screen_hazard_axes(etest["H"], etest["hazard_axes"])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[scenario_discovery] E_test='{spec.inflow_type}' "
          f"R={etest['H'].shape[0]} | retained hazard axes: {screen['retained']} "
          f"(dropped degenerate: {screen['degenerate']})")

    pd.DataFrame([{
        "axis": a,
        "retained": a in screen["retained"],
        "degenerate": a in screen["degenerate"],
        "cluster": next((i for i, c in enumerate(screen["clusters"]) if a in c), -1),
    } for a in etest["hazard_axes"]]).to_csv(TABLE_DIR / "axis_screen.csv", index=False)

    results: list[DesignResult] = []
    for name in designs:
        rdir = _resolve_reeval_dir(name, slug, tag, seed)
        if rdir is None:
            warnings.warn(f"[{name}] no re-eval matrix under "
                          f"outputs/{name}/{slug}/reeval/{tag} -- skipping.")
            continue
        raw = rob.load_raw(rdir)
        if not raw.is_ensemble or raw.n_realizations <= 1:
            warnings.warn(f"[{name}] single-trace re-eval (R={raw.n_realizations}): "
                          f"failure labels are undefined per realization -- skipping.")
            continue
        try:
            res = discover_for_design(name, raw, etest, screen, draw=draw)
        except (KeyError, ValueError) as exc:
            warnings.warn(f"[{name}] scenario discovery failed: {exc}")
            continue
        results.append(res)
        plot_factor_map(res, slug)
        print(f"[scenario_discovery] {name}: solution {res.solution_id}, "
              f"failures {int(res.y.sum())}/{len(res.y)} | "
              f"top axis '{res.model.axes[int(np.argmax(res.model.importances))]}' | "
              f"{res.stats.get('verdict')}")

    if not results:
        warnings.warn("No design produced a usable re-eval cube; nothing written.")
        return {"designs": [], "tables": [], "n_designs": 0}

    imp = pd.DataFrame(
        [{"design": r.design, "solution_id": r.solution_id,
          "backend": r.model.backend, "train_accuracy": r.model.train_accuracy,
          **dict(zip(r.model.axes, r.model.importances))} for r in results]
    )
    imp.to_csv(TABLE_DIR / "factor_importances.csv", index=False)

    pd.concat([r.shifts.assign(design=r.design) for r in results]) \
        .to_csv(TABLE_DIR / "hazard_shift_ks.csv", index=False)
    pd.DataFrame([r.stats for r in results]) \
        .to_csv(TABLE_DIR / "coverage_deficit_association.csv", index=False)
    pd.DataFrame([{"design": r.design, **r.compromise} for r in results]) \
        .to_csv(TABLE_DIR / "compromise_solutions.csv", index=False)
    bins = [r.bins.assign(design=r.design) for r in results if not r.bins.empty]
    if bins:
        pd.concat(bins).to_csv(TABLE_DIR / "coverage_deficit_deciles.csv", index=False)

    (TABLE_DIR / "scenario_discovery_meta.json").write_text(json.dumps({
        "formulation": formulation, "moea_slug": slug, "etest": spec.inflow_type,
        "reeval_tag": tag, "seed": seed, "ensemble_draw": draw,
        "compromise_rule": COMPROMISE_RULE, "classifier_backend": results[0].model.backend,
        "gbc": {"n_estimators": GBC_N_ESTIMATORS, "max_depth": GBC_MAX_DEPTH,
                "learning_rate": GBC_LEARNING_RATE},
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "hazard_axes_all": list(etest["hazard_axes"]),
        "hazard_axes_retained": screen["retained"],
        "redundancy_clusters": screen["clusters"],
        "residual_max_rho": screen["residual_max_rho"],
        "designs": [r.design for r in results],
    }, indent=2))

    plot_mechanism(results, slug)
    print(f"[scenario_discovery] tables -> {TABLE_DIR}")
    return {"designs": [r.design for r in results], "n_designs": len(results),
            "tables": sorted(p.name for p in TABLE_DIR.glob("*"))}


def main() -> None:
    """CLI. Identifiers only — settings live in module constants / env."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--formulation", default="ffmp")
    p.add_argument("--reeval-tag", default=None,
                   help="Re-eval ensemble preset id (default: configured E_test).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--draw", type=int, default=0,
                   help="Ensemble-draw index of the search ensembles.")
    p.add_argument("--designs", default=None,
                   help="Comma-separated design ids (default: campaign designs).")
    args = p.parse_args()

    designs = ([d.strip() for d in args.designs.split(",") if d.strip()]
               if args.designs else campaign_designs())
    run(args.formulation, designs, args.reeval_tag, args.seed, args.draw)


if __name__ == "__main__":
    main()
