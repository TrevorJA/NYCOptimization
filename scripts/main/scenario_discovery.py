"""scenario_discovery.py - Scenario discovery on E_test failures, IN HAZARD SPACE.

This is the **mechanism test** for the study's central claim, not a decorative
post-processing step. The claim is that covering *hazard space* during MOEA
search produces policies that are more robust on the held-out test ensemble
``E_test``. The falsifiable prediction it implies is::

    A design's policies should FAIL on E_test in the hazard region that design
    UNDER-COVERED during search.

so the script does two things, per scenario design:

1. **Scenario discovery.** Label each E_test state of the world (SOW)
   success/failure for the design's compromise policy (the multivariate Starr
   domain criterion on the per-SOW annual-unit objective values — the same
   all-criteria conjunction that defines the primary robustness metric, so
   discovery inherits exactly the robustness criteria, as is standard), then fit
   a gradient-boosted classifier of failure on the SOW's HAZARD coordinates
   (the within-SOW mean of its realizations' realized-sequence descriptors).
   Reports factor importances, a 2-D factor map, and the failure/success
   distributional shift per axis (two-sample KS).

2. **The mechanism test.** For each E_test SOW, compute the **coverage
   deficit** — the distance, in the E_test hazard image's empirical-CDF/rank
   space, to the nearest member of that design's SEARCH ensemble — and test
   whether failure probability is POSITIVELY associated with it (AUC of deficit
   as a failure predictor, its excess over a RANDOM-COVERAGE null, an empirical
   p-value, the logistic slope, and failure rate by deficit decile).
   Hazard-filling designs, having filled the space, should show NO excess
   association; ``historic`` / ``fixed_probabilistic`` should show failures
   concentrating where their ensembles left hazard space unsampled. A null is a
   real, reportable result and is written as such.

   The random-coverage null is load-bearing, not a nicety: nearest-neighbor
   distance is systematically larger near the boundary of the hazard manifold, and
   failures sit in a tail, so a uniformly-covering ensemble still scores AUC ~ 0.62
   from geometry alone. The verdict is therefore read off AUC MINUS its null, never
   off the raw AUC (see :func:`random_coverage_null`).

Where this sits. The study's PRIMARY scenario-discovery factor maps run in the
sampled DU input space (theta), via ``src.factor_mapping`` and
``scripts/main/factor_mapping_run.py`` -- the standard setting of Hadjimichael
et al. (2020) and Gold et al. (2022). THIS script is the hazard-space
supplement: the mechanism test lives here because the coverage hypothesis is
stated in hazard space -- the only space in which "the design under-covered
here" is even definable -- and the hazard-space classifier/factor map is the
supplemental view of the same labels.

Classifier. ``src.factor_mapping.fit_classifier`` (gradient boosting, 250
trees, ``max_depth=2``, ``learning_rate=0.1``, stratified-CV AUC reported).
Trees are monotone-invariant, so the fit is done in rank space (where the
factor map is plotted); importances are unchanged by that choice.

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

Outputs (namespaced by slug, re-eval tag, and label -- runs never overwrite
each other):
  * tables  -> ``outputs/comparison/{slug}/{tag}/scenario_discovery/{label}/*.csv``
  * figures -> ``{figure root}/{design or comparison}/{slug}/scenario_discovery/{tag}/{label}/``

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
from dataclasses import dataclass
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
from src import factor_mapping as fm  # noqa: E402
from src import robustness as rob  # noqa: E402
from src.factor_mapping import (  # noqa: E402
    FactorMapFit, align_hazard_to_cube, cdf_transform, fit_classifier,
    screen_hazard_axes, select_compromise,
)
from src.plotting.style import apply_style, save_figure  # noqa: E402
from src.satisficing_criteria import criterion_by_key, focal_criterion  # noqa: E402
from src.scenario_designs import campaign_designs, get_scenario_design  # noqa: E402

try:  # sklearn is optional here; used only for the descriptive logistic slope.
    from sklearn.linear_model import LogisticRegression

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only on a stripped env
    _HAS_SKLEARN = False


###############################################################################
# Settings (module constants; env-overridable. NO CLI value flags — repo rule)
###############################################################################

#: Rule selecting the per-design analysis policy. Scenario discovery is run on a
#: small number of COMPROMISE solutions, not the whole front (Kasprzyk et al.
#: 2013). Both rules are always computed and written to
#: ``compromise_solutions.csv``; this one names the policy actually analyzed.
#:   "best_satisficing" -- highest satisficing fraction under the label's
#:                         criterion set; ties broken by min-distance-to-ideal.
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

_FIG_KIND = "scenario_discovery"


###############################################################################
# Labels
###############################################################################

def resolve_sd_label() -> str:
    """The label the classifier is fit to, from ``NYCOPT_SD_LABEL``.

    ``criterion:<set_key>`` (default ``criterion:<focal set>``) labels a SOW a
    failure when the named criterion set's conjunction is False -- "where does
    this policy fail this stakeholder framing's standard?". ``regret`` is the
    incumbent-relative label -- "where would the Decree parties REGRET having
    adopted this policy rather than keeping the FFMP?". They are different
    questions and can localise in different hazard regions. The all-axes
    reference conjunction stays reachable as ``criterion:reference_all8``.

    Raises:
        SystemExit: For an unrecognized label form or unknown criterion key.
    """
    label = os.environ.get("NYCOPT_SD_LABEL",
                           f"criterion:{focal_criterion().key}")
    if label == "regret":
        return label
    if label.startswith("criterion:"):
        try:
            criterion_by_key(label.split(":", 1)[1])
        except KeyError as exc:
            sys.exit(f"[scenario_discovery] {exc}")
        return label
    sys.exit(f"[scenario_discovery] unknown NYCOPT_SD_LABEL={label!r}; "
             f"expected 'criterion:<set_key>' or 'regret'.")


def label_thresholds(label: str, raw: rob.RawCube) -> dict | None:
    """The criterion-set threshold vector a ``criterion:`` label implies."""
    if not label.startswith("criterion:"):
        return None
    cset = criterion_by_key(label.split(":", 1)[1])
    return cset.thresholds(raw.thresholds, raw.kinds)


def _label_matrix(raw: rob.RawCube, baseline: rob.RawCube | None,
                  label: str) -> np.ndarray:
    """The ``(S, G)`` failure/regret label matrix for ``label``.

    Raises:
        SystemExit: If the regret label is requested without a status-quo cube.
            Falling back to a satisficing label would answer a different
            question under the same figure caption.
    """
    if label.startswith("criterion:"):
        return fm.failure_matrix(raw, label_thresholds(label, raw))
    if baseline is None:
        sys.exit(
            "[scenario_discovery] NYCOPT_SD_LABEL=regret needs the status-quo "
            "re-eval cube beside the run (workflow step 05 with the SAME "
            "NYCOPT_REEVAL_ENSEMBLE_PRESET as step 08). Refusing to fall back to "
            "a satisficing label -- that answers a different question."
        )
    return fm.regret_matrix(raw, baseline)


def _load_baseline(reeval_dir: Path) -> rob.RawCube | None:
    """Load the status-quo cube written beside a re-eval dir, or None."""
    bdir = Path(reeval_dir) / "baseline"
    if any((bdir / f).exists() for f in ("reeval_raw.parquet", "reeval_raw.csv.gz")):
        return rob.load_raw(bdir)
    return None


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
    """Distance from each E_test SOW to the nearest SEARCH-ensemble member.

    Both point sets must already be in the SAME normalized hazard space (E_test's
    empirical-CDF/rank space — see :func:`cdf_transform`). This is the per-SOW
    operationalization of "the design under-covered this part of hazard space":
    large deficit = no search scenario resembled this state's typical realized
    hazard.

    Args:
        X_test: ``(G, m)`` E_test SOW hazard coordinates, normalized.
        X_search: ``(n, m)`` search-ensemble hazard coordinates, normalized.

    Returns:
        Length-``G`` array of nearest-neighbor distances.
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
        "n_sow": int(y.size),
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
        "verdict": "no discrimination (all SOWs same class)",
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
    from scengen.hazard_metrics import _REFERENCE_START, DEFAULT_NYC_INFLOW_NODES
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
    from src.ensembles import get_ensemble_spec
    H, axes = _hazard_block(
        inflow_by_real, ordered, DEFAULT_NYC_INFLOW_NODES,
        daily_to_monthly(ref_daily, agg="mean"), ref_daily.to_numpy(dtype=float),
        n_years=int(get_ensemble_spec(slug).realization_years),
    )
    rows = np.arange(len(ordered))
    save_hazard_image(cached, H=H, hazard_axes=axes,
                      realization_ids=ordered, selected_rows=rows,
                      reference_start=_REFERENCE_START)
    return {"H": H, "hazard_axes": list(axes), "chosen_axes": list(axes),
            "realization_ids": np.asarray(ordered, dtype=int), "selected_rows": rows}


def _historic_hazard_points(n_years: int) -> dict:
    """Hazard image of the historical record, as its rolling ``L``-year windows.

    The ``historic`` design searches on ONE continuous trace, so it stages no
    ensemble and has no hazard image — yet the mechanism test needs its search
    coverage, and it is the design the prediction is sharpest for. The hazard
    content the search actually saw is the set of L-year windows the record
    contains, so the record is imaged as its rolling windows anchored at the
    month every scenario window starts in (``config.ENSEMBLE_START_DATE``;
    1-year step; windows truncated to a common length so the POT/SSI operators
    see rectangular input). This is a modeling choice and is reported as one.

    Each window is scored on the pool's exact metric span: the trailing
    partial FFMP year is cut (the scored content ends May 31 of the window's
    final year) and the leading ``config.METRIC_EXCLUSION_MONTHS`` are cut by
    date from the daily series for the wet axes, so every window scores the
    same [Jun 1 year 1, May 31 year L] span the objectives use; the monthly
    series keeps its leading months as the SSI accumulation input.
    """
    from scengen.hazard_filling import daily_to_monthly
    from scengen.hazard_metrics import (
        DEFAULT_NYC_INFLOW_NODES, compute_candidate_hazard_image,
    )
    from src.load.historical_flows import load_historical_flows

    ref = load_historical_flows(gage=False, period="full")
    agg = ref.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    years = sorted({t.year for t in agg.index})
    anchor_month = pd.Timestamp(config.ENSEMBLE_START_DATE).month
    windows = []
    for y0 in years:
        start = pd.Timestamp(year=y0, month=anchor_month, day=1)
        end = pd.Timestamp(year=y0 + n_years, month=anchor_month, day=1)
        if start < agg.index[0] or end > agg.index[-1]:
            continue
        # Pool convention: score only the metric span — cut the trailing
        # partial FFMP year, so the content ends May 31 of the final year.
        cutoff = start + pd.DateOffset(months=config.METRIC_EXCLUSION_MONTHS)
        metric_end = cutoff + pd.DateOffset(years=n_years - 1)
        windows.append(agg.loc[start:metric_end - pd.Timedelta(days=1)])
    if not windows:
        return {}
    # Month counts are equal across windows by construction; day counts differ
    # only by leap days, so the daily block truncates to the common length.
    monthly = np.vstack([daily_to_monthly(w, agg="mean") for w in windows])
    cuts = [
        int((w.index < w.index[0] + pd.DateOffset(months=config.METRIC_EXCLUSION_MONTHS)).sum())
        for w in windows
    ]
    n_wet = min(len(w) - c for w, c in zip(windows, cuts))
    daily = np.vstack([w.to_numpy(dtype=float)[c:c + n_wet] for w, c in zip(windows, cuts)])
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
    model: FactorMapFit
    shifts: pd.DataFrame
    stats: dict
    bins: pd.DataFrame
    X: np.ndarray          # (G, m) E_test SOW hazard coords, rank space
    H: np.ndarray          # (G, m) E_test SOW hazard coords, raw units
    y: np.ndarray          # (G,) failure labels
    deficit: np.ndarray | None
    n_search: int


def discover_for_design(design_name: str, raw: rob.RawCube, etest: dict,
                        screen: dict, label: str, draw: int = 0,
                        baseline: rob.RawCube | None = None) -> DesignResult:
    """Run scenario discovery + the mechanism test for one design.

    Args:
        design_name: Registered scenario-design name.
        raw: The design's re-eval cube on E_test.
        etest: E_test's hazard image.
        screen: Output of :func:`screen_hazard_axes` on E_test's image.
        label: The resolved SD label (``criterion:<set_key>`` | ``regret``).
        draw: Ensemble-draw index for resolving the search ensemble.
        baseline: The status-quo cube, required when ``label`` is
            ``"regret"`` and ignored otherwise.

    Returns:
        A :class:`DesignResult`.
    """
    axes = screen["retained"]
    H = align_hazard_to_cube(raw, etest, screen["retained_idx"])       # raw units
    # The CDF reference is the SOW-level E_test cloud itself, so the SOW points
    # and the search-ensemble members land in one common rank space.
    H_ref = H
    X = cdf_transform(H, H_ref)                                        # rank space

    # The analyzed policy is chosen under the label's own criterion set, so a
    # criterion label analyzes the policy that framing favors (the compromise
    # is set-specific; the set key is recorded in the stats row).
    compromise = select_compromise(raw, rule=COMPROMISE_RULE,
                                   thresholds=label_thresholds(label, raw))
    y = _label_matrix(raw, baseline, label)[compromise["index"]]       # (G,)
    if y.all() or not y.any():
        warnings.warn(
            f"[{design_name}] label {label!r} has one class on this cube "
            f"({int(y.sum())}/{len(y)} failures) -- the criterion is degenerate "
            f"here; classifier and mechanism test will report no discrimination."
        )

    model = fit_classifier(X, y, axes, space="hazard")
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

    # The label is recorded next to every number it produced: a factor map fit to
    # regret and one fit to a criterion set look identical and mean different
    # things.
    stats = {"design": design_name, "solution_id": compromise["solution_id"],
             "label": label, "n_search": n_search, **stats}
    return DesignResult(
        design=design_name, solution_id=compromise["solution_id"],
        compromise=compromise, model=model, shifts=shifts, stats=stats, bins=bins,
        X=X, H=H, y=y, deficit=deficit, n_search=n_search,
    )


###############################################################################
# Figures
###############################################################################

def plot_factor_map(res: DesignResult, slug: str, fig_sub: Path) -> Path | None:
    """Factor map on the top-2 hazard axes + the factor-importance bars.

    The hazard-space SUPPLEMENTAL view (theta-space factor maps are the
    primary; see ``scripts/main/factor_mapping_run.py``). Left: the
    classifier's predicted failure-probability surface over the two most
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

    out = (config.figure_dir_for(res.design, slug, _FIG_KIND) / fig_sub
           / "factor_map")
    save_figure(fig, out)
    plt.close(fig)
    return out


def plot_mechanism(results: list[DesignResult], slug: str,
                   fig_sub: Path) -> Path | None:
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

    out = (config.figure_dir_for("comparison", slug, _FIG_KIND) / fig_sub
           / "coverage_deficit_vs_failure")
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
    slug = config.results_slug(tag, formulation)
    label = resolve_sd_label()
    label_slug = label.replace(":", "_")
    # Namespaced by slug/tag/label so successive runs never overwrite each
    # other, and resolved here (not at import) because the tag is run identity.
    table_dir = (config.OUTPUTS_DIR / "comparison" / slug / tag
                 / "scenario_discovery" / label_slug)
    fig_sub = Path(tag) / label_slug

    # After the E_test guard, so a failed pre-flight leaves no empty output dir.
    etest = load_etest_hazard_image(spec)
    screen = screen_hazard_axes(etest["H"], etest["hazard_axes"])
    table_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scenario_discovery] E_test='{spec.inflow_type}' "
          f"R={etest['H'].shape[0]} | label={label} | retained hazard axes: "
          f"{screen['retained']} (dropped degenerate: {screen['degenerate']})")

    pd.DataFrame([{
        "axis": a,
        "retained": a in screen["retained"],
        "degenerate": a in screen["degenerate"],
        "cluster": next((i for i, c in enumerate(screen["clusters"]) if a in c), -1),
    } for a in etest["hazard_axes"]]).to_csv(table_dir / "axis_screen.csv", index=False)

    results: list[DesignResult] = []
    meta_thresholds: dict | None = None
    for name in designs:
        rdir = _resolve_reeval_dir(name, slug, tag, seed)
        if rdir is None:
            warnings.warn(f"[{name}] no re-eval matrix under "
                          f"outputs/{name}/{slug}/reeval/{tag} -- skipping.")
            continue
        raw = rob.load_raw(rdir)
        if not raw.is_ensemble or raw.n_sow <= 1:
            warnings.warn(f"[{name}] single-trace re-eval (G={raw.n_sow}): "
                          f"per-SOW failure labels are undefined -- skipping.")
            continue
        if meta_thresholds is None:
            meta_thresholds = label_thresholds(label, raw)
        try:
            res = discover_for_design(name, raw, etest, screen, label,
                                      draw=draw, baseline=_load_baseline(rdir))
        except (KeyError, ValueError) as exc:
            warnings.warn(f"[{name}] scenario discovery failed: {exc}")
            continue
        results.append(res)
        plot_factor_map(res, slug, fig_sub)
        label_word = "regret SOWs" if label == "regret" else "failures"
        print(f"[scenario_discovery] {name}: solution {res.solution_id}, "
              f"{label_word} {int(res.y.sum())}/{len(res.y)} | "
              f"top axis '{res.model.axes[int(np.argmax(res.model.importances))]}' | "
              f"{res.stats.get('verdict')}")

    if not results:
        warnings.warn("No design produced a usable re-eval cube; nothing written.")
        return {"designs": [], "tables": [], "n_designs": 0}

    imp = pd.DataFrame(
        [{"design": r.design, "solution_id": r.solution_id,
          "backend": r.model.backend, "train_accuracy": r.model.train_accuracy,
          "cv_auc": r.model.cv_auc, "cv_auc_std": r.model.cv_auc_std,
          "cv_accuracy": r.model.cv_accuracy, "cv_note": r.model.cv_note,
          **dict(zip(r.model.axes, r.model.importances))} for r in results]
    )
    imp.to_csv(table_dir / "factor_importances.csv", index=False)

    pd.concat([r.shifts.assign(design=r.design) for r in results]) \
        .to_csv(table_dir / "hazard_shift_ks.csv", index=False)
    pd.DataFrame([r.stats for r in results]) \
        .to_csv(table_dir / "coverage_deficit_association.csv", index=False)
    pd.DataFrame([{"design": r.design, **r.compromise} for r in results]) \
        .to_csv(table_dir / "compromise_solutions.csv", index=False)
    bins = [r.bins.assign(design=r.design) for r in results if not r.bins.empty]
    if bins:
        pd.concat(bins).to_csv(table_dir / "coverage_deficit_deciles.csv", index=False)

    meta = {
        "formulation": formulation, "moea_slug": slug, "etest": spec.inflow_type,
        "reeval_tag": tag, "seed": seed, "ensemble_draw": draw,
        "label": label,
        "compromise_rule": COMPROMISE_RULE,
        "classifier_backend": results[0].model.backend,
        "gbc": {"n_estimators": fm.FM_N_TREES, "max_depth": fm.FM_MAX_DEPTH,
                "learning_rate": fm.FM_LEARNING_RATE},
        "redundancy_threshold": fm.FM_RHO_THRESHOLD,
        "hazard_axes_all": list(etest["hazard_axes"]),
        "hazard_axes_retained": screen["retained"],
        "redundancy_clusters": screen["clusters"],
        "residual_max_rho": screen["residual_max_rho"],
        "designs": [r.design for r in results],
    }
    if meta_thresholds is not None:
        meta["criterion_thresholds"] = {
            n: (None if not np.isfinite(v) else v)
            for n, v in meta_thresholds.items()
        }
    (table_dir / "scenario_discovery_meta.json").write_text(
        json.dumps(meta, indent=2))

    plot_mechanism(results, slug, fig_sub)
    print(f"[scenario_discovery] tables -> {table_dir}")
    return {"designs": [r.design for r in results], "n_designs": len(results),
            "label": label,
            "tables": sorted(p.name for p in table_dir.glob("*"))}


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
