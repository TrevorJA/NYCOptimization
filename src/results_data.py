"""
results_data.py - Shared data substrate for the results-figure sequence.

Loads, once, everything the cross-design results figures consume: each campaign
design's per-SOW raw cube (``robustness.RawCube``), its robustness scorecard,
and the FFMP incumbent's per-SOW vector aligned to the cube's SOW labels. Also
provides the satisficing-criterion helpers the figures share: joint/univariate
satisficing under ANY threshold vector (pure post-processing on the persisted
cube -- no simulation), conjunction-collapse curves, and threshold-response
curves. Thresholds default to the ``reeval_raw_meta.json`` snapshot so figures
cannot drift from the adopted criteria; alternative criterion sets pass
explicit vectors instead.

Satisficing here is always the SOW-counting Starr domain criterion of
``src.robustness``, re-expressed under varied threshold vectors and axis
subsets, using ``math.inf`` bounds to make an axis non-binding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import config
from src import robustness as rob
from src.plotting.style import DESIGN_ORDER


@dataclass
class DesignResults:
    """One design's re-evaluation artifacts on the common E_test tag.

    Attributes:
        design: Scenario-design name (output-tree partition).
        path: The re-eval leaf directory holding the artifacts.
        raw: The per-SOW cube (``S`` solutions x ``G`` SOWs x ``M`` objectives).
        scorecard: ``robustness_scorecard.csv`` indexed by ``solution_id``.
        incumbent: FFMP status-quo per-SOW matrix ``(G, M)`` aligned to
            ``raw.sow_labels`` by SOW label (NaN where unscored), or None when
            the run has no ``baseline/`` cube.
    """

    design: str
    path: Path
    raw: rob.RawCube
    scorecard: pd.DataFrame
    incumbent: Optional[np.ndarray]


def load_design_results(
    reeval_tag: str,
    slug: str = "ffmp_obj8",
    designs: Sequence[str] = DESIGN_ORDER,
    outputs_root: Optional[Path] = None,
) -> dict[str, DesignResults]:
    """Load every design's re-eval cube + scorecard + aligned incumbent.

    Args:
        reeval_tag: The common held-out ensemble tag (carries subset status --
            figures must not re-annotate it).
        slug: The moea slug shared by the campaign runs.
        designs: Designs to load, in display order; a design with no re-eval on
            this tag raises (the figure sequence needs all of them).
        outputs_root: Root of the output tree; defaults to ``config.OUTPUTS_DIR``.

    Returns:
        ``{design: DesignResults}`` in the order of ``designs``.
    """
    root = Path(outputs_root) if outputs_root is not None else config.OUTPUTS_DIR
    out: dict[str, DesignResults] = {}
    for design in designs:
        leaf = root / design / slug / "reeval" / reeval_tag
        raw = rob.load_raw(leaf)
        scorecard = pd.read_csv(leaf / "robustness_scorecard.csv",
                                index_col="solution_id")
        incumbent = None
        baseline_dir = leaf / "baseline"
        if (baseline_dir / "reeval_raw_meta.json").exists():
            incumbent = rob._aligned_baseline(raw, rob.load_raw(baseline_dir))
        out[design] = DesignResults(design, leaf, raw, scorecard, incumbent)
    return out


def load_threshold_snapshot(reeval_tag: str, slug: str = "ffmp_obj8",
                            designs: Sequence[str] = DESIGN_ORDER,
                            outputs_root: Optional[Path] = None) -> tuple:
    """The adopted objective order + threshold/kind snapshot, WITHOUT the cube.

    Reads only ``reeval_raw_meta.json`` (a few KB) from the first design that
    has one, so scorecard-backed figures can print the exact criteria in their
    footer without paying for -- or declaring a need on -- the per-SOW cubes.
    It is the same snapshot ``load_raw`` would expose, so the footer cannot
    drift from the criteria the scores were computed under.

    Args:
        reeval_tag: The held-out ensemble tag.
        slug: The moea slug shared by the campaign runs.
        designs: Designs to try, in order.
        outputs_root: Root of the output tree; defaults to ``config.OUTPUTS_DIR``.

    Returns:
        ``(obj_names, thresholds, kinds)``.

    Raises:
        FileNotFoundError: No design carries a re-eval meta on this tag.
    """
    import json

    root = Path(outputs_root) if outputs_root is not None else config.OUTPUTS_DIR
    for design in designs:
        meta_path = (root / design / slug / "reeval" / reeval_tag
                     / "reeval_raw_meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            return (list(meta["obj_names"]), dict(meta["thresholds"]),
                    dict(meta["kinds"]))
    raise FileNotFoundError(
        f"no reeval_raw_meta.json under any of {list(designs)} for slug "
        f"'{slug}' and tag '{reeval_tag}'."
    )


###############################################################################
# Criterion vectors
###############################################################################

def criterion_thresholds(res: DesignResults, cset) -> dict:
    """A criterion set's full threshold vector for one design's cube.

    The one-liner every figure shares, so the kinds plumbing (non-member axes
    non-binding at ``+/-inf``) is never re-implemented at a call site.

    Args:
        res: The design's loaded results.
        cset: A :class:`src.satisficing_criteria.CriterionSet`.
    """
    return cset.thresholds(res.raw.thresholds, res.raw.kinds)


def relax_axes(thresholds: dict, kinds: dict, drop: Iterable[str]) -> dict:
    """Return ``thresholds`` with the axes in ``drop`` made non-binding.

    A dropped "ge" axis gets ``-inf``, a dropped "le" axis ``+inf`` -- every
    finite value passes, so the joint criterion simply ignores the axis while
    :func:`robustness._satisfaction_cube`'s missing-threshold guard stays armed.
    """
    out = dict(thresholds)
    for name in drop:
        out[name] = -math.inf if kinds[name] == "ge" else math.inf
    return out


###############################################################################
# Satisficing under arbitrary criteria (SOW-counting, per solution)
###############################################################################

def satisfaction(raw: rob.RawCube, thresholds: Optional[dict] = None,
                 kinds: Optional[dict] = None) -> np.ndarray:
    """Per-cell pass/fail ``(S, G, M)`` under a criterion vector (meta default)."""
    return rob._satisfaction_cube(raw, thresholds=thresholds, kinds=kinds)


def incumbent_satisfaction(res: DesignResults,
                           thresholds: Optional[dict] = None) -> Optional[np.ndarray]:
    """Incumbent per-cell pass/fail ``(G, M)``, or None without a baseline cube."""
    if res.incumbent is None:
        return None
    sat = rob._satisfy(res.incumbent[np.newaxis, :, :], res.raw.obj_names,
                       thresholds or res.raw.thresholds, res.raw.kinds)
    return sat[0]


def joint_fraction(sat: np.ndarray) -> np.ndarray:
    """Per-solution joint (all-axes) satisficing fraction from ``(S, G, M)``."""
    return sat.all(axis=2).mean(axis=1)


def univariate_fraction(sat: np.ndarray) -> np.ndarray:
    """Per-solution per-axis satisficing fractions ``(S, M)`` from ``(S, G, M)``."""
    return sat.mean(axis=1)


###############################################################################
# Conjunction-collapse curves
###############################################################################

def collapse_curve(sat: np.ndarray, obj_names: Sequence[str],
                   order: Sequence[str]) -> pd.DataFrame:
    """Joint satisficing as axes are conjoined one at a time, in ``order``.

    At each depth d the criterion is the conjunction of the first d axes of
    ``order``. Two statistics per depth:

    - ``best_policy``: the best single solution's joint SOW fraction -- one
      policy must clear every conjoined axis in the same SOW.
    - ``any_policy``: the fraction of SOWs where SOME solution clears the
      conjunction (the attainability ceiling; the gap to ``best_policy`` is
      cross-SOW policy conflict).

    Args:
        sat: Pass/fail cube ``(S, G, M)``.
        obj_names: Axis names matching ``sat``'s last dimension.
        order: The conjunction order (a permutation or subset of ``obj_names``).

    Returns:
        One row per depth: ``axis`` (the axis added), ``depth`` (1-based),
        ``best_policy``, ``any_policy``.
    """
    idx = {n: k for k, n in enumerate(obj_names)}
    rows = []
    running = np.ones(sat.shape[:2], dtype=bool)   # (S, G)
    for depth, name in enumerate(order, start=1):
        running = running & sat[:, :, idx[name]]
        rows.append({
            "axis": name,
            "depth": depth,
            "best_policy": float(running.mean(axis=1).max()),
            "any_policy": float(running.any(axis=0).mean()),
        })
    return pd.DataFrame(rows)


#: Fixed global conjunction order for collapse figures: pooled difficulty
#: across designs, easiest first, so the three designs' curves are directly
#: comparable at every depth.
COLLAPSE_ORDER: tuple[str, ...] = (
    "nyc_delivery_reliability_annual",
    "nyc_delivery_deficit_p99_pct",
    "montague_flow_reliability_annual",
    "montague_flow_deficit_p99_pct",
    "nj_delivery_reliability_annual",
    "nyc_storage_min_p01_pct",
    "downstream_flood_exceedance_annual",
    "trenton_flow_reliability_annual",
)


###############################################################################
# Threshold-response curves
###############################################################################

def threshold_response(values: np.ndarray, kind: str,
                       grid: np.ndarray) -> np.ndarray:
    """Satisficing fraction of a per-SOW value vector at each candidate threshold.

    Args:
        values: Per-SOW objective values ``(G,)`` (NaN = unsatisfied, per the
            scoring rule).
        kind: "ge" or "le" -- the satisficing direction.
        grid: Candidate threshold values (natural units).

    Returns:
        ``fraction_passing(grid)``: for "ge" the survival curve, for "le" the
        CDF, so the value always reads as the SOW fraction meeting a threshold
        placed there.
    """
    v = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.full(len(grid), np.nan)
    if kind == "ge":
        return np.array([(v >= t).sum() / n for t in grid])
    return np.array([(v <= t).sum() / n for t in grid])
