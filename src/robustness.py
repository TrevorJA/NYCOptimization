"""
robustness.py - Offline robustness scoring from the persisted re-eval matrix.

The re-eval drivers (``src.reevaluate`` / ``src.reevaluate_mpi``) persist the raw
per-realization base-objective matrix (``reeval_raw.parquet`` + a self-describing
``reeval_raw_meta.json``) rather than a single collapsed satisficing fraction.
This module scores robustness metrics from that matrix *offline*, so different
metrics — which rank solutions differently (McPhail et al. 2018; Giuliani &
Castelletti 2016; Herman et al. 2015; Bonham et al. 2024) — are computed without
re-simulating. The matrix is the McPhail T1×T2×T3 substrate: persisting natural
units preserves enough to recompute any T1 (identity/regret/threshold), T2
(scenario subset), and T3 (moment) composition later.

Two robustness UNITS, both computable from the same cube
--------------------------------------------------------
The test ensemble E_test is ``N_theta`` deeply-uncertain states of the world (SOWs
— LHS points in the forcing space) x ``R`` stochastic realizations each. That gives
two legitimate units for a satisficing criterion, and they are different quantities:

  - the **realization unit** — every one of the ``N_theta x R`` realizations is a
    scenario, and the criterion is applied across all of them;
  - the **SOW unit** — the R realizations sharing a theta are collapsed FIRST (an
    explicit risk attitude within the state of the world), and the criterion is then
    applied across the ``N_theta`` SOWs. This is the Triangle-lineage construction
    (Herman et al. 2014; Trindade et al. 2017; Gold et al. 2022, 2023), and a
    reviewer from that school expects it.

Both are reported. The SOW path needs ``sow_ids`` in the meta (persisted by
``src.reeval_core.sow_grouping``); without it the SOW metrics are N/A, never a
silent fallback to the realization unit.

The finalized metric set. **No perfect-foresight optimization appears anywhere.**

  - **Multivariate (Starr 1962) domain criterion [PRIMARY]** — fraction of
    realizations meeting *all* thresholds jointly. The standard measure of the
    Herman (2014/2015) / Trindade (2017) / Gold (2022, 2023) lineage. Converges at
    50-300 scenarios (Bonham 2024).
  - **The same criterion on the SOW unit** (``sat_multivariate_sow``) — the R
    realizations within each theta collapsed first, then Starr across SOWs.
  - **Univariate satisficing** — its per-objective decomposition.
  - **Laplace / mean** (McPhail T3 = mean) — the risk-neutral anchor.
  - **Maximin** (McPhail T3 = worst-case) — the risk-averse anchor. Both are free,
    and their absence would be asked about: metric choice changes rankings
    (Herman 2015; McPhail 2018), so one robustness family is never sufficient.
  - **Improvement over the status quo** — SIGNED deviation from the default FFMP
    policy on the *same* ensemble, oriented so **positive = better** for every
    objective. A FIXED external reference (it does not move when designs are added
    or dropped), costing one policy simulation that workflow step 05 already
    performs.
  - **Incumbent-relative regret** — the one-sided (adverse) half of that same
    deviation, in NATURAL UNITS, plus the unit-free harm frequencies. This is the
    metric that answers "what would the Decree parties give up by adopting?", and
    it discriminates exactly where the domain criterion saturates. See the
    "Incumbent-relative regret" section below.
  - **Threshold spectrum** — satisficing vs the magnitude threshold. Robustness is
    threshold-dependent (Hadjimichael et al. 2020), and rank agreement ACROSS
    scenario designs degrades as the criterion tightens (Quinn et al. 2020), so a
    single threshold could manufacture or hide the entire design effect.
  - **Attainability screen** — which realizations no policy can win, separating a
    bad design from an impossible scenario (Shavazipour et al. 2021).
  - **Ranking-stability** — Kendall τ_b across metrics (McPhail 2020; Bonham 2024).

Deliberately absent: **regret-from-best** (set-relative and design-coupled, so
dropping one design changes every other design's score; needs 400+ scenarios and
never converges on a tail objective — Bonham 2024) and the **search-vs-test
overfitting gap** (undefined in Brodeur 2020, and structurally invalid under a
measure change). See the comments at each site.

Composable by design: ``aggregate_over_realizations(values, agg)`` is the seam —
``agg=nanmean`` is Laplace, ``agg=nanmin`` on an oriented slab is maximin, and a
percentile or Hurwicz blend is a thin wrapper over the same matrix.

Self-describing: thresholds/kinds/directions and the base-objective column order
are read from ``reeval_raw_meta.json`` (snapshotted at simulation time), so
scoring never depends on the live objective registry or a changed
``NYCOPT_SAT_THRESHOLDS`` (the moving-measuring-stick guard, McPhail et al. 2020).
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


###############################################################################
# Loading
###############################################################################

@dataclass
class RawCube:
    """Dense ``(S, R, M)`` re-eval matrix plus its self-describing metadata.

    Attributes:
        cube: float array ``(n_solutions, n_realizations, n_base_objs)`` in
            natural units; NaN where a (solution, realization) is missing/failed.
        solution_ids: length-S solution ids (sorted).
        realization_ids: length-R realization indices (sorted union).
        base_names: length-M base objective names (column order = meta order).
        thresholds: base_name -> satisficing magnitude threshold (or None).
        kinds: base_name -> "ge" | "le" (satisficing direction).
        directions: base_name -> "maximize" | "minimize".
        is_ensemble: False for single-trace re-eval (R == 1, robustness N/A).
        sow_ids: length-R SOW (forcing-profile) index of each realization, aligned
            to ``realization_ids``; ``None`` when the ensemble has no forcing
            profiles (stationary / historic), in which case the SOW-unit metrics
            are N/A rather than silently reverting to the realization unit.
        realizations_per_sow: R per SOW, or ``None``.
        meta: the full parsed ``reeval_raw_meta.json``.
    """

    cube: np.ndarray
    solution_ids: list
    realization_ids: list
    base_names: list
    thresholds: dict
    kinds: dict
    directions: dict
    is_ensemble: bool
    meta: dict
    sow_ids: list | None = None
    realizations_per_sow: int | None = None

    @property
    def n_realizations(self) -> int:
        return self.cube.shape[1]

    @property
    def n_sow(self) -> int | None:
        """Number of distinct SOWs on the realization axis, or ``None`` if ungrouped."""
        return None if self.sow_ids is None else len(set(self.sow_ids))

    def direction_signs(self) -> np.ndarray:
        """+1 for maximize, -1 for minimize, aligned to ``base_names``."""
        return np.array(
            [1 if self.directions.get(n) == "maximize" else -1
             for n in self.base_names],
            dtype=int,
        )

    def sow_groups(self) -> list[tuple[int, np.ndarray]]:
        """Group the realization axis by SOW.

        Returns:
            ``[(sow_id, column_positions), ...]`` in ascending SOW order, where
            ``column_positions`` indexes axis 1 of :attr:`cube`. Empty when the
            ensemble carries no SOW grouping.
        """
        if self.sow_ids is None:
            return []
        order: dict[int, list[int]] = {}
        for j, s in enumerate(self.sow_ids):
            order.setdefault(int(s), []).append(j)
        return [(s, np.asarray(order[s], dtype=int)) for s in sorted(order)]


def _read_long(reeval_dir: Path) -> pd.DataFrame:
    parquet = reeval_dir / "reeval_raw.parquet"
    csvgz = reeval_dir / "reeval_raw.csv.gz"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csvgz.exists():
        return pd.read_csv(csvgz)
    raise FileNotFoundError(
        f"No reeval_raw.parquet or reeval_raw.csv.gz in {reeval_dir}"
    )


def load_raw(reeval_dir) -> RawCube:
    """Load the long-format raw matrix + meta and densify to an ``(S,R,M)`` cube.

    Missing ``(solution, realization, objective)`` cells (failed solutions,
    ragged realization coverage) are NaN-filled so metrics align on the union of
    realization ids — never on positional index.
    """
    reeval_dir = Path(reeval_dir)
    meta_path = reeval_dir / "reeval_raw_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text())

    df = _read_long(reeval_dir)
    base_names = list(meta["base_names"])
    # Prefer the full attempted solution-id list from meta so fully-failed
    # solutions (no rows) survive as all-NaN cube slices rather than vanishing.
    # Fall back to the ids present in the rows for metas written before this key
    # existed (and for the test fixtures).
    meta_sids = meta.get("solution_ids")
    if meta_sids is not None:
        solution_ids = sorted(int(x) for x in meta_sids)
    else:
        solution_ids = sorted(int(x) for x in df["solution_id"].unique())
    realization_ids = sorted(int(x) for x in df["realization_id"].unique())

    s_ix = {s: i for i, s in enumerate(solution_ids)}
    r_ix = {r: i for i, r in enumerate(realization_ids)}
    o_ix = {o: k for k, o in enumerate(base_names)}

    cube = np.full((len(solution_ids), len(realization_ids), len(base_names)),
                   np.nan, dtype=float)
    for sid, rid, obj, val in zip(
        df["solution_id"], df["realization_id"], df["objective"], df["value"]
    ):
        k = o_ix.get(obj)
        if k is None:
            continue
        cube[s_ix[int(sid)], r_ix[int(rid)], k] = val

    return RawCube(
        cube=cube,
        solution_ids=solution_ids,
        realization_ids=realization_ids,
        base_names=base_names,
        thresholds={k: meta.get("thresholds", {}).get(k) for k in base_names},
        kinds={k: meta.get("kinds", {}).get(k) for k in base_names},
        directions={k: meta.get("directions", {}).get(k) for k in base_names},
        is_ensemble=bool(meta.get("is_ensemble", True)),
        sow_ids=_aligned_sow_ids(meta, realization_ids),
        realizations_per_sow=meta.get("realizations_per_sow"),
        meta=meta,
    )


def _aligned_sow_ids(meta: dict, realization_ids: list) -> Optional[list]:
    """Re-align the meta's ``sow_ids`` (keyed to ``realization_indices``) to the cube's axis.

    The cube's realization axis is the union of ids actually present in the rows, which
    can be a subset of the ensemble's ``realization_indices`` (failed batches). Aligning
    positionally would silently mis-assign realizations to SOWs, so the join is on
    ``realization_id``. If any cube realization is unmapped the grouping is dropped
    entirely (fail closed) rather than partially applied.
    """
    sow_ids = meta.get("sow_ids")
    if sow_ids is None:
        return None
    idx = meta.get("realization_indices") or []
    by_rid = {int(r): int(s) for r, s in zip(idx, sow_ids)}
    aligned = [by_rid.get(int(r)) for r in realization_ids]
    if any(s is None for s in aligned):
        warnings.warn(
            "reeval_raw_meta carries sow_ids but some realizations in the matrix are "
            "not covered by them; SOW-unit robustness metrics will be reported N/A."
        )
        return None
    return aligned


###############################################################################
# Composable aggregation
###############################################################################

def aggregate_over_realizations(values: np.ndarray,
                                agg: Callable = np.nanmean) -> np.ndarray:
    """Reduce an ``(S, R)`` slice over realizations (axis 1), NaN-safe.

    The single seam that makes risk attitudes composable: ``agg=np.nanmean`` is
    expected value / Laplace, ``agg=np.nanmin`` (on a maximize objective) is
    maximin/Wald, ``lambda v: np.nanpercentile(v, q, axis=1)`` is percentile, and
    a best/worst blend is Hurwicz. Only mean is used by the metrics below; the
    rest are future wrappers.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return agg(values, axis=1)


###############################################################################
# Satisficing (univariate + multivariate Starr domain criterion)
###############################################################################

def _satisfaction_cube(raw: RawCube, thresholds: dict = None,
                       kinds: dict = None) -> np.ndarray:
    """Boolean ``(S, R, M)`` cube: does cell meet its objective's threshold?

    Replicates ``src.objectives_ensemble.SatisficingAgg``: non-finite values are
    **unsatisfied** (a degenerate realization can't masquerade as satisficing).

    A missing threshold or kind is a HARD ERROR, not a skipped column. The
    multivariate criterion is a conjunction over all objectives, so an all-False
    column would silently drive ``sat_multivariate`` to 0 for every solution --
    the primary robustness metric, reading as "nothing is ever robust", with no
    other symptom. The only legitimate case (a single-trace re-eval, where a
    realization-fraction criterion is undefined) is caught upstream by the R == 1
    gate in :func:`score_robustness`, which NaNs the whole scorecard.

    Raises:
        ValueError: If any base objective lacks a threshold or a kind.
    """
    thresholds = thresholds if thresholds is not None else raw.thresholds
    kinds = kinds if kinds is not None else raw.kinds
    missing = [
        n for n in raw.base_names
        if thresholds.get(n) is None or kinds.get(n) is None
    ]
    if missing:
        raise ValueError(
            f"No satisficing threshold/kind for {missing}. The multivariate "
            f"criterion is a conjunction, so a missing column would silently "
            f"zero the primary metric for every solution. Supply a threshold for "
            f"every base objective (use +/-inf to make one non-binding)."
        )

    return _satisfy(raw.cube, raw.base_names, thresholds, kinds)


def _satisfy(cube: np.ndarray, base_names: list, thresholds: dict,
             kinds: dict) -> np.ndarray:
    """Boolean satisfaction cube for ANY ``(S, U, M)`` array, whatever the unit U is.

    Shared by the realization-unit path (U = realizations) and the SOW-unit path
    (U = SOWs, after the within-SOW collapse), so the two units differ only in what
    the middle axis MEANS -- never in how the criterion is applied.
    """
    S, U, M = cube.shape
    sat = np.zeros((S, U, M), dtype=bool)
    for k, name in enumerate(base_names):
        thr, kind = thresholds[name], kinds[name]
        slab = cube[:, :, k]
        finite = np.isfinite(slab)
        if kind == "ge":
            sat[:, :, k] = finite & (slab >= thr)
        else:
            sat[:, :, k] = finite & (slab <= thr)
    return sat


def satisficing_univariate(raw: RawCube, thresholds: dict = None,
                           kinds: dict = None) -> pd.DataFrame:
    """Per-objective satisficing fraction. Reproduces the search-time metric.

    Returns a ``(S × M)`` DataFrame indexed by solution_id; NaN denominator
    includes missing realizations, so a failed realization counts as unsatisfied.
    """
    sat = _satisfaction_cube(raw, thresholds, kinds)
    frac = sat.mean(axis=1)  # (S, M)
    return pd.DataFrame(
        frac, index=pd.Index(raw.solution_ids, name="solution_id"),
        columns=[f"sat_uni__{n}" for n in raw.base_names],
    )


def satisficing_multivariate(raw: RawCube, thresholds: dict = None,
                             kinds: dict = None) -> pd.Series:
    """Starr (1962) domain criterion: fraction of realizations meeting ALL
    thresholds jointly. The PRIMARY robustness measure (Herman et al. 2015).

    A non-finite value in any objective fails that realization's joint criterion,
    mirroring the univariate non-finite-as-unsatisfied rule.

    The unit here is the REALIZATION. For the SOW unit — the Triangle-lineage
    construction, in which the stochastic traces within a deeply-uncertain state of
    the world are collapsed first — see :func:`satisficing_multivariate_sow`.
    """
    sat = _satisfaction_cube(raw, thresholds, kinds)
    joint = sat.all(axis=2).mean(axis=1)  # (S,)
    return pd.Series(
        joint, index=pd.Index(raw.solution_ids, name="solution_id"),
        name="sat_multivariate",
    )


###############################################################################
# The SOW unit (Herman 2014; Trindade 2017; Gold 2022, 2023)
###############################################################################

#: Within-SOW aggregators. This is a REAL methodological choice, not a detail: it is
#: the risk attitude applied to natural variability *inside* a deeply-uncertain state
#: of the world, before any uncertainty across states is considered.
WITHIN_SOW_AGGREGATORS: tuple[str, ...] = ("mean", "worst")


def collapse_within_sow(raw: RawCube, within_sow_agg: str = "mean"
                        ) -> tuple[np.ndarray, list]:
    """Stage 1 of the SOW unit: collapse the R realizations of each SOW to one vector.

    Args:
        raw: The re-eval cube, which must carry ``sow_ids``.
        within_sow_agg: ``"mean"`` (risk-NEUTRAL within the state of the world — the
            default, and what the Triangle-lineage papers do) or ``"worst"``
            (risk-AVERSE: the worst realization in the SOW, in each objective's own
            direction). Reported in the output, because the choice moves the number.

    Returns:
        ``(cube_sow, sow_labels)`` — ``cube_sow`` is ``(S, n_sow, M)`` in natural
        units, ``sow_labels`` the ascending SOW ids it is indexed by.

    Raises:
        ValueError: If the cube carries no SOW grouping, or the aggregator is unknown.
    """
    if raw.sow_ids is None:
        raise ValueError(
            "This re-eval cube carries no sow_ids, so the SOW unit is undefined. Its "
            "ensemble has no forcing profiles (a stationary ensemble, or the historic "
            "trace). Do NOT substitute the realization unit -- they are different "
            "quantities."
        )
    if within_sow_agg not in WITHIN_SOW_AGGREGATORS:
        raise ValueError(
            f"unknown within_sow_agg {within_sow_agg!r}; expected one of "
            f"{WITHIN_SOW_AGGREGATORS}"
        )

    groups = raw.sow_groups()
    signs = raw.direction_signs()
    S, _R, M = raw.cube.shape
    out = np.full((S, len(groups), M), np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN SOW -> NaN
        for g, (_sow, cols) in enumerate(groups):
            block = raw.cube[:, cols, :]                     # (S, R_g, M)
            if within_sow_agg == "mean":
                out[:, g, :] = np.nanmean(block, axis=1)
            else:  # worst, in each objective's own direction
                oriented = block * signs[None, None, :]
                out[:, g, :] = np.nanmin(oriented, axis=1) * signs[None, :]
    return out, [s for s, _ in groups]


def satisficing_multivariate_sow(raw: RawCube, thresholds: dict = None,
                                 kinds: dict = None,
                                 within_sow_agg: str = "mean") -> pd.Series:
    """Starr domain criterion on the SOW unit: two stages, in this order.

    1. **Collapse** the R realizations within each theta into one performance vector
       per SOW (:func:`collapse_within_sow`; ``mean`` = risk-neutral by default,
       ``worst`` = risk-averse sensitivity).
    2. **Apply the Starr (1962) domain criterion across the N_theta SOWs**: the
       fraction of deeply-uncertain states in which the collapsed vector meets ALL
       thresholds jointly.

    This is what Herman et al. (2014), Trindade et al. (2017) and Gold et al. (2022,
    2023) compute, and it is a DIFFERENT quantity from the realization-unit
    :func:`satisficing_multivariate` whenever the within-SOW spread is non-trivial: a
    policy that fails half the traces in every SOW scores ~0.5 on realizations and
    ~0.0 on SOWs (under ``worst``), or somewhere else entirely (under ``mean``).

    **Precision is governed by N_theta, not by N_test.** Adding realizations per SOW
    sharpens each SOW's collapsed estimate but adds no new states of the world; only
    more thetas do. That is why the theta count is the sizing knob that matters.
    """
    cube_sow, sow_labels = collapse_within_sow(raw, within_sow_agg)
    thresholds = thresholds if thresholds is not None else raw.thresholds
    kinds = kinds if kinds is not None else raw.kinds
    missing = [n for n in raw.base_names
               if thresholds.get(n) is None or kinds.get(n) is None]
    if missing:
        raise ValueError(
            f"No satisficing threshold/kind for {missing}; the multivariate criterion "
            f"is a conjunction, so a missing column would zero the metric silently."
        )
    sat = _satisfy(cube_sow, raw.base_names, thresholds, kinds)   # (S, n_sow, M)
    joint = sat.all(axis=2).mean(axis=1)                          # (S,)
    del sow_labels
    return pd.Series(
        joint, index=pd.Index(raw.solution_ids, name="solution_id"),
        name="sat_multivariate_sow",
    )


###############################################################################
# Risk-attitude anchors (McPhail et al. 2018, T3)
###############################################################################

def _oriented(raw: RawCube, k: int) -> np.ndarray:
    """Return objective ``k``'s ``(S, R)`` slab re-signed so HIGHER IS BETTER.

    Lets one code path serve both directions: a minimize objective is negated, so
    ``nanmean`` is always Laplace and ``nanmin`` is always maximin.
    """
    return raw.cube[:, :, k] * raw.direction_signs()[k]


def laplace_mean(raw: RawCube) -> pd.DataFrame:
    """Mean performance across realizations, per objective (McPhail T3 = mean).

    The risk-neutral anchor (Laplace's principle of insufficient reason). Reported
    in the objective's NATURAL orientation, so "higher is better" follows the
    objective's own direction. Free to compute, and its absence would be asked
    about: metric choice changes rankings (Herman et al. 2015; McPhail et al.
    2018), so a single robustness family is never sufficient.
    """
    S, R, M = raw.cube.shape
    out = np.full((S, M), np.nan)
    for k in range(M):
        out[:, k] = aggregate_over_realizations(raw.cube[:, :, k], np.nanmean)
    return pd.DataFrame(
        out, index=pd.Index(raw.solution_ids, name="solution_id"),
        columns=[f"laplace__{n}" for n in raw.base_names],
    )


def maximin(raw: RawCube) -> pd.DataFrame:
    """Worst-case performance across realizations, per objective (McPhail T3 = worst).

    The risk-averse anchor (Wald). Computed on the direction-oriented slab and
    returned in natural units, so it is the worst realization's value: the minimum
    for a maximize objective, the maximum for a minimize objective.
    """
    signs = raw.direction_signs()
    S, R, M = raw.cube.shape
    out = np.full((S, M), np.nan)
    for k in range(M):
        worst_oriented = aggregate_over_realizations(_oriented(raw, k), np.nanmin)
        out[:, k] = worst_oriented * signs[k]  # back to natural units
    return pd.DataFrame(
        out, index=pd.Index(raw.solution_ids, name="solution_id"),
        columns=[f"maximin__{n}" for n in raw.base_names],
    )


###############################################################################
# Improvement over the status quo
###############################################################################

def improvement_vs_baseline(raw: RawCube, baseline: RawCube,
                            normalize: str = "best",
                            agg: Callable = np.nanmean) -> pd.DataFrame:
    """Per-objective shortfall against the status-quo policy, on the SAME realizations.

    "How much worse than current operations." This is DELIBERATELY NOT a Savage
    regret, and it is the only baseline-relative measure this module computes.

    Why not regret-from-best. Best-in-set regret defines the reference as the best
    value achieved by any policy in the pooled solution set, which makes it
    **set-relative and design-coupled**: in a cross-design comparison, dropping one
    design changes every other design's regret, so it is not a design-independent
    quantity. Bonham et al. (2024) additionally show it needs 400+ scenarios to
    converge (vs 50-300 for satisficing) and that on a max-over-time objective it
    never converges at all -- they caution against ranking policies on it in
    isolation. Our two deficit objectives are worst-percentile operators, which is
    exactly that pathology.

    Why not Cohen et al. (2021) baseline regret. That measure needs one
    perfect-foresight MOEA run per scenario (97 optimizations there), which is
    formulation-specific and unscalable to a candidate pool of 1e5-1e6. Cohen is
    cited as motivation for the contribution, never as a metric computed here.

    The status quo, by contrast, is a FIXED external reference: it does not move
    when designs are added or removed, and it costs one extra policy simulation
    that workflow step 05 already performs. The reference is licensed by McPhail
    et al. (2018) §3.1, whose regret T1 admits "some type of baseline performance
    for a given scenario instead of the performance of the best decision
    alternative".

    NOT Kasprzyk et al. (2013). Their Eq. (9) percent deviation is
    ``(f_i,90 - f_i,base) / f_i,base`` where ``base`` is the SAME solution's value
    in the baseline STATE OF THE WORLD and the percentile runs over the SOW
    ensemble -- a within-policy, across-SOW sensitivity measure (Herman et al.
    2015's R1), not a comparison against a status-quo POLICY. It was miscited here
    as the precedent for exactly that. Kasprzyk remains the precedent for R1, and
    R1 is not computed in this study.

    **SIGNED, not clipped, and the sign convention is the point.** The value is

        delta_i(p, s) = sign_i * (f_i(p, s) - f_i(b, s)) / |f_i(b, s)|

    where ``sign_i`` is +1 for a maximize objective and -1 for a minimize one, so
    **positive always means BETTER than current operations**, whatever the
    objective's direction. Averaged over realizations. Higher is better.

    Clipping at 0 would be degenerate here: optimized policies are expected to
    dominate the status quo on most objectives in most realizations, so a
    clipped quantity collapses to ~0 for every policy of every design and
    discriminates nothing (the failure mode of Bonham et al. 2024 and Huang et
    al. 2025).

    The baseline cube must carry the same objectives; realizations are joined on
    ``realization_id`` (not position). ``normalize="best"`` divides by
    ``|baseline|`` (a fractional improvement); ``"none"`` leaves natural units.
    Where the baseline lacks a realization, that cell is NaN.
    """
    signs = raw.direction_signs()
    # The baseline is one policy: collapse its solution axis to a per-realization
    # vector keyed by realization_id.
    base_by_rid = {
        rid: baseline.cube[:, j, :]
        for j, rid in enumerate(baseline.realization_ids)
    }
    S, R, M = raw.cube.shape
    out = np.full((S, M), np.nan)
    base_aligned = np.full((R, M), np.nan)
    for j, rid in enumerate(raw.realization_ids):
        if rid in base_by_rid:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                base_aligned[j, :] = np.nanmean(base_by_rid[rid], axis=0)

    for k in range(M):
        slab = raw.cube[:, :, k]            # (S, R)
        bvec = base_aligned[:, k]           # (R,)
        # Signed and direction-oriented: positive = better than the status quo.
        delta = signs[k] * (slab - bvec[None, :])
        if normalize == "best":
            denom = np.where(np.abs(bvec) > 0, np.abs(bvec), np.nan)
            delta = delta / denom[None, :]
        out[:, k] = aggregate_over_realizations(delta, agg)
    return pd.DataFrame(
        out, index=pd.Index(raw.solution_ids, name="solution_id"),
        columns=[f"vs_baseline__{n}" for n in raw.base_names],
    )


###############################################################################
# Incumbent-relative regret
###############################################################################
# WHAT THIS IS, in the McPhail et al. (2018) T1/T2/T3 scheme:
#
#   T1 = regret from a fixed BASELINE DECISION ALTERNATIVE, evaluated per
#        scenario. McPhail license this reference explicitly -- "Alternative
#        metrics that are based on the relative performance of decision
#        alternatives use some type of baseline performance for a given scenario
#        instead of the performance of the best decision alternative" -- but never
#        name, tabulate, or test it. Making it explicit is the contribution; it is
#        not a new metric family.
#   T2 = the ADVERSE SUBSET (the scenarios in which the policy is worse than the
#        incumbent). This is not ad-hoc clipping: it is exactly the subset
#        selection of McPhail's "undesirable deviations" metric (Kwakkel et al.
#        2016b: T1 = regret from median, T2 = worst-half, T3 = sum), with the
#        reference changed from the policy's own median to the incumbent.
#   T3 = mean / 90th percentile over SOWs. The 90th percentile follows Herman et
#        al. (2015) R1/R2, "intended to reflect the tail end of poor performance
#        while reducing susceptibility to outliers".
#
# WHY IT IS NOT REDUNDANT WITH SATISFICING. The satisficing criteria are fixed
# scalars anchored on the incumbent's HISTORIC attainment; the regret reference is
# the incumbent's performance in THAT SOW, a bar that moves with the forcing.
# Where the fixed criterion drives the domain criterion to 0 for every policy
# (the status quo already fails NYC reliability in ~84% of E_test SOWs),
# satisficing ties everything -- Bonham et al. (2024)'s saturation failure mode --
# and regret still separates policies.
#
# WHY NATURAL UNITS, AND NO CROSS-OBJECTIVE SCALAR. Dividing by the baseline's own
# per-cell value (what `improvement_vs_baseline` does) is degenerate for this
# objective set: flood exceedance is EXACTLY 0 in a large share of realizations
# and both CVaR90 deficits are 0 in wet ones, so the cell is NaN'd and silently
# dropped -- and the dropped cells are precisely the benign ones, biasing the
# estimator toward the flood-active subset. Working in natural units dissolves
# that rather than patching it: no denominator, no dropped cell, every SOW
# contributes. Herman et al. (2015) hit the same wall and say so ("normalized by
# the objective value itself rather than the best value because the latter often
# approaches zero"); Eker & Kwakkel (2018) add +1 to both terms for the same
# reason; McPhail et al. (2018) give no normalization guidance at all. The two
# published scales are both unusable here -- Cohen et al. (2021) normalize on the
# per-scenario span to a PERFECT-FORESIGHT optimum (one MOEA run per scenario, out
# of budget), and Sunkara et al. (2023) rescale over the ALTERNATIVE SET, which is
# design-coupled and so carries the exact defect that disqualifies best-in-set
# regret. The cost of natural units is that there is no cross-objective regret
# scalar; the unit-free harm FREQUENCIES below carry that role instead.
#
# NO MAX REGRET. Bonham et al. (2024): regret families need 400+ scenarios and
# never converge on extreme-of-extremes operators. McPhail et al. document the
# tie-degeneracy directly ("many of the decision alternatives have a reliability
# of 0% in the worst-case scenario ... the maximin metric ... ranks many of the
# decision alternatives as equal").

#: Decree-party grouping of the base objectives, for the party-level harm
#: frequencies. The DRB renegotiation is unanimity-bound, so one party's loss is
#: NOT compensable by another's gain -- which is why the party summary is a
#: FREQUENCY over a disjunction and never a summed or averaged party score.
#: Sunkara et al. (2023) document what the compensating form costs: their
#: all-actor metric looks stable only because "the water supply sector may fail in
#: certain scenarios, but those failures are in aggregate countered by increasing
#: levels of success for the ecology-MEF sector".
#: Judgment call recorded: NYC aggregate storage sits under `nyc` because the
#: objective is NYC's own supply security, though the same storage also underwrites
#: downstream release capability.
DECREE_PARTY_OBJECTIVES: dict[str, tuple[str, ...]] = {
    "nyc": (
        "nyc_delivery_reliability_weekly",
        "nyc_delivery_deficit_cvar90_pct",
        "nyc_storage_p5_pct",
    ),
    "nj": (
        "nj_delivery_reliability_weekly",
    ),
    "downstream_flow": (
        "montague_flow_reliability_weekly",
        "montague_flow_deficit_cvar90_pct",
        "trenton_flow_reliability_weekly",
    ),
    "flood_exposed": (
        "downstream_flood_exceedance_minor",
        "downstream_flood_days_minor",
    ),
}

#: Multiplier ``k`` on the per-objective just-noticeable difference that defines
#: the no-harm tolerance ``tau_i = k * eps_i``. MUST be declared before results are
#: inspected (the manuscript pre-specifies its other endpoints); override with
#: ``NYCOPT_REGRET_TAU_K``. k = 0 is the strict weak-Pareto-improvement form.
REGRET_TAU_K: float = float(os.environ.get("NYCOPT_REGRET_TAU_K", "1"))


def _regret_unit(raw: RawCube, baseline: RawCube, unit: str,
                 within_sow_agg: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Align policy and incumbent onto a common unit axis.

    Both cubes are reduced independently -- the policy cube by its own SOW
    grouping, the incumbent cube by its own -- and then joined on the unit LABEL
    (SOW id or realization id), never on position. A cube whose labels do not
    cover the other's contributes NaN for the missing units rather than a silent
    mis-pairing.

    Args:
        raw: The policy re-eval cube.
        baseline: The status-quo cube on the same ensemble (``S == 1``).
        unit: ``"sow"`` or ``"realization"``.
        within_sow_agg: Within-SOW risk attitude, when ``unit == "sow"``.

    Returns:
        ``(policy, incumbent, unit_name)`` where ``policy`` is ``(S, U, M)`` and
        ``incumbent`` is ``(U, M)``, both in natural units.

    Raises:
        ValueError: If ``unit`` is unknown, or the SOW unit is requested for a
            cube that carries no SOW grouping.
    """
    if unit not in ("sow", "realization"):
        raise ValueError(f"unit must be 'sow' or 'realization', got {unit!r}")

    if unit == "sow":
        if raw.sow_ids is None or baseline.sow_ids is None:
            raise ValueError(
                "SOW-unit regret needs sow_ids on BOTH the policy and the "
                "baseline cube. Do NOT substitute the realization unit -- it is a "
                "different quantity."
            )
        pol, pol_labels = collapse_within_sow(raw, within_sow_agg)
        base, base_labels = collapse_within_sow(baseline, within_sow_agg)
    else:
        pol, pol_labels = raw.cube, list(raw.realization_ids)
        base, base_labels = baseline.cube, list(baseline.realization_ids)

    # The incumbent cube holds ONE policy; collapse its solution axis (nanmean is
    # an identity for S == 1 and averages any accidental duplicates).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_vec = np.nanmean(base, axis=0)                       # (U_base, M)

    by_label = {lab: base_vec[j, :] for j, lab in enumerate(base_labels)}
    aligned = np.full((pol.shape[1], pol.shape[2]), np.nan, dtype=float)
    for j, lab in enumerate(pol_labels):
        row = by_label.get(lab)
        if row is not None:
            aligned[j, :] = row
    return pol, aligned, unit


def incumbent_advantage(raw: RawCube, baseline: RawCube, unit: str = "sow",
                        within_sow_agg: str = "mean") -> np.ndarray:
    """Signed advantage over the status quo, ``(S, U, M)``, in NATURAL UNITS.

    ``D_i(x, u) = sign_i * (f_i(x, u) - f_i(b, u))`` where ``sign_i`` is +1 for a
    maximize objective and -1 for a minimize one, so **positive always means
    BETTER than current operations** whatever the objective's own direction.

    This is the substrate every other quantity in this section is a view of; it is
    a pure function of two cubes that already exist, so nothing needs persisting.
    """
    pol, base, _ = _regret_unit(raw, baseline, unit, within_sow_agg)
    signs = raw.direction_signs()
    return signs[None, None, :] * (pol - base[None, :, :])


def regret_magnitudes(raw: RawCube, baseline: RawCube, unit: str = "sow",
                      within_sow_agg: str = "mean") -> pd.DataFrame:
    """Per-objective regret and gain magnitudes, in each objective's OWN units.

    Reads directly: "1.8 more failing weeks per year", "0.4 more ft-days/yr above
    minor flood stage". These columns are never summed, averaged, or compared
    ACROSS objectives -- see the section header for why there is no scalar.

    Columns, per base objective:
      - ``regret_mean__``  mean over units of ``max(0, -D)``  (risk-neutral)
      - ``regret_q90__``   90th percentile of ``max(0, -D)``  (tail; Herman R1/R2)
      - ``regret_cond__``  mean regret GIVEN a shortfall, i.e. ``mean(-D | D < 0)``;
        **NaN when the policy is never worse than the incumbent** on that
        objective, which is a real distinction from a shortfall of size zero and
        is why it is not filled with 0. Read it beside ``harm_freq__``.
      - ``gain_mean__``    mean over units of ``max(0, +D)``; the degeneracy
        companion, because a policy scores zero regret by BEING the incumbent.
    """
    D = incumbent_advantage(raw, baseline, unit, within_sow_agg)   # (S, U, M)
    G = np.where(np.isfinite(D), np.maximum(0.0, -D), np.nan)
    P = np.where(np.isfinite(D), np.maximum(0.0, D), np.nan)
    adverse = np.where(np.isfinite(D) & (D < 0), -D, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> NaN
        cols = {}
        for k, name in enumerate(raw.base_names):
            cols[f"regret_mean__{name}"] = np.nanmean(G[:, :, k], axis=1)
            cols[f"regret_q90__{name}"] = np.nanquantile(G[:, :, k], 0.90, axis=1)
            cols[f"regret_cond__{name}"] = np.nanmean(adverse[:, :, k], axis=1)
            cols[f"gain_mean__{name}"] = np.nanmean(P[:, :, k], axis=1)

    ordered = [f"{pre}__{n}"
               for n in raw.base_names
               for pre in ("regret_mean", "regret_q90", "regret_cond", "gain_mean")]
    return pd.DataFrame(
        {c: cols[c] for c in ordered},
        index=pd.Index(raw.solution_ids, name="solution_id"),
    )


def tau_ladder(base_names: list, k: float = None, floors: dict = None) -> dict:
    """Per-objective no-harm tolerance in natural units: ``tau_i = k * u_i``.

    The tolerance UNIT ``u_i`` is:

    - the objective's §1 base-metric epsilon from ``src.objectives.OBJECTIVES`` --
      a signal-scale just-noticeable difference (Reed et al. 2013: ~IQR/10 over
      random DV policies), calibrated in exactly the whole-trace metric space the
      re-eval cube lives in, so the ladder reads "no objective degraded by more
      than k just-noticeable differences";
    - ``max(eps_i, floor_i)`` when ``floors`` is supplied, where ``floor_i`` is the
      measured noise floor of that objective's SOW-mean estimator.

    **Why floors exist.** An epsilon BELOW its objective's noise floor makes every
    rung meaningless on that axis -- the criterion fires on Monte Carlo noise
    rather than on harm -- and because one ``k`` is shared across objectives, a
    single such axis silently sets what every rung means. Flood exceedance is the
    live case: its epsilon (0.01 ft-days/yr) sits an order of magnitude under its
    floor. Taking the max keeps epsilon where resolution binds and the floor where
    noise binds, so ``k`` means the same thing on every axis. Floors are measured
    by ``scripts/supplemental/regret_tolerance_diagnostics.py`` (pass A) and are a
    property of the incumbent alone, so using them is not circular.

    Provenance caveat for the methods text: the §1 epsilons were measured on the
    historic reference trace, not on E_test, and the CAMPAIGN epsilon vector is the
    separate annual-unit one in ``src.objectives_ensemble``. They are the right
    SPACE and the right ORDER; they are not an E_test-derived quantity.

    Env override ``NYCOPT_REGRET_TAU`` (JSON ``{base_name: tau}``) replaces the
    WHOLE vector, for the case where an adopted vector is recorded rather than
    derived. It must cover every objective: a partial override would leave the rest
    on a different tolerance basis without saying so.

    Args:
        base_names: Base objective names, in cube column order.
        k: Multiplier; defaults to :data:`REGRET_TAU_K`. ``k = 0`` gives the strict
            weak-Pareto-improvement form.
        floors: Optional ``{base_name: noise_floor}`` in natural units.

    Returns:
        ``{base_name: tau}``.

    Raises:
        KeyError: If a base name has neither a registered epsilon nor an override
            -- a silent 0 would turn a tolerance into a strict criterion without
            saying so -- or if the env override is partial or names a stranger.
    """
    from src.objectives import OBJECTIVES

    raw = os.environ.get("NYCOPT_REGRET_TAU", "").strip()
    if raw:
        override = json.loads(raw)
        unknown = [n for n in override if n not in base_names]
        if unknown:
            raise KeyError(
                f"NYCOPT_REGRET_TAU names objectives absent from this cube: {unknown}"
            )
        absent = [n for n in base_names if n not in override]
        if absent:
            raise KeyError(
                f"NYCOPT_REGRET_TAU is a WHOLE-vector override but omits {absent}; a "
                f"partial vector would leave those objectives on a different "
                f"tolerance basis than the rest."
            )
        return {n: float(override[n]) for n in base_names}

    k = REGRET_TAU_K if k is None else float(k)
    missing = [n for n in base_names if n not in OBJECTIVES]
    if missing:
        raise KeyError(
            f"no registered base objective for {missing}, so no epsilon to build a "
            f"tolerance from. Pass an explicit tau dict instead of defaulting to 0, "
            f"which would silently harden the criterion."
        )
    floors = floors or {}
    return {n: k * max(float(OBJECTIVES[n].epsilon), float(floors.get(n, 0.0)))
            for n in base_names}


def regret_frequencies(raw: RawCube, baseline: RawCube, tau: dict = None,
                       unit: str = "sow", within_sow_agg: str = "mean",
                       parties: dict = None) -> pd.DataFrame:
    """Unit-free harm frequencies. These carry the scalar role.

    Because the magnitudes above stay in natural units, the cross-objective and
    cross-policy summaries are frequencies, which need no normalization at all.

    Columns:
      - ``harm_freq__{obj}``        fraction of units with ``D_i < 0``
      - ``party_harm_freq__{party}``  fraction of units in which ANY objective of
        that Decree party is worse off (a disjunction, never a sum -- see
        :data:`DECREE_PARTY_OBJECTIVES`)
      - ``no_harm_freq``            fraction of units with ``D_i >= 0`` for ALL i:
        a weak Pareto improvement on the incumbent. Cohen et al. (2021) apply this
        exact condition as a hard FILTER ("solutions that will at a minimum
        outperform the status quo in all re-evaluations"); reporting it as a graded
        frequency keeps the policies whose trade-off against the incumbent is the
        interesting part of the question.
      - ``no_harm_freq_tau``        fraction of units with ``D_i >= -tau_i`` for
        ALL i -- the literal reading of "improves some outcomes without degrading
        others below current performance".
      - ``n_degraded_mean``         mean number of objectives simultaneously worse
        off, the informative decomposition of ``no_harm_freq`` (which is small by
        construction when there are 8 objectives with genuine trade-offs).

    A non-finite ``D`` counts as HARM, mirroring the non-finite-as-unsatisfied rule
    of the satisficing path: a degenerate unit must not read as "no harm".
    """
    D = incumbent_advantage(raw, baseline, unit, within_sow_agg)   # (S, U, M)
    tau = tau_ladder(raw.base_names) if tau is None else tau
    missing = [n for n in raw.base_names if n not in tau]
    if missing:
        raise KeyError(f"no tolerance supplied for {missing}")
    tau_vec = np.array([float(tau[n]) for n in raw.base_names], dtype=float)

    finite = np.isfinite(D)
    harm = (~finite) | (D < 0)                                     # (S, U, M)
    harm_tau = (~finite) | (D < -tau_vec[None, None, :])

    index = pd.Index(raw.solution_ids, name="solution_id")
    out = pd.DataFrame(index=index)
    for k, name in enumerate(raw.base_names):
        out[f"harm_freq__{name}"] = harm[:, :, k].mean(axis=1)

    parties = DECREE_PARTY_OBJECTIVES if parties is None else parties
    col_of = {n: k for k, n in enumerate(raw.base_names)}
    for party, members in parties.items():
        idx = [col_of[n] for n in members if n in col_of]
        if not idx:
            continue
        out[f"party_harm_freq__{party}"] = harm[:, :, idx].any(axis=2).mean(axis=1)

    out["no_harm_freq"] = (~harm).all(axis=2).mean(axis=1)
    out["no_harm_freq_tau"] = (~harm_tau).all(axis=2).mean(axis=1)
    out["n_degraded_mean"] = harm.sum(axis=2).mean(axis=1)
    return out


def incumbent_spread(baseline: RawCube, within_sow_agg: str = "mean") -> dict:
    """Per-objective ``q90 - q10`` of the incumbent over the ensemble.

    The OPTIONAL normalization scale, offered so a reviewer asking for a
    dimensionless regret can be answered without re-simulating -- and so rank
    agreement between the natural-unit and normalized orderings can be checked with
    Kendall's tau_b. It is never the reported primary (see the section header).

    Unlike a per-cell baseline denominator it is a single fixed vector: non-zero by
    construction (the incumbent's performance varies across the DU box), publishable
    as a table, and independent of which policies or designs are in the comparison.
    It is the simulation-free stand-in for Cohen et al. (2021)'s achievable span.

    Raises:
        ValueError: If any objective's spread is zero -- silently dividing by it
            would produce infinities that read as catastrophic regret.
    """
    if baseline.sow_ids is not None:
        vals, _ = collapse_within_sow(baseline, within_sow_agg)
    else:
        vals = baseline.cube
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        flat = np.nanmean(vals, axis=0)                            # (U, M)
        hi = np.nanquantile(flat, 0.90, axis=0)
        lo = np.nanquantile(flat, 0.10, axis=0)
    spread = hi - lo
    dead = [n for n, s in zip(baseline.base_names, spread)
            if not np.isfinite(s) or s <= 0]
    if dead:
        raise ValueError(
            f"the incumbent's q90-q10 spread is zero/non-finite for {dead}, so it "
            f"cannot serve as a regret scale. Report natural units for these."
        )
    return {n: float(s) for n, s in zip(baseline.base_names, spread)}


###############################################################################
# Attainability screen
###############################################################################

def attainability_screen(raw: RawCube, thresholds: dict = None,
                         kinds: dict = None) -> pd.DataFrame:
    """Per-realization: can ANY solution in this set meet all the criteria?

    Separates "this design searched badly" from "this test realization is
    unwinnable for anyone" -- a distinction that is otherwise invisible, and that
    matters: Shavazipour et al. (2021) found 23% of their test scenarios could not
    meet the reliability criterion under ANY feasible policy, so the satisficing
    ceiling was structural rather than a search failure.

    This is the free substitute for a per-scenario oracle. It costs zero extra
    simulation (the cube already exists), but it is an EMPIRICAL attainability
    bound, not a true ceiling: it says only that no policy *in this set* wins the
    realization, not that none exists. Report it as such. Pool the cubes of all
    designs before calling this if the question is "unwinnable by anyone."

    Returns a tidy frame (realization_id, n_satisficing_solutions, attainable,
    plus per-objective ``anysat__{name}`` columns showing WHICH criterion is
    binding where nothing attains the joint criterion).
    """
    sat = _satisfaction_cube(raw, thresholds, kinds)     # (S, R, M)
    joint = sat.all(axis=2)                              # (S, R)
    n_sat = joint.sum(axis=0)                            # (R,)
    frame = pd.DataFrame({
        "realization_id": raw.realization_ids,
        "n_satisficing_solutions": n_sat.astype(int),
        "attainable": n_sat > 0,
    })
    for k, name in enumerate(raw.base_names):
        frame[f"anysat__{name}"] = sat[:, :, k].any(axis=0)
    return frame


###############################################################################
# Threshold spectrum (Hadjimichael et al. 2020)
###############################################################################

def threshold_spectrum(raw: RawCube, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90)
                       ) -> pd.DataFrame:
    """Satisficing fraction as a function of the magnitude threshold.

    Robustness depends on the threshold (Hadjimichael et al. 2020): a solution
    robust at one magnitude can be fragile at a neighbor. For each objective the
    threshold grid is the pooled-distribution quantiles (plus the labeled default
    from the meta), and satisficing is reported at each. Returns a tidy DataFrame
    (solution_id, objective, threshold, is_default, satisficing).
    """
    rows = []
    for k, name in enumerate(raw.base_names):
        kind = raw.kinds.get(name)
        if kind is None:
            continue
        slab = raw.cube[:, :, k]  # (S, R)
        pooled = slab[np.isfinite(slab)]
        if pooled.size == 0:
            continue
        grid = list(np.quantile(pooled, quantiles))
        default = raw.thresholds.get(name)
        labeled = {round(float(g), 6): False for g in grid}
        if default is not None:
            labeled[round(float(default), 6)] = True
        for thr, is_default in sorted(labeled.items()):
            finite = np.isfinite(slab)
            if kind == "ge":
                sat = (finite & (slab >= thr)).mean(axis=1)
            else:
                sat = (finite & (slab <= thr)).mean(axis=1)
            for sid, frac in zip(raw.solution_ids, sat):
                rows.append((sid, name, thr, is_default, float(frac)))
    return pd.DataFrame(
        rows,
        columns=["solution_id", "objective", "threshold",
                 "is_default", "satisficing"],
    )


###############################################################################
# Distributional reporting (Hadjimichael et al. 2023)
###############################################################################

def realization_quantiles(raw: RawCube,
                          quantiles=(0.05, 0.25, 0.50, 0.75, 0.95)
                          ) -> pd.DataFrame:
    """Per-solution per-objective distribution of base values over realizations.

    A scalar scorecard discards the distribution the matrix exists to preserve;
    this keeps it (Hadjimichael et al. 2023). Tidy: (solution_id, objective, qXX...).
    """
    rows = []
    qcols = [f"q{int(q * 100):02d}" for q in quantiles]
    for k, name in enumerate(raw.base_names):
        slab = raw.cube[:, :, k]  # (S, R)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            qs = np.nanquantile(slab, quantiles, axis=1)  # (Q, S)
        for si, sid in enumerate(raw.solution_ids):
            rows.append([sid, name] + [float(qs[qi, si])
                                       for qi in range(len(quantiles))])
    return pd.DataFrame(rows, columns=["solution_id", "objective"] + qcols)


###############################################################################
# Ranking stability (McPhail 2020; Bonham 2024)
###############################################################################

def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    a, b = a[mask], b[mask]
    try:
        from scipy.stats import kendalltau
        return float(kendalltau(a, b).correlation)
    except Exception:  # noqa: BLE001 - scipy missing -> O(n^2) fallback
        n = len(a)
        num = 0
        for i in range(n):
            for j in range(i + 1, n):
                num += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        denom = n * (n - 1) / 2
        return float(num / denom) if denom else np.nan


def ranking_stability(scorecard: pd.DataFrame,
                      higher_better: dict) -> pd.DataFrame:
    """Kendall τ_b between every pair of metric columns over the solution set.

    Metric rankings disagree (McPhail 2020); this quantifies how much. Columns
    where lower is better (regret) are negated so all are "higher = more robust"
    before correlating. Bonham (2024) treats τ_b ≥ 0.975 as effectively stable.
    """
    cols = list(scorecard.columns)
    mat = scorecard.to_numpy(dtype=float).copy()
    for ci, c in enumerate(cols):
        if not higher_better.get(c, True):
            mat[:, ci] = -mat[:, ci]
    n = len(cols)
    tau = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            tau[i, j] = 1.0 if i == j else _kendall_tau(mat[:, i], mat[:, j])
    return pd.DataFrame(tau, index=cols, columns=cols)


# There is deliberately NO search-vs-test "overfitting gap" here.
#
# Two independent reasons, and the first alone is disqualifying:
#
# 1. Brodeur et al. (2020) DEFINES NO SUCH METRIC. Overfitting is diagnosed there
#    *graphically*, by plotting cost distributions over the training and held-out
#    ensembles side by side. There is no gap equation, no gap magnitude, and no
#    gap-based ranking anywhere in that paper. Citing it for a defined gap metric
#    would not survive review.
#
# 2. It is structurally invalid for THIS study. The hazard-filling designs compute
#    their in-sample objectives under a deliberately distorted (coverage-weighted)
#    measure, while the re-evaluation is under E_test's natural measure. The
#    difference of the two is a difference of two expectations under two DIFFERENT
#    measures -- an artifact of the measure change, not an overfitting quantity.
#    It would GROW with the very coverage the method advocates. Brodeur's own
#    caveat is the citation: they restrict all claims to *relative* rankings across
#    the two periods and never interpret the absolute train-vs-test difference,
#    precisely because their two ensembles are not drawn from the same
#    distribution.


###############################################################################
# Orchestration
###############################################################################

_DEFAULT_METRICS = (
    "satisficing_multivariate",     # PRIMARY (Starr domain criterion, realization unit)
    "satisficing_multivariate_sow", # the same criterion on the SOW unit
    "satisficing_univariate",       # its per-objective decomposition
    "laplace_mean",                 # McPhail T3 = mean   (risk-neutral anchor)
    "maximin",                      # McPhail T3 = worst  (risk-averse anchor)
    "improvement_vs_baseline",      # fixed external reference; no optimization
    "regret_magnitudes",            # incumbent-relative regret, natural units
    "regret_frequencies",           # its unit-free harm frequencies (the scalars)
)

#: Metrics that need the status-quo cube. Requested without one they are skipped
#: with a warning rather than silently returning zeros.
_BASELINE_METRICS = (
    "improvement_vs_baseline", "regret_magnitudes", "regret_frequencies",
)


def score_robustness(raw: RawCube, baseline: Optional[RawCube] = None,
                     metrics=_DEFAULT_METRICS, thresholds: dict = None,
                     within_sow_agg: str = "mean",
                     ) -> tuple[pd.DataFrame, dict]:
    """Assemble the per-solution scorecard for the requested metrics.

    Args:
        raw: The re-eval cube.
        baseline: Status-quo cube on the SAME ensemble (enables every metric in
            :data:`_BASELINE_METRICS` — the signed improvement and the
            incumbent-relative regret family).
        metrics: Metric ids to compute.
        thresholds: Optional threshold override (the meta's are used otherwise).
        within_sow_agg: Within-SOW risk attitude for ``satisficing_multivariate_sow``
            (``"mean"`` | ``"worst"``).

    Returns:
        ``(scorecard, higher_better)`` where ``higher_better`` maps each column to
        whether larger = more robust (for ranking-stability orientation).

    Realization-defined metrics are N/A (NaN) for a single-trace re-eval
    (``is_ensemble`` False / R == 1) — a historical-record design is a reference, not
    a controlled robustness comparison. SOW-defined metrics are N/A when the ensemble
    carries no ``sow_ids`` (no forcing profiles): the realization unit is a DIFFERENT
    quantity and is never substituted for it.
    """
    pieces = []
    higher_better: dict = {}
    index = pd.Index(raw.solution_ids, name="solution_id")
    signs = raw.direction_signs()

    # EVERY metric here is defined ACROSS realizations, so all of them are N/A on
    # a single-trace re-eval. Two things depend on this gate:
    #   - It must cover the WHOLE scorecard, including the baseline-relative
    #     metric (meaningless at R == 1).
    #   - Gated metrics must not be COMPUTED, only NaN-filled afterwards: a
    #     single-trace re-eval resolves a non-ensemble objective set that may
    #     carry no satisficing thresholds, and _satisfaction_cube raises on a
    #     missing threshold.
    r1 = (not raw.is_ensemble) or raw.n_realizations <= 1

    # The SOW unit needs a SOW grouping. Its absence is a property of the ensemble
    # (no forcing profiles), not a failure -- but it must NaN the SOW columns rather
    # than fall back to the realization unit, which measures something else.
    no_sow = raw.sow_ids is None

    def _nan_frame(cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(np.nan, index=index, columns=cols)

    def _add(name: str, compute, cols: list[str], higher: dict,
             gated: bool = False) -> None:
        if name not in metrics:
            return
        pieces.append(_nan_frame(cols) if (r1 or gated) else compute())
        higher_better.update(higher)

    _add(
        "satisficing_multivariate",
        lambda: satisficing_multivariate(raw, thresholds).to_frame(),
        ["sat_multivariate"],
        {"sat_multivariate": True},
    )
    _add(
        "satisficing_multivariate_sow",
        lambda: satisficing_multivariate_sow(
            raw, thresholds, within_sow_agg=within_sow_agg).to_frame(),
        ["sat_multivariate_sow"],
        {"sat_multivariate_sow": True},
        gated=no_sow,
    )
    _add(
        "satisficing_univariate",
        lambda: satisficing_univariate(raw, thresholds),
        [f"sat_uni__{n}" for n in raw.base_names],
        {f"sat_uni__{n}": True for n in raw.base_names},
    )
    # Laplace and maximin are in NATURAL units, so orientation follows each
    # objective's own direction rather than being uniformly "higher is better".
    _add(
        "laplace_mean",
        lambda: laplace_mean(raw),
        [f"laplace__{n}" for n in raw.base_names],
        {f"laplace__{n}": signs[k] > 0 for k, n in enumerate(raw.base_names)},
    )
    _add(
        "maximin",
        lambda: maximin(raw),
        [f"maximin__{n}" for n in raw.base_names],
        {f"maximin__{n}": signs[k] > 0 for k, n in enumerate(raw.base_names)},
    )

    wanted_baseline = [m for m in metrics if m in _BASELINE_METRICS]
    if wanted_baseline and baseline is None:
        warnings.warn(
            f"{', '.join(wanted_baseline)} requested but no baseline re-eval was "
            "found; skipping. The baseline must be simulated on the SAME "
            "re-eval ensemble (workflow step 05 with the same "
            "NYCOPT_REEVAL_ENSEMBLE_PRESET as step 08), or it lands under a "
            "different reeval tag and auto-detection silently finds nothing."
        )
    elif baseline is not None:
        _add(
            "improvement_vs_baseline",
            lambda: improvement_vs_baseline(raw, baseline),
            [f"vs_baseline__{n}" for n in raw.base_names],
            # Signed and direction-oriented, so HIGHER IS BETTER for every
            # objective regardless of its own direction.
            {f"vs_baseline__{n}": True for n in raw.base_names},
        )

        # The regret family is defined on the SOW unit (the incumbent's bar moves
        # with the forcing, so the unit must be the deeply-uncertain state), and is
        # gated off when either cube carries no SOW grouping -- never silently
        # recomputed on realizations, which is a different quantity.
        no_regret_sow = no_sow or baseline.sow_ids is None
        mag_cols = [f"{pre}__{n}"
                    for n in raw.base_names
                    for pre in ("regret_mean", "regret_q90",
                                "regret_cond", "gain_mean")]
        _add(
            "regret_magnitudes",
            lambda: regret_magnitudes(raw, baseline, within_sow_agg=within_sow_agg),
            mag_cols,
            # Regret is a loss: lower is better. Gain is the mirror.
            {c: c.startswith("gain_mean__") for c in mag_cols},
            gated=no_regret_sow,
        )

        party_cols = [f"party_harm_freq__{p}" for p, members in
                      DECREE_PARTY_OBJECTIVES.items()
                      if any(n in raw.base_names for n in members)]
        freq_cols = ([f"harm_freq__{n}" for n in raw.base_names] + party_cols
                     + ["no_harm_freq", "no_harm_freq_tau", "n_degraded_mean"])
        # The tolerance ladder is resolved HERE, once, so a cube whose objectives
        # are not in the registry (synthetic fixtures) gates the whole block off
        # instead of half-computing it. tau_ladder itself still raises, because a
        # caller supplying a real cube deserves the error rather than a silent 0.
        try:
            tau = tau_ladder(raw.base_names)
        except KeyError:
            tau = None
        _add(
            "regret_frequencies",
            lambda: regret_frequencies(raw, baseline, tau=tau,
                                       within_sow_agg=within_sow_agg),
            freq_cols,
            {c: c.startswith("no_harm_freq") for c in freq_cols},
            gated=no_regret_sow or tau is None,
        )

    scorecard = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=index)

    # A solution with no successful realizations at all (its whole cube slice is
    # non-finite, e.g. every re-eval batch failed) is scored NaN across every
    # metric, matching its NaN row in objectives_summary.csv. Otherwise
    # satisficing would read 0.0 (worst) and make a *failed* run indistinguishable
    # from a *ran-but-bad* run, distorting cross-solution comparison.
    all_missing = np.all(~np.isfinite(raw.cube), axis=(1, 2))  # (S,)
    if all_missing.any() and len(scorecard.columns):
        scorecard.iloc[all_missing, :] = np.nan
    return scorecard, higher_better


def run(reeval_dir, baseline_dir=None, metrics=_DEFAULT_METRICS,
        within_sow_agg: str = "mean") -> Path:
    """Score a re-eval output dir and write the robustness artifacts.

    Writes ``robustness_scorecard.csv``, ``robustness_ranking_stability.csv``,
    ``robustness_threshold_spectrum.csv``, ``robustness_quantiles.csv``,
    ``robustness_attainability.csv``, and ``robustness_meta.json`` (which records the
    SOW grouping and the within-SOW aggregator, so a scorecard is never read without
    knowing which risk attitude produced its SOW column). Returns the scorecard path.
    """
    reeval_dir = Path(reeval_dir)
    raw = load_raw(reeval_dir)
    baseline = load_raw(baseline_dir) if baseline_dir else None

    scorecard, higher_better = score_robustness(
        raw, baseline, metrics, within_sow_agg=within_sow_agg)

    out = reeval_dir / "robustness_scorecard.csv"
    scorecard.to_csv(out)

    # Every scoring-time choice that MOVES a number is recorded next to the numbers
    # rather than left implicit in a default: the within-SOW aggregator, and the
    # no-harm tolerance ladder that defines `no_harm_freq_tau`.
    meta = {
        "metrics": list(metrics),
        "n_solutions": len(raw.solution_ids),
        "n_realizations": raw.n_realizations,
        "n_sow": raw.n_sow,
        "realizations_per_sow": raw.realizations_per_sow,
        "within_sow_aggregator": within_sow_agg if raw.sow_ids is not None else None,
        "sow_metrics_available": raw.sow_ids is not None,
        "regret_available": baseline is not None and raw.sow_ids is not None
                            and baseline.sow_ids is not None,
        "regret_unit": "sow",
        "regret_tau_k": REGRET_TAU_K,
    }
    try:
        meta["regret_tau"] = tau_ladder(raw.base_names)
    except KeyError:
        meta["regret_tau"] = None
    (reeval_dir / "robustness_meta.json").write_text(json.dumps(meta, indent=2))

    ranking_stability(scorecard, higher_better).to_csv(
        reeval_dir / "robustness_ranking_stability.csv")

    # The threshold spectrum is the substrate for the design-ranking threshold
    # sweep: rank agreement ACROSS scenario designs degrades as the satisficing
    # criterion tightens (Quinn et al. 2020), so a single threshold could
    # manufacture or hide the entire design effect.
    threshold_spectrum(raw).to_csv(
        reeval_dir / "robustness_threshold_spectrum.csv", index=False)

    # Raw distributions, always: a robustness scalar can be stable, optimizable,
    # and still perverse (Huang et al. 2025: a deviation metric is driven to zero
    # by being uniformly terrible; Bonham et al. 2024: a saturated criterion ties
    # everything). Co-reporting the distribution is the sanity check.
    realization_quantiles(raw).to_csv(
        reeval_dir / "robustness_quantiles.csv", index=False)

    attainability_screen(raw).to_csv(
        reeval_dir / "robustness_attainability.csv", index=False)

    print(f"[robustness] scorecard -> {out}")
    print(f"[robustness] ranking-stability, threshold-spectrum, quantiles, "
          f"attainability -> {reeval_dir}")
    return out


def _resolve_default_reeval_dir(formulation: str, seed=None) -> Path:
    from config import (REEVAL_ENSEMBLE_SPEC, active_scenario_name,
                        derive_slug)
    from src.reeval_core import reeval_output_dir
    return reeval_output_dir(active_scenario_name(), derive_slug(formulation),
                             REEVAL_ENSEMBLE_SPEC, seed)


def main():
    # A plain ASCII description, not __doc__: the module docstring carries
    # non-ASCII (tau, arrows) and argparse writes help to a cp1252 console on
    # Windows, which raises UnicodeEncodeError on --help.
    parser = argparse.ArgumentParser(
        description="Score robustness offline from a persisted re-eval matrix."
    )
    parser.add_argument("--reeval-dir", default=None,
                        help="Re-eval output dir. Default: resolved from config.")
    parser.add_argument("--formulation", default=None,
                        help="Used to resolve --reeval-dir when omitted.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--baseline-dir", default=None,
                        help="Status-quo re-eval dir (enables improvement_vs_baseline). "
                             "Auto-detected at <reeval-dir>/baseline.")
    parser.add_argument("--metrics", default=None,
                        help="Comma-separated metric ids. Default: config "
                             "REEVALUATION_SETTINGS['robustness_metrics'].")
    args = parser.parse_args()

    from config import REEVALUATION_SETTINGS
    if args.metrics:
        metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    else:
        metrics = tuple(REEVALUATION_SETTINGS.get("robustness_metrics",
                                                  _DEFAULT_METRICS))
    within_sow_agg = REEVALUATION_SETTINGS.get("within_sow_aggregator", "mean")

    if args.reeval_dir:
        reeval_dir = Path(args.reeval_dir)
    elif args.formulation:
        reeval_dir = _resolve_default_reeval_dir(args.formulation, args.seed)
    else:
        parser.error("provide --reeval-dir or --formulation")

    # Auto-detect the status-quo re-eval matrix (written by
    # `run_baseline.py --reeval` under `<reeval_dir>/baseline`) so
    # improvement_vs_baseline works without setting NYCOPT_REEVAL_BASELINE_DIR.
    # NOTE: step 05 must be run with the SAME NYCOPT_REEVAL_ENSEMBLE_PRESET as
    # step 08, or the baseline lands under a different reeval tag and this finds
    # nothing -- the metric is then silently skipped.
    baseline_dir = args.baseline_dir
    if baseline_dir is None:
        auto = reeval_dir / "baseline"
        if any((auto / f).exists() for f in
               ("reeval_raw.parquet", "reeval_raw.csv.gz")):
            baseline_dir = str(auto)
            print(f"[robustness] auto-detected baseline dir -> {baseline_dir}")

    run(reeval_dir, baseline_dir, metrics, within_sow_agg=within_sow_agg)


if __name__ == "__main__":
    main()
