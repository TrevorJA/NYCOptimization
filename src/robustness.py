"""
robustness.py - Offline robustness scoring from the persisted per-SOW matrix.

The re-eval drivers (``src.reevaluate`` / ``src.reevaluate_mpi`` /
``src.chunk_reeval``) persist the per-SOW annual-unit objective matrix
(``reeval_raw.parquet`` + a self-describing ``reeval_raw_meta.json``): each
E_test state of the world's R realizations pooled through the §2 unit
operators, giving the SEARCH OBJECTIVES recomputed per deeply-uncertain state.
This module scores robustness metrics from that matrix *offline*, so different
metrics — which rank solutions differently (McPhail et al. 2018; Giuliani &
Castelletti 2016; Herman et al. 2015; Bonham et al. 2024) — are computed
without re-simulating.

One metric currency
-------------------
Every quantity here is a transformation of the per-SOW values of the SAME
annual-unit objectives the MOEA optimized — the standard construction of the
robustness literature, where the performance measure inside the robustness
calculation IS the optimization objective re-evaluated per state of the world
(Herman et al. 2014, 2015 Eq. 1-7; Trindade et al. 2017 "as calculated with
Eqs. (16)-(20)"; Quinn et al. 2018; Gold et al. 2023; McPhail et al. 2018,
whose f(x, S) is titled "objectives and performance metrics"). What differs
between search and re-evaluation is the ENSEMBLE (search ensemble vs held-out
E_test) and the OUTER aggregation across SOWs (satisficing / maximin / regret)
— never the definition of the statistic.

The SOW is the only scoring unit: pooling each state's R realizations' unit
years through the unit operator IS the within-state collapse, so there is no
separate within-SOW risk-attitude knob and no realization-unit metric family.
E_test carries no probability measure over the forcing space (Lamontagne et
al. 2018), so counting SOWs keeps designed LHS coverage separate from fitted
stochastic variability.

The metric set. **No perfect-foresight optimization appears anywhere.**

  - **Multivariate (Starr 1962) domain criterion [PRIMARY]**
    (``sat_multivariate_sow``) — the fraction of SOWs whose objective vector
    meets *all* thresholds jointly. The standard measure of the Herman
    (2014/2015) / Trindade (2017, 2019) / Gold (2023) lineage. Ranking
    converges at 50-300 *distinct* scenarios (Bonham 2024).
  - **Univariate satisficing** (``sat_uni_sow__``) — its per-objective
    decomposition.
  - **Laplace / mean** (McPhail T3 = mean) — the risk-neutral anchor.
  - **Maximin** (McPhail T3 = worst-case) — the risk-averse anchor. Both are
    free, and their absence would be asked about: metric choice changes
    rankings (Herman 2015; McPhail 2018), so one robustness family is never
    sufficient.
  - **Incumbent-relative regret** — the one-sided (adverse) deviation from the
    status-quo FFMP policy on the *same* SOWs, in NATURAL UNITS, plus the
    unit-free harm frequencies. A FIXED external reference (it does not move
    when designs are added or dropped), costing one policy simulation that
    workflow step 05 already performs. It answers "what would the Decree
    parties give up by adopting?", and it discriminates exactly where the
    domain criterion saturates.
  - **Threshold spectrum** — satisficing vs the magnitude threshold. Robustness
    is threshold-dependent (Hadjimichael et al. 2020), and rank agreement
    ACROSS scenario designs degrades as the criterion tightens (Quinn et al.
    2020), so a single threshold could manufacture or hide the entire design
    effect.
  - **Attainability screen** — which SOWs no policy can win, separating a bad
    design from an impossible state (Shavazipour et al. 2021).
  - **Ranking-stability** — Kendall τ_b across metrics (McPhail 2020; Bonham
    2024).

Deliberately absent: **regret-from-best** (set-relative and design-coupled, so
dropping one design changes every other design's score; needs 400+ scenarios
and never converges on a tail objective — Bonham 2024); the **search-vs-test
overfitting gap** (undefined in Brodeur 2020, and structurally invalid under a
measure change); and any **|baseline|-normalized deviation** (the denominator
is zero exactly in the benign SOWs — flood exceedance and the deficit tails —
so the dropped cells bias the estimator, and Herman et al. 2015 show the
normalized form selects poor-baseline solutions as a "mathematical artifact").
The retired ``improvement_vs_baseline`` was that normalized form; its signed
information survives as ``gain_mean__`` / ``regret_mean__``, the two one-sided
halves of the same incumbent advantage on the same unit.

Self-describing: thresholds/kinds/directions and the objective column order
are read from ``reeval_raw_meta.json`` (snapshotted at simulation time), so
scoring never depends on the live objective registry or a changed
``NYCOPT_SAT_THRESHOLDS`` (the moving-measuring-stick guard, McPhail et al.
2020).
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
    """Dense ``(S, G, M)`` per-SOW objective matrix plus its metadata.

    Attributes:
        cube: float array ``(n_solutions, n_sow, n_objs)`` of per-SOW
            annual-unit objective values in natural units; NaN where a
            (solution, SOW) is missing/failed.
        solution_ids: length-S solution ids (sorted).
        sow_labels: length-G SOW (forcing-profile) ids (sorted union).
        obj_names: length-M annual objective names (column order = meta order).
        thresholds: obj_name -> satisficing threshold (or None).
        kinds: obj_name -> "ge" | "le" (satisficing direction; follows the
            objective's own direction).
        directions: obj_name -> "maximize" | "minimize".
        is_ensemble: False for single-trace re-eval (G == 1, robustness N/A).
        realizations_per_sow: R pooled into each SOW value, or ``None``.
        meta: the full parsed ``reeval_raw_meta.json``.
    """

    cube: np.ndarray
    solution_ids: list
    sow_labels: list
    obj_names: list
    thresholds: dict
    kinds: dict
    directions: dict
    is_ensemble: bool
    meta: dict
    realizations_per_sow: int | None = None

    @property
    def n_sow(self) -> int:
        return self.cube.shape[1]

    def direction_signs(self) -> np.ndarray:
        """+1 for maximize, -1 for minimize, aligned to ``obj_names``."""
        return np.array(
            [1 if self.directions.get(n) == "maximize" else -1
             for n in self.obj_names],
            dtype=int,
        )


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
    """Load the long-format per-SOW matrix + meta and densify to ``(S,G,M)``.

    Missing ``(solution, sow, objective)`` cells (failed solutions, ragged SOW
    coverage) are NaN-filled so metrics align on the union of SOW labels —
    never on positional index.
    """
    reeval_dir = Path(reeval_dir)
    meta_path = reeval_dir / "reeval_raw_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text())

    if "obj_names" not in meta:
        raise ValueError(
            f"{meta_path} predates the per-SOW annual-unit substrate (it has "
            f"no 'obj_names'). Old whole-trace cubes are not scoreable under "
            f"the unified metric currency — re-run workflow steps 05 and "
            f"08/09 to regenerate this re-eval."
        )

    df = _read_long(reeval_dir)
    obj_names = list(meta["obj_names"])
    # Prefer the full attempted solution-id list from meta so fully-failed
    # solutions (no rows) survive as all-NaN cube slices rather than vanishing.
    # Fall back to the ids present in the rows (test fixtures).
    meta_sids = meta.get("solution_ids")
    if meta_sids is not None:
        solution_ids = sorted(int(x) for x in meta_sids)
    else:
        solution_ids = sorted(int(x) for x in df["solution_id"].unique())
    sow_labels = sorted(int(x) for x in df["sow_id"].unique())

    s_ix = {s: i for i, s in enumerate(solution_ids)}
    g_ix = {g: i for i, g in enumerate(sow_labels)}
    o_ix = {o: k for k, o in enumerate(obj_names)}

    cube = np.full((len(solution_ids), len(sow_labels), len(obj_names)),
                   np.nan, dtype=float)
    for sid, gid, obj, val in zip(
        df["solution_id"], df["sow_id"], df["objective"], df["value"]
    ):
        k = o_ix.get(obj)
        if k is None:
            continue
        cube[s_ix[int(sid)], g_ix[int(gid)], k] = val

    return RawCube(
        cube=cube,
        solution_ids=solution_ids,
        sow_labels=sow_labels,
        obj_names=obj_names,
        thresholds={k: meta.get("thresholds", {}).get(k) for k in obj_names},
        kinds={k: meta.get("kinds", {}).get(k) for k in obj_names},
        directions={k: meta.get("directions", {}).get(k) for k in obj_names},
        is_ensemble=bool(meta.get("is_ensemble", True)),
        realizations_per_sow=meta.get("realizations_per_sow"),
        meta=meta,
    )


###############################################################################
# Satisficing (univariate + multivariate Starr domain criterion)
###############################################################################

def _satisfaction_cube(raw: RawCube, thresholds: dict = None,
                       kinds: dict = None) -> np.ndarray:
    """Boolean ``(S, G, M)`` cube: does the per-SOW value meet its threshold?

    Non-finite values are **unsatisfied** (a failed SOW can't masquerade as
    satisficing).

    A missing threshold or kind is a HARD ERROR, not a skipped column. The
    multivariate criterion is a conjunction over all objectives, so an
    all-False column would silently drive the primary metric to 0 for every
    solution, reading as "nothing is ever robust", with no other symptom. The
    only legitimate case (a single-trace re-eval, where a fraction-of-SOWs
    criterion is undefined) is caught upstream by the G == 1 gate in
    :func:`score_robustness`, which NaNs the whole scorecard.

    Raises:
        ValueError: If any objective lacks a threshold or a kind.
    """
    thresholds = thresholds if thresholds is not None else raw.thresholds
    kinds = kinds if kinds is not None else raw.kinds
    missing = [
        n for n in raw.obj_names
        if thresholds.get(n) is None or kinds.get(n) is None
    ]
    if missing:
        raise ValueError(
            f"No satisficing threshold/kind for {missing}. The multivariate "
            f"criterion is a conjunction, so a missing column would silently "
            f"zero the primary metric for every solution. Supply a threshold "
            f"for every objective (use +/-inf to make one non-binding)."
        )
    return _satisfy(raw.cube, raw.obj_names, thresholds, kinds)


def _satisfy(cube: np.ndarray, obj_names: list, thresholds: dict,
             kinds: dict) -> np.ndarray:
    """Boolean satisfaction cube for an ``(S, G, M)`` per-SOW array."""
    S, G, M = cube.shape
    sat = np.zeros((S, G, M), dtype=bool)
    for k, name in enumerate(obj_names):
        thr, kind = thresholds[name], kinds[name]
        slab = cube[:, :, k]
        finite = np.isfinite(slab)
        if kind == "ge":
            sat[:, :, k] = finite & (slab >= thr)
        else:
            sat[:, :, k] = finite & (slab <= thr)
    return sat


def satisficing_multivariate_sow(raw: RawCube, thresholds: dict = None,
                                 kinds: dict = None) -> pd.Series:
    """Starr (1962) domain criterion [THE ADOPTED PRIMARY].

    The fraction of deeply-uncertain states of the world in which the per-SOW
    annual-unit objective vector meets ALL thresholds jointly — the Herman et
    al. (2014) / Trindade et al. (2017) / Gold et al. (2023) construction,
    computed on the search's own objective statistics recomputed per SOW.

    **Precision is governed by N_theta, not by N_test.** Adding realizations
    per SOW sharpens each SOW's pooled estimate but adds no new states of the
    world; only more thetas do.
    """
    sat = _satisfaction_cube(raw, thresholds, kinds)
    joint = sat.all(axis=2).mean(axis=1)  # (S,)
    return pd.Series(
        joint, index=pd.Index(raw.solution_ids, name="solution_id"),
        name="sat_multivariate_sow",
    )


def satisficing_univariate_sow(raw: RawCube, thresholds: dict = None,
                               kinds: dict = None) -> pd.DataFrame:
    """Per-objective satisficing fraction: the PRIMARY's decomposition.

    Same unit, same per-SOW values, conjunction dropped.

    Returns:
        ``(S x M)`` DataFrame of ``sat_uni_sow__{name}`` columns.
    """
    sat = _satisfaction_cube(raw, thresholds, kinds)
    return pd.DataFrame(
        sat.mean(axis=1), index=pd.Index(raw.solution_ids, name="solution_id"),
        columns=[f"sat_uni_sow__{n}" for n in raw.obj_names],
    )


###############################################################################
# Risk-attitude anchors (McPhail et al. 2018, T3)
###############################################################################

def _reduce_over_sows(values: np.ndarray, agg: Callable) -> np.ndarray:
    """Reduce an ``(S, G)`` slice over SOWs (axis 1), NaN-safe.

    The composability seam: ``agg=np.nanmean`` is Laplace, ``agg=np.nanmin``
    on an oriented slab is maximin, and a percentile or Hurwicz blend is a
    thin wrapper over the same matrix.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return agg(values, axis=1)


def laplace_mean(raw: RawCube) -> pd.DataFrame:
    """Mean per-SOW performance, per objective (McPhail T3 = mean).

    The risk-neutral anchor (Laplace's principle of insufficient reason).
    Reported in the objective's NATURAL orientation, so "higher is better"
    follows the objective's own direction.
    """
    S, G, M = raw.cube.shape
    out = np.full((S, M), np.nan)
    for k in range(M):
        out[:, k] = _reduce_over_sows(raw.cube[:, :, k], np.nanmean)
    return pd.DataFrame(
        out, index=pd.Index(raw.solution_ids, name="solution_id"),
        columns=[f"laplace__{n}" for n in raw.obj_names],
    )


def maximin(raw: RawCube) -> pd.DataFrame:
    """Worst-SOW performance, per objective (McPhail T3 = worst).

    The risk-averse anchor (Wald). Computed on the direction-oriented slab and
    returned in natural units, so it is the worst SOW's value: the minimum for
    a maximize objective, the maximum for a minimize objective.
    """
    signs = raw.direction_signs()
    S, G, M = raw.cube.shape
    out = np.full((S, M), np.nan)
    for k in range(M):
        oriented = raw.cube[:, :, k] * signs[k]
        out[:, k] = _reduce_over_sows(oriented, np.nanmin) * signs[k]
    return pd.DataFrame(
        out, index=pd.Index(raw.solution_ids, name="solution_id"),
        columns=[f"maximin__{n}" for n in raw.obj_names],
    )


###############################################################################
# Incumbent-relative regret
###############################################################################
# WHAT THIS IS, in the McPhail et al. (2018) T1/T2/T3 scheme:
#
#   T1 = regret from a fixed BASELINE DECISION ALTERNATIVE, evaluated per
#        SOW. McPhail license this reference explicitly -- "Alternative
#        metrics that are based on the relative performance of decision
#        alternatives use some type of baseline performance for a given
#        scenario instead of the performance of the best decision alternative"
#        -- but never name, tabulate, or test it. Making it explicit is the
#        contribution; it is not a new metric family.
#   T2 = the ADVERSE SUBSET (the SOWs in which the policy is worse than the
#        incumbent). This is not ad-hoc clipping: it is exactly the subset
#        selection of McPhail's "undesirable deviations" metric (Kwakkel et
#        al. 2016b: T1 = regret from median, T2 = worst-half, T3 = sum), with
#        the reference changed from the policy's own median to the incumbent.
#   T3 = mean / 90th percentile over SOWs. The 90th percentile follows Herman
#        et al. (2015) R1/R2, "intended to reflect the tail end of poor
#        performance while reducing susceptibility to outliers".
#
# WHY IT IS NOT REDUNDANT WITH SATISFICING. The satisficing criteria are fixed
# scalars; the regret reference is the incumbent's performance in THAT SOW, a
# bar that moves with the forcing. Where the fixed criterion drives the domain
# criterion to 0 for every policy, satisficing ties everything -- Bonham et
# al. (2024)'s saturation failure mode -- and regret still separates policies.
#
# WHY NATURAL UNITS, AND NO CROSS-OBJECTIVE SCALAR. Dividing by the baseline's
# own per-SOW value is degenerate for this objective set: flood exceedance is
# EXACTLY 0 in a large share of SOWs and both deficit tails are 0 in wet ones,
# so the cell would be dropped -- and the dropped cells are precisely the
# benign ones, biasing the estimator toward the flood-active subset. Working
# in natural units dissolves that rather than patching it: no denominator, no
# dropped cell, every SOW contributes. Herman et al. (2015) hit the same wall
# ("normalized by the objective value itself rather than the best value
# because the latter often approaches zero"), and show the normalized form
# rewards poor baseline performance outright; Eker & Kwakkel (2018) add +1 to
# both terms for the same reason. The two published scales are both unusable
# here -- Cohen et al. (2021) normalize on the per-scenario span to a
# PERFECT-FORESIGHT optimum (one MOEA run per scenario, out of budget), and
# Sunkara et al. (2023) rescale over the ALTERNATIVE SET, which is
# design-coupled and so carries the exact defect that disqualifies best-in-set
# regret. The cost of natural units is that there is no cross-objective regret
# scalar; the unit-free harm FREQUENCIES below carry that role instead.
#
# NO MAX REGRET. Bonham et al. (2024): regret families need 400+ scenarios and
# never converge on extreme-of-extremes operators. McPhail et al. document the
# tie-degeneracy directly ("many of the decision alternatives have a
# reliability of 0% in the worst-case scenario ... the maximin metric ...
# ranks many of the decision alternatives as equal").

#: Decree-party grouping of the annual objectives, for the party-level harm
#: frequencies. The DRB renegotiation is unanimity-bound, so one party's loss
#: is NOT compensable by another's gain -- which is why the party summary is a
#: FREQUENCY over a disjunction and never a summed or averaged party score.
#: Sunkara et al. (2023) document what the compensating form costs: their
#: all-actor metric looks stable only because "the water supply sector may
#: fail in certain scenarios, but those failures are in aggregate countered by
#: increasing levels of success for the ecology-MEF sector".
#: Judgment call recorded: NYC aggregate storage sits under `nyc` because the
#: objective is NYC's own supply security, though the same storage also
#: underwrites downstream release capability.
DECREE_PARTY_OBJECTIVES: dict[str, tuple[str, ...]] = {
    "nyc": (
        "nyc_delivery_reliability_annual",
        "nyc_delivery_deficit_p99_pct",
        "nyc_storage_min_p01_pct",
    ),
    "nj": (
        "nj_delivery_reliability_annual",
    ),
    "downstream_flow": (
        "montague_flow_reliability_annual",
        "montague_flow_deficit_p99_pct",
        "trenton_flow_reliability_annual",
    ),
    # One active flood objective; the inactive day-count diagnostic joins this
    # disjunction only if it ever enters the active set.
    "flood_exposed": (
        "downstream_flood_exceedance_annual",
    ),
}

#: Multiplier ``k`` on the per-objective just-noticeable difference that defines
#: the no-harm tolerance ``tau_i = k * eps_i``. MUST be declared before results are
#: inspected (the manuscript pre-specifies its other endpoints); override with
#: ``NYCOPT_REGRET_TAU_K``. k = 0 is the strict weak-Pareto-improvement form.
REGRET_TAU_K: float = float(os.environ.get("NYCOPT_REGRET_TAU_K", "1"))


def _aligned_baseline(raw: RawCube, baseline: RawCube) -> np.ndarray:
    """The incumbent's ``(G, M)`` per-SOW vector, aligned to ``raw.sow_labels``.

    The baseline cube holds ONE policy; its solution axis is collapsed
    (``nanmean`` is an identity for S == 1 and averages any accidental
    duplicates), and the join is on the SOW LABEL, never on position. A
    baseline missing SOWs the policy cube covers is a HARD ERROR: a NaN
    incumbent row would count as harm for EVERY policy in
    ``regret_frequencies`` (non-finite differences are harm by convention),
    so a partially-failed incumbent would silently degrade the whole
    comparison rather than one cell.
    """
    if list(baseline.obj_names) != list(raw.obj_names):
        raise ValueError(
            f"baseline objectives {baseline.obj_names} do not match the "
            f"policy cube's {raw.obj_names}"
        )
    uncovered = sorted(set(raw.sow_labels) - set(baseline.sow_labels))
    if uncovered:
        raise ValueError(
            f"the incumbent baseline cube covers {len(baseline.sow_labels)} "
            f"SOWs but the policy cube scores {len(raw.sow_labels)}; "
            f"{len(uncovered)} SOWs are uncovered (first few: "
            f"{uncovered[:8]}). A NaN incumbent row reads as harm for every "
            f"policy — re-run step 05 on the same test ensemble."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_vec = np.nanmean(baseline.cube, axis=0)  # (G_base, M)
    by_label = {lab: base_vec[j, :] for j, lab in enumerate(baseline.sow_labels)}
    aligned = np.full((raw.cube.shape[1], raw.cube.shape[2]), np.nan, dtype=float)
    for j, lab in enumerate(raw.sow_labels):
        row = by_label.get(lab)
        if row is not None:
            aligned[j, :] = row
    return aligned


def incumbent_advantage(raw: RawCube, baseline: RawCube) -> np.ndarray:
    """Signed advantage over the status quo, ``(S, G, M)``, in NATURAL UNITS.

    ``D_i(x, theta) = sign_i * (J_i(x, theta) - J_i(b, theta))`` where
    ``sign_i`` is +1 for a maximize objective and -1 for a minimize one, so
    **positive always means BETTER than current operations** whatever the
    objective's own direction. ``J`` is the per-SOW annual-unit objective —
    the search statistic recomputed per state of the world.

    This is the substrate every other quantity in this section is a view of;
    it is a pure function of two cubes that already exist, so nothing needs
    persisting.
    """
    base = _aligned_baseline(raw, baseline)
    signs = raw.direction_signs()
    return signs[None, None, :] * (raw.cube - base[None, :, :])


def regret_magnitudes(raw: RawCube, baseline: RawCube) -> pd.DataFrame:
    """Per-objective regret and gain magnitudes, in each objective's OWN units.

    Reads directly: "the worst 1% of unit-years' minimum storage is 3.2
    percentage points lower than under current operations". These columns are
    never summed, averaged, or compared ACROSS objectives -- see the section
    header for why there is no scalar.

    Columns, per objective:
      - ``regret_mean__``  mean over SOWs of ``max(0, -D)``  (risk-neutral)
      - ``regret_q90__``   90th percentile of ``max(0, -D)``  (tail; Herman R1/R2)
      - ``regret_cond__``  mean regret GIVEN a shortfall, i.e. ``mean(-D | D < 0)``;
        **NaN when the policy is never worse than the incumbent** on that
        objective, which is a real distinction from a shortfall of size zero and
        is why it is not filled with 0. Read it beside ``harm_freq__``.
      - ``gain_mean__``    mean over SOWs of ``max(0, +D)``; the degeneracy
        companion, because a policy scores zero regret by BEING the incumbent.
    """
    D = incumbent_advantage(raw, baseline)                          # (S, G, M)
    G_ = np.where(np.isfinite(D), np.maximum(0.0, -D), np.nan)
    P = np.where(np.isfinite(D), np.maximum(0.0, D), np.nan)
    adverse = np.where(np.isfinite(D) & (D < 0), -D, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> NaN
        cols = {}
        for k, name in enumerate(raw.obj_names):
            cols[f"regret_mean__{name}"] = np.nanmean(G_[:, :, k], axis=1)
            cols[f"regret_q90__{name}"] = np.nanquantile(G_[:, :, k], 0.90, axis=1)
            cols[f"regret_cond__{name}"] = np.nanmean(adverse[:, :, k], axis=1)
            cols[f"gain_mean__{name}"] = np.nanmean(P[:, :, k], axis=1)

    ordered = [f"{pre}__{n}"
               for n in raw.obj_names
               for pre in ("regret_mean", "regret_q90", "regret_cond", "gain_mean")]
    return pd.DataFrame(
        {c: cols[c] for c in ordered},
        index=pd.Index(raw.solution_ids, name="solution_id"),
    )


def tau_ladder(obj_names: list, k: float = None, floors: dict = None) -> dict:
    """Per-objective no-harm tolerance in natural units: ``tau_i = k * u_i``.

    The tolerance UNIT ``u_i`` is:

    - the objective's ANNUAL-UNIT epsilon from
      ``src.objectives_ensemble.ENSEMBLE_OBJECTIVES`` -- the campaign's own
      calibrated just-noticeable difference (epsilon-calibration experiment:
      clean-ceil of max(signal IQR/10, bootstrap noise floor, frequency
      granularity)), measured in exactly the annual-unit metric space the
      per-SOW cube lives in. The regret tolerance and the search resolution
      are on ONE calibration, so the ladder reads "no objective degraded by
      more than k search-resolution steps";
    - ``max(eps_i, floor_i)`` when ``floors`` is supplied, where ``floor_i`` is
      the measured noise floor of that objective's per-SOW estimator.

    **Why floors exist.** An epsilon BELOW its objective's noise floor makes
    every rung meaningless on that axis -- the criterion fires on Monte Carlo
    noise rather than on harm -- and because one ``k`` is shared across
    objectives, a single such axis silently sets what every rung means. Taking
    the max keeps epsilon where resolution binds and the floor where noise
    binds, so ``k`` means the same thing on every axis. Floors are measured by
    ``scripts/supplemental/regret_tolerance_diagnostics.py`` (pass A) and are
    a property of the incumbent alone, so using them is not circular.

    Env override ``NYCOPT_REGRET_TAU`` (JSON ``{obj_name: tau}``) replaces the
    WHOLE vector, for the case where an adopted vector is recorded rather than
    derived. It must cover every objective: a partial override would leave the
    rest on a different tolerance basis without saying so. The override states
    the tolerance at the ADOPTED rung ``NYCOPT_REGRET_TAU_K``, so it is scaled
    by ``k / REGRET_TAU_K`` -- at the adopted rung that is the identity, and a
    k-sweep still sweeps.

    Args:
        obj_names: Annual objective names, in cube column order.
        k: Multiplier; defaults to :data:`REGRET_TAU_K`. ``k = 0`` gives the
            strict weak-Pareto-improvement form.
        floors: Optional ``{obj_name: noise_floor}`` in natural units.

    Returns:
        ``{obj_name: tau}``.

    Raises:
        KeyError: If an objective name has neither a registered epsilon nor an
            override -- a silent 0 would turn a tolerance into a strict
            criterion without saying so -- or if the env override is partial
            or names a stranger.
    """
    from src.objectives_ensemble import ENSEMBLE_OBJECTIVES

    raw = os.environ.get("NYCOPT_REGRET_TAU", "").strip()
    if raw:
        override = json.loads(raw)
        unknown = [n for n in override if n not in obj_names]
        if unknown:
            raise KeyError(
                f"NYCOPT_REGRET_TAU names objectives absent from this cube: {unknown}"
            )
        absent = [n for n in obj_names if n not in override]
        if absent:
            raise KeyError(
                f"NYCOPT_REGRET_TAU is a WHOLE-vector override but omits {absent}; a "
                f"partial vector would leave those objectives on a different "
                f"tolerance basis than the rest."
            )
        # The override is the tolerance AT THE ADOPTED RUNG, so it supplies the
        # ladder's UNIT, not a constant. Returning it unscaled made every rung
        # of a k-sweep identical, which silently turned the tolerance profile
        # (and the figure drawn from it) into a flat line whose shape was an
        # artifact of the override rather than a property of the metric.
        scale = REGRET_TAU_K if k is None else float(k)
        unit = 1.0 if REGRET_TAU_K == 0 else scale / REGRET_TAU_K
        return {n: float(override[n]) * unit for n in obj_names}

    if not floors:
        # No adopted vector AND no measured floors: this is the eps-only
        # ladder, which is NOT the adopted basis (six of eight adopted taus
        # are floor-bound, not epsilon-bound). Legitimate for the pass-B
        # k-sweep, which unsets the override on purpose -- but silent drift
        # onto a different tolerance basis is exactly what the whole-vector
        # override exists to prevent, so say so.
        warnings.warn(
            "regret tau: no NYCOPT_REGRET_TAU override and no measured "
            "floors, so the tolerance falls back to k*epsilon. This is NOT "
            "the adopted vector (see workflow/envs/*.env); set "
            "NYCOPT_REGRET_TAU, or pass floors=, unless you intend the "
            "eps-only ladder."
        )

    k = REGRET_TAU_K if k is None else float(k)
    missing = [n for n in obj_names if n not in ENSEMBLE_OBJECTIVES]
    if missing:
        raise KeyError(
            f"no registered annual objective for {missing}, so no epsilon to build "
            f"a tolerance from. Pass an explicit tau dict instead of defaulting to "
            f"0, which would silently harden the criterion."
        )
    floors = floors or {}
    return {n: k * max(float(ENSEMBLE_OBJECTIVES[n].epsilon),
                       float(floors.get(n, 0.0)))
            for n in obj_names}


def adopted_floors() -> dict | None:
    """Measured per-objective noise floors from pass A, or ``None`` if absent.

    Reads the ``rtol_floors.json`` written by
    ``scripts/supplemental/regret_tolerance_diagnostics.run_pass_a`` so that
    k-sweeps which deliberately unset ``NYCOPT_REGRET_TAU`` still sweep the
    adopted ``max(eps, floor)`` basis via ``tau_ladder(floors=...)`` instead of
    silently dropping to the eps-only ladder (most adopted taus are
    floor-bound, not epsilon-bound).

    Returns:
        ``{obj_name: tau_floor}`` in natural units, or ``None`` when pass A has
        not written its floors table.
    """
    import supplemental_config as sc

    path = sc.RTOL_TABLES_DIR / "rtol_floors.json"
    if not path.exists():
        return None
    return {str(name): float(v)
            for name, v in json.loads(path.read_text()).items()}


def regret_frequencies(raw: RawCube, baseline: RawCube, tau: dict = None,
                       parties: dict = None, axes=None) -> pd.DataFrame:
    """Unit-free harm frequencies. These carry the scalar role.

    Because the magnitudes above stay in natural units, the cross-objective and
    cross-policy summaries are frequencies, which need no normalization at all.

    Columns:
      - ``harm_freq__{obj}``        fraction of SOWs with ``D_i < 0``
      - ``party_harm_freq__{party}``  fraction of SOWs in which ANY objective of
        that Decree party is worse off (a disjunction, never a sum -- see
        :data:`DECREE_PARTY_OBJECTIVES`)
      - ``no_harm_freq``            fraction of SOWs with ``D_i >= 0`` for ALL i:
        a weak Pareto improvement on the incumbent. Cohen et al. (2021) apply this
        exact condition as a hard FILTER ("solutions that will at a minimum
        outperform the status quo in all re-evaluations"); reporting it as a graded
        frequency keeps the policies whose trade-off against the incumbent is the
        interesting part of the question.
      - ``no_harm_freq_tau``        fraction of SOWs with ``D_i >= -tau_i`` for
        ALL i -- the literal reading of "improves some outcomes without degrading
        others below current performance".
      - ``n_degraded_mean``         mean number of objectives simultaneously worse
        off, the informative decomposition of ``no_harm_freq`` (which is small by
        construction when there are 8 objectives with genuine trade-offs).

    A non-finite ``D`` counts as HARM, mirroring the non-finite-as-unsatisfied rule
    of the satisficing path: a degenerate SOW must not read as "no harm".

    ``axes`` restricts the whole computation to a subset of objectives (a
    criterion set's member axes): per-objective columns are emitted only for
    those axes, party disjunctions only for parties with a member among them,
    and the joint no-harm conjunctions run over the subset. Default None =
    all objectives (the global frequencies).
    """
    D = incumbent_advantage(raw, baseline)                          # (S, G, M)
    tau = tau_ladder(raw.obj_names) if tau is None else tau
    missing = [n for n in raw.obj_names if n not in tau]
    if missing:
        raise KeyError(f"no tolerance supplied for {missing}")
    tau_vec = np.array([float(tau[n]) for n in raw.obj_names], dtype=float)

    if axes is not None:
        unknown = [n for n in axes if n not in raw.obj_names]
        if unknown:
            raise KeyError(f"axes not in this cube: {unknown}")
        keep = [k for k, n in enumerate(raw.obj_names) if n in set(axes)]
        D = D[:, :, keep]
        tau_vec = tau_vec[keep]
        names = [raw.obj_names[k] for k in keep]
    else:
        names = list(raw.obj_names)

    finite = np.isfinite(D)
    harm = (~finite) | (D < 0)                                      # (S, G, M')
    harm_tau = (~finite) | (D < -tau_vec[None, None, :])

    index = pd.Index(raw.solution_ids, name="solution_id")
    out = pd.DataFrame(index=index)
    for k, name in enumerate(names):
        out[f"harm_freq__{name}"] = harm[:, :, k].mean(axis=1)

    parties = DECREE_PARTY_OBJECTIVES if parties is None else parties
    col_of = {n: k for k, n in enumerate(names)}
    for party, members in parties.items():
        idx = [col_of[n] for n in members if n in col_of]
        if not idx:
            continue
        out[f"party_harm_freq__{party}"] = harm[:, :, idx].any(axis=2).mean(axis=1)

    out["no_harm_freq"] = (~harm).all(axis=2).mean(axis=1)
    out["no_harm_freq_tau"] = (~harm_tau).all(axis=2).mean(axis=1)
    out["n_degraded_mean"] = harm.sum(axis=2).mean(axis=1)
    return out


def incumbent_spread(baseline: RawCube) -> dict:
    """Per-objective ``q90 - q10`` of the incumbent's per-SOW values.

    The OPTIONAL normalization scale, offered so a reviewer asking for a
    dimensionless regret can be answered without re-simulating -- and so rank
    agreement between the natural-unit and normalized orderings can be checked
    with Kendall's tau_b. It is never the reported primary (see the section
    header).

    Unlike a per-cell baseline denominator it is a single fixed vector: non-zero
    by construction (the incumbent's performance varies across the DU box),
    publishable as a table, and independent of which policies or designs are in
    the comparison. It is the simulation-free stand-in for Cohen et al.
    (2021)'s achievable span.

    Raises:
        ValueError: If any objective's spread is zero -- silently dividing by it
            would produce infinities that read as catastrophic regret.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        flat = np.nanmean(baseline.cube, axis=0)                    # (G, M)
        hi = np.nanquantile(flat, 0.90, axis=0)
        lo = np.nanquantile(flat, 0.10, axis=0)
    spread = hi - lo
    dead = [n for n, s in zip(baseline.obj_names, spread)
            if not np.isfinite(s) or s <= 0]
    if dead:
        raise ValueError(
            f"the incumbent's q90-q10 spread is zero/non-finite for {dead}, so it "
            f"cannot serve as a regret scale. Report natural units for these."
        )
    return {n: float(s) for n, s in zip(baseline.obj_names, spread)}


###############################################################################
# Criterion-set scoring (Quinn 2017 subset criteria)
###############################################################################

def criterion_shortfall(raw: RawCube, thresholds: dict,
                        kinds: dict = None) -> pd.DataFrame:
    """Satisficing-regret: how far below a criterion the failing SOWs sit.

    The McPhail et al. (2021) satisficing-regret transform ``max(0, c - f)``:
    per axis with a FINITE threshold, the shortfall of the per-SOW value from
    the criterion -- ``max(0, thr - v)`` for "ge" axes, ``max(0, v - thr)``
    for "le" -- aggregated over SOWs. Where the binary Starr count saturates
    (all-pass or all-fail), the shortfall still discriminates: two policies
    failing the same SOWs differ in how badly they miss.

    Values stay in each objective's NATURAL units and are never summed across
    objectives (the module's no-composite-scalar rule). Non-finite cells are
    NaN -- a failed SOW has no defined shortfall magnitude; its frequency is
    already carried by the satisficing fraction.

    Args:
        raw: The per-SOW re-eval cube.
        thresholds: Full threshold vector; only axes with finite values are
            scored (non-binding ``+/-inf`` axes are skipped).
        kinds: ``{objective: "ge"|"le"}``; defaults to the cube's snapshot.

    Returns:
        Per-solution frame with ``shortfall_mean__{name}`` (mean over SOWs,
        passes contributing 0) and ``shortfall_q90__{name}`` (90th percentile
        over SOWs, the Herman et al. (2015) tail emphasis) for each scored
        axis.
    """
    kinds = kinds if kinds is not None else raw.kinds
    index = pd.Index(raw.solution_ids, name="solution_id")
    out = pd.DataFrame(index=index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, name in enumerate(raw.obj_names):
            thr = thresholds.get(name)
            if thr is None or not np.isfinite(thr):
                continue
            slab = raw.cube[:, :, k]                                # (S, G)
            miss = thr - slab if kinds[name] == "ge" else slab - thr
            shortfall = np.where(np.isfinite(slab),
                                 np.maximum(0.0, miss), np.nan)
            out[f"shortfall_mean__{name}"] = np.nanmean(shortfall, axis=1)
            out[f"shortfall_q90__{name}"] = np.nanquantile(shortfall, 0.90,
                                                           axis=1)
    return out


def score_criteria(raw: RawCube, baseline: Optional[RawCube] = None,
                   sets=None) -> tuple[pd.DataFrame, dict]:
    """Per-solution scorecard under the named criterion sets.

    For each :class:`~src.satisficing_criteria.CriterionSet`:

    - ``sat_set__{key}``: the Starr domain criterion under the set's
      threshold vector (member axes at their placement, all others
      non-binding) -- the same SOW-counting unit as the primary metric.
      ``reference_all8`` reproduces ``sat_multivariate_sow`` exactly.
    - ``shortfall_{mean,q90}__{key}__{obj}``: satisficing-regret magnitudes
      for the set's member axes (:func:`criterion_shortfall`), namespaced by
      set because placements for a shared axis may differ between sets.
    - ``no_harm_freq_tau__{key}`` (baseline runs only): the incumbent-relative
      no-harm frequency with the harm conjunction restricted to the set's
      member axes -- "does the policy avoid harming the incumbent on THIS
      framing's axes", the criterion-conditional companion of the global
      ``no_harm_freq_tau``.

    Args:
        raw: The per-SOW re-eval cube.
        baseline: Status-quo cube on the same ensemble (enables the per-set
            no-harm columns).
        sets: Criterion sets; defaults to
            ``satisficing_criteria.ALL_SETS``.

    Returns:
        ``(scorecard, higher_better)`` in the :func:`score_robustness` sense.
    """
    from src.satisficing_criteria import ALL_SETS

    sets = ALL_SETS if sets is None else sets
    # A set whose member axes are absent from this cube (synthetic fixtures,
    # alternative formulations) is skipped with a warning rather than raised:
    # the named sets describe the production 8-objective schema, and run()
    # scores every cube it is pointed at.
    skipped = [c.key for c in sets if not c.reference
               and any(a not in raw.obj_names for a in c.axes)]
    if skipped:
        warnings.warn(
            f"criterion sets {skipped} name objectives absent from this cube "
            f"({list(raw.obj_names)}); skipping them."
        )
        sets = [c for c in sets if c.key not in skipped]

    index = pd.Index(raw.solution_ids, name="solution_id")
    if (not raw.is_ensemble) or raw.n_sow <= 1 or not sets:
        cols = [f"sat_set__{c.key}" for c in sets]
        return (pd.DataFrame(np.nan, index=index, columns=cols),
                {c: True for c in cols})

    try:
        tau = tau_ladder(raw.obj_names)
    except KeyError:
        tau = None

    pieces, higher_better = [], {}
    for cset in sets:
        thresholds = cset.thresholds(raw.thresholds, raw.kinds)
        sat = satisficing_multivariate_sow(raw, thresholds)
        col = f"sat_set__{cset.key}"
        pieces.append(sat.rename(col).to_frame())
        higher_better[col] = True

        if not cset.reference:
            shortfall = criterion_shortfall(raw, thresholds)
            shortfall.columns = [
                c.replace("__", f"__{cset.key}__", 1)
                for c in shortfall.columns
            ]
            pieces.append(shortfall)
            higher_better.update({c: False for c in shortfall.columns})

        if baseline is not None and tau is not None and not cset.reference:
            freq = regret_frequencies(raw, baseline, tau=tau, axes=cset.axes)
            col = f"no_harm_freq_tau__{cset.key}"
            pieces.append(freq["no_harm_freq_tau"].rename(col).to_frame())
            higher_better[col] = True

    scorecard = pd.concat(pieces, axis=1)
    all_missing = np.all(~np.isfinite(raw.cube), axis=(1, 2))
    if all_missing.any():
        scorecard.iloc[all_missing, :] = np.nan
    return scorecard, higher_better


def criterion_ranking_stability(per_set: pd.DataFrame) -> pd.DataFrame:
    """Kendall τ_b between the solution rankings of each criterion set.

    The Quinn et al. (2017) conclusion-invariance check: if the same policies
    rank as most robust under every stakeholder framing, the criteria choice
    eases rather than raises tension. Operates on the ``sat_set__*`` columns
    of :func:`score_criteria`'s scorecard (all higher-better).
    """
    cols = [c for c in per_set.columns if c.startswith("sat_set__")]
    return ranking_stability(per_set[cols], {c: True for c in cols})


###############################################################################
# Attainability screen
###############################################################################

def attainability_screen(raw: RawCube, thresholds: dict = None,
                         kinds: dict = None) -> pd.DataFrame:
    """Per-SOW: can ANY solution in this set meet all the criteria?

    Separates "this design searched badly" from "this state of the world is
    unwinnable for anyone" -- a distinction that is otherwise invisible, and
    that matters: Shavazipour et al. (2021) found 23% of their test scenarios
    could not meet the reliability criterion under ANY feasible policy, so the
    satisficing ceiling was structural rather than a search failure.

    This is the free substitute for a per-scenario oracle. It costs zero extra
    simulation (the cube already exists), but it is an EMPIRICAL attainability
    bound, not a true ceiling: it says only that no policy *in this set* wins
    the SOW, not that none exists. Report it as such. Pool the cubes of all
    designs before calling this if the question is "unwinnable by anyone."

    Returns a tidy frame (sow_id, n_satisficing_solutions, attainable, plus
    per-objective ``anysat__{name}`` columns showing WHICH criterion is
    binding where nothing attains the joint criterion).
    """
    sat = _satisfaction_cube(raw, thresholds, kinds)     # (S, G, M)
    joint = sat.all(axis=2)                              # (S, G)
    n_sat = joint.sum(axis=0)                            # (G,)
    frame = pd.DataFrame({
        "sow_id": raw.sow_labels,
        "n_satisficing_solutions": n_sat.astype(int),
        "attainable": n_sat > 0,
    })
    for k, name in enumerate(raw.obj_names):
        frame[f"anysat__{name}"] = sat[:, :, k].any(axis=0)
    return frame


###############################################################################
# Threshold spectrum (Hadjimichael et al. 2020)
###############################################################################

def threshold_spectrum(raw: RawCube, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90)
                       ) -> pd.DataFrame:
    """Satisficing fraction as a function of the magnitude threshold.

    Robustness depends on the threshold (Hadjimichael et al. 2020): a solution
    robust at one magnitude can be fragile at a neighbor. For each objective
    the threshold grid is the pooled per-SOW-distribution quantiles (plus the
    labeled default from the meta), and satisficing is reported at each.
    Returns a tidy DataFrame (solution_id, objective, threshold, is_default,
    satisficing).
    """
    rows = []
    for k, name in enumerate(raw.obj_names):
        kind = raw.kinds.get(name)
        if kind is None:
            continue
        slab = raw.cube[:, :, k]  # (S, G)
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

def sow_quantiles(raw: RawCube,
                  quantiles=(0.05, 0.25, 0.50, 0.75, 0.95)) -> pd.DataFrame:
    """Per-solution per-objective distribution of per-SOW values.

    A scalar scorecard discards the distribution the matrix exists to preserve;
    this keeps it (Hadjimichael et al. 2023). Tidy: (solution_id, objective, qXX...).
    """
    rows = []
    qcols = [f"q{int(q * 100):02d}" for q in quantiles]
    for k, name in enumerate(raw.obj_names):
        slab = raw.cube[:, :, k]  # (S, G)
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
    "satisficing_multivariate_sow", # PRIMARY (Starr domain criterion, SOW unit)
    "satisficing_univariate_sow",   # the PRIMARY's per-objective decomposition
    "laplace_mean",                 # McPhail T3 = mean   (risk-neutral anchor)
    "maximin",                      # McPhail T3 = worst  (risk-averse anchor)
    "regret_magnitudes",            # incumbent-relative regret, natural units
    "regret_frequencies",           # its unit-free harm frequencies (the scalars)
)

#: Metrics that need the status-quo cube. Requested without one they are skipped
#: with a warning rather than silently returning zeros.
_BASELINE_METRICS = ("regret_magnitudes", "regret_frequencies")


def score_robustness(raw: RawCube, baseline: Optional[RawCube] = None,
                     metrics=_DEFAULT_METRICS, thresholds: dict = None,
                     ) -> tuple[pd.DataFrame, dict]:
    """Assemble the per-solution scorecard for the requested metrics.

    Args:
        raw: The per-SOW re-eval cube.
        baseline: Status-quo cube on the SAME ensemble (enables the
            incumbent-relative regret family).
        metrics: Metric ids to compute.
        thresholds: Optional threshold override (the meta's are used otherwise).

    Returns:
        ``(scorecard, higher_better)`` where ``higher_better`` maps each column to
        whether larger = more robust (for ranking-stability orientation).

    Every metric is defined ACROSS SOWs, so all of them are N/A (NaN) for a
    single-trace re-eval (``is_ensemble`` False / G == 1) — a historical-record
    re-eval is a reference, not a controlled robustness comparison.
    """
    pieces = []
    higher_better: dict = {}
    index = pd.Index(raw.solution_ids, name="solution_id")
    signs = raw.direction_signs()

    # The G == 1 gate must cover the WHOLE scorecard, and gated metrics must
    # not be COMPUTED, only NaN-filled afterwards: a single-trace re-eval may
    # carry no satisficing thresholds, and _satisfaction_cube raises on a
    # missing threshold.
    g1 = (not raw.is_ensemble) or raw.n_sow <= 1

    def _nan_frame(cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(np.nan, index=index, columns=cols)

    def _add(name: str, compute, cols: list[str], higher: dict,
             gated: bool = False) -> None:
        if name not in metrics:
            return
        pieces.append(_nan_frame(cols) if (g1 or gated) else compute())
        higher_better.update(higher)

    _add(
        "satisficing_multivariate_sow",
        lambda: satisficing_multivariate_sow(raw, thresholds).to_frame(),
        ["sat_multivariate_sow"],
        {"sat_multivariate_sow": True},
    )
    _add(
        "satisficing_univariate_sow",
        lambda: satisficing_univariate_sow(raw, thresholds),
        [f"sat_uni_sow__{n}" for n in raw.obj_names],
        {f"sat_uni_sow__{n}": True for n in raw.obj_names},
    )
    # Laplace and maximin are in NATURAL units, so orientation follows each
    # objective's own direction rather than being uniformly "higher is better".
    _add(
        "laplace_mean",
        lambda: laplace_mean(raw),
        [f"laplace__{n}" for n in raw.obj_names],
        {f"laplace__{n}": signs[k] > 0 for k, n in enumerate(raw.obj_names)},
    )
    _add(
        "maximin",
        lambda: maximin(raw),
        [f"maximin__{n}" for n in raw.obj_names],
        {f"maximin__{n}": signs[k] > 0 for k, n in enumerate(raw.obj_names)},
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
        mag_cols = [f"{pre}__{n}"
                    for n in raw.obj_names
                    for pre in ("regret_mean", "regret_q90",
                                "regret_cond", "gain_mean")]
        _add(
            "regret_magnitudes",
            lambda: regret_magnitudes(raw, baseline),
            mag_cols,
            # Regret is a loss: lower is better. Gain is the mirror.
            {c: c.startswith("gain_mean__") for c in mag_cols},
        )

        party_cols = [f"party_harm_freq__{p}" for p, members in
                      DECREE_PARTY_OBJECTIVES.items()
                      if any(n in raw.obj_names for n in members)]
        freq_cols = ([f"harm_freq__{n}" for n in raw.obj_names] + party_cols
                     + ["no_harm_freq", "no_harm_freq_tau", "n_degraded_mean"])
        # The tolerance ladder is resolved HERE, once, so a cube whose objectives
        # are not in the registry (synthetic fixtures) gates the whole block off
        # instead of half-computing it. tau_ladder itself still raises, because a
        # caller supplying a real cube deserves the error rather than a silent 0.
        try:
            tau = tau_ladder(raw.obj_names)
        except KeyError:
            tau = None
        _add(
            "regret_frequencies",
            lambda: regret_frequencies(raw, baseline, tau=tau),
            freq_cols,
            {c: c.startswith("no_harm_freq") for c in freq_cols},
            gated=tau is None,
        )

    scorecard = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=index)

    # A solution with no successful SOWs at all (its whole cube slice is
    # non-finite, e.g. every re-eval batch failed) is scored NaN across every
    # metric, matching its NaN row in objectives_summary.csv. Otherwise
    # satisficing would read 0.0 (worst) and make a *failed* run indistinguishable
    # from a *ran-but-bad* run, distorting cross-solution comparison.
    all_missing = np.all(~np.isfinite(raw.cube), axis=(1, 2))  # (S,)
    if all_missing.any() and len(scorecard.columns):
        scorecard.iloc[all_missing, :] = np.nan
    return scorecard, higher_better


def run(reeval_dir, baseline_dir=None, metrics=_DEFAULT_METRICS) -> Path:
    """Score a re-eval output dir and write the robustness artifacts.

    Writes ``robustness_scorecard.csv``, ``robustness_scorecard_criteria.csv``
    (the per-criterion-set companion), ``robustness_criterion_stability.csv``,
    ``robustness_ranking_stability.csv``, ``robustness_threshold_spectrum.csv``,
    ``robustness_quantiles.csv``, ``robustness_attainability.csv``, and
    ``robustness_meta.json``. Returns the scorecard path.
    """
    from src.satisficing_criteria import ALL_SETS

    reeval_dir = Path(reeval_dir)
    raw = load_raw(reeval_dir)
    baseline = load_raw(baseline_dir) if baseline_dir else None

    scorecard, higher_better = score_robustness(raw, baseline, metrics)

    out = reeval_dir / "robustness_scorecard.csv"
    scorecard.to_csv(out)

    # Per-criterion-set companion scorecard (Quinn 2017 subset criteria) and
    # the cross-set conclusion-invariance matrix. Written beside -- never
    # into -- the main scorecard, so existing consumers are unaffected.
    criteria_scorecard, _ = score_criteria(raw, baseline, ALL_SETS)
    criteria_scorecard.to_csv(reeval_dir / "robustness_scorecard_criteria.csv")
    criterion_ranking_stability(criteria_scorecard).to_csv(
        reeval_dir / "robustness_criterion_stability.csv")

    # Every scoring-time choice that MOVES a number is recorded next to the
    # numbers rather than left implicit in a default: the no-harm tolerance
    # ladder that defines `no_harm_freq_tau`.
    meta = {
        "metrics": list(metrics),
        "substrate": raw.meta.get("substrate"),
        "n_solutions": len(raw.solution_ids),
        "n_sow": raw.n_sow,
        "realizations_per_sow": raw.realizations_per_sow,
        "regret_available": baseline is not None,
        "regret_unit": "sow",
        "regret_tau_k": REGRET_TAU_K,
    }
    try:
        meta["regret_tau"] = tau_ladder(raw.obj_names)
    except KeyError:
        meta["regret_tau"] = None
    # The criterion sets are scoring-time choices that move numbers, so the
    # full resolved vectors are snapshotted (moving-measuring-stick guard,
    # extended to sets). Sets naming axes absent from this cube are omitted,
    # matching score_criteria's skip.
    meta["criterion_sets"] = {
        c.key: {
            "axes": list(c.axes),
            "thresholds": {
                n: (None if not np.isfinite(v) else v)
                for n, v in c.thresholds(raw.thresholds, raw.kinds).items()
            },
            "reference": c.reference,
        }
        for c in ALL_SETS
        if c.reference or all(a in raw.obj_names for a in c.axes)
    }
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
    sow_quantiles(raw).to_csv(
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
                        help="Status-quo re-eval dir (enables the regret family). "
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

    if args.reeval_dir:
        reeval_dir = Path(args.reeval_dir)
    elif args.formulation:
        reeval_dir = _resolve_default_reeval_dir(args.formulation, args.seed)
    else:
        parser.error("provide --reeval-dir or --formulation")

    # Auto-detect the status-quo re-eval matrix (written by
    # `run_baseline.py --reeval` under `<reeval_dir>/baseline`) so the regret
    # family works without setting NYCOPT_REEVAL_BASELINE_DIR.
    # NOTE: step 05 must be run with the SAME NYCOPT_REEVAL_ENSEMBLE_PRESET as
    # step 08, or the baseline lands under a different reeval tag and this finds
    # nothing -- the metrics are then silently skipped.
    baseline_dir = args.baseline_dir
    if baseline_dir is None:
        auto = reeval_dir / "baseline"
        if any((auto / f).exists() for f in
               ("reeval_raw.parquet", "reeval_raw.csv.gz")):
            baseline_dir = str(auto)
            print(f"[robustness] auto-detected baseline dir -> {baseline_dir}")

    run(reeval_dir, baseline_dir, metrics)


if __name__ == "__main__":
    main()
