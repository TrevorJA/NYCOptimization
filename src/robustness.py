"""
robustness.py - Offline robustness scoring from the persisted per-SOW matrix.

The re-eval drivers (``src.reevaluate`` / ``src.reevaluate_mpi`` /
``src.chunk_reeval``) persist the per-SOW annual-unit objective matrix
(``reeval_raw.parquet`` + ``reeval_raw_meta.json``): each E_test SOW's R
realizations pooled through the §2 unit operators, i.e. the search objectives
recomputed per state of the world. This module scores robustness from that
matrix offline, without re-simulating. Rationale for the metric set lives in
docs/notes/methods/objective_definitions.md and experimental_design.md.

The SOW is the only scoring unit. Metrics:

  - multivariate Starr satisficing (``sat_multivariate_sow``) [PRIMARY]: the
    fraction of SOWs whose objective vector meets all thresholds jointly;
  - univariate satisficing (``sat_uni_sow__``): its per-objective decomposition;
  - Laplace mean and maximin over SOWs, in natural units;
  - incumbent-relative regret: one-sided adverse deviation from the status-quo
    FFMP policy on the same SOWs, in natural units, plus unit-free harm
    frequencies;
  - threshold spectrum, attainability screen, and ranking stability
    (Kendall tau_b across metrics).

Thresholds, kinds, directions, and the objective column order are read from
``reeval_raw_meta.json`` (snapshotted at simulation time), so scoring never
depends on the live registry or a changed ``NYCOPT_SAT_THRESHOLDS``.
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
    """Starr (1962) domain criterion [PRIMARY].

    The fraction of SOWs in which the per-SOW annual-unit objective vector
    meets ALL thresholds jointly. Precision is governed by the SOW count, not
    by the realizations per SOW.
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
# Risk-attitude references (Laplace mean, maximin)
###############################################################################

def _reduce_over_sows(values: np.ndarray, agg: Callable) -> np.ndarray:
    """Reduce an ``(S, G)`` slice over SOWs (axis 1), NaN-safe."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return agg(values, axis=1)


def laplace_mean(raw: RawCube) -> pd.DataFrame:
    """Mean per-SOW performance, per objective (risk-neutral reference).

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
    """Worst-SOW performance, per objective (risk-averse reference).

    Computed on the direction-oriented slab and returned in natural units:
    the minimum for a maximize objective, the maximum for a minimize one.
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
# Regret = the one-sided adverse deviation D_i from the incumbent (status-quo
# FFMP policy) per SOW, in natural units, never summed across objectives; the
# unit-free harm frequencies carry the cross-objective role. There is no
# max-regret and no baseline-normalized form (justification in
# docs/notes/methods/objective_definitions.md §4).

#: Decree-party grouping of the annual objectives for the party-level harm
#: frequencies. Party harm is a frequency over a disjunction (the
#: renegotiation is unanimity-bound), never a summed score. NYC storage sits
#: under `nyc`.
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

#: Multiplier ``k`` on the per-objective tolerance unit, ``tau_i = k * u_i``.
#: Fixed before results are inspected; override with ``NYCOPT_REGRET_TAU_K``.
#: k = 0 is the strict weak-Pareto-improvement form.
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

    These columns are never summed, averaged, or compared ACROSS objectives.

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

    ``u_i = max(eps_i, floor_i)``, where ``eps_i`` is the objective's
    annual-unit epsilon (``src.objectives_ensemble.ENSEMBLE_OBJECTIVES``) and
    ``floor_i`` the measured noise floor of its per-SOW estimator
    (``scripts/supplemental/regret_tolerance_diagnostics.py`` pass A), or 0
    when ``floors`` is not supplied. Rationale in
    docs/notes/methods/regret_tolerance_diagnostics.md.

    ``NYCOPT_REGRET_TAU`` (JSON ``{obj_name: tau}``) replaces the WHOLE vector
    with the tolerance at rung ``REGRET_TAU_K`` and is rescaled by
    ``k / REGRET_TAU_K``, so a k-sweep still sweeps.

    Args:
        obj_names: Annual objective names, in cube column order.
        k: Multiplier; defaults to :data:`REGRET_TAU_K`. ``k = 0`` gives the
            strict weak-Pareto-improvement form.
        floors: Optional ``{obj_name: noise_floor}`` in natural units.

    Returns:
        ``{obj_name: tau}``.

    Raises:
        KeyError: If an objective name has neither a registered epsilon nor an
            override, or if the env override is partial or names an unknown
            objective.
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
        # The override is the tolerance at rung REGRET_TAU_K and scales with k.
        scale = REGRET_TAU_K if k is None else float(k)
        unit = 1.0 if REGRET_TAU_K == 0 else scale / REGRET_TAU_K
        return {n: float(override[n]) * unit for n in obj_names}

    if not floors:
        # Eps-only ladder: not the adopted basis, so warn.
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
    k-sweeps which unset ``NYCOPT_REGRET_TAU`` still sweep the
    ``max(eps, floor)`` basis via ``tau_ladder(floors=...)``.

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
    """Unit-free harm frequencies; these carry the cross-objective scalar role.

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

    An OPTIONAL normalization scale for a dimensionless regret; never the
    reported primary.

    Raises:
        ValueError: If any objective's spread is zero (dividing by it would
            read as catastrophic regret).
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

    The McPhail et al. (2021) transform ``max(0, c - f)``: per axis with a
    FINITE threshold, ``max(0, thr - v)`` for "ge" axes and ``max(0, v - thr)``
    for "le", aggregated over SOWs. Values stay in natural units and are never
    summed across objectives. Non-finite cells are NaN.

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

    The Quinn et al. (2017) conclusion-invariance check over the ``sat_set__*``
    columns of :func:`score_criteria`'s scorecard (all higher-better).
    """
    cols = [c for c in per_set.columns if c.startswith("sat_set__")]
    return ranking_stability(per_set[cols], {c: True for c in cols})


###############################################################################
# Attainability screen
###############################################################################

def attainability_screen(raw: RawCube, thresholds: dict = None,
                         kinds: dict = None) -> pd.DataFrame:
    """Per-SOW: can ANY solution in this set meet all the criteria?

    An EMPIRICAL attainability bound, not a ceiling: no policy in this set
    wins the SOW, which does not mean none exists. Pool the cubes of all
    designs before calling this for "unwinnable by anyone".

    Returns a tidy frame (sow_id, n_satisficing_solutions, attainable, plus
    per-objective ``anysat__{name}`` columns showing which criterion binds
    where nothing attains the joint criterion).
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

    For each objective the threshold grid is the pooled per-SOW-distribution
    quantiles plus the labeled default from the meta. Returns a tidy DataFrame
    (solution_id, objective, threshold, is_default, satisficing).
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
    """Per-solution per-objective quantiles of the per-SOW values.

    Tidy: (solution_id, objective, qXX...).
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

    Columns where lower is better (regret) are negated so all are
    "higher = more robust" before correlating.
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
    from src.satisficing_criteria import ALL_SETS, active_variant

    reeval_dir = Path(reeval_dir)
    raw = load_raw(reeval_dir)
    baseline = load_raw(baseline_dir) if baseline_dir else None

    scorecard, higher_better = score_robustness(raw, baseline, metrics)

    out = reeval_dir / "robustness_scorecard.csv"
    scorecard.to_csv(out)

    # Per-criterion-set companion scorecard (Quinn 2017 subset criteria) and
    # the cross-set conclusion-invariance matrix, written beside the main
    # scorecard.
    criteria_scorecard, _ = score_criteria(raw, baseline, ALL_SETS)
    criteria_scorecard.to_csv(reeval_dir / "robustness_scorecard_criteria.csv")
    criterion_ranking_stability(criteria_scorecard).to_csv(
        reeval_dir / "robustness_criterion_stability.csv")

    # Every scoring-time choice that moves a number (tolerance ladder,
    # criterion sets) is snapshotted beside the numbers.
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
    # Sets naming axes absent from this cube are omitted, matching
    # score_criteria's skip.
    meta["criteria_variant"] = active_variant()
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

    # Substrate for the design-ranking threshold sweep.
    threshold_spectrum(raw).to_csv(
        reeval_dir / "robustness_threshold_spectrum.csv", index=False)

    # Raw per-SOW distributions are always co-reported beside the scalars.
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
