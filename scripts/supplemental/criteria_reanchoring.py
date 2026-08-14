"""criteria_reanchoring.py - Audit table for subset-criteria threshold placement.

Measures, per objective, everything the pre-declared placement rules of
``docs/notes/methods/robustness_threshold_diagnostics.md`` need in order to
(re-)anchor the named criterion sets of ``src.satisficing_criteria``:

- the FFMP incumbent's per-SOW q10/q50/q90 on E_test (rule 1 anchors);
- the incumbent's pass fraction at the ADOPTED threshold (the rule-1 trigger:
  re-anchor any criterion the status quo itself fails);
- the pooled-cell stringency of the adopted threshold (how much of the pooled
  candidate distribution it excludes -- the degeneracy diagnostic);
- a stricter-side round of the incumbent median at the objective's epsilon
  granularity (``proposed_reanchor``), plus the raw median, so the transcribed
  literal can use a finer step where epsilon is coarse (e.g. storage).

The output CSV is the AUDIT TRAIL: the placements in
``src.satisficing_criteria`` are transcribed literals citing this table --
thresholds stay frozen in code per repo convention; this script never edits
them.

Anvil-side (needs the raw per-SOW cubes). Output:
``outputs/comparison/{slug}/{tag}/criteria_reanchoring.csv``.

Run::

    python scripts/supplemental/criteria_reanchoring.py --formulation ffmp
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from src import results_data as rd  # noqa: E402
from src import robustness as rob  # noqa: E402

#: Rule-2 external goalposts (these beat round numbers and are never
#: re-anchored): the observed WY2001-2023 minor-flood exceedance.
EXTERNAL_GOALPOSTS: dict = {"downstream_flood_exceedance_annual": 1.17}

#: Incumbent pass fraction below which rule 1 fires (the status quo itself
#: fails the criterion in most states of the world).
RULE1_PASS_FLOOR: float = 0.5


def _stricter_round(value: float, step: float, kind: str) -> float:
    """Round ``value`` to the stricter side at ``step`` granularity."""
    if not np.isfinite(value) or step <= 0:
        return float("nan")
    n = value / step
    return float((math.ceil(n) if kind == "ge" else math.floor(n)) * step)


def measure_anchors(baseline_values: np.ndarray, obj_names: list,
                    thresholds: dict, kinds: dict,
                    pooled: dict) -> pd.DataFrame:
    """The per-objective re-anchoring audit table.

    Args:
        baseline_values: Incumbent per-SOW matrix ``(G, M)`` on E_test.
        obj_names: Objective names matching the matrix columns.
        thresholds: The adopted threshold snapshot.
        kinds: ``{objective: "ge"|"le"}``.
        pooled: Per-objective pooled finite candidate cells (all designs).

    Returns:
        One row per objective: incumbent quantiles, incumbent pass fraction at
        the adopted threshold, pooled stringency, epsilon, the stricter-side
        rounded re-anchor proposal, and the governing rule.
    """
    from src.objectives_ensemble import ENSEMBLE_OBJECTIVES

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, name in enumerate(obj_names):
            v = baseline_values[:, k]
            finite = v[np.isfinite(v)]
            thr, kind = thresholds.get(name), kinds[name]
            if finite.size and thr is not None:
                passing = (finite >= thr) if kind == "ge" else (finite <= thr)
                pass_frac = float(passing.sum() / len(v))
            else:
                pass_frac = float("nan")
            pool = np.asarray(pooled.get(name, []), dtype=float)
            if pool.size and thr is not None:
                stringency = float(np.mean(pool < thr) if kind == "ge"
                                   else np.mean(pool > thr))
            else:
                stringency = float("nan")
            eps = (float(ENSEMBLE_OBJECTIVES[name].epsilon)
                   if name in ENSEMBLE_OBJECTIVES else float("nan"))
            q50 = float(np.nanmedian(finite)) if finite.size else float("nan")

            if name in EXTERNAL_GOALPOSTS:
                rule = "rule2_external_goalpost"
                proposal = float(EXTERNAL_GOALPOSTS[name])
            elif np.isfinite(pass_frac) and pass_frac < RULE1_PASS_FLOOR:
                rule = "rule1_reanchor"
                proposal = _stricter_round(q50, eps, kind)
            else:
                rule = "keep"
                proposal = float(thr) if thr is not None else float("nan")

            rows.append({
                "objective": name,
                "kind": kind,
                "adopted_threshold": thr,
                "incumbent_q10": (float(np.nanquantile(finite, 0.10))
                                  if finite.size else float("nan")),
                "incumbent_q50": q50,
                "incumbent_q90": (float(np.nanquantile(finite, 0.90))
                                  if finite.size else float("nan")),
                "incumbent_pass_frac": pass_frac,
                "pooled_stringency": stringency,
                "epsilon": eps,
                "proposed_reanchor": proposal,
                "rule": rule,
            })
    return pd.DataFrame(rows)


def run(formulation: str, reeval_tag: str | None) -> Path:
    """Measure and write the audit table for the campaign designs' common tag."""
    from src.reeval_core import reeval_tag as tag_of
    from src.ensembles import get_ensemble_spec

    spec = (get_ensemble_spec(reeval_tag) if reeval_tag
            else config.REEVAL_ENSEMBLE_SPEC)
    tag = tag_of(spec)
    slug = config.derive_slug(formulation)
    results = rd.load_design_results(tag, slug=slug)

    first = next(iter(results.values()))
    incumbent = next((r.incumbent for r in results.values()
                      if r.incumbent is not None), None)
    if incumbent is None:
        sys.exit("[criteria_reanchoring] no design carries a baseline cube on "
                 "this tag; the incumbent anchors cannot be measured.")

    pooled: dict = {}
    for res in results.values():
        for k, name in enumerate(res.raw.obj_names):
            v = res.raw.cube[:, :, k]
            pooled.setdefault(name, []).append(v[np.isfinite(v)])
    pooled = {n: np.concatenate(v) for n, v in pooled.items()}

    table = measure_anchors(incumbent, first.raw.obj_names,
                            first.raw.thresholds, first.raw.kinds, pooled)
    out = config.OUTPUTS_DIR / "comparison" / slug / tag
    out.mkdir(parents=True, exist_ok=True)
    path = out / "criteria_reanchoring.csv"
    table.to_csv(path, index=False)
    print(f"[criteria_reanchoring] audit table -> {path}")
    print(table.to_string(index=False))
    return path


def main() -> None:
    """CLI. Identifiers only -- no value flags (repo rule)."""
    p = argparse.ArgumentParser(
        description="Measure the subset-criteria re-anchoring audit table.")
    p.add_argument("--formulation", default="ffmp")
    p.add_argument("--reeval-tag", default=None,
                   help="Re-eval ensemble preset id (default: configured E_test).")
    args = p.parse_args()
    run(args.formulation, args.reeval_tag)


if __name__ == "__main__":
    main()
