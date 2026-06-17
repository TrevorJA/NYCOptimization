"""
objectives_ensemble.py - Ensemble (multi-realization) objective framework.

Wraps the single-trace metrics in `src.objectives` with a per-realization
aggregator so the same underlying metric functions can serve both
`run_simulation_inmemory` (1 data dict) and `run_simulation_ensemble_inmemory`
(N data dicts). v1 ships **multivariate satisficing only**: each ensemble
objective scores the fraction of realizations that meet a per-objective
threshold. Mean / percentile / regret aggregators are intentionally deferred
(see plan §8.4 in `in-this-session-i-tranquil-feigenbaum.md`).

Design choices:

- **Don't fold into `src/objectives.py`.** The legacy module remains the
  single-trace baseline and the manuscript-baseline path. New ensemble work
  lives here.
- **Reuse single-trace metric functions.** Every `EnsembleObjective` wraps an
  existing `Objective` from the `OBJECTIVES` registry; no metric math is
  duplicated. Updating the per-day computation in `objectives.py` flows
  through to ensemble runs automatically.
- **Direction = "maximize" for every satisficing objective.** Satisficing
  rates live in [0, 1] with higher better, regardless of whether the base
  metric is maximize (reliability) or minimize (deficit).
- **Threshold overrides via env.** `NYCOPT_SAT_THRESHOLDS` accepts a JSON
  object `{"<obj_name>": <threshold>, ...}` so manuscript-default vs
  sensitivity-sub-experiment is a config switch, not a code edit.

Naming convention: `<base_objective_name>__sat<threshold>[unit]`. The unit
fragment is for human readability only — the registered name is the source
of truth.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Literal

import numpy as np

from src.objectives import OBJECTIVES, Objective, ObjectiveSet


###############################################################################
# Aggregator
###############################################################################

class SatisficingAgg:
    """Fraction of finite per-realization values that meet the threshold.

    `kind="ge"` ⇒ raw >= threshold (use for maximize-base metrics).
    `kind="le"` ⇒ raw <= threshold (use for minimize-base metrics).

    NaN / non-finite values count as **unsatisfied** so a degenerate
    realization can't masquerade as satisficing. If every value is non-finite
    the result is 0.0 rather than NaN, since downstream Borg minimisation
    needs a finite vector.
    """

    def __init__(self, threshold: float, kind: Literal["ge", "le"]):
        if kind not in ("ge", "le"):
            raise ValueError(f"kind must be 'ge' or 'le', got '{kind}'")
        self.threshold = float(threshold)
        self.kind = kind

    def __call__(self, values) -> float:
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            return 0.0
        finite = np.isfinite(arr)
        if self.kind == "ge":
            sat = finite & (arr >= self.threshold)
        else:
            sat = finite & (arr <= self.threshold)
        return float(sat.sum()) / float(arr.size)


###############################################################################
# EnsembleObjective wrapper
###############################################################################

class EnsembleObjective:
    """Wraps a base single-trace `Objective` with a per-realization aggregator.

    Implements the same `compute(...)`/`compute_for_borg(...)` interface as
    `Objective` so an `ObjectiveSet` of `EnsembleObjective` instances works
    with `ObjectiveSet.compute_for_borg_ensemble(data_per_real)` (added in
    `src/objectives.py`).
    """

    def __init__(self, base: Objective, aggregator: Callable,
                 name: str, epsilon: float):
        self.base = base
        self.aggregator = aggregator
        self.name = name
        self.epsilon = float(epsilon)
        # Satisficing fractions are uniformly higher-is-better.
        self.direction = "maximize"

    @property
    def sign(self) -> int:
        return 1  # always maximize for satisficing v1

    def compute(self, data_per_real: list) -> float:
        """Run the base metric on each realization and aggregate."""
        values = [self.base.compute(d) for d in data_per_real]
        return self.aggregator(values)

    def compute_for_borg(self, data_per_real: list) -> float:
        """Borg minimises, so negate the maximised satisficing rate."""
        raw = self.compute(data_per_real)
        return -raw if self.direction == "maximize" else raw


###############################################################################
# Threshold defaults & env override
###############################################################################
# Documented in `local_notes/decisions/2026-05-06_satisficing_thresholds.md`.

# Per-objective satisficing thresholds (the across-realization "acceptable"
# level). These are the analyst-chosen satisficing levels, distinct from the
# Decree thresholds baked into the temporal metrics. Placeholder values pending
# the random-DV sensitivity experiment; override via NYCOPT_SAT_THRESHOLDS.
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "nyc_delivery_reliability_weekly__sat95":     0.95,
    "nyc_delivery_deficit_cvar90_pct__sat10":     10.0,
    "montague_flow_reliability_weekly__sat85":    0.85,
    "montague_flow_deficit_cvar90_pct__sat25":    25.0,
    "trenton_flow_reliability_weekly__sat85":     0.85,
    "nj_delivery_reliability_weekly__sat95":      0.95,
    "downstream_flood_days_minor__sat10":         10.0,
    "nyc_storage_p5_pct__sat25":                  25.0,
}


def _resolve_thresholds() -> dict[str, float]:
    """Apply NYCOPT_SAT_THRESHOLDS JSON env override on top of defaults."""
    thresholds = dict(_DEFAULT_THRESHOLDS)
    raw = os.environ.get("NYCOPT_SAT_THRESHOLDS", "").strip()
    if raw:
        overrides = json.loads(raw)
        for k, v in overrides.items():
            if k not in thresholds:
                raise KeyError(
                    f"NYCOPT_SAT_THRESHOLDS: unknown ensemble objective '{k}'. "
                    f"Available: {sorted(thresholds)}"
                )
            thresholds[k] = float(v)
    return thresholds


###############################################################################
# Ensemble objective registry
###############################################################################
# (base_objective_name, ensemble_name, kind, default_epsilon)
# `kind` is the satisficing direction relative to the BASE objective:
#   - maximize-base  -> "ge"  (raw >= threshold satisfies)
#   - minimize-base  -> "le"  (raw <= threshold satisfies)
_REGISTRY_SPEC: list[tuple[str, str, Literal["ge", "le"], float]] = [
    ("nyc_delivery_reliability_weekly",
     "nyc_delivery_reliability_weekly__sat95",   "ge", 0.02),
    ("nyc_delivery_deficit_cvar90_pct",
     "nyc_delivery_deficit_cvar90_pct__sat10",   "le", 0.02),
    ("montague_flow_reliability_weekly",
     "montague_flow_reliability_weekly__sat85",  "ge", 0.02),
    ("montague_flow_deficit_cvar90_pct",
     "montague_flow_deficit_cvar90_pct__sat25",  "le", 0.02),
    ("trenton_flow_reliability_weekly",
     "trenton_flow_reliability_weekly__sat85",   "ge", 0.02),
    ("nj_delivery_reliability_weekly",
     "nj_delivery_reliability_weekly__sat95",    "ge", 0.02),
    ("downstream_flood_days_minor",
     "downstream_flood_days_minor__sat10",       "le", 0.02),
    ("nyc_storage_p5_pct",
     "nyc_storage_p5_pct__sat25",                "ge", 0.02),
]


def _build_registry() -> dict[str, EnsembleObjective]:
    thresholds = _resolve_thresholds()
    registry: dict[str, EnsembleObjective] = {}
    for base_name, ens_name, kind, eps in _REGISTRY_SPEC:
        if base_name not in OBJECTIVES:
            raise KeyError(
                f"ensemble registry references unknown base objective "
                f"'{base_name}'"
            )
        if ens_name not in thresholds:
            raise KeyError(
                f"ensemble registry references threshold '{ens_name}' "
                f"missing from _DEFAULT_THRESHOLDS"
            )
        agg = SatisficingAgg(threshold=thresholds[ens_name], kind=kind)
        registry[ens_name] = EnsembleObjective(
            base=OBJECTIVES[base_name],
            aggregator=agg,
            name=ens_name,
            epsilon=eps,
        )
    return registry


ENSEMBLE_OBJECTIVES: dict[str, EnsembleObjective] = _build_registry()


###############################################################################
# Assembler
###############################################################################

def build_ensemble_objective_set(items) -> ObjectiveSet:
    """Assemble an ObjectiveSet from a list of ensemble objective names.

    Mirrors `src.objectives.build_objective_set` but resolves names against
    `ENSEMBLE_OBJECTIVES`. Items may be:
      - str:                ensemble objective name
      - EnsembleObjective:  use directly

    Returns:
        ObjectiveSet whose contained objectives all expose
        `compute(data_per_real)` and `compute_for_borg(data_per_real)`.
    """
    resolved = []
    for item in items:
        if isinstance(item, EnsembleObjective):
            resolved.append(item)
        elif isinstance(item, str):
            if item not in ENSEMBLE_OBJECTIVES:
                raise KeyError(
                    f"Unknown ensemble objective '{item}'. "
                    f"Available: {sorted(ENSEMBLE_OBJECTIVES)}"
                )
            resolved.append(ENSEMBLE_OBJECTIVES[item])
        else:
            raise TypeError(
                f"build_ensemble_objective_set items must be str or "
                f"EnsembleObjective; got {type(item).__name__}"
            )
    return ObjectiveSet(resolved)


def list_available_ensemble_objectives() -> str:
    lines = [f"Available ensemble objectives ({len(ENSEMBLE_OBJECTIVES)}):"]
    for name, obj in ENSEMBLE_OBJECTIVES.items():
        agg = obj.aggregator
        thr_str = f"{agg.threshold:g}" if isinstance(agg, SatisficingAgg) else "?"
        kind_str = agg.kind if isinstance(agg, SatisficingAgg) else "?"
        lines.append(
            f"  {name}: base={obj.base.name}, kind={kind_str}, "
            f"threshold={thr_str}, eps={obj.epsilon}"
        )
    return "\n".join(lines)
