"""
satisficing_criteria.py - Named satisficing criterion sets (explicit subsets).

Each criterion set thresholds only a small subset (1-3) of the eight annual-unit
objectives and leaves every other axis unconstrained, following the satisficing
pattern of Quinn et al. (2017): one robustness metric (the Starr domain
criterion, ``src.robustness.satisficing_multivariate_sow``) evaluated under
several alternative stakeholder framings of "acceptable performance". Reporting
robustness under multiple criterion sets -- and whether the design ranking is
invariant across them -- is itself a result, not a sensitivity afterthought.

The previous all-8-axis conjunction is retained ONLY as ``reference_all8``:
on E_test it is degenerate (joint Starr = 0.0 for every design and for the
FFMP incumbent; see ``outputs/comparison/{slug}/{tag}/default_thresholds.csv``
and the incumbent pass fractions recorded in ``src.objectives_ensemble``),
which is reported as a finding, never used for selection.

Axes deliberately excluded from every named set (visible in the univariate
decomposition and the reference set, but never conjoined):

- ``nyc_delivery_deficit_p99_pct`` (incumbent pass 0.980) and
  ``nj_delivery_reliability_annual`` (> 0.98): saturated non-discriminators
  (Bonham-style saturation flags fire for every design).
- ``montague_flow_deficit_p99_pct`` (incumbent pass 0.965): likewise.

Threshold placements follow the pre-declared rules of
``docs/notes/methods/robustness_threshold_diagnostics.md`` (rule 1: re-anchor
any criterion the status quo itself fails, rounding to the stricter side;
rule 2: external goalposts beat round numbers). Placements marked PROVISIONAL
below are finalized from the Anvil-side audit table
``outputs/comparison/{slug}/{tag}/criteria_reanchoring.csv``
(``scripts/supplemental/criteria_reanchoring.py``).

These are POST-PROCESSING criteria only: they re-count the persisted per-SOW
cube and never touch the search-time registry (``src.objectives_ensemble``) or
its ``NYCOPT_SAT_THRESHOLDS`` env override. The adopted vector always comes
from the run's own ``reeval_raw_meta.json`` snapshot (the
moving-measuring-stick guard).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field


def nonbinding_threshold(kind: str) -> float:
    """The threshold value that makes an axis pass for every finite value.

    A "ge" axis becomes non-binding at ``-inf``, a "le" axis at ``+inf`` --
    every finite objective value passes, so a joint criterion simply ignores
    the axis while ``robustness._satisfaction_cube``'s missing-threshold guard
    stays armed.
    """
    if kind == "ge":
        return -math.inf
    if kind == "le":
        return math.inf
    raise ValueError(f"unknown satisficing kind '{kind}'")


@dataclass(frozen=True)
class CriterionSet:
    """One named stakeholder framing of the satisficing criterion.

    Attributes:
        key: Stable identifier used in file names and tables.
        label: Display name for legends/titles.
        rationale: Evidence anchor for each thresholded axis.
        criteria: ``{objective: threshold}`` for ONLY the thresholded axes
            (1-3 of them); every other axis is unconstrained. Kinds (ge/le)
            always come from the run's meta snapshot and are never changed.
        reference: True only for the all-axes reference set, whose
            ``thresholds`` are the adopted snapshot itself.
    """

    key: str
    label: str
    rationale: str
    criteria: dict = field(default_factory=dict)
    reference: bool = False

    @property
    def axes(self) -> tuple[str, ...]:
        """The thresholded (member) axes, in declaration order."""
        return tuple(self.criteria)

    def thresholds(self, adopted: dict, kinds: dict) -> dict:
        """The full threshold vector for this set.

        Args:
            adopted: The run's adopted threshold snapshot (all axes).
            kinds: ``{objective: "ge"|"le"}`` from the same snapshot.

        Returns:
            ``{objective: threshold}`` over every axis of ``adopted``:
            member axes at their set placement, all others non-binding.
            For a ``reference`` set, the adopted snapshot unchanged.
        """
        if self.reference:
            return dict(adopted)
        for name in self.criteria:
            if name not in adopted:
                raise KeyError(f"{self.key}: unknown objective '{name}'")
        out = {name: nonbinding_threshold(kinds[name]) for name in adopted}
        out.update(self.criteria)
        return out


#: The named criterion sets, in display order, followed by the all-axes
#: reference. Placements per the pre-declared rules (module docstring);
#: incumbent statistics cited from the interim-tag re-evaluation
#: (criteria_reanchoring.csv is the audit trail).
CRITERION_SETS: tuple[CriterionSet, ...] = (
    CriterionSet(
        key="nyc_supply",
        label="NYC supply security",
        rationale=("NYC delivery reliability at the adopted historic anchor "
                   "(0.65, discriminating: pooled stringency 0.32); storage "
                   "re-anchored per rule 1 from the aspirational FFMP L5 "
                   "goalpost (26%, incumbent pass 0.014) to the incumbent's "
                   "median year (12.9 -> 13.0, stricter side). The 26% "
                   "goalpost stays reported in the univariate decomposition."),
        criteria={
            "nyc_delivery_reliability_annual": 0.65,
            "nyc_storage_min_p01_pct": 13.0,
        },
    ),
    CriterionSet(
        key="downstream_flows",
        label="Downstream flow targets",
        rationale=("Montague re-anchored per rule 1: the adopted 0.79 lies "
                   "outside the incumbent's E_test support (pass 0.000); "
                   "placed at the incumbent's median year rounded stricter at "
                   "epsilon granularity (0.482 -> 0.50; pooled stringency "
                   "0.53), transcribed from criteria_reanchoring.csv. Trenton "
                   "moved from 0.87 (excludes 90% of pooled cells) to the "
                   "incumbent's median year (0.73 -> 0.75, stricter side)."),
        criteria={
            "montague_flow_reliability_annual": 0.50,
            "trenton_flow_reliability_annual": 0.75,
        },
    ),
    CriterionSet(
        key="flood",
        label="Flood exposure",
        rationale=("Rule-2 external goalpost: the observed WY2001-2023 minor "
                   "flood exceedance (1.17 ft-d/yr); incumbent pass 0.443 -- "
                   "discriminating, unchanged."),
        criteria={
            "downstream_flood_exceedance_annual": 1.17,
        },
    ),
    CriterionSet(
        key="compromise",
        label="All-parties compromise",
        rationale=("One axis per Decree-party interest (Quinn et al. 2017 "
                   "small-conjunction pattern): NYC delivery at the adopted "
                   "anchor, Trenton at the incumbent median-year placement, "
                   "flood at the rule-2 external goalpost."),
        criteria={
            "nyc_delivery_reliability_annual": 0.65,
            "trenton_flow_reliability_annual": 0.75,
            "downstream_flood_exceedance_annual": 1.17,
        },
    ),
    CriterionSet(
        key="reference_all8",
        label="Reference: all axes (adopted)",
        rationale=("The adopted search-time snapshot conjoined over every "
                   "axis. Degenerate on E_test (joint Starr = 0.0 for every "
                   "design and the incumbent) -- reported as a finding, "
                   "never used for selection."),
        reference=True,
    ),
)

#: The named (non-reference) sets, in display order.
NAMED_SETS: tuple[CriterionSet, ...] = tuple(
    c for c in CRITERION_SETS if not c.reference)

#: All sets including the reference, reference last.
ALL_SETS: tuple[CriterionSet, ...] = NAMED_SETS + tuple(
    c for c in CRITERION_SETS if c.reference)


def criterion_by_key(key: str) -> CriterionSet:
    """Look up a criterion set by its stable key."""
    for c in CRITERION_SETS:
        if c.key == key:
            return c
    raise KeyError(f"unknown criterion set '{key}'")


#: The default FOCAL criterion for policy-level robustness figures and the
#: step-11 satisficing label: the all-parties compromise set.
DEFAULT_FOCAL_KEY = "compromise"

#: Env var selecting the FOCAL criterion. Default = :data:`DEFAULT_FOCAL_KEY`.
#: Outputs carry the criterion key in their filenames, so runs under different
#: focal criteria coexist rather than overwrite.
FOCAL_CRITERION_ENV = "NYCOPT_FOCAL_CRITERION"


def focal_criterion() -> CriterionSet:
    """The focal criterion set for policy-level robustness figures."""
    return criterion_by_key(os.environ.get(FOCAL_CRITERION_ENV,
                                           DEFAULT_FOCAL_KEY))
