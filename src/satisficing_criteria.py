"""
satisficing_criteria.py - Named satisficing criterion sets (explicit subsets).

Quinn et al. (2017)-style subset criteria: each named set thresholds only its
member axes of the eight annual-unit objectives and leaves every other axis
non-binding, so one robustness metric (the Starr domain criterion,
``src.robustness.satisficing_multivariate_sow``) is evaluated under several
stakeholder framings. ``reference_all8`` is the all-axes reference only
(reported, never used for selection).

Threshold placements follow the rules of
``docs/notes/methods/robustness_threshold_diagnostics.md`` (external goalposts
where one exists, otherwise the stricter side of the incumbent's attainment,
with any criterion the status quo itself fails re-anchored at
maintain-status-quo performance) and the ``criteria_reanchoring.csv`` audit
(``scripts/supplemental/criteria_reanchoring.py``).

Variants are selected by ``NYCOPT_CRITERIA_VARIANT``; scoring and every
criteria figure read the same selection, so rescore each design after
changing it.

These are POST-PROCESSING criteria only: they re-count the persisted per-SOW
cube and never touch the search-time registry (``src.objectives_ensemble``)
or its ``NYCOPT_SAT_THRESHOLDS`` override. The adopted vector always comes
from the run's own ``reeval_raw_meta.json`` snapshot.
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
        criteria: ``{objective: threshold}`` for ONLY the thresholded axes;
            every other axis is unconstrained. Kinds (ge/le) always come from
            the run's meta snapshot and are never changed.
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


#: VARIANT ``adopted``: the re-anchored placements, in display order, followed
#: by the all-axes reference (criteria_reanchoring.csv is the audit trail).
_ADOPTED_SETS: tuple[CriterionSet, ...] = (
    CriterionSet(
        key="nyc_supply",
        label="NYC supply security",
        rationale=("NYC delivery reliability at the historic anchor (0.65); "
                   "storage re-anchored per rule 1 from the FFMP L5 goalpost "
                   "(26%) to the incumbent's median year (13.0, stricter "
                   "side)."),
        criteria={
            "nyc_delivery_reliability_annual": 0.65,
            "nyc_storage_min_p01_pct": 13.0,
        },
    ),
    CriterionSet(
        key="downstream_flows",
        label="Downstream flow targets",
        rationale=("Montague and Trenton re-anchored per rule 1 to the "
                   "incumbent's median year, rounded to the stricter side "
                   "(0.50 and 0.75)."),
        criteria={
            "montague_flow_reliability_annual": 0.50,
            "trenton_flow_reliability_annual": 0.75,
        },
    ),
    CriterionSet(
        key="flood",
        label="Flood exposure",
        rationale=("Rule-2 external goalpost: the observed WY2001-2023 minor "
                   "flood exceedance (1.17 ft-d/yr)."),
        criteria={
            "downstream_flood_exceedance_annual": 1.17,
        },
    ),
    CriterionSet(
        key="compromise",
        label="All-parties compromise",
        rationale=("One axis per Decree-party interest: NYC delivery at the "
                   "historic anchor, Trenton at the incumbent median-year "
                   "placement, flood at the rule-2 external goalpost."),
        criteria={
            "nyc_delivery_reliability_annual": 0.65,
            "trenton_flow_reliability_annual": 0.75,
            "downstream_flood_exceedance_annual": 1.17,
        },
    ),
    CriterionSet(
        key="reference_all8",
        label="Reference: all axes (adopted)",
        rationale=("The search-time threshold snapshot conjoined over every "
                   "axis; reported as a finding, never used for selection."),
        reference=True,
    ),
)


#: The all-axes reference set, shared by every variant.
_REFERENCE_SET = next(c for c in _ADOPTED_SETS if c.reference)

#: VARIANT ``v2_20260821``: a broader reading of each stakeholder framing.
#: The deficit axes enter the NYC and downstream sets, storage tightens, and
#: the compromise set carries one axis per party plus storage.
_V2_20260821_SETS: tuple[CriterionSet, ...] = (
    CriterionSet(
        key="nyc_supply",
        label="NYC supply security",
        rationale=("Delivery reliability at the historic anchor, P99 deficit "
                   "capped at half the Decree allocation, storage between the "
                   "incumbent median (13%) and the FFMP L5 goalpost (26%)."),
        criteria={
            "nyc_delivery_reliability_annual": 0.65,
            "nyc_delivery_deficit_p99_pct": 50.0,
            "nyc_storage_min_p01_pct": 20.0,
        },
    ),
    CriterionSet(
        key="downstream_flows",
        label="Downstream flow targets",
        rationale=("Montague reliability at 0.65 with a P99 deficit cap at "
                   "50%; Trenton at 0.75."),
        criteria={
            "montague_flow_reliability_annual": 0.65,
            "montague_flow_deficit_p99_pct": 50.0,
            "trenton_flow_reliability_annual": 0.75,
        },
    ),
    CriterionSet(
        key="flood",
        label="Flood exposure",
        rationale=("Minor-flood exceedance at 1.25 ft-d/yr, slightly above "
                   "the observed WY2001-2023 value (1.17)."),
        criteria={
            "downstream_flood_exceedance_annual": 1.25,
        },
    ),
    CriterionSet(
        key="compromise",
        label="All-parties compromise",
        rationale=("One reliability axis each for NYC and Montague at 0.65, "
                   "flood at 1.5 ft-d/yr, and a 15% storage floor."),
        criteria={
            "nyc_delivery_reliability_annual": 0.65,
            "montague_flow_reliability_annual": 0.65,
            "downstream_flood_exceedance_annual": 1.5,
            "nyc_storage_min_p01_pct": 15.0,
        },
    ),
    _REFERENCE_SET,
)

#: Every saved criteria variant, keyed by name. Never edit a saved variant in
#: place once figures have been rendered from it.
CRITERION_VARIANTS: dict[str, tuple[CriterionSet, ...]] = {
    "adopted": _ADOPTED_SETS,
    "v2_20260821": _V2_20260821_SETS,
}

#: Env var selecting the active criteria variant (a key of
#: :data:`CRITERION_VARIANTS`); rescore each design after changing it.
CRITERIA_VARIANT_ENV = "NYCOPT_CRITERIA_VARIANT"

#: The default variant when the env var is unset.
DEFAULT_CRITERIA_VARIANT = "v2_20260821"


def active_variant() -> str:
    """The selected criteria-variant name (validated)."""
    name = os.environ.get(CRITERIA_VARIANT_ENV, DEFAULT_CRITERIA_VARIANT)
    if name not in CRITERION_VARIANTS:
        raise KeyError(f"{CRITERIA_VARIANT_ENV}={name!r} is not a saved "
                       f"variant; choose from {sorted(CRITERION_VARIANTS)}")
    return name


#: The active criterion sets, in display order, reference last.
CRITERION_SETS: tuple[CriterionSet, ...] = CRITERION_VARIANTS[active_variant()]

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
