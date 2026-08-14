"""
satisficing_criteria.py - Named alternative satisficing criterion sets.

Phase-2 of the results sequence recomputes robustness under criterion vectors
reflecting distinct stakeholder framings (approved at the 2026-08-14 check-in).
Each set is a FULL 8-axis threshold dict in the same units/kinds as the adopted
snapshot; only the listed axes deviate from it. Threshold placements are
anchored in the phase-1 threshold-response curves
(``outputs/figures/comparison/ffmp_obj8/satisficing/threshold_response.csv``)
and in the incumbent's per-SOW medians, so every deviation has a stated,
data-grounded rationale rather than a tuned number.

These are POST-PROCESSING criteria only: they re-count the persisted per-SOW
cube (``src.results_data.satisfaction``) and never touch the search-time
registry (``src.objectives_ensemble``) or its ``NYCOPT_SAT_THRESHOLDS`` env
override. The adopted vector always comes from the run's own
``reeval_raw_meta.json`` snapshot (the moving-measuring-stick guard); the
deviations below are absolute values on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CriterionSet:
    """One named stakeholder framing of the satisficing criterion.

    Attributes:
        key: Stable identifier used in file names and tables.
        label: Display name for legends/titles.
        rationale: One-sentence justification of the deviations (anchors).
        deviations: ``{objective: threshold}`` replacing the adopted values;
            axes not listed keep the adopted snapshot value. Kinds (ge/le)
            are never changed.
    """

    key: str
    label: str
    rationale: str
    deviations: dict = field(default_factory=dict)

    def thresholds(self, adopted: dict) -> dict:
        """The full 8-axis vector: the adopted snapshot plus the deviations."""
        out = dict(adopted)
        for name, value in self.deviations.items():
            if name not in out:
                raise KeyError(f"{self.key}: unknown objective '{name}'")
            out[name] = value
        return out


#: The phase-2 criterion sets, in display order. "adopted" (first) is the
#: DEFAULT analysis criterion -- the search-time meta snapshot itself (the
#: "uniform" round-number set was trialed as default 2026-08-14 and REVERTED
#: same day: too strict on this ensemble; it stays here as a comparison
#: framing that documents that stringency).
CRITERION_SETS: tuple[CriterionSet, ...] = (
    CriterionSet(
        key="adopted",
        label="Adopted (search-time criteria)",
        rationale=("The thresholds adopted 2026-08-08 and snapshotted in "
                   "reeval_raw_meta.json; historic-record anchors plus the "
                   "NWS minor-flood and FFMP L5 storage goalposts."),
    ),
    CriterionSet(
        key="uniform",
        label="Uniform round-number criteria",
        rationale=("Thresholds equalized across like objectives and rounded "
                   "for interpretability (Trevor, 2026-08-14): every "
                   "reliability >= 0.70, every deficit <= 30%, flood "
                   "exceedance <= 1.5 ft-d/yr, storage P1 >= 25%."),
        deviations={
            "nyc_delivery_reliability_annual": 0.70,
            "nyc_delivery_deficit_p99_pct": 30.0,
            "montague_flow_reliability_annual": 0.70,
            "montague_flow_deficit_p99_pct": 30.0,
            "trenton_flow_reliability_annual": 0.70,
            "downstream_flood_exceedance_annual": 1.5,
            "nyc_storage_min_p01_pct": 25.0,
            "nj_delivery_reliability_annual": 0.70,
        },
    ),
    CriterionSet(
        key="nyc_supply",
        label="A: NYC supply security",
        rationale=("NYC delivery/storage axes kept at adopted values; the "
                   "downstream Trenton and flood requirements relaxed to the "
                   "placements the best ensemble-design policy meets in ~90% "
                   "of E_test SOWs (threshold-response curves)."),
        deviations={
            "trenton_flow_reliability_annual": 0.65,
            "downstream_flood_exceedance_annual": 2.0,
        },
    ),
    CriterionSet(
        key="downstream",
        label="B: Downstream / Decree parties",
        rationale=("Montague, NJ, and flood axes kept at adopted values; "
                   "Trenton moved to the best-policy 75% placement (= the "
                   "incumbent's median year, 0.73); NYC delivery relaxed to "
                   "the 0.5 stakeholder floor and storage to the incumbent's "
                   "median year (12.9%)."),
        deviations={
            "nyc_delivery_reliability_annual": 0.50,
            "nyc_storage_min_p01_pct": 13.0,
            "trenton_flow_reliability_annual": 0.75,
        },
    ),
    CriterionSet(
        key="compromise",
        label="C: All-parties compromise",
        rationale=("Every party's axes kept at adopted values except the two "
                   "structurally-conflicting downstream axes, each moved to "
                   "match-or-beat the incumbent's median year: Trenton 0.75, "
                   "flood 1.5 ft·d/yr; storage stays at the aspirational "
                   "FFMP L5 anchor (26%)."),
        deviations={
            "trenton_flow_reliability_annual": 0.75,
            "downstream_flood_exceedance_annual": 1.5,
        },
    ),
)


def criterion_by_key(key: str) -> CriterionSet:
    """Look up a criterion set by its stable key."""
    for c in CRITERION_SETS:
        if c.key == key:
            return c
    raise KeyError(f"unknown criterion set '{key}'")


#: The default analysis criterion for the phase-1 diagnostics: the adopted
#: search-time snapshot ("uniform" was trialed and reverted 2026-08-14).
DEFAULT_CRITERION_KEY = "adopted"

#: The default FOCAL criterion for the phase-3/4 policy-robustness figures:
#: criterion B, selected at the 2026-08-14 check-in.
DEFAULT_FOCAL_KEY = "downstream"

#: Env var selecting the FOCAL criterion for the phase-3/4 figures. Default =
#: :data:`DEFAULT_FOCAL_KEY`. Changing it re-parameterizes the whole figure
#: tranche -- outputs carry the criterion key in their filenames, so runs
#: under different focal criteria coexist rather than overwrite.
FOCAL_CRITERION_ENV = "NYCOPT_FOCAL_CRITERION"


def default_criterion() -> CriterionSet:
    """The default analysis criterion set (phase-1 diagnostics)."""
    return criterion_by_key(DEFAULT_CRITERION_KEY)


def focal_criterion() -> CriterionSet:
    """The focal criterion set for policy-level robustness figures."""
    import os
    return criterion_by_key(
        os.environ.get(FOCAL_CRITERION_ENV, DEFAULT_FOCAL_KEY))
