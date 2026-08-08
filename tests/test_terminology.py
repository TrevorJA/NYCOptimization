"""
tests/test_terminology.py - Guards against retired methods and terminology lingering.

This suite exists because of a specific, repeated failure mode. When a method
changes, the *name* of the old method survives in code, notes, and slides long
after the thing itself is gone, and a reader cannot tell the difference between
"this is what we do" and "this is what we used to do". Real instances, all found
by a human noticing rather than by a test:

  - The notes specified a simulated-annealing selector that was NEVER implemented.
  - "Master ensemble" survived the deletion of the shared-master architecture,
    including in the proposal deck, which is how the whole rewrite started.
  - ``objectives_summary.csv`` columns were named ``*_p99_pct`` (promising a
    percentile in percent) while containing satisficing fractions in [0, 1].
  - ``improvement_vs_baseline`` computed a *shortfall*, so a policy that beat the
    status quo scored 0 and the name said the opposite of the quantity.

Two kinds of check:

  1. RETIRED TERMS must not appear as live claims anywhere in the method surface.
     A line is allowed to name a retired term only if it also carries a negation
     marker -- i.e. it is saying "we do NOT do this, and here is why", which is
     exactly the framing we want to keep.
  2. NAME/SEMANTICS invariants on the metric columns: a column prefix must mean
     what it says, and its higher-is-better orientation must be declared
     correctly. A wrong orientation flag silently inverts an objective in every
     ranking correlation, with no other symptom.

When a method is retired in future, add its term here in the SAME commit. That is
the cheap half of the discipline; the test is the half that does not forget.
"""

import re
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import src.robustness as rob


# ---------------------------------------------------------------------------
# 1. Retired terminology
# ---------------------------------------------------------------------------

#: Term -> what replaced it (shown in the failure message so the fix is obvious).
RETIRED_TERMS: dict[str, str] = {
    "master ensemble": "candidate pool (search side) or test ensemble (re-eval)",
    "simulated annealing": "deterministic LHS + nearest-neighbour snap "
                           "(the annealer was never implemented)",
    "support_points": "deleted from the campaign",
    "fixed_probabilistic_short": "fixed_probabilistic",
    "fixed_probabilistic_long": "deleted from the campaign",
    "regret_from_best": "deleted -- set-relative and design-coupled",
    # The NAME stays retired even though incumbent-relative regret is now computed.
    # "Regret from baseline" is ambiguous in exactly the way that produced the
    # miscitation this project shipped: Herman et al. (2015)'s R1 is regret from a
    # baseline STATE OF THE WORLD (same policy, reference SOW), while what we
    # compute is regret from a baseline POLICY (the incumbent, per SOW). The live
    # names say which: incumbent_advantage / regret_magnitudes / regret_frequencies.
    "regret_from_baseline": "incumbent_advantage + regret_magnitudes / "
                            "regret_frequencies (regret from the incumbent POLICY, "
                            "per SOW) -- the old name conflated that with Herman's "
                            "R1, regret from a baseline STATE OF THE WORLD",
    "overfitting gap": "deleted -- undefined in Brodeur (2020) and invalid "
                       "under a measure change",
    "ensemble objective-sensitivity experiment":
        "framing-convention diagnostics (framing_convention_diagnostics.md; "
        "cube reductions of the epsilon-calibration policy populations + the "
        "satisfaction-factor sweep). Retired 2026-07-30: its per-realization "
        "matrix predated the annual-unit scheme and every decision it gated "
        "is now answered by the newer machinery.",
    "ensemble_objective_sensitivity":
        "framing_convention_analysis.py / satisfaction_factor_{run,figures}.py "
        "over the epsilon-calibration cubes (the retired experiment's scripts, "
        "launchers, and ENS_* config are deleted)",
    "nested axis sets":
        "the two diagnostic axis sets: campaign (config.HAZARD_SELECTION_AXES) "
        "vs the full retained set. The m4/m6 diagnostic nestings were retired "
        "2026-07-30 (assess_m6_axis_sets.py and its launcher deleted).",
    "nyc_drought_factor":
        "nyc_allocation_reduction_* (stage-wise ADDITIONAL reduction DVs, "
        "decoded to delivery factors by cumulative subtraction from 1.0; "
        "re-parameterized 2026-08-06). Pywr-DRB's model-side "
        "drought_factor_delivery_* / {level}_factor_delivery_* parameter "
        "names are unaffected.",
    "nj_drought_factor":
        "nj_allocation_reduction_* (see nyc_drought_factor)",
    "delivery_monotonicity":
        "deleted -- monotonicity is structural under the non-negative "
        "allocation-reduction DVs; flood_zone_ordering is the only DV-space "
        "Borg constraint",
    # 2026-08-07 unified-metric-currency refactor: robustness/regret moved onto
    # the per-SOW annual-unit objective substrate; the whole-trace / realization-
    # unit machinery below dissolved into it.
    "improvement_vs_baseline":
        "the incumbent-advantage regret family (regret_mean__/gain_mean__ are "
        "its one-sided halves, natural units, SOW unit)",
    "vs_baseline":
        "the incumbent-advantage regret family (regret_mean__/gain_mean__ are "
        "its one-sided halves, natural units, SOW unit)",
    "within_sow_agg":
        "retired: pooling each SOW's unit-years through the §2 unit operator "
        "IS the within-state collapse",
    "collapse_within_sow":
        "retired: pooling each SOW's unit-years through the §2 unit operator "
        "IS the within-state collapse",
    "satisficing_from_raw":
        "persist-time per-SOW pooling (reeval_core.sow_objective_matrix)",
    "SatisficingAgg":
        "AnnualUnitObjective.sat_threshold/sat_kind + robustness._satisfy",
}

#: A line may name a retired term if it is explicitly disclaiming it.
_NEGATION = re.compile(
    r"\b(no|not|never|non-|deleted|retired|removed|abandoned|excluded|absent|"
    r"without|superseded|instead of|rather than|used to|no longer|gone|banned|"
    r"deliberately|previously)\b",
    re.IGNORECASE,
)

#: Lines to look at around a hit when deciding whether it is disclaimed. A
#: disclaimer routinely wraps: "Deliberately absent: ... and the search-vs-test\n
#: overfitting gap (undefined in Brodeur 2020...)". Scoping the negation check to
#: the hit's own line would flag that as a live claim.
_NEGATION_WINDOW = 2

#: The method surface a reader would trust. Scratch/temporary code is exempt.
_SCAN_GLOBS = (
    "src/**/*.py",
    "config.py",
    "scripts/main/*.py",
    "workflow/**/*.sh",
    "workflow/envs/*.env",
    "docs/notes/methods/*.md",
    "docs/notes/terminology.md",
    "docs/research_project_summary.md",
)

#: This file necessarily names every retired term.
_EXEMPT = {Path(__file__).name}


def _scan_files() -> list[Path]:
    seen: list[Path] = []
    for pattern in _SCAN_GLOBS:
        for p in PROJECT_DIR.glob(pattern):
            if p.is_file() and p.name not in _EXEMPT:
                seen.append(p)
    return seen


@pytest.mark.parametrize("term,replacement", sorted(RETIRED_TERMS.items()))
def test_retired_term_is_not_a_live_claim(term, replacement):
    """A retired term may only appear inside an explicit disclaimer."""
    offenders: list[str] = []
    for path in _scan_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if term.lower() not in line.lower():
                continue
            lo = max(0, i - _NEGATION_WINDOW)
            hi = min(len(lines), i + _NEGATION_WINDOW + 1)
            if _NEGATION.search("\n".join(lines[lo:hi])):
                continue  # disclaimed ("we do NOT do this"), possibly across lines
            rel = path.relative_to(PROJECT_DIR)
            offenders.append(f"  {rel}:{i + 1}: {line.strip()[:110]}")

    assert not offenders, (
        f"\n'{term}' is retired -- use: {replacement}.\n"
        f"It may only appear in an explicit 'we do NOT do this' disclaimer.\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 2. Metric name / semantics invariants
# ---------------------------------------------------------------------------

#: column prefix -> (what the cells contain, higher_is_better)
#: Every metric column must be reachable from this table, so a new metric cannot
#: be added without declaring what it means.
METRIC_CONTRACT = {
    # ONE metric currency: every column is a transformation of the per-SOW
    # annual-unit objective values, and the SOW is the only scoring unit.
    "sat_multivariate_sow": ("fraction of SOWs whose per-SOW annual-unit objective "
                             "vector meets ALL criteria jointly [PRIMARY]", True),
    "sat_uni_sow__": ("fraction of SOWs whose per-SOW annual-unit objective meets "
                      "one criterion", True),
    "laplace__": ("mean SOW performance, natural units, own direction", None),
    "maximin__": ("worst SOW performance, natural units, own direction", None),
    # Incumbent-relative regret. The magnitude columns are in each objective's OWN
    # natural units and are never compared across objectives; the frequency columns
    # are unit-free and are what the cross-objective comparison actually uses.
    "regret_mean__": ("mean shortfall vs the incumbent over SOWs, natural units",
                      False),
    "regret_q90__": ("90th-percentile shortfall vs the incumbent over SOWs, "
                     "natural units", False),
    "regret_cond__": ("mean shortfall GIVEN a shortfall, natural units; NaN when "
                      "the policy is never worse than the incumbent", False),
    "gain_mean__": ("mean improvement over the incumbent, natural units; the "
                    "degeneracy companion to regret", True),
    "harm_freq__": ("fraction of SOWs in which this objective is worse than the "
                    "incumbent", False),
    "party_harm_freq__": ("fraction of SOWs in which ANY objective of this Decree "
                          "party is worse than the incumbent (a disjunction, never "
                          "a sum)", False),
    "no_harm_freq": ("fraction of SOWs in which NO objective is degraded (beyond "
                     "tolerance, for the _tau column)", True),
    "n_degraded_mean": ("mean number of objectives simultaneously worse than the "
                        "incumbent", False),
}


def _fixture_cube(tmp_path, worse_than_baseline: bool = False):
    """A tiny two-objective per-SOW cube plus a status-quo baseline."""
    import json

    import pandas as pd

    meta = {
        "is_ensemble": True,
        "obj_names": ["A", "B"],
        "thresholds": {"A": 0.9, "B": 10.0},
        "kinds": {"A": "ge", "B": "le"},
        "directions": {"A": "maximize", "B": "minimize"},
        "sow_labels": [0, 1],
        "realizations_per_sow": 2,
        "substrate": "sow_annual_unit",
    }

    def write(d, records):
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records, columns=["solution_id", "sow_id",
                                       "objective", "value"]).to_csv(
            d / "reeval_raw.csv.gz", index=False, compression="gzip")
        (d / "reeval_raw_meta.json").write_text(json.dumps(meta))

    write(tmp_path, [
        (0, 0, "A", 0.95), (0, 0, "B", 5.0),
        (0, 1, "A", 0.85), (0, 1, "B", 8.0),
        (1, 0, "A", 0.92), (1, 0, "B", 12.0),
        (1, 1, "A", 0.99), (1, 1, "B", 9.0),
    ])
    # A status quo that is either dominated (default) or dominant.
    b = (1.00, 1.0) if worse_than_baseline else (0.80, 15.0)
    write(tmp_path / "baseline", [
        (0, 0, "A", b[0]), (0, 0, "B", b[1]),
        (0, 1, "A", b[0]), (0, 1, "B", b[1]),
    ])
    return rob.load_raw(tmp_path), rob.load_raw(tmp_path / "baseline")


def test_every_scorecard_column_declares_its_meaning(tmp_path):
    """No metric column may exist without an entry in METRIC_CONTRACT."""
    raw, baseline = _fixture_cube(tmp_path)
    scorecard, _ = rob.score_robustness(raw, baseline, metrics=rob._DEFAULT_METRICS)
    for col in scorecard.columns:
        assert any(col.startswith(p) for p in METRIC_CONTRACT), (
            f"scorecard column '{col}' has no METRIC_CONTRACT entry -- declare "
            f"what its cells contain and whether higher is better."
        )


def test_higher_better_flags_match_the_contract(tmp_path):
    """A wrong orientation flag silently inverts an objective in every ranking."""
    raw, baseline = _fixture_cube(tmp_path)
    _, higher_better = rob.score_robustness(
        raw, baseline, metrics=rob._DEFAULT_METRICS)
    for col, flag in higher_better.items():
        for prefix, (_meaning, expected) in METRIC_CONTRACT.items():
            if col.startswith(prefix) and expected is not None:
                assert flag is expected, (
                    f"'{col}' is declared higher_better={flag}, contract says "
                    f"{expected}."
                )


def test_gain_and_regret_are_the_one_sided_halves_of_the_advantage(tmp_path):
    """gain_mean__ / regret_mean__ split the SIGNED incumbent advantage in two.

    The retired baseline-improvement metric shipped with the exact bug this
    guards: a name promising 'improvement' over a number that clipped it away.
    Its signed information now lives in the two one-sided halves, so the names
    must track the sign: a policy that DOMINATES the status quo carries all of
    its signal in gain (regret exactly 0), and a policy the status quo
    dominates carries all of it in regret (gain exactly 0). Anything else means
    a column is reporting the opposite of what its name says.
    """
    raw_good, base_good = _fixture_cube(tmp_path / "good")
    dom = rob.regret_magnitudes(raw_good, base_good)
    gain_cols = [c for c in dom.columns if c.startswith("gain_mean__")]
    regret_cols = [c for c in dom.columns if c.startswith("regret_mean__")]
    assert gain_cols and regret_cols
    assert (dom[gain_cols].to_numpy() > 0).all(), (
        "policies that beat the status quo must carry POSITIVE gain"
    )
    assert (dom[regret_cols].to_numpy() == 0).all(), (
        "a dominating policy has NO shortfall; regret must be exactly 0"
    )

    raw_bad, base_bad = _fixture_cube(tmp_path / "bad", worse_than_baseline=True)
    sub = rob.regret_magnitudes(raw_bad, base_bad)
    assert (sub[regret_cols].to_numpy() > 0).all(), (
        "policies the status quo dominates must carry POSITIVE regret, not be "
        "clipped to 0 -- clipping destroys the discrimination the family exists for"
    )
    assert (sub[gain_cols].to_numpy() == 0).all(), (
        "a dominated policy has NO improvement; gain must be exactly 0"
    )
