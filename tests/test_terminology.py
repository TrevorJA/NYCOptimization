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

import numpy as np
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
    "regret_from_baseline": "improvement_vs_baseline (signed, positive = better)",
    "overfitting gap": "deleted -- undefined in Brodeur (2020) and invalid "
                       "under a measure change",
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
    "sat_multivariate": ("fraction of realizations meeting ALL criteria", True),
    "sat_multivariate_sow": ("fraction of SOWs whose within-SOW-collapsed performance "
                             "meets ALL criteria", True),
    "sat_uni__": ("fraction of realizations meeting one criterion", True),
    "laplace__": ("mean performance, natural units, own direction", None),
    "maximin__": ("worst realization, natural units, own direction", None),
    "vs_baseline__": ("signed improvement over status quo; positive = better", True),
}


def _fixture_cube(tmp_path, worse_than_baseline: bool = False):
    """A tiny two-objective cube plus a status-quo baseline."""
    import json

    import pandas as pd

    meta = {
        "is_ensemble": True,
        "base_names": ["A", "B"],
        "thresholds": {"A": 0.9, "B": 10.0},
        "kinds": {"A": "ge", "B": "le"},
        "directions": {"A": "maximize", "B": "minimize"},
        "realization_indices": [0, 1],
    }

    def write(d, records):
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records, columns=["solution_id", "realization_id",
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


def test_improvement_vs_baseline_name_matches_its_sign(tmp_path):
    """The name says 'improvement'. The number must go UP when the policy is better.

    This is the exact bug the metric shipped with: it computed a shortfall clipped
    at zero, so beating the status quo scored 0 and the name said the opposite of
    the quantity -- while also collapsing to ~0 for every policy, since optimized
    policies dominate the status quo nearly everywhere.
    """
    raw_good, base_good = _fixture_cube(tmp_path / "good")
    better = rob.improvement_vs_baseline(raw_good, base_good, normalize="none")
    assert (better.to_numpy()[np.isfinite(better.to_numpy())] > 0).all(), (
        "policies that beat the status quo must score POSITIVE"
    )

    raw_bad, base_bad = _fixture_cube(tmp_path / "bad", worse_than_baseline=True)
    worse = rob.improvement_vs_baseline(raw_bad, base_bad, normalize="none")
    assert (worse.to_numpy()[np.isfinite(worse.to_numpy())] < 0).all(), (
        "policies that lose to the status quo must score NEGATIVE, not be clipped "
        "to 0 -- clipping destroys the discrimination the metric exists for"
    )
