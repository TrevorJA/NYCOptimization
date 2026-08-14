"""
factor_maps.py - Focal-criterion pass/fail maps over the theta forcing space.

For a small set of frontier policies (plus the FFMP incumbent), each E_test
SOW is classified pass/fail under the FOCAL criterion set (the conjunction of
that set's member axes only; ``NYCOPT_FOCAL_CRITERION``) and mapped at its
forcing-space coordinates ``theta = (m, r1, r2)`` -- the water-year log-mean
shift and the annual/semiannual harmonic amplitudes of the DU forcing
parameterization (``src.plotting.forcing_space``). Small multiples per policy
with a shared legend show WHERE each policy fails and how the robust
optimized policies differ from the incumbent. The fitted probability-surface
companions live in ``src.factor_mapping`` /
``scripts/main/factor_mapping_run.py``; this figure is the raw-label view.

Policy selection is rule-based so it restates automatically under a different
focal criterion: per design, the maximum-robustness policy on the
(joint Starr, no-harm frequency) frontier plus, when different, the most
robust policy that is (near-)never harmful vs the incumbent; a design whose
joint Starr is identically zero contributes its maximin policy instead (the
one whose WORST single-axis satisficing fraction is largest -- no mean
aggregation). ``NYCOPT_FACTOR_POLICIES`` (JSON ``{design: [solution_id, ...]}``)
overrides the rule.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config
from src import results_data as rd
from src.satisficing_criteria import focal_criterion
from src.plotting.forcing_space import load_etest_sample
from src.plotting.satisficing_diagnostics import _add_footer, _designs
from src.plotting.style import design_label, save_figure

#: Pass/fail mark styling (Okabe-Ito green for the rare pass class; light grey
#: for the failing majority so the success region pops).
PASS_COLOR = "#009E73"
FAIL_COLOR = "0.62"

#: A policy counts as "never harmful" on the frontier at or above this
#: no-harm frequency.
NO_HARM_FLOOR = 0.99


def select_focal_policies(results: dict) -> list[dict]:
    """Rule-based factor-map policy selection (env-overridable).

    Returns dicts with ``design``, ``solution_id`` (None = incumbent), and a
    display ``label``.
    """
    focal = focal_criterion()
    override = os.environ.get("NYCOPT_FACTOR_POLICIES")
    picks: list[dict] = []
    for d in _designs(results):
        res = results[d]
        thr = rd.criterion_thresholds(res, focal)
        sat = rd.satisfaction(res.raw, thresholds=thr)
        joint = rd.joint_fraction(sat)
        ids = np.asarray(res.raw.solution_ids, dtype=int)
        if override:
            chosen = [int(s) for s in json.loads(override).get(d, [])]
        elif joint.max() <= 0:
            # All-zero joint Starr: fall back to the maximin policy (largest
            # worst-axis satisficing fraction) -- no mean aggregation.
            worst_axis = rd.univariate_fraction(sat).min(axis=1)
            chosen = [int(ids[np.argmax(worst_axis)])]
        else:
            no_harm = res.scorecard["no_harm_freq_tau"].reindex(ids).to_numpy(float)
            # Max joint Starr, ties broken by no-harm frequency (else a
            # dominated tie-mate can displace the frontier policy).
            best = np.lexsort((no_harm, joint))[-1]
            chosen = [int(ids[best])]
            safe = no_harm >= NO_HARM_FLOOR
            if safe.any():
                j_safe = np.where(safe, joint, -np.inf)
                best_safe = int(ids[np.argmax(j_safe)])
                if best_safe not in chosen and j_safe.max() > 0:
                    chosen.append(best_safe)
        for sid in chosen:
            short = design_label(d).split(" (")[0]
            picks.append({"design": d, "solution_id": sid,
                          "label": f"{short}\npolicy #{sid}"})
    picks.append({"design": _designs(results)[0], "solution_id": None,
                  "label": "FFMP incumbent\n(status quo)"})
    return picks


def _pass_vector(results: dict, pick: dict) -> np.ndarray:
    """Per-SOW joint pass/fail (focal criterion) for one selected policy."""
    focal = focal_criterion()
    res = results[pick["design"]]
    thr = rd.criterion_thresholds(res, focal)
    if pick["solution_id"] is None:
        inc = rd.incumbent_satisfaction(res, thresholds=thr)
        return inc.all(axis=1)
    sat = rd.satisfaction(res.raw, thresholds=thr)
    row = list(res.raw.solution_ids).index(pick["solution_id"])
    return sat[row].all(axis=1)


def fig_factor_maps_theta_focal(results: dict, out_dir: Path,
                                table_dir: Path) -> dict:
    """Pass/fail of the focal joint criterion over the theta forcing space.

    Columns: the selected frontier policies + the incumbent. Rows: the
    (volume multiplier e^m, annual amplitude r1) plane on top and the
    (e^m, semiannual amplitude r2) plane below. Green = the SOW meets all
    criteria in the focal set under that policy; grey = it fails at least
    one. Column titles carry the pass count (repo rule: titles carry at most
    the pass fraction).
    """
    focal = focal_criterion()
    designs = _designs(results)
    first = results[designs[0]].raw
    tag = first.meta["reeval_tag"]
    sample = load_etest_sample(config.STAGED_ENSEMBLE_DIR / tag)
    theta, names = sample["theta"], sample["theta_names"]
    if sample["n_sow"] != first.n_sow:
        raise ValueError(f"theta sample has {sample['n_sow']} SOWs, cube has "
                         f"{first.n_sow} -- ensemble/tag mismatch")
    em = np.exp(theta[:, names.index("m")])
    r1 = theta[:, names.index("r1")]
    r2 = theta[:, names.index("r2")]

    picks = select_focal_policies(results)
    n = len(picks)
    rows_spec = [(r1, "Seasonal amplitude, $r_1$"),
                 (r2, "Semiannual amplitude, $r_2$")]

    rows = []
    fig, axes = plt.subplots(2, n, figsize=(2.75 * n, 6.2),
                             sharex=True, sharey="row")
    for ci, pick in enumerate(picks):
        ok = _pass_vector(results, pick)
        for ri, (yv, ylab) in enumerate(rows_spec):
            ax = axes[ri, ci]
            ax.scatter(em[~ok], yv[~ok], s=13, color=FAIL_COLOR, lw=0,
                       alpha=0.85, zorder=3)
            ax.scatter(em[ok], yv[ok], s=20, color=PASS_COLOR, lw=0, zorder=4)
            ax.grid(color="0.93", lw=0.6)
            ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title(f"{pick['label']}\n({int(ok.sum())}/{len(ok)} pass)",
                             fontsize=8.5)
            if ci == 0:
                ax.set_ylabel(ylab)
            if ri == 1:
                ax.set_xlabel("Water-year volume\nmultiplier, $e^{m}$")
        rows += [{"design": pick["design"],
                  "solution_id": (-1 if pick["solution_id"] is None
                                  else pick["solution_id"]),
                  "sow_id": int(s), "em": float(a), "r1": float(b),
                  "r2": float(c), "pass": bool(p)}
                 for s, a, b, c, p in zip(first.sow_labels, em, r1, r2, ok)]

    n_axes = len(focal.axes) if not focal.reference else len(first.obj_names)
    handles = [
        Line2D([], [], marker="o", ls="none", ms=7, color=PASS_COLOR,
               label=(f"SOW meets all focal criteria "
                      f"({n_axes} axes) under this policy")),
        Line2D([], [], marker="o", ls="none", ms=6, color=FAIL_COLOR,
               label="SOW fails at least one focal criterion"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    sel_txt = "; ".join(
        f"{design_label(p['design']).split(' (')[0]} #{p['solution_id']}"
        for p in picks if p["solution_id"] is not None)
    policies = (f"Policies shown: rule-selected focal-criterion frontier "
                f"champions ({sel_txt}) and the FFMP incumbent, each judged "
                f"per SOW on {first.n_sow} held-out E_test SOWs.")
    _add_footer(results, fig, y=-0.10, policies=policies,
                criteria=rd.criterion_thresholds(results[designs[0]], focal),
                criteria_header=(f"Focal satisficing criteria — {focal.label} "
                                 f"(all must hold):"))

    save_figure(fig, out_dir / f"factor_maps_theta_{focal.key}")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(table_dir / f"factor_maps_theta_{focal.key}.csv",
                              index=False)
    return {"criterion": focal.key,
            "policies": [p["label"].replace("\n", " ") for p in picks]}
