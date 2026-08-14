"""factor_mapping_run.py - Success/failure surfaces across designs x criterion sets.

The study's PRIMARY scenario-discovery factor maps: for each campaign design's
per-criterion-set compromise policy (plus the FFMP incumbent), label every
E_test SOW success/failure under the criterion set and fit the boosted-tree
success classifier of ``src.factor_mapping`` on the SOW's coordinates --
THETA space ``(e^m, r1, r2)`` primary (the sampled DU input space, the
Hadjimichael et al. 2020 / Gold et al. 2022 setting), hazard space as the
supplemental view (screened realized-sequence descriptors, rank space).

This script only orchestrates and persists; all computation lives in
``src.factor_mapping``. The written artifacts are the figure-render inputs --
figures never need the raw cubes:

  outputs/comparison/{slug}/{tag}/factor_mapping/{criterion}/
      factor_map_fits.csv        one row per fit (importances, CV skill)
      factor_map_labels.csv      per-SOW coordinates + pass/fail per policy
      factor_map_surfaces.npz    top-2-axis probability grids per fit
      factor_map_reference_sows.csv   expected / dry marker coordinates
      factor_mapping_meta.json

Settings via env (repo rule: no CLI value flags):
  NYCOPT_FM_CRITERIA   comma-separated criterion-set keys (default: all named)
  NYCOPT_FM_SPACES     comma-separated in {theta, hazard} (default: both)
  NYCOPT_FM_*          classifier settings (see src.factor_mapping)

Run::

    python scripts/main/factor_mapping_run.py --formulation ffmp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from src import factor_mapping as fm  # noqa: E402
from src import results_data as rd  # noqa: E402
from src.satisficing_criteria import NAMED_SETS, criterion_by_key  # noqa: E402


def _requested_criteria():
    raw = os.environ.get("NYCOPT_FM_CRITERIA", "").strip()
    if not raw:
        return list(NAMED_SETS)
    return [criterion_by_key(k.strip()) for k in raw.split(",") if k.strip()]


def _requested_spaces() -> list[str]:
    raw = os.environ.get("NYCOPT_FM_SPACES", "theta,hazard")
    spaces = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in spaces if s not in ("theta", "hazard")]
    if unknown:
        sys.exit(f"[factor_mapping] unknown NYCOPT_FM_SPACES entries {unknown}; "
                 f"expected 'theta' and/or 'hazard'.")
    return spaces


def _hazard_coordinates(results: dict, tag: str) -> tuple | None:
    """E_test SOW hazard coordinates (rank space) + retained axis names.

    Returns ``(X, names)`` aligned to the first design's cube (all designs
    share the E_test SOW axis), or None -- with a warning -- when the hazard
    image is not staged, so a theta-only run still completes.
    """
    from scripts.main.scenario_discovery import load_etest_hazard_image
    from src.ensembles import get_ensemble_spec

    try:
        etest = load_etest_hazard_image(get_ensemble_spec(tag))
    except SystemExit:
        warnings.warn("[factor_mapping] E_test hazard image not staged; "
                      "hazard-space fits skipped.")
        return None
    screen = fm.screen_hazard_axes(etest["H"], etest["hazard_axes"])
    first = next(iter(results.values())).raw
    H = fm.align_hazard_to_cube(first, etest, screen["retained_idx"])
    return fm.cdf_transform(H, H), screen["retained"]


def run(formulation: str, reeval_tag: str | None) -> dict:
    """Fit and persist the factor-map artifacts for every (design x set x space)."""
    from src.reeval_core import reeval_tag as tag_of
    from src.ensembles import get_ensemble_spec

    spec = (get_ensemble_spec(reeval_tag) if reeval_tag
            else config.REEVAL_ENSEMBLE_SPEC)
    tag = tag_of(spec)
    slug = config.derive_slug(formulation)
    criteria = _requested_criteria()
    spaces = _requested_spaces()

    results = rd.load_design_results(tag, slug=slug)
    first = next(iter(results.values())).raw

    features: dict[str, tuple] = {}
    if "theta" in spaces:
        X_theta, theta_names = fm.theta_features(tag)
        fm.assert_theta_alignment(X_theta, first)
        features["theta"] = (X_theta, theta_names)
    if "hazard" in spaces:
        hz = _hazard_coordinates(results, tag)
        if hz is not None:
            features["hazard"] = hz

    root = config.OUTPUTS_DIR / "comparison" / slug / tag / "factor_mapping"
    n_fits = 0
    for cset in criteria:
        fit_rows, label_rows, surfaces = [], [], {}
        for design, res in results.items():
            thr = cset.thresholds(res.raw.thresholds, res.raw.kinds)
            compromise = fm.select_compromise(res.raw, thresholds=thr)
            policies = [(str(compromise["solution_id"]),
                         fm.success_labels(res.raw, thr,
                                           solution_index=compromise["index"]))]
            if res.incumbent is not None:
                policies.append(("incumbent", fm.matrix_success_labels(
                    res.incumbent, res.raw.obj_names, thr, res.raw.kinds)))
            for policy, ok in policies:
                if policy != "incumbent" or design == next(iter(results)):
                    # The incumbent is design-independent; label it once.
                    label_rows += _label_records(design, policy, cset.key,
                                                 features, res, ok)
                for space, (X, names) in features.items():
                    fit = fm.fit_classifier(X, ok, names, space=space)
                    a1, a2 = fm.top_axes(fit, 2) if len(names) > 1 else (0, 0)
                    key = f"{design}__{policy}__{space}"
                    if fit.predict_proba is not None and len(names) > 1:
                        g1, g2, P = fm.probability_surface(fit, X, a1, a2)
                        surfaces[f"{key}__g1"] = g1
                        surfaces[f"{key}__g2"] = g2
                        surfaces[f"{key}__P"] = P
                        surfaces[f"{key}__axes"] = np.array(
                            [names[a1], names[a2]])
                    fit_rows.append({
                        "design": design, "policy": policy,
                        "criterion": cset.key, "space": space,
                        "backend": fit.backend,
                        "n_pos": fit.n_pos, "n_neg": fit.n_neg,
                        "train_accuracy": fit.train_accuracy,
                        "cv_auc": fit.cv_auc, "cv_auc_std": fit.cv_auc_std,
                        "cv_accuracy": fit.cv_accuracy, "cv_note": fit.cv_note,
                        "top_axis_1": names[a1], "top_axis_2": names[a2],
                        **{f"imp__{n}": float(v) for n, v in
                           zip(names, np.atleast_1d(fit.importances))},
                    })
                    n_fits += 1

        out = root / cset.key
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fit_rows).to_csv(out / "factor_map_fits.csv", index=False)
        pd.DataFrame(label_rows).to_csv(out / "factor_map_labels.csv",
                                        index=False)
        if surfaces:
            np.savez_compressed(out / "factor_map_surfaces.npz", **surfaces)
        if "theta" in features:
            fm.reference_sows(*features["theta"]).to_csv(
                out / "factor_map_reference_sows.csv", index=False)
        (out / "factor_mapping_meta.json").write_text(json.dumps({
            "formulation": formulation, "moea_slug": slug, "reeval_tag": tag,
            "criterion": cset.key, "criterion_axes": list(cset.axes),
            "criterion_thresholds": cset.criteria,
            "criterion_kinds": {a: first.kinds[a] for a in cset.axes},
            "criterion_label": cset.label,
            "spaces": sorted(features), "n_sow": first.n_sow,
            "gbm": {"n_estimators": fm.FM_N_TREES,
                    "max_depth": fm.FM_MAX_DEPTH,
                    "learning_rate": fm.FM_LEARNING_RATE,
                    "cv_folds": fm.FM_CV_FOLDS},
            "designs": list(results),
        }, indent=2))
        print(f"[factor_mapping] {cset.key}: "
              f"{len(fit_rows)} fits -> {out}")

    return {"criteria": [c.key for c in criteria], "spaces": sorted(features),
            "n_fits": n_fits, "root": str(root)}


def _label_records(design: str, policy: str, criterion: str, features: dict,
                   res, ok: np.ndarray) -> list[dict]:
    """Tidy per-SOW label rows carrying the theta coordinates when present."""
    rows = []
    theta = features.get("theta")
    for g, sow in enumerate(res.raw.sow_labels):
        row = {"design": design, "policy": policy, "criterion": criterion,
               "sow_id": int(sow), "pass": bool(ok[g])}
        if theta is not None:
            X, names = theta
            row.update({n: float(X[g, k]) for k, n in enumerate(names)})
        rows.append(row)
    return rows


def main() -> None:
    """CLI. Identifiers only -- settings live in env (see module docstring)."""
    p = argparse.ArgumentParser(
        description="Fit success/failure factor-map surfaces per criterion set.")
    p.add_argument("--formulation", default="ffmp")
    p.add_argument("--reeval-tag", default=None,
                   help="Re-eval ensemble preset id (default: configured E_test).")
    args = p.parse_args()
    run(args.formulation, args.reeval_tag)


if __name__ == "__main__":
    main()
