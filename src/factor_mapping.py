"""
factor_mapping.py - Success/failure surface classification over SOW spaces.

The scenario-discovery classification layer, shared by the scenario-discovery mechanism
test (`scripts/main/scenario_discovery.py`) and the factor-map figure sequence:
label each held-out E_test SOW success/failure for a policy under a satisficing
criterion set, fit a classifier of that label on the SOW's coordinates, and
expose the fitted probability surface, factor importances, and cross-validated
skill for figures and tables. Pure computation -- no file or figure output
(orchestration lives in ``scripts/main/factor_mapping_run.py``).

Two SOW coordinate systems are supported:

- **theta space** (PRIMARY): the sampled DU coordinates of the E_test
  design, the three climate-forcing amplitudes ``(e^m, r1, r2)`` plus, when
  the staged E_test carries it, the NYC entitlement-utilization factor ``u``
  (``src.demand_utilization``) -- the input space of the design (Hadjimichael
  et al. 2020; Gold et al. 2022; Lau et al. 2023). Designed-orthogonal LHS
  axes, so no redundancy screen is needed.
- **hazard space** (SUPPLEMENTAL): realized-sequence drought/flood descriptors,
  screened for redundancy first (:func:`screen_hazard_axes`); the space of the
  scenario-discovery coverage-deficit mechanism test.

Two classifier families share one interface (:func:`fit_classifier`), selected
by ``NYCOPT_FM_MODEL`` or the ``model=`` argument:

- ``"gbt"`` (DEFAULT): ``GradientBoostingClassifier`` with 250 trees,
  ``max_depth=2``, ``learning_rate=0.1`` (between Lau et al. 2023 and Gold
  et al. 2022). Importances are the trees' impurity-reduction importances.
- ``"logit"``: logistic regression in the scenario-discovery idiom of Quinn
  (2018, Water Programming, "Logistic regression for scenario discovery",
  https://waterprogramming.wpcomstaging.com/2018/05/04/logistic-regression-for-scenario-discovery/):
  ``ln(p / (1 - p))`` linear in the standardized factors plus, by default,
  every pairwise interaction. Factors are ranked by greedy forward selection
  on McFadden's pseudo-R^2, and each factor's importance is its pseudo-R^2
  gain on that path.

Stratified k-fold CV AUC is reported with every fit, computed identically for
both families -- the like-for-like skill comparison. :func:`compare_models`
fits both on one label; :func:`model_suffix` namespaces the non-default
family's artifacts so the two never overwrite each other.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp

import config
from src import robustness as rob

try:  # sklearn is the intended backend; the fallback is declared, never silent.
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only on a stripped env
    _HAS_SKLEARN = False


###############################################################################
# Settings (module constants; env-overridable. NO CLI value flags -- repo rule)
###############################################################################

#: Classifier family: ``"gbt"`` (gradient-boosted trees) or ``"logit"``
#: (logistic regression). Read through :func:`resolve_model`, never directly.
FM_MODEL: str = os.environ.get("NYCOPT_FM_MODEL", "gbt").strip().lower()

#: The supported classifier families and the ``FactorMapFit.backend`` each
#: one records.
MODELS: tuple = ("gbt", "logit")
MODEL_BACKENDS: dict = {"gbt": "gradient_boosting",
                        "logit": "logistic_regression"}

#: Logistic regression: inverse L2 penalty strength on the STANDARDIZED terms.
#: Large = effectively maximum likelihood; the residual penalty only keeps the
#: coefficients finite when a label is (quasi-)separable, where the MLE does
#: not exist.
FM_LOGIT_C: float = float(os.environ.get("NYCOPT_FM_LOGIT_C", "1e4"))

#: Logistic regression: include every pairwise interaction term (``x_i * x_j``
#: of the standardized factors) alongside the main effects. ``0`` = main
#: effects only (the linear-boundary model of the blog post).
FM_LOGIT_INTERACTIONS: bool = (
    os.environ.get("NYCOPT_FM_LOGIT_INTERACTIONS", "1").strip() != "0")

#: Logistic regression: solver iteration cap (near-separable labels need many).
FM_LOGIT_MAX_ITER: int = int(os.environ.get("NYCOPT_FM_LOGIT_MAX_ITER", "5000"))

#: Gradient-boosted classifier hyperparameters (see module docstring).
FM_N_TREES: int = int(os.environ.get("NYCOPT_FM_N_TREES", "250"))
FM_MAX_DEPTH: int = int(os.environ.get("NYCOPT_FM_MAX_DEPTH", "2"))
FM_LEARNING_RATE: float = float(os.environ.get("NYCOPT_FM_LEARNING_RATE", "0.1"))

#: Stratified CV folds for the reported AUC (reduced to the minority-class
#: count when smaller; below 2 usable folds CV is skipped with a reason).
FM_CV_FOLDS: int = int(os.environ.get("NYCOPT_FM_CV_FOLDS", "5"))

#: Spearman |rho| above which two hazard axes are collinear (Dormann et al. 2013).
FM_RHO_THRESHOLD: float = float(os.environ.get("NYCOPT_FM_RHO_THRESHOLD", "0.7"))

#: Probability-surface grid resolution per axis.
FM_GRID_RES: int = int(os.environ.get("NYCOPT_FM_GRID_RES", "60"))


###############################################################################
# SOW labels (success = the criterion set's conjunction holds in that SOW)
###############################################################################

def matrix_success_labels(values: np.ndarray, obj_names: list,
                          thresholds: dict, kinds: dict) -> np.ndarray:
    """Per-SOW success labels for one ``(G, M)`` objective matrix.

    Success = every FINITE-thresholded axis meets its criterion (non-binding
    ``+/-inf`` axes pass automatically); non-finite values are unsatisfied,
    mirroring ``robustness._satisfaction_cube``.
    """
    sat = rob._satisfy(np.asarray(values, dtype=float)[np.newaxis, :, :],
                       obj_names, thresholds, kinds)
    return sat[0].all(axis=1)                                       # (G,)


def success_labels(raw: rob.RawCube, thresholds: dict, kinds: dict = None,
                   solution_index: int = 0) -> np.ndarray:
    """Per-SOW success labels ``(G,)`` for one solution of a re-eval cube.

    Args:
        raw: The per-SOW re-eval cube.
        thresholds: Full threshold vector (a
            :meth:`~src.satisficing_criteria.CriterionSet.thresholds` result).
        kinds: ``{objective: "ge"|"le"}``; defaults to the cube's snapshot.
        solution_index: Row into the cube (NOT a solution id).
    """
    kinds = kinds if kinds is not None else raw.kinds
    return matrix_success_labels(raw.cube[solution_index], raw.obj_names,
                                 thresholds, kinds)


def failure_matrix(raw: rob.RawCube, thresholds: dict = None,
                   kinds: dict = None) -> np.ndarray:
    """Boolean ``(S, G)`` failure matrix: the criterion conjunction, negated.

    A SOW FAILS for a solution when the joint satisficing conjunction on its
    per-SOW annual-unit objective values is False (the Starr domain
    criterion). Discovery therefore inherits exactly the robustness criteria
    AND the robustness unit (the SOW). Default thresholds = the cube's meta
    snapshot (the all-axes reference); pass a criterion set's vector for the
    subset labels.
    """
    return ~rob._satisfaction_cube(raw, thresholds, kinds).all(axis=2)


def regret_matrix(raw: rob.RawCube, baseline: rob.RawCube,
                  tau: dict = None, axes=None) -> np.ndarray:
    """Boolean ``(S, G)`` regret matrix: some objective is worse than the incumbent.

    A SOW is labelled REGRET for a solution when at least one per-SOW objective
    is degraded by more than its tolerance ``tau_i`` relative to the status-quo
    FFMP policy in that same state -- the complement of the low-regret
    condition in ``robustness.regret_frequencies``, on the same unit as the
    reported regret family. Non-finite values count as regret, mirroring the
    non-finite-as-unsatisfied rule of the satisficing label.

    Args:
        raw: The policy cube.
        baseline: The incumbent cube.
        tau: Per-objective tolerance; defaults to ``robustness.tau_ladder``.
        axes: Restrict the disjunction to these objectives (a criterion set's
            member axes), exactly as ``regret_frequencies(axes=...)`` does.
            Passing a set's axes makes this the PER-SOW decomposition of that
            set's ``no_harm_freq_tau__{key}`` scorecard column. Default None =
            every objective.

    Returns:
        ``(S, G)`` boolean; True = the SOW is a regret SOW for that solution.
    """
    D = rob.incumbent_advantage(raw, baseline)                      # (S, G, M)
    tau = rob.tau_ladder(raw.obj_names) if tau is None else tau
    tau_vec = np.array([float(tau[n]) for n in raw.obj_names], dtype=float)
    if axes is not None:
        unknown = [n for n in axes if n not in raw.obj_names]
        if unknown:
            raise KeyError(f"axes not in this cube: {unknown}")
        keep = [k for k, n in enumerate(raw.obj_names) if n in set(axes)]
        D = D[:, :, keep]
        tau_vec = tau_vec[keep]
    finite = np.isfinite(D)
    return ((~finite) | (D < -tau_vec[None, None, :])).any(axis=2)


###############################################################################
# SOW feature spaces
###############################################################################

def theta_features(tag: str) -> tuple[np.ndarray, list]:
    """The E_test DU coordinates ``(G, 3)`` or ``(G, 4)`` and their names.

    Columns ``["em", "r1", "r2"]``: the water-year volume multiplier
    ``e^m`` (the plotting convention of ``src.plotting.forcing_space``) and
    the annual/semiannual harmonic amplitudes. When the staged E_test carries
    the NYC entitlement-utilization factor (``src.demand_utilization``; the
    ``u`` key of :func:`~src.plotting.forcing_space.load_etest_sample`), a
    fourth column ``"u"`` holds each SOW's nominal utilization; a legacy
    staging without it returns the three climate columns only.

    Args:
        tag: The staged E_test ensemble tag (= the re-eval tag).

    Returns:
        ``(X, names)``.
    """
    from src.plotting.forcing_space import load_etest_sample

    sample = load_etest_sample(config.STAGED_ENSEMBLE_DIR / tag)
    theta, names = sample["theta"], sample["theta_names"]
    cols = [
        np.exp(theta[:, names.index("m")]),
        theta[:, names.index("r1")],
        theta[:, names.index("r2")],
    ]
    out_names = ["em", "r1", "r2"]
    if "u" in sample:
        cols.append(np.asarray(sample["u"], dtype=float))
        out_names.append("u")
    return np.column_stack(cols), out_names


def assert_theta_alignment(X: np.ndarray, raw: rob.RawCube) -> None:
    """Hard error when the theta sample and the cube disagree on SOW count."""
    if len(X) != raw.n_sow:
        raise ValueError(
            f"theta sample has {len(X)} SOWs, cube has {raw.n_sow} -- "
            f"ensemble/tag mismatch"
        )


def cdf_transform(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map columns of ``values`` into the empirical-CDF/rank space of ``reference``.

    Reference-anchored (unlike ``scengen.subsample.empirical_cdf_normalize``,
    which ranks a matrix against itself): a search ensemble and E_test must
    share one coordinate system before their distances mean anything.

    Args:
        values: ``(n, m)`` raw hazard coordinates to transform.
        reference: ``(R, m)`` raw hazard coordinates defining the CDF (E_test).

    Returns:
        ``(n, m)`` array in ``[0, 1]``.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    reference = np.asarray(reference, dtype=float)
    out = np.empty_like(values, dtype=float)
    n_ref = reference.shape[0]
    for a in range(values.shape[1]):
        ref_sorted = np.sort(reference[:, a])
        out[:, a] = np.searchsorted(ref_sorted, values[:, a],
                                    side="right") / n_ref
    return np.clip(out, 0.0, 1.0)


def align_hazard_to_cube(raw: rob.RawCube, image: dict,
                         axis_idx: list) -> np.ndarray:
    """SOW-level hazard coordinates aligned to the re-eval cube's SOW axis.

    The hazard image carries one row per REALIZATION; the cube's unit is the
    SOW. Each SOW's coordinate is the MEAN over its realizations' descriptors.
    The join is on ``realization_id // realizations_per_sow == sow_id``, never
    positional.

    Returns:
        ``(G_cube, len(axis_idx))`` raw hazard coordinates aligned to
        ``raw.sow_labels``.

    Raises:
        KeyError: If a re-evaluated SOW has no hazard coordinates.
        ValueError: If the cube carries no ``realizations_per_sow``.
    """
    rps = raw.realizations_per_sow
    if not rps:
        raise ValueError(
            "the re-eval cube records no realizations_per_sow, so the hazard "
            "image's realization rows cannot be grouped into its SOWs."
        )
    H = np.asarray(image["H"], dtype=float)[:, axis_idx]
    by_sow: dict = {}
    for i, rid in enumerate(image["realization_ids"]):
        by_sow.setdefault(int(rid) // int(rps), []).append(i)
    missing = [g for g in raw.sow_labels if int(g) not in by_sow]
    if missing:
        raise KeyError(
            f"{len(missing)} re-evaluated SOW(s) have no hazard coordinates in "
            f"the E_test hazard image (e.g. {missing[:5]}). The hazard image "
            f"and the re-eval ensemble are not the same ensemble."
        )
    return np.vstack([H[by_sow[int(g)], :].mean(axis=0) for g in raw.sow_labels])


def screen_hazard_axes(H: np.ndarray, axes: list,
                       threshold: float = FM_RHO_THRESHOLD) -> dict:
    """Drop degenerate axes and keep one representative per redundant cluster.

    Stronger than the selection-side screen
    (``scengen.hazard_filling.screen_hazard_axes``, |rho_S| >= 0.95):
    importances over correlated axes are not interpretable (Quinn et al.
    2020), so discovery clusters at the lower cut before the fit.

    Args:
        H: ``(R, m)`` hazard image (raw metric values).
        axes: Length-``m`` axis names.
        threshold: Spearman ``|rho|`` redundancy cut.

    Returns:
        Dict with ``retained`` (axis names), ``retained_idx``, ``clusters``,
        ``degenerate``, ``rho`` (over the non-degenerate axes),
        ``screened_axes``, and ``residual_max_rho``.
    """
    from scengen.diagnostics import per_metric_spread, spearman_clusters
    from scengen.hazard_filling import DEFAULT_AXIS_PRIORITY

    H = np.asarray(H, dtype=float)
    spread = per_metric_spread(H, axes)
    degenerate = [a for a in axes if spread[a]["degenerate"]]
    kept = [a for a in axes if a not in degenerate]
    if not kept:  # every axis degenerate: fall back to the full set, loudly
        warnings.warn(
            "Every hazard axis was flagged degenerate by the spread screen; "
            "falling back to the unscreened axis set. Importances over these "
            "axes are not interpretable."
        )
        kept = list(axes)
        degenerate = []

    keep_idx = [axes.index(a) for a in kept]
    clusters = spearman_clusters(
        H[:, keep_idx], kept, threshold=threshold,
        priority=DEFAULT_AXIS_PRIORITY,
    )
    retained = list(clusters["representatives"])
    retained_idx = [axes.index(a) for a in retained]

    rho = np.atleast_2d(clusters["rho"])
    sub = [kept.index(a) for a in retained]
    resid = rho[np.ix_(sub, sub)].copy() if len(sub) > 1 else np.ones((1, 1))
    np.fill_diagonal(resid, 0.0)
    residual_max = float(np.abs(resid).max()) if len(sub) > 1 else 0.0
    if residual_max >= threshold:
        warnings.warn(
            f"Retained hazard axes still contain a pair with |rho_S| = "
            f"{residual_max:.2f} >= {threshold}. Factor importances over "
            f"correlated axes are unstable (Quinn et al. 2020) -- read the "
            f"importance ranking as indicative only."
        )
    return {
        "retained": retained,
        "retained_idx": retained_idx,
        "clusters": clusters["clusters"],
        "degenerate": degenerate,
        "rho": rho,
        "screened_axes": kept,
        "residual_max_rho": residual_max,
    }


###############################################################################
# Classifier family selection (gbt | logit)
###############################################################################

def resolve_model(model: str = None) -> str:
    """The classifier family to use: ``model`` or the :data:`FM_MODEL` default.

    Raises:
        ValueError: For a name outside :data:`MODELS`.
    """
    name = (FM_MODEL if model is None else str(model)).strip().lower()
    if name not in MODELS:
        raise ValueError(
            f"unknown classifier model {name!r}; expected one of {MODELS} "
            f"(set NYCOPT_FM_MODEL)."
        )
    return name


def model_suffix(model: str = None) -> str:
    """Artifact-name suffix for a family: ``""`` for gbt, ``"_logit"`` for logit.

    The gbt default keeps its historical, unsuffixed paths; every other
    family is namespaced so a comparison run never overwrites the gbt
    artifacts.
    """
    name = resolve_model(model)
    return "" if name == "gbt" else f"_{name}"


def artifact_dirname(model: str = None) -> str:
    """The ``comparison/{slug}/{tag}/`` subdirectory holding a family's fits."""
    return f"factor_mapping{model_suffix(model)}"


def classifier_settings(model: str = None) -> dict:
    """The active family's settings, for run metadata (JSON-serializable)."""
    name = resolve_model(model)
    out = {"model": name, "backend": MODEL_BACKENDS[name],
           "cv_folds": FM_CV_FOLDS}
    if name == "gbt":
        out.update({"n_estimators": FM_N_TREES, "max_depth": FM_MAX_DEPTH,
                    "learning_rate": FM_LEARNING_RATE})
    else:
        out.update({"C": FM_LOGIT_C, "interactions": FM_LOGIT_INTERACTIONS,
                    "max_iter": FM_LOGIT_MAX_ITER, "standardized": True,
                    "importance": "forward-selection McFadden pseudo-R2 gain"})
    return out


###############################################################################
# Classifier (Gold et al. 2022; Lau et al. 2023; Quinn 2018 for logit)
###############################################################################

@dataclass
class FactorMapFit:
    """A fitted success/failure classifier over SOW coordinates.

    ``predict_proba`` maps ``(n, m)`` points to the probability of the
    POSITIVE class of the labels it was fit on (success for factor maps,
    failure for the scenario-discovery mechanism test -- orientation is the caller's).

    Attributes:
        space: ``"theta"`` or ``"hazard"``.
        axes: Feature names, in feature order.
        importances: Per-axis factor importance, summing to 1. Impurity
            reduction for ``gradient_boosting``; forward-selection McFadden
            pseudo-R^2 gain for ``logistic_regression``.
        backend: ``"gradient_boosting"``, ``"logistic_regression"``,
            ``"single_class"``, or the declared ks/k-NN fallback.
        predict_proba: Probability surface callable.
        train_accuracy: Resubstitution accuracy (a fit diagnostic, not skill).
        cv_auc: Stratified k-fold ROC AUC -- the over-trust guard. NaN when
            CV was skipped (see ``cv_note``).
        cv_auc_std: Fold-to-fold standard deviation of the CV AUC.
        cv_accuracy: Stratified k-fold accuracy.
        n_pos: Positive-class count in the training labels.
        n_neg: Negative-class count.
        cv_note: Why CV was skipped, when it was ("" otherwise).
        meta: Hyperparameters and fold count actually used. A logistic fit
            also carries its diagnostics here (``pseudo_r2``, ``terms``,
            ``coef_std``, ``intercept_std``, ``forward_order``,
            ``forward_pseudo_r2``, ``converged``); :func:`fit_diagnostics`
            flattens them for a table row.
    """

    space: str
    axes: list
    importances: np.ndarray
    backend: str
    predict_proba: object = field(repr=False, default=None)
    train_accuracy: float = float("nan")
    cv_auc: float = float("nan")
    cv_auc_std: float = float("nan")
    cv_accuracy: float = float("nan")
    n_pos: int = 0
    n_neg: int = 0
    cv_note: str = ""
    meta: dict = field(default_factory=dict)


def _knn_probability(X: np.ndarray, y: np.ndarray, k: int = 25):
    """k-NN probability surface -- the declared no-sklearn fallback."""
    tree = cKDTree(X)
    k = int(min(max(3, k), len(X)))

    def _predict(grid: np.ndarray) -> np.ndarray:
        _, idx = tree.query(np.atleast_2d(grid), k=k)
        return y[np.atleast_2d(idx)].mean(axis=1)

    return _predict


def _cross_validate(X: np.ndarray, y: np.ndarray, folds: int,
                    make_clf) -> tuple[float, float, float, str]:
    """Stratified k-fold (AUC mean, AUC std, accuracy, note)."""
    n_min = int(min((y == 1).sum(), (y == 0).sum()))
    usable = min(folds, n_min)
    if usable < 2:
        return (np.nan, np.nan, np.nan,
                f"CV skipped: minority class has {n_min} member(s)")
    aucs, accs = [], []
    skf = StratifiedKFold(n_splits=usable, shuffle=True, random_state=0)
    for train, test in skf.split(X, y):
        clf = make_clf().fit(X[train], y[train])
        p = clf.predict_proba(X[test])[:, 1]
        aucs.append(roc_auc_score(y[test], p))
        accs.append(float(((p >= 0.5).astype(int) == y[test]).mean()))
    return (float(np.mean(aucs)), float(np.std(aucs)),
            float(np.mean(accs)), "")


def mcfadden_r2(y: np.ndarray, p: np.ndarray) -> float:
    """McFadden's pseudo-R^2: ``1 - lnL(model) / lnL(intercept only)``.

    The model-comparison metric of the logistic scenario-discovery idiom
    (Quinn 2018). Probabilities are clipped away from 0/1 so a separable fit
    scores just under 1 rather than overflowing.

    Args:
        y: Length-``n`` 0/1 labels.
        p: Length-``n`` fitted ``P(y = 1)``.

    Returns:
        The pseudo-R^2, or NaN for a single-class label (no null deviance).
    """
    y = np.asarray(y).astype(int)

    def _loglik(prob: np.ndarray) -> float:
        prob = np.clip(np.asarray(prob, dtype=float), 1e-12, 1.0 - 1e-12)
        return float(np.where(y == 1, np.log(prob), np.log1p(-prob)).sum())

    ll_null = _loglik(np.full(len(y), y.mean()))
    if not np.isfinite(ll_null) or ll_null == 0.0 or y.min() == y.max():
        return float("nan")
    return 1.0 - _loglik(p) / ll_null


def _fit_gbt(X: np.ndarray, y: np.ndarray, axes: list, space: str,
             n_estimators: int, max_depth: int, learning_rate: float,
             cv: int, random_state: int, n_pos: int, n_neg: int,
             meta: dict) -> FactorMapFit:
    """The gradient-boosted fit (``model="gbt"``) of :func:`fit_classifier`."""

    def make_clf():
        return GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state,
        )

    clf = make_clf().fit(X, y)
    cv_auc, cv_auc_std, cv_acc, note = _cross_validate(X, y, cv, make_clf)
    return FactorMapFit(
        space=space, axes=list(axes),
        importances=np.asarray(clf.feature_importances_, dtype=float),
        backend="gradient_boosting",
        predict_proba=lambda g: clf.predict_proba(np.atleast_2d(g))[:, 1],
        train_accuracy=float(clf.score(X, y)),
        cv_auc=cv_auc, cv_auc_std=cv_auc_std, cv_accuracy=cv_acc,
        n_pos=n_pos, n_neg=n_neg, cv_note=note, meta=meta,
    )


def _make_logit(interactions: bool, C: float):
    """Standardize -> (pairwise interactions) -> near-unpenalized logit.

    Standardizing FIRST centres the factors, so each interaction term is the
    product of two z-scores: it is not collinear with its main effects and
    its coefficient is on the same scale as theirs.
    """
    steps = [("scale", StandardScaler())]
    if interactions:
        steps.append(("interact", PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False)))
    steps.append(("logit", LogisticRegression(
        C=C, max_iter=FM_LOGIT_MAX_ITER)))
    return Pipeline(steps)


def _logit_term_names(clf, axes: list) -> list:
    """Model-term names in coefficient order: ``"em"``, ..., ``"em:r1"``, ..."""
    if "interact" not in clf.named_steps:
        return list(axes)
    powers = clf.named_steps["interact"].powers_
    return [":".join(a for a, p in zip(axes, row) if p) for row in powers]


def _logit_forward_selection(X: np.ndarray, y: np.ndarray,
                             make_clf) -> tuple[list, list]:
    """Greedy forward selection of FACTORS on McFadden's pseudo-R^2.

    Start from the intercept-only model; at each step add the factor whose
    inclusion raises pseudo-R^2 the most. Adding a factor brings in its main
    effect AND (when the model has interactions) its interaction with every
    factor already in the model, so the path is hierarchical and its last
    step is exactly the full model.

    Returns:
        ``(order, cumulative)``: feature indices in entry order and the
        model's pseudo-R^2 after each entry.
    """
    remaining = list(range(X.shape[1]))
    order, cumulative = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        while remaining:
            scores = []
            for k in remaining:
                cols = order + [k]
                sub = make_clf().fit(X[:, cols], y)
                scores.append(
                    mcfadden_r2(y, sub.predict_proba(X[:, cols])[:, 1]))
            best = int(np.argmax(scores))
            order.append(remaining.pop(best))
            cumulative.append(float(scores[best]))
    return order, cumulative


def _fit_logit(X: np.ndarray, y: np.ndarray, axes: list, space: str,
               interactions: bool, cv: int, n_pos: int, n_neg: int,
               meta: dict) -> FactorMapFit:
    """The logistic-regression fit (``model="logit"``) of :func:`fit_classifier`.

    The surface, CV skill, and coefficients come from the FULL model (every
    factor, plus every pairwise interaction when enabled). The forward
    selection only ranks the factors: each one's importance is the
    pseudo-R^2 it added when it entered, normalized to sum to 1. An
    interaction's contribution is therefore credited to whichever of its two
    factors entered LATER -- the ranking is path-dependent wherever
    interactions matter, which the recorded ``forward_order`` makes explicit.
    """

    def make_clf():
        return _make_logit(interactions, FM_LOGIT_C)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = make_clf().fit(X, y)
        cv_auc, cv_auc_std, cv_acc, note = _cross_validate(X, y, cv, make_clf)
    logit = clf.named_steps["logit"]
    converged = bool(np.max(logit.n_iter_) < FM_LOGIT_MAX_ITER)
    if not converged:
        warnings.warn(
            f"logistic regression hit the {FM_LOGIT_MAX_ITER}-iteration cap "
            f"on a {space}-space fit ({n_pos} positive / {n_neg} negative): "
            f"the label is probably (quasi-)separable. Raise "
            f"NYCOPT_FM_LOGIT_MAX_ITER or lower NYCOPT_FM_LOGIT_C."
        )

    order, cumulative = _logit_forward_selection(X, y, make_clf)
    # Nested near-MLE models cannot lose likelihood; enforce that against
    # optimizer noise so no factor is credited a negative gain.
    cum = np.maximum.accumulate(np.clip(np.nan_to_num(cumulative), 0.0, None))
    importances = np.zeros(X.shape[1])
    importances[order] = np.diff(cum, prepend=0.0)
    total = float(importances.sum())
    importances = (importances / total if total > 0
                   else np.full(len(importances), 1.0 / len(importances)))

    terms = _logit_term_names(clf, axes)
    meta = {
        **meta,
        "pseudo_r2": mcfadden_r2(y, clf.predict_proba(X)[:, 1]),
        "terms": terms,
        # Coefficients on the standardized terms (z-scores and their products).
        "coef_std": {t: float(c) for t, c in zip(terms, logit.coef_[0])},
        "intercept_std": float(logit.intercept_[0]),
        "forward_order": [axes[k] for k in order],
        "forward_pseudo_r2": [float(v) for v in cum],
        "converged": converged,
    }
    return FactorMapFit(
        space=space, axes=list(axes), importances=importances,
        backend="logistic_regression",
        predict_proba=lambda g: clf.predict_proba(np.atleast_2d(g))[:, 1],
        train_accuracy=float(clf.score(X, y)),
        cv_auc=cv_auc, cv_auc_std=cv_auc_std, cv_accuracy=cv_acc,
        n_pos=n_pos, n_neg=n_neg, cv_note=note, meta=meta,
    )


def fit_classifier(X: np.ndarray, y: np.ndarray, axes: list,
                   space: str = "theta", n_estimators: int = None,
                   max_depth: int = None, learning_rate: float = None,
                   cv: int = None, random_state: int = 0,
                   model: str = None,
                   interactions: bool = None) -> FactorMapFit:
    """Fit ``P(y = 1)`` on SOW coordinates + CV skill, with either family.

    ``model="gbt"`` (the default) fits boosted trees. Trees are
    monotone-invariant, so rank-space and raw-value fits are equivalent; pass
    whichever coordinates the figure will be drawn in.

    ``model="logit"`` fits a logistic regression on the standardized factors
    and, by default, every pairwise interaction (:func:`_fit_logit`). Unlike
    trees it is NOT monotone-invariant: the log-odds are linear (bilinear with
    interactions) in the coordinates as passed, so a fit on ``e^m`` differs
    from one on ``m``, and a rank-space fit from a raw-value one.

    If sklearn is unavailable the model degrades -- loudly, never silently --
    to KS-statistic importances with a k-NN probability surface, which
    preserves the map and the ranking but not the interaction structure.

    Args:
        X: ``(G, m)`` SOW coordinates.
        y: Length-``G`` boolean/int labels (1 = the class whose probability
            ``predict_proba`` reports).
        axes: Length-``m`` feature names.
        space: Recorded on the fit (``"theta"`` | ``"hazard"``).
        n_estimators, max_depth, learning_rate: gbt-only overrides of the
            module defaults (``FM_*``); ignored by ``"logit"``.
        cv: Override of :data:`FM_CV_FOLDS` (both families).
        random_state: Seed for the gbt fit. The CV shuffle is fixed, so both
            families are scored on the SAME folds.
        model: ``"gbt"`` | ``"logit"``; default :data:`FM_MODEL`
            (``NYCOPT_FM_MODEL``).
        interactions: logit-only override of :data:`FM_LOGIT_INTERACTIONS`.

    Returns:
        A :class:`FactorMapFit`.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    model = resolve_model(model)
    n_estimators = FM_N_TREES if n_estimators is None else int(n_estimators)
    max_depth = FM_MAX_DEPTH if max_depth is None else int(max_depth)
    learning_rate = (FM_LEARNING_RATE if learning_rate is None
                     else float(learning_rate))
    interactions = (FM_LOGIT_INTERACTIONS if interactions is None
                    else bool(interactions))
    cv = FM_CV_FOLDS if cv is None else int(cv)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if model == "gbt":
        meta = {"model": model, "n_estimators": n_estimators,
                "max_depth": max_depth, "learning_rate": learning_rate,
                "cv_folds_requested": cv}
    else:
        meta = {"model": model, "C": FM_LOGIT_C, "interactions": interactions,
                "max_iter": FM_LOGIT_MAX_ITER, "cv_folds_requested": cv}

    if not _HAS_SKLEARN:
        warnings.warn(
            "scikit-learn is not installed: falling back to KS-statistic "
            "importances + a k-NN probability surface. The reported "
            f"importances are NOT {MODEL_BACKENDS[model]} factor importances "
            "-- say so when reporting."
        )
        ks = np.array([
            ks_2samp(X[y == 1, a], X[y == 0, a]).statistic
            if n_pos and n_neg else 0.0
            for a in range(X.shape[1])
        ])
        imp = ks / ks.sum() if ks.sum() > 0 else np.full(len(ks), 1.0 / len(ks))
        return FactorMapFit(
            space=space, axes=list(axes), importances=imp,
            backend="ks_knn_fallback", predict_proba=_knn_probability(X, y),
            n_pos=n_pos, n_neg=n_neg, cv_note="CV skipped: no sklearn",
            meta=meta,
        )

    if n_pos == 0 or n_neg == 0:
        # A single-class label cannot be fit; return the degenerate constant
        # surface with a recorded reason rather than raising, so orchestration
        # over near-saturated criterion sets reports the degeneracy as data.
        p = float(n_pos > 0)
        return FactorMapFit(
            space=space, axes=list(axes),
            importances=np.full(len(axes), np.nan), backend="single_class",
            predict_proba=lambda g: np.full(len(np.atleast_2d(g)), p),
            train_accuracy=1.0, n_pos=n_pos, n_neg=n_neg,
            cv_note="label has one class -- criterion degenerate on this cube",
            meta=meta,
        )

    if model == "logit":
        return _fit_logit(X, y, axes, space, interactions, cv, n_pos, n_neg,
                          meta)
    return _fit_gbt(X, y, axes, space, n_estimators, max_depth, learning_rate,
                    cv, random_state, n_pos, n_neg, meta)


def fit_diagnostics(fit: FactorMapFit) -> dict:
    """Flat, table-ready diagnostics a fit carries beyond the shared fields.

    Empty for every backend but ``logistic_regression``, so a fits table gains
    columns only in a logit run (and a degenerate single-class fit inside one
    leaves them blank).

    Returns:
        ``pseudo_r2`` (McFadden, full model), ``converged``, ``fs_order``
        (factors in forward-selection entry order, ``">"``-joined),
        ``fs_r2__{axis}`` (cumulative pseudo-R^2 once that factor entered),
        ``intercept_std`` and ``coef__{term}`` (coefficients on the
        standardized terms; interactions are named ``a:b``).
    """
    meta = fit.meta or {}
    if "pseudo_r2" not in meta:
        return {}
    out = {"pseudo_r2": meta["pseudo_r2"], "converged": meta["converged"],
           "fs_order": ">".join(meta["forward_order"])}
    out.update({f"fs_r2__{a}": v for a, v in
                zip(meta["forward_order"], meta["forward_pseudo_r2"])})
    out["intercept_std"] = meta["intercept_std"]
    out.update({f"coef__{t}": c for t, c in meta["coef_std"].items()})
    return out


def compare_models(X: np.ndarray, y: np.ndarray, axes: list,
                   space: str = "theta", models: tuple = MODELS,
                   **kwargs) -> dict:
    """Fit every classifier family on ONE label, for a side-by-side read.

    The CV folds are identical across families (fixed shuffle), so the
    ``cv_auc`` / ``cv_accuracy`` of the returned fits compare like for like.

    Args:
        X, y, axes, space: As :func:`fit_classifier`.
        models: Families to fit (default: all of :data:`MODELS`).
        **kwargs: Forwarded to :func:`fit_classifier`.

    Returns:
        ``{model: FactorMapFit}`` in ``models`` order.
    """
    return {m: fit_classifier(X, y, axes, space=space, model=m, **kwargs)
            for m in models}


def probability_surface(fit: FactorMapFit, X: np.ndarray, ax1: int, ax2: int,
                        grid_res: int = None, fixed: str = "median",
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fitted probability over a 2-D grid of two features.

    Off-plane features are held at their observed median (the
    Hadjimichael et al. 2020 / Gold et al. 2022 convention) or mean.

    Args:
        fit: A :class:`FactorMapFit`.
        X: ``(G, m)`` training coordinates (defines grid bounds and the
            off-plane anchors).
        ax1: Feature index on the horizontal axis.
        ax2: Feature index on the vertical axis.
        grid_res: Grid points per axis (default :data:`FM_GRID_RES`).
        fixed: ``"median"`` or ``"mean"`` off-plane anchor.

    Returns:
        ``(g1, g2, P)`` where ``g1``/``g2`` are the axis grids and ``P`` is
        the ``(grid_res, grid_res)`` probability of the fit's positive class,
        indexed ``P[j, i]`` = (g2[j], g1[i]).
    """
    X = np.asarray(X, dtype=float)
    grid_res = FM_GRID_RES if grid_res is None else int(grid_res)
    anchor = (np.nanmedian(X, axis=0) if fixed == "median"
              else np.nanmean(X, axis=0))
    g1 = np.linspace(np.nanmin(X[:, ax1]), np.nanmax(X[:, ax1]), grid_res)
    g2 = np.linspace(np.nanmin(X[:, ax2]), np.nanmax(X[:, ax2]), grid_res)
    G1, G2 = np.meshgrid(g1, g2)
    pts = np.tile(anchor, (G1.size, 1))
    pts[:, ax1] = G1.ravel()
    pts[:, ax2] = G2.ravel()
    P = np.asarray(fit.predict_proba(pts), dtype=float).reshape(G1.shape)
    return g1, g2, P


def top_axes(fit: FactorMapFit, k: int = 2) -> list:
    """Indices of the ``k`` most important features, importance-descending."""
    imp = np.asarray(fit.importances, dtype=float)
    if not np.isfinite(imp).any():
        return list(range(min(k, len(fit.axes))))
    order = np.argsort(np.nan_to_num(imp, nan=-1.0))[::-1]
    return [int(i) for i in order[:k]]


###############################################################################
# Analysis-policy selection
###############################################################################

def _normalized_mean_objectives(raw: rob.RawCube) -> np.ndarray:
    """``(S, M)`` mean re-evaluated objectives, oriented so 0 = ideal, 1 = worst.

    Direction-oriented (maximize objectives flipped) and min-max normalized
    over the solution set, so the ideal point is the origin.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(raw.cube, axis=1)  # (S, M)
    signs = raw.direction_signs()
    loss = means * -signs[None, :]  # lower = better, for every objective
    lo = np.nanmin(loss, axis=0)
    hi = np.nanmax(loss, axis=0)
    span = np.where(np.abs(hi - lo) > 0, hi - lo, 1.0)
    return (loss - lo[None, :]) / span[None, :]


def select_compromise(raw: rob.RawCube, rule: str = "best_satisficing",
                      thresholds: dict = None) -> dict:
    """Choose the per-design analysis policy, and report both candidate rules.

    Scenario discovery is run on a small number of compromise solutions, not
    the whole Pareto front (Kasprzyk et al. 2013; Gold et al. 2022).

    Args:
        raw: The design's re-eval cube.
        rule: ``"best_satisficing"`` or ``"min_dist_ideal"``.
        thresholds: Satisficing criterion vector for the ``best_satisficing``
            ranking; defaults to the cube's meta snapshot. Under a subset
            criterion set the compromise policy is set-specific -- always
            record which set selected the analyzed ``solution_id``.

    Returns:
        Dict with ``solution_id`` (the analyzed policy), ``rule``, the two
        candidate ids (``best_satisficing_id``, ``min_dist_ideal_id``), the
        chosen policy's ``satisficing`` fraction and ``distance_to_ideal``,
        and ``index`` (row into the cube).

    Raises:
        ValueError: For an unknown ``rule``, or when every solution is all-NaN.
    """
    if rule not in ("best_satisficing", "min_dist_ideal"):
        raise ValueError(
            f"unknown compromise rule {rule!r}; expected 'best_satisficing' "
            f"or 'min_dist_ideal' (set NYCOPT_SD_COMPROMISE_RULE)."
        )
    sat = rob.satisficing_multivariate_sow(
        raw, thresholds).to_numpy(dtype=float)                      # (S,)
    dist = np.linalg.norm(_normalized_mean_objectives(raw), axis=1)  # (S,)

    alive = np.any(np.isfinite(raw.cube), axis=(1, 2))
    if not alive.any():
        raise ValueError("every solution in this re-eval cube is all-NaN (failed).")
    masked_dist = np.where(alive, dist, np.inf)
    masked_sat = np.where(alive, sat, -np.inf)

    # Ties on satisficing are common (a saturated criterion ties everything),
    # so break them on distance-to-ideal rather than on solution-id order.
    best_sat = int(np.lexsort((masked_dist, -masked_sat))[0])
    best_dist = int(np.argmin(masked_dist))
    idx = best_sat if rule == "best_satisficing" else best_dist
    return {
        "rule": rule,
        "index": idx,
        "solution_id": int(raw.solution_ids[idx]),
        "best_satisficing_id": int(raw.solution_ids[best_sat]),
        "min_dist_ideal_id": int(raw.solution_ids[best_dist]),
        "satisficing": float(sat[idx]),
        "distance_to_ideal": float(dist[idx]),
    }
