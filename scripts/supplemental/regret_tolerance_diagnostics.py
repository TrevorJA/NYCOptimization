"""regret_tolerance_diagnostics.py - Fix the two free parameters of the regret comparison.

The incumbent-relative regret comparison has exactly two numbers that are not
determined by the data: the no-harm tolerance ``tau_i = k * eps_i`` and the
non-inferiority margin ``delta`` on ``no_harm_freq_tau``. Both are
pre-registration quantities. This script measures everything that is admissible
as an anchor for them, and nothing that is not.

**The trap it exists to avoid.** A tolerance read off the distribution of
candidate-policy regret guarantees its own answer, exactly as a satisficing
threshold read off the baseline's own E_test quantiles would
(``robustness_threshold_diagnostics.md`` section 0b, rule 4 — those candidates
are reported in every table and never adopted). Worse, the hypothesis here is a
NON-INFERIORITY claim ("hazard filling is not worse in regret"), and
non-inferiority claims are flattered by insensitivity: a loose tolerance
saturates ``no_harm_freq_tau`` at 1 for every design and makes the claim
trivially true. Every choice below is therefore biased toward the DISCRIMINATING
end, and the assay-sensitivity check (pass B) exists to prove the comparison
could have detected a difference had one been there.

Two passes, deliberately separable.

**Pass A — needs only the incumbent's cube, so it can run the moment step 05
lands and long before any search finishes.** It answers "how small can a
tolerance be before it is measuring Monte Carlo noise rather than harm?" Under
the null that a policy is operationally identical to the incumbent, the SOW-mean
difference is pure estimator noise; a tolerance below that scale reports noise as
harm. Reported two ways:

  - parametric: ``tau_floor = z * sqrt(2) * median_theta[ sigma_i(theta)/sqrt(R) ]``
  - non-parametric: a split-half null, partitioning each SOW's R realizations into
    two halves and treating one as a pseudo-policy against the other

Both are UNPAIRED and therefore CONSERVATIVE (they overstate the floor). The real
comparison is paired — a policy and the incumbent are simulated on the same
inflow sequences, so their difference cancels most of the shared natural
variability — and the paired floor is smaller. It cannot be estimated without a
second policy on E_test, so pass A reports the unpaired bound explicitly labelled
as an upper bound, and pass B replaces it the moment any policy cube exists. An
overstated floor pushes ``k`` up, i.e. toward less discrimination, which is the
direction that flatters the hypothesis: hence the label, and hence pass B.

**Pass B — needs the re-evaluated policy sets.** It answers "over what range of
tolerance is the comparison informative at all, and how big is a difference that
means nothing?"

  - the discrimination band: the ``k`` at which ``no_harm_freq_tau`` is starved
    (< RTOL_SATURATION_LO for every design) or saturated (> RTOL_SATURATION_HI)
  - the paired SOW-level bootstrap standard error of the BETWEEN-DESIGN
    difference (both designs are scored on the same SOWs, so the difference is
    paired and its error is smaller than either margin's)
  - two empirical nulls: seed-pairs within a draw (pure search stochasticity) and
    draw-pairs within a design (search + ensemble construction). The draw is the
    declared unit of analysis, so the draw-level null is the denominator the
    design contrast must beat, and it is what ``delta`` is anchored on
  - assay sensitivity against ``historic``, the unmatched reference expected to be
    worse
  - the binding objective, and how much of the joint no-harm frequency is
    co-occurrence rather than independent accumulation across eight objectives

Neither pass uses the between-design contrast to set either parameter. Pass B's
nulls are within-design quantities, which is what makes them admissible: they fix
the scale of "no difference" without touching the direction of the answer.

Zero simulation; every number is a reduction of persisted cubes.

Run:
    venv/Scripts/python.exe scripts/supplemental/regret_tolerance_diagnostics.py
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as sc                                    # noqa: E402
import src.robustness as rob                                        # noqa: E402


###############################################################################
# Pass A - the noise floor (incumbent cube only)
###############################################################################

def within_sow_noise(baseline: rob.RawCube) -> pd.DataFrame:
    """Per objective: the within-SOW spread of the incumbent and its SOW-mean error.

    ``sigma`` is the median across SOWs of the within-SOW standard deviation of the
    whole-record base metric; ``se_sow_mean = sigma / sqrt(R)`` is the standard
    error of one SOW's collapsed value; ``null_sd_unpaired = sqrt(2) * se`` is the
    standard deviation of a difference between two INDEPENDENT policies' SOW means
    under the null that they behave alike.

    The last column is an upper bound. The real comparison is paired (a policy and
    the incumbent share inflow sequences), which cancels most of the shared
    variability, so the paired null SD is smaller and the true floor lower.

    Raises:
        ValueError: If the cube carries no SOW grouping, in which case the
            per-SOW estimator whose noise this measures does not exist.
    """
    if baseline.sow_ids is None:
        raise ValueError(
            "the incumbent cube carries no sow_ids, so there is no SOW-mean "
            "estimator to characterise. Re-evaluate on an ensemble with forcing "
            "profiles."
        )
    rows = []
    groups = baseline.sow_groups()
    for k, name in enumerate(baseline.base_names):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sds, ns = [], []
            for _sow, cols in groups:
                v = baseline.cube[:, cols, k].ravel()
                v = v[np.isfinite(v)]
                if v.size >= 2:
                    sds.append(float(np.std(v, ddof=1)))
                    ns.append(v.size)
        if not sds:
            rows.append({"objective": name, "n_sow": 0, "R_median": np.nan,
                         "sigma_within_sow": np.nan, "se_sow_mean": np.nan,
                         "null_sd_unpaired": np.nan})
            continue
        sigma = float(np.median(sds))
        r = float(np.median(ns))
        se = sigma / np.sqrt(r)
        rows.append({"objective": name, "n_sow": len(sds), "R_median": r,
                     "sigma_within_sow": sigma, "se_sow_mean": se,
                     "null_sd_unpaired": float(np.sqrt(2.0) * se)})
    return pd.DataFrame(rows)


def tolerance_floor(noise: pd.DataFrame, z: float = None) -> pd.DataFrame:
    """Per-objective tolerance floor in natural units, and the ladder rung that clears it.

    ``tau_floor = z * null_sd_unpaired`` is the smallest tolerance at which a
    policy identical to the incumbent is flagged as harming that objective in at
    most ``Phi(-z)`` of SOWs. Below it, ``harm_freq`` is measuring the estimator.

    Returns the noise table plus ``eps`` (the objective's just-noticeable
    difference), ``tau_floor``, and ``k_floor = tau_floor / eps`` — the ladder rung
    at which that objective becomes defensible.
    """
    from src.objectives import OBJECTIVES

    z = sc.RTOL_FALSE_HARM_Z if z is None else float(z)
    out = noise.copy()
    out["z"] = z
    out["tau_floor"] = z * out["null_sd_unpaired"]
    out["eps"] = [float(OBJECTIVES[n].epsilon) if n in OBJECTIVES else np.nan
                  for n in out["objective"]]
    out["k_floor"] = out["tau_floor"] / out["eps"]
    return out


def split_half_null(baseline: rob.RawCube, grid=None, reps: int = None,
                    seed: int = None) -> pd.DataFrame:
    """Non-parametric null: the incumbent compared against itself.

    Each SOW's realizations are split into two halves; one half's mean plays the
    policy, the other the incumbent. Every flagged "harm" is then false by
    construction, so the resulting frequencies are the false-harm rate of the
    tolerance ladder.

    The two halves are disjoint, so the difference is noisier than the real
    (paired, full-sample) comparison. Differences are rescaled by
    ``sqrt( (2/R) / (1/n1 + 1/n2) )`` to correspond to two independent full-R
    estimates, which removes the half-sample penalty but NOT the pairing penalty;
    what remains is the same unpaired upper bound the parametric floor reports,
    measured without a normality assumption.

    Returns:
        Tidy frame: tau_k, objective (or ``__joint__``), false_harm_freq.
    """
    from src.objectives import OBJECTIVES

    grid = sc.RTOL_TAU_GRID if grid is None else grid
    reps = sc.RTOL_SPLIT_HALF_REPS if reps is None else int(reps)
    rng = np.random.default_rng(sc.RTOL_SPLIT_HALF_SEED if seed is None else seed)

    names = baseline.base_names
    signs = baseline.direction_signs()
    groups = baseline.sow_groups()
    eps = np.array([float(OBJECTIVES[n].epsilon) if n in OBJECTIVES else np.nan
                    for n in names])

    acc = {k: {"joint": [], "per_obj": []} for k in grid}
    for _rep in range(reps):
        diffs = np.full((len(groups), len(names)), np.nan)
        for g, (_sow, cols) in enumerate(groups):
            cols = np.asarray(cols)
            perm = rng.permutation(cols.size)
            a, b = cols[perm[: cols.size // 2]], cols[perm[cols.size // 2:]]
            if a.size == 0 or b.size == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ma = np.nanmean(baseline.cube[:, a, :], axis=(0, 1))
                mb = np.nanmean(baseline.cube[:, b, :], axis=(0, 1))
            scale = np.sqrt((2.0 / cols.size) / (1.0 / a.size + 1.0 / b.size))
            diffs[g, :] = signs * (ma - mb) * scale
        finite = np.isfinite(diffs)
        for k in grid:
            tau = k * eps
            harm = (~finite) | (diffs < -tau[None, :])
            acc[k]["per_obj"].append(harm.mean(axis=0))
            acc[k]["joint"].append(float(harm.any(axis=1).mean()))

    rows = []
    for k in grid:
        rows.append({"tau_k": float(k), "objective": "__joint__",
                     "false_harm_freq": float(np.mean(acc[k]["joint"]))})
        per = np.mean(np.vstack(acc[k]["per_obj"]), axis=0)
        for name, v in zip(names, per):
            rows.append({"tau_k": float(k), "objective": name,
                         "false_harm_freq": float(v)})
    return pd.DataFrame(rows)


def ladder_shape_table(floors: pd.DataFrame) -> pd.DataFrame:
    """Compare the candidate ladder SHAPES, not just the scale.

    The ladder ``tau_i = k * u_i`` shares ONE ``k`` across objectives, so the choice
    of unit ``u_i`` decides what a rung means on each axis. Three candidates:

      - ``eps``   : the just-noticeable difference. Fails where an epsilon sits
                    below its objective's noise floor, because then even k = 1 is
                    inside the estimator's noise on that axis while being far
                    outside it on others.
      - ``floor`` : the noise floor. Equalises the false-harm rate across
                    objectives but throws away the resolution information.
      - ``max``   : ``max(eps_i, floor_i)``. Keeps epsilon where resolution binds
                    and the floor where noise binds, so one ``k`` means the same
                    thing everywhere. This is the recommended shape.

    The ``ratio`` column is ``floor_i / eps_i``: any objective far above 1 is one
    whose epsilon cannot carry a tolerance, and is the reason this table exists.
    """
    out = floors[["objective", "eps", "tau_floor"]].copy()
    out["ratio_floor_over_eps"] = out["tau_floor"] / out["eps"]
    out["unit_eps"] = out["eps"]
    out["unit_floor"] = out["tau_floor"]
    out["unit_max"] = np.maximum(out["eps"], out["tau_floor"])
    out["eps_below_floor"] = out["ratio_floor_over_eps"] > 1.0
    return out


def floors_dict(floors: pd.DataFrame) -> dict:
    """``{objective: tau_floor}`` for :func:`src.robustness.tau_ladder`."""
    return {str(r["objective"]): float(r["tau_floor"])
            for _, r in floors.iterrows() if np.isfinite(r["tau_floor"])}


def headline_k(floors: pd.DataFrame, grid=None) -> dict:
    """The pre-registered headline rung: the SMALLEST k clearing every floor.

    Smallest rather than largest because the hypothesis is a non-inferiority
    claim, which a loose tolerance flatters (see :data:`supplemental_config.RTOL_TAU_RULE`).

    Under the recommended ``max`` shape the unit already absorbs the floor, so
    every rung clears by construction and ``k`` = the smallest non-zero rung; the
    ``eps``-shape answer is still reported because it is what diagnoses a
    mis-scaled epsilon in the first place.

    Returns:
        ``{k, binding_objective, k_floor_max, cleared}``; ``k`` is None when no rung
        on the grid clears every objective, which is itself the finding — the ladder
        needs extending upward, or its shape is wrong.
    """
    grid = sorted(sc.RTOL_TAU_GRID if grid is None else grid)
    kf = floors.dropna(subset=["k_floor"])
    if kf.empty:
        return {"k": None, "binding_objective": None, "k_floor_max": np.nan,
                "cleared": False}
    worst = kf.loc[kf["k_floor"].idxmax()]
    need = float(worst["k_floor"])
    ok = [k for k in grid if k >= need]
    return {"k": float(ok[0]) if ok else None,
            "binding_objective": str(worst["objective"]),
            "k_floor_max": need, "cleared": bool(ok)}


###############################################################################
# Pass B - discrimination, nulls, and the margin (needs policy cubes)
###############################################################################

def discrimination_band(profile: pd.DataFrame) -> pd.DataFrame:
    """Per tolerance rung: is the comparison starved, informative, or saturated?

    A rung is SATURATED when every design's no-harm frequency exceeds
    ``RTOL_SATURATION_HI`` — the non-inferiority claim is then trivially true and
    carries no information — and STARVED when every design falls below
    ``RTOL_SATURATION_LO``. Only the informative band can support the claim.
    """
    rows = []
    for k, g in profile.groupby("tau_k"):
        per_design = g.groupby("design")["best"].mean()
        lo, hi = float(per_design.min()), float(per_design.max())
        if lo > sc.RTOL_SATURATION_HI:
            verdict = "saturated"
        elif hi < sc.RTOL_SATURATION_LO:
            verdict = "starved"
        else:
            verdict = "informative"
        rows.append({"tau_k": float(k), "n_designs": int(per_design.size),
                     "min_design": lo, "max_design": hi,
                     "spread": hi - lo, "verdict": verdict})
    return pd.DataFrame(rows).sort_values("tau_k")


def null_differences(profile: pd.DataFrame) -> pd.DataFrame:
    """Empirical nulls: differences that carry no design effect, by construction.

    Two levels, both WITHIN a design:

      - ``seed``: two seeds of the same draw. They share the ensemble, so they
        differ only by MOEA stochasticity.
      - ``draw``: two draws of the same design. They differ by search stochasticity
        AND ensemble construction, which is exactly the variability a between-design
        difference must exceed, because the draw is the declared unit of analysis.

    ``delta`` is anchored on the draw level. Using the seed level would understate
    it by omitting construction variance and would make the comparison look more
    decisive than the replication scheme supports.

    Returns:
        Tidy frame: level, design, tau_k, abs_diff.
    """
    rows = []
    for (design, k), g in profile.groupby(["design", "tau_k"]):
        for (draw,), h in g.groupby(["draw"]):
            for a, b in itertools.combinations(h["best"].dropna().tolist(), 2):
                rows.append({"level": "seed", "design": design, "draw": draw,
                             "tau_k": float(k), "abs_diff": abs(a - b)})
        by_draw = g.groupby("draw")["best"].mean().dropna().tolist()
        for a, b in itertools.combinations(by_draw, 2):
            rows.append({"level": "draw", "design": design, "draw": -1,
                         "tau_k": float(k), "abs_diff": abs(a - b)})
    return pd.DataFrame(rows)


def paired_bootstrap_se(runs, design_a: str, design_b: str, k: float,
                        n_boot: int = None, seed: int = None) -> dict:
    """SOW-level bootstrap SE of the PAIRED between-design difference in Pi_tau.

    Both designs are scored on the same states of the world, so the difference is
    paired: resampling SOWs and recomputing BOTH designs on the resample keeps that
    pairing and gives a standard error smaller than differencing two independently
    bootstrapped margins would.

    The statistic is the difference of each design's best-policy no-harm frequency,
    which is the reported endpoint.

    Returns:
        ``{k, design_a, design_b, diff, se, ci_lo, ci_hi, n_sow, n_boot}``; ``se``
        is NaN when either design has no usable run.
    """
    n_boot = sc.RTOL_BOOTSTRAP_N if n_boot is None else int(n_boot)
    rng = np.random.default_rng(sc.RTOL_BOOTSTRAP_SEED if seed is None else seed)

    def _harm_free(design: str):
        """(n_run, n_sow) boolean stacks of 'this policy harms nothing here'."""
        stacks = []
        for r in runs:
            if r.design != design:
                continue
            raw = rob.load_raw(r.path)
            bdir = r.path / "baseline"
            if not any((bdir / f).exists()
                       for f in ("reeval_raw.parquet", "reeval_raw.csv.gz")):
                continue
            base = rob.load_raw(bdir)
            if raw.sow_ids is None or base.sow_ids is None:
                continue
            D = rob.incumbent_advantage(raw, base,
                                        within_sow_agg=sc.RTOL_WITHIN_SOW_AGG)
            tau = rob.tau_ladder(raw.base_names, k=k)
            tv = np.array([tau[n] for n in raw.base_names], dtype=float)
            finite = np.isfinite(D)
            stacks.append((~((~finite) | (D < -tv[None, None, :]))).all(axis=2))
        return stacks

    sa, sb = _harm_free(design_a), _harm_free(design_b)
    out = {"tau_k": float(k), "design_a": design_a, "design_b": design_b,
           "diff": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
           "n_sow": 0, "n_boot": n_boot}
    if not sa or not sb:
        return out
    n_sow = sa[0].shape[1]
    if any(s.shape[1] != n_sow for s in sa + sb):
        warnings.warn("[rtol] designs disagree on the SOW count; the difference "
                      "is not paired and the bootstrap is skipped.")
        return out

    def _endpoint(stacks, cols):
        # Best policy per run on the resampled SOWs, then the mean over runs.
        return float(np.mean([s[:, cols].mean(axis=1).max() for s in stacks]))

    allc = np.arange(n_sow)
    out["diff"] = _endpoint(sa, allc) - _endpoint(sb, allc)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        cols = rng.integers(0, n_sow, n_sow)
        draws[i] = _endpoint(sa, cols) - _endpoint(sb, cols)
    out["se"] = float(np.std(draws, ddof=1))
    out["ci_lo"], out["ci_hi"] = (float(np.quantile(draws, 0.025)),
                                  float(np.quantile(draws, 0.975)))
    out["n_sow"] = int(n_sow)
    return out


def assay_sensitivity(profile: pd.DataFrame,
                      control: str = None) -> pd.DataFrame:
    """Can the metric separate the design that SHOULD be worse, at each tolerance?

    A non-inferiority claim is only interpretable if the comparison had the power
    to find a difference. ``historic`` is the unmatched prevailing-practice
    reference; if it is indistinguishable from the matched designs at a given
    tolerance, then "no difference" at that tolerance is evidence of an insensitive
    measurement, not of equivalent designs.

    Returns:
        Tidy frame: tau_k, design, control_gap (design minus control), and
        ``separates`` (whether the gap exceeds the draw-level null spread).
    """
    control = sc.RTOL_ASSAY_CONTROL_DESIGN if control is None else control
    nulls = null_differences(profile)
    draw_null = (nulls[nulls["level"] == "draw"]
                 .groupby("tau_k")["abs_diff"].max())
    rows = []
    for k, g in profile.groupby("tau_k"):
        per_design = g.groupby("design")["best"].mean()
        if control not in per_design.index:
            continue
        floor = float(draw_null.get(k, np.nan))
        for design, v in per_design.items():
            if design == control:
                continue
            gap = float(v - per_design[control])
            rows.append({"tau_k": float(k), "design": design,
                         "control": control, "control_gap": gap,
                         "draw_null_max": floor,
                         "separates": bool(np.isfinite(floor) and abs(gap) > floor)})
    return pd.DataFrame(rows)


def joint_vs_independent(runs, k: float) -> pd.DataFrame:
    """How much of the joint no-harm frequency is co-occurrence, not accumulation.

    With eight objectives, a small per-objective harm rate compounds: if harms were
    independent across objectives the joint no-harm frequency would be
    ``prod_i (1 - phi_i)``. The gap between that product and the observed
    ``no_harm_freq_tau`` says whether the joint metric is dominated by one binding
    objective (harms co-occur; observed >> independent) or by accumulation across
    many (observed ~ independent), which changes how it should be read entirely.
    """
    rows = []
    for r in runs:
        raw = rob.load_raw(r.path)
        bdir = r.path / "baseline"
        if not any((bdir / f).exists()
                   for f in ("reeval_raw.parquet", "reeval_raw.csv.gz")):
            continue
        base = rob.load_raw(bdir)
        if raw.sow_ids is None or base.sow_ids is None:
            continue
        tau = rob.tau_ladder(raw.base_names, k=k)
        f = rob.regret_frequencies(raw, base, tau=tau,
                                   within_sow_agg=sc.RTOL_WITHIN_SOW_AGG)
        phi = f[[f"harm_freq__{n}" for n in raw.base_names]].to_numpy()
        indep = np.prod(1.0 - phi, axis=1)
        obs = f["no_harm_freq_tau"].to_numpy()
        binding = [raw.base_names[i] for i in np.argmax(phi, axis=1)]
        for j, sid in enumerate(f.index):
            rows.append({"design": r.design, "draw": r.draw, "seed": r.seed,
                         "solution_id": int(sid), "tau_k": float(k),
                         "no_harm_observed": float(obs[j]),
                         "no_harm_if_independent": float(indep[j]),
                         "co_occurrence_gap": float(obs[j] - indep[j]),
                         "binding_objective": binding[j]})
    return pd.DataFrame(rows)


###############################################################################
# Orchestration
###############################################################################

def run_pass_a(baseline_dir=None) -> dict:
    """Everything computable from the incumbent alone. No policy runs needed."""
    baseline_dir = Path(sc.RTOL_REEVAL_BASELINE_DIR if baseline_dir is None
                        else baseline_dir)
    if not any((baseline_dir / f).exists()
               for f in ("reeval_raw.parquet", "reeval_raw.csv.gz")):
        raise SystemExit(
            f"[rtol] no incumbent cube at {baseline_dir}. Pass A needs only the "
            f"step-05 baseline re-evaluated on the test ensemble; run that first."
        )
    base = rob.load_raw(baseline_dir)
    noise = within_sow_noise(base)
    floors = tolerance_floor(noise)
    shapes = ladder_shape_table(floors)
    split = split_half_null(base)
    head = headline_k(floors)

    sc.RTOL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    floors.to_csv(sc.rtol_table_path("rtol_noise_floor"), index=False)
    shapes.to_csv(sc.rtol_table_path("rtol_ladder_shapes"), index=False)
    split.to_csv(sc.rtol_table_path("rtol_split_half_null"), index=False)
    (sc.RTOL_TABLES_DIR / "rtol_floors.json").write_text(
        json.dumps(floors_dict(floors), indent=2))

    print(f"[rtol] pass A: noise floor over {base.n_sow} SOWs x "
          f"{base.realizations_per_sow} realizations")
    for _, r in floors.iterrows():
        print(f"[rtol]   {r['objective']:38s} tau_floor={r['tau_floor']:.4g} "
              f"eps={r['eps']:.4g} -> k_floor={r['k_floor']:.2f}")

    bad = shapes[shapes["eps_below_floor"]]
    if not bad.empty:
        print(f"[rtol] LADDER SHAPE: {len(bad)} objective(s) have an epsilon BELOW "
              f"their noise floor, so a shared k does not mean the same thing on "
              f"every axis:")
        for _, r in bad.iterrows():
            print(f"[rtol]   {r['objective']:38s} floor/eps = "
                  f"{r['ratio_floor_over_eps']:.1f}x")
        print(f"[rtol]   -> use the 'max' shape: pass rtol_floors.json to "
              f"robustness.tau_ladder(floors=...), or record the resulting vector "
              f"in NYCOPT_REGRET_TAU. The eps-only ladder below is what that fixes.")

    if head["cleared"]:
        print(f"[rtol] eps-shape headline rung k = {head['k']:g} "
              f"(binding: {head['binding_objective']}, needs k >= "
              f"{head['k_floor_max']:.2f})")
    else:
        print(f"[rtol] NO RUNG CLEARS THE FLOOR under the eps shape: the binding "
              f"objective {head['binding_objective']} needs k >= "
              f"{head['k_floor_max']:.2f}, above the top of RTOL_TAU_GRID. Adopt the "
              f"'max' shape rather than stretching the ladder -- a rung that large "
              f"is far outside the noise on every other axis.")
    return {"noise": noise, "floors": floors, "shapes": shapes,
            "split_half": split, "headline": head}


def run_pass_b(formulation: str = "ffmp", reeval_tag: str | None = None) -> dict:
    """Everything needing the re-evaluated policy sets. Skipped when absent."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compare_designs", PROJECT_DIR / "scripts" / "main" / "compare_designs.py")
    cd = importlib.util.module_from_spec(spec)
    sys.modules["compare_designs"] = cd
    spec.loader.exec_module(cd)

    import config
    tag = reeval_tag or cd.reeval_tag_of(config.REEVAL_ENSEMBLE_SPEC)
    runs = cd.discover_runs(formulation, tag)
    if not runs:
        print(f"[rtol] pass B skipped: no re-eval runs for '{formulation}' / '{tag}'.")
        return {}

    profile = cd.regret_tolerance_sweep(runs, grid=sc.RTOL_TAU_GRID)
    if profile.empty:
        print("[rtol] pass B skipped: no run carries a status-quo baseline.")
        return {}

    band = discrimination_band(profile)
    nulls = null_differences(profile)
    assay = assay_sensitivity(profile)

    sc.RTOL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    profile.to_csv(sc.rtol_table_path("rtol_tolerance_profile"), index=False)
    band.to_csv(sc.rtol_table_path("rtol_discrimination_band"), index=False)
    nulls.to_csv(sc.rtol_table_path("rtol_null_differences"), index=False)
    assay.to_csv(sc.rtol_table_path("rtol_assay_sensitivity"), index=False)

    k = sc.RTOL_ADOPTED_K
    boots, coocc = [], pd.DataFrame()
    if k is not None:
        designs = sorted(profile["design"].unique())
        for a, b in itertools.combinations(designs, 2):
            boots.append(paired_bootstrap_se(runs, a, b, k))
        coocc = joint_vs_independent(runs, k)
        pd.DataFrame(boots).to_csv(
            sc.rtol_table_path("rtol_paired_bootstrap"), index=False)
        coocc.to_csv(sc.rtol_table_path("rtol_joint_vs_independent"), index=False)

        draw_null = nulls[(nulls["level"] == "draw") & (nulls["tau_k"] == k)]
        se = max((b["se"] for b in boots if np.isfinite(b["se"])), default=np.nan)
        delta = np.nanmax([2.0 * se, draw_null["abs_diff"].max()
                           if not draw_null.empty else np.nan])
        print(f"[rtol] margin at k={k:g}: delta = {delta:.4f} "
              f"(2 x paired bootstrap SE = {2 * se:.4f}, "
              f"draw-level null max = {draw_null['abs_diff'].max():.4f})")
        (sc.RTOL_TABLES_DIR / "rtol_margin.json").write_text(json.dumps(
            {"tau_k": float(k), "delta": float(delta),
             "paired_bootstrap_se": float(se),
             "draw_null_max": float(draw_null["abs_diff"].max())
             if not draw_null.empty else None,
             "rule": sc.RTOL_MARGIN_RULE}, indent=2))
    else:
        print("[rtol] RTOL_ADOPTED_K is unset, so the margin, the paired bootstrap "
              "and the co-occurrence table are not computed. Adopt the pass-A "
              "headline rung first — deriving the margin at a rung chosen after "
              "seeing the profile would be exactly the circularity this script "
              "exists to prevent.")

    for _, r in band.iterrows():
        print(f"[rtol]   k={r['tau_k']:<5g} designs [{r['min_design']:.3f}, "
              f"{r['max_design']:.3f}]  {r['verdict']}")
    if not assay.empty and not assay["separates"].any():
        print("[rtol] ASSAY SENSITIVITY FAILS at every tolerance: the metric cannot "
              "separate the unmatched reference design from the matched ones, so a "
              "null between the matched designs is uninformative rather than a "
              "finding. Report it as such.")
    return {"profile": profile, "band": band, "nulls": nulls, "assay": assay,
            "bootstrap": boots, "co_occurrence": coocc}


def main() -> None:
    a = run_pass_a()
    b = run_pass_b()
    print(f"[rtol] tables -> {sc.RTOL_TABLES_DIR}")
    del a, b


if __name__ == "__main__":
    main()
