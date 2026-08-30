"""regret_tolerance_passb_candidates.py - pass B scoring of round candidate tau vectors.

MEASUREMENT ONLY (adopts nothing, edits no config); companion to
``regret_tolerance_diagnostics.py`` and
``docs/notes/methods/regret_tolerance_diagnostics.md``. Everything reduces the
persisted campaign re-eval cubes (``REEVAL_TAG``); zero simulation:

  1. Pass B on both ladder shapes (eps-only and ``max(eps, floor)``).
  2. The per-objective magnitude of the incumbent-relative signed difference
     D_i over all policies x SOWs.
  3. A grid of round candidate tau vectors scored for an informative rung
     (neither saturated nor starved) and assay sensitivity (``historic``
     separates from the two matched designs), on the 8-axis conjunction and
     the ``compromise`` 3-axis subset.
  4. A paired near-tie floor: the spread of D_i among policies that are
     near-ties with the incumbent on that axis.

Run under an allocation with NYCOPT_REGRET_TAU unset (the whole-vector override
would pin every rung of the sweep to the single adopted vector).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import src.robustness as rob                                        # noqa: E402
import supplemental_config as sc                                    # noqa: E402
from src.etest import campaign_reeval_preset                        # noqa: E402

DESIGNS = ("historic", "monte_carlo", "hazard_filling_stationary")
MATCHED = ("monte_carlo", "hazard_filling_stationary")
CONTROL = "historic"
#: Re-eval tag of the cubes scored: the campaign preset unless overridden.
REEVAL_TAG = os.environ.get("NYCOPT_REEVAL_ENSEMBLE_PRESET") or campaign_reeval_preset()
FORMULATION_SLUG = "ffmp_obj8"

OUT = sc.RTOL_TABLES_DIR
BOOT_N = 2000
BOOT_SEED = 7

RELIABILITY = ("nyc_delivery_reliability_annual",
               "montague_flow_reliability_annual",
               "trenton_flow_reliability_annual",
               "nj_delivery_reliability_annual")
DEFICIT = ("nyc_delivery_deficit_p99_pct", "montague_flow_deficit_p99_pct")
FLOOD = ("downstream_flood_exceedance_annual",)
STORAGE = ("nyc_storage_min_p01_pct",)
COMPROMISE_AXES = ("nyc_delivery_reliability_annual",
                   "trenton_flow_reliability_annual",
                   "downstream_flood_exceedance_annual")


###############################################################################
# Load
###############################################################################

class DesignCube:
    """One design's policy cube, its aligned incumbent, and D = advantage."""

    def __init__(self, design: str):
        p = (PROJECT_DIR / "outputs" / design / FORMULATION_SLUG / "reeval"
             / REEVAL_TAG)
        self.design = design
        self.raw = rob.load_raw(p)
        self.base = rob.load_raw(p / "baseline")
        self.D = rob.incumbent_advantage(self.raw, self.base)      # (S, G, M)
        self.finite = np.isfinite(self.D)
        self.names = list(self.raw.obj_names)
        self.n_sol, self.n_sow, _ = self.D.shape

    def harm_free(self, tau_vec: np.ndarray, keep=None) -> np.ndarray:
        """(S, G) bool: this policy harms NO kept objective beyond tau, here."""
        D, fin, t = self.D, self.finite, tau_vec
        if keep is not None:
            D, fin, t = D[:, :, keep], fin[:, :, keep], t[keep]
        return (fin & (D >= -t[None, None, :])).all(axis=2)

    def pi(self, tau_vec: np.ndarray, keep=None) -> np.ndarray:
        """(S,) Pi_tau per policy."""
        return self.harm_free(tau_vec, keep).mean(axis=1)


def tau_vec_of(names, tau: dict) -> np.ndarray:
    return np.array([float(tau[n]) for n in names], dtype=float)


###############################################################################
# Ladders
###############################################################################

def eps_vector(names) -> dict:
    from src.objectives_ensemble import ENSEMBLE_OBJECTIVES
    return {n: float(ENSEMBLE_OBJECTIVES[n].epsilon) for n in names}


def floor_vector() -> dict:
    return json.loads((OUT / "rtol_floors.json").read_text())


###############################################################################
# Reporting helpers
###############################################################################

def band_verdict(vals) -> str:
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if lo > sc.RTOL_SATURATION_HI:
        return "saturated"
    if hi < sc.RTOL_SATURATION_LO:
        return "starved"
    return "informative"


def paired_bootstrap(cubes: dict, tau: dict, a: str, b: str, keep=None,
                     stat: str = "best", n_boot: int = BOOT_N) -> dict:
    """SOW-level paired bootstrap SE of Pi_tau(a) - Pi_tau(b).

    Both designs are scored on the same SOWs, so the SOW resample is shared
    across designs - that is what makes the difference paired and its SE
    smaller than differencing two independent margins.
    """
    rng = np.random.default_rng(BOOT_SEED)
    ha = cubes[a].harm_free(tau_vec_of(cubes[a].names, tau), keep)
    hb = cubes[b].harm_free(tau_vec_of(cubes[b].names, tau), keep)
    n_sow = ha.shape[1]
    red = (lambda v: float(np.max(v))) if stat == "best" else \
          (lambda v: float(np.median(v)))

    def ep(h, cols):
        return red(h[:, cols].mean(axis=1))

    allc = np.arange(n_sow)
    diff = ep(ha, allc) - ep(hb, allc)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        cols = rng.integers(0, n_sow, n_sow)
        draws[i] = ep(ha, cols) - ep(hb, cols)
    return {"design_a": a, "design_b": b, "stat": stat, "diff": diff,
            "se": float(np.std(draws, ddof=1)),
            "ci_lo": float(np.quantile(draws, 0.025)),
            "ci_hi": float(np.quantile(draws, 0.975)),
            "n_sow": int(n_sow), "n_boot": int(n_boot)}


def score_vector(cubes: dict, tau: dict, keep=None) -> dict:
    """Per-design best/median Pi_tau plus the band and assay verdicts."""
    row = {}
    for d in DESIGNS:
        v = cubes[d].pi(tau_vec_of(cubes[d].names, tau), keep)
        row[f"best__{d}"] = float(np.max(v))
        row[f"median__{d}"] = float(np.median(v))
    for stat in ("best", "median"):
        vals = [row[f"{stat}__{d}"] for d in DESIGNS]
        row[f"verdict_{stat}"] = band_verdict(vals)
        row[f"spread_{stat}"] = float(max(vals) - min(vals))
        for d in MATCHED:
            row[f"assay_gap_{stat}__{d}"] = (row[f"{stat}__{d}"]
                                             - row[f"{stat}__{CONTROL}"])
    return row


###############################################################################
# Main
###############################################################################

def main() -> None:                                        # noqa: C901
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 600)
    pd.set_option("display.max_columns", 60)
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("LOAD")
    print("=" * 78)
    cubes = {d: DesignCube(d) for d in DESIGNS}
    names = cubes[DESIGNS[0]].names
    for d in DESIGNS:
        c = cubes[d]
        print(f"  {d:28s} S={c.n_sol:5d}  G={c.n_sow}  "
              f"R/SOW={c.raw.realizations_per_sow}  "
              f"baseline G={c.base.n_sow}  NaN={np.mean(~c.finite):.3g}")
    keep_comp = [i for i, n in enumerate(names) if n in COMPROMISE_AXES]

    eps = eps_vector(names)
    flo = floor_vector()
    umax = {n: max(eps[n], flo[n]) for n in names}
    print("\n  units:")
    for n in names:
        print(f"    {n:38s} eps={eps[n]:<8g} floor={flo[n]:<12.4g} "
              f"max={umax[n]:<12.4g}")

    ###########################################################################
    # 1. Pass B - both ladder shapes
    ###########################################################################
    print("\n" + "=" * 78)
    print("1. PASS B - k-curve, discrimination band, assay")
    print("=" * 78)
    prof_rows = []
    for shape, unit in (("eps", eps), ("max", umax)):
        for k in sc.RTOL_TAU_GRID:
            tau = {n: k * unit[n] for n in names}
            r = {"shape": shape, "tau_k": float(k)}
            r.update(score_vector(cubes, tau))
            r_c = score_vector(cubes, tau, keep_comp)
            for key, v in r_c.items():
                r[f"comp_{key}"] = v
            prof_rows.append(r)
    prof = pd.DataFrame(prof_rows)
    prof.to_csv(OUT / "rtolB_kcurve_both_shapes.csv", index=False)

    for shape in ("eps", "max"):
        g = prof[prof["shape"] == shape]
        print(f"\n  --- ladder shape = {shape} "
              f"(tau_i = k * {'eps_i' if shape == 'eps' else 'max(eps_i, floor_i)'}) ---")
        print("   k     hist_best  fixp_best  hazf_best  verdict(best)   "
              "hist_med  fixp_med  hazf_med  verdict(med)")
        for _, r in g.iterrows():
            print(f"  {r['tau_k']:<5g} "
                  f"{r['best__historic']:9.3f}  "
                  f"{r['best__monte_carlo']:9.3f}  "
                  f"{r['best__hazard_filling_stationary']:9.3f}  "
                  f"{r['verdict_best']:<14s}  "
                  f"{r['median__historic']:8.3f}  "
                  f"{r['median__monte_carlo']:8.3f}  "
                  f"{r['median__hazard_filling_stationary']:8.3f}  "
                  f"{r['verdict_median']}")

    ###########################################################################
    # 2. Empirical nulls - what the replication scheme can and cannot supply
    ###########################################################################
    print("\n" + "=" * 78)
    print("2. EMPIRICAL NULLS (note section 4.2)")
    print("=" * 78)
    reps = {d: {"draws": {0}, "seeds": {None}} for d in DESIGNS}
    print("  replication actually on disk:")
    for d in DESIGNS:
        print(f"    {d:28s} draws={sorted(reps[d]['draws'])} "
              f"seeds={sorted(str(s) for s in reps[d]['seeds'])} -> "
              f"0 seed pairs, 0 draw pairs")
    print("  SEED-LEVEL null : NOT ESTIMABLE (1 seed per draw).")
    print("  DRAW-LEVEL null : NOT ESTIMABLE (1 draw per design).")
    print("  => delta's second term (draw spread) is undefined; only the")
    print("     2 x paired-bootstrap-SE term can be computed. Reported as a")
    print("     LOWER BOUND on delta, never as delta.")

    ###########################################################################
    # 3. Paired bootstrap at the adopted rung, and at the eps rung
    ###########################################################################
    print("\n" + "=" * 78)
    print("3. PAIRED BOOTSTRAP SE (note section 4.3) + section-5 margin term")
    print("=" * 78)
    boots = []
    for label, unit in (("adopted max-shape k=1", umax), ("eps-shape k=1", eps)):
        tau = {n: 1.0 * unit[n] for n in names}
        for a, b in itertools.combinations(DESIGNS, 2):
            for stat in ("best", "median"):
                r = paired_bootstrap(cubes, tau, a, b, stat=stat)
                r["ladder"] = label
                r["axes"] = "all8"
                boots.append(r)
                rc = paired_bootstrap(cubes, tau, a, b, keep_comp, stat=stat)
                rc["ladder"] = label
                rc["axes"] = "compromise3"
                boots.append(rc)
    bt = pd.DataFrame(boots)
    bt.to_csv(OUT / "rtolB_paired_bootstrap_candidates.csv", index=False)
    print(bt[["ladder", "axes", "stat", "design_a", "design_b", "diff", "se",
              "ci_lo", "ci_hi"]].to_string(index=False))
    for label in bt["ladder"].unique():
        for ax in ("all8", "compromise3"):
            for stat in ("best", "median"):
                s = bt[(bt["ladder"] == label) & (bt["axes"] == ax)
                       & (bt["stat"] == stat)]["se"].max()
                print(f"  delta LOWER BOUND ({label}, {ax}, {stat}) = "
                      f"2 x max SE = {2 * s:.4f}   [draw term NOT ESTIMABLE]")

    ###########################################################################
    # 4. Co-occurrence (note section 4.5)
    ###########################################################################
    print("\n" + "=" * 78)
    print("4. CO-OCCURRENCE: one binding objective, or accumulation?")
    print("=" * 78)
    co_rows = []
    for label, unit in (("adopted max-shape k=1", umax), ("eps-shape k=1", eps)):
        tau = tau_vec_of(names, {n: unit[n] for n in names})
        for d in DESIGNS:
            c = cubes[d]
            harm_tau = (~c.finite) | (c.D < -tau[None, None, :])
            phi = harm_tau.mean(axis=1)                     # (S, M)
            obs = (~harm_tau).all(axis=2).mean(axis=1)      # (S,)
            indep = np.prod(1.0 - phi, axis=1)
            for i, n in enumerate(names):
                co_rows.append({"ladder": label, "design": d, "objective": n,
                                "mean_harm_freq_tau": float(phi[:, i].mean()),
                                "frac_policies_ever_harming":
                                    float((phi[:, i] > 0).mean())})
            co_rows.append({"ladder": label, "design": d, "objective": "__JOINT__",
                            "mean_harm_freq_tau": np.nan,
                            "frac_policies_ever_harming": np.nan,
                            "mean_no_harm_observed": float(obs.mean()),
                            "mean_no_harm_if_independent": float(indep.mean()),
                            "mean_co_occurrence_gap": float((obs - indep).mean())})
    co = pd.DataFrame(co_rows)
    co.to_csv(OUT / "rtolB_co_occurrence_candidates.csv", index=False)
    for label in co["ladder"].unique():
        print(f"\n  --- {label} ---")
        g = co[(co["ladder"] == label) & (co["objective"] != "__JOINT__")]
        print(g.pivot(index="objective", columns="design",
                      values="mean_harm_freq_tau").to_string())
        j = co[(co["ladder"] == label) & (co["objective"] == "__JOINT__")]
        print(j[["design", "mean_no_harm_observed",
                 "mean_no_harm_if_independent",
                 "mean_co_occurrence_gap"]].to_string(index=False))

    ###########################################################################
    # 5. Magnitudes actually in play
    ###########################################################################
    print("\n" + "=" * 78)
    print("5. MAGNITUDES: distribution of D_i over all policies x SOWs")
    print("=" * 78)
    mag_rows = []
    for d in DESIGNS:
        c = cubes[d]
        for i, n in enumerate(names):
            v = c.D[:, :, i].ravel()
            v = v[np.isfinite(v)]
            neg = v[v < 0]
            row = {"design": d, "objective": n, "n_cells": int(v.size),
                   "frac_negative": float(neg.size / v.size),
                   "frac_exact_zero": float(np.mean(v == 0.0)),
                   "median_D": float(np.median(v)),
                   "q25_D": float(np.quantile(v, 0.25)),
                   "q75_D": float(np.quantile(v, 0.75))}
            for q in (0.01, 0.05, 0.10, 0.25):
                row[f"negtail_p{int(q * 100):02d}"] = (
                    float(np.quantile(neg, q)) if neg.size else np.nan)
            row["median_negative_D"] = (float(np.median(neg))
                                        if neg.size else np.nan)
            row["worst_D"] = float(v.min())
            mag_rows.append(row)
    mag = pd.DataFrame(mag_rows)
    mag.to_csv(OUT / "rtolB_D_magnitudes.csv", index=False)
    print("\n  (a) tail of the NEGATIVE part of D (real degradations), "
          "percentiles OF THE NEGATIVE SUBSET:")
    print(mag[["design", "objective", "frac_negative", "frac_exact_zero",
               "negtail_p01", "negtail_p05", "negtail_p10", "negtail_p25",
               "median_negative_D", "worst_D"]].to_string(index=False))
    print("\n  (b) full D distribution:")
    print(mag[["design", "objective", "q25_D", "median_D", "q75_D"]]
          .to_string(index=False))

    # Pooled over designs
    print("\n  (c) POOLED over all three designs:")
    pooled_rows = []
    for i, n in enumerate(names):
        v = np.concatenate([cubes[d].D[:, :, i].ravel() for d in DESIGNS])
        v = v[np.isfinite(v)]
        neg = v[v < 0]
        pooled_rows.append({
            "objective": n, "n_cells": int(v.size),
            "frac_negative": float(neg.size / v.size),
            "median_D": float(np.median(v)),
            "negtail_p01": float(np.quantile(neg, 0.01)) if neg.size else np.nan,
            "negtail_p05": float(np.quantile(neg, 0.05)) if neg.size else np.nan,
            "negtail_p10": float(np.quantile(neg, 0.10)) if neg.size else np.nan,
            "negtail_p25": float(np.quantile(neg, 0.25)) if neg.size else np.nan,
            "median_negative_D": float(np.median(neg)) if neg.size else np.nan,
            "worst_D": float(v.min())})
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUT / "rtolB_D_magnitudes_pooled.csv", index=False)
    print(pooled.to_string(index=False))

    # Incumbent's own per-SOW scale
    print("\n  (d) INCUMBENT per-SOW quantiles (its own scale):")
    b = cubes[DESIGNS[0]].base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        flat_full = np.nanmean(b.cube, axis=0)                    # (1000, M)
    lab = {int(s): j for j, s in enumerate(b.sow_labels)}
    idx200 = [lab[int(s)] for s in cubes[DESIGNS[0]].raw.sow_labels]
    inc_rows = []
    for i, n in enumerate(names):
        for tag, arr in (("E_test_1000", flat_full[:, i]),
                         ("subset_200", flat_full[idx200, i])):
            q = np.nanquantile(arr, [0.10, 0.50, 0.90])
            inc_rows.append({"objective": n, "sows": tag,
                             "q10": float(q[0]), "q50": float(q[1]),
                             "q90": float(q[2]),
                             "q90_minus_q10": float(q[2] - q[0])})
    inc = pd.DataFrame(inc_rows)
    inc.to_csv(OUT / "rtolB_incumbent_scale.csv", index=False)
    print(inc.to_string(index=False))

    # Measurement granularity of each axis (distinct-value spacing)
    print("\n  (e) MEASUREMENT GRANULARITY (smallest non-zero |D| observed):")
    for i, n in enumerate(names):
        v = np.concatenate([cubes[d].D[:, :, i].ravel() for d in DESIGNS])
        v = np.abs(v[np.isfinite(v)])
        nz = v[v > 0]
        print(f"    {n:38s} min|D|>0 = {nz.min():.6g}   "
              f"(frac exact ties = {float(np.mean(v == 0)):.4f})")

    ###########################################################################
    # 6. Paired sanity check on the pass-A floor
    ###########################################################################
    print("\n" + "=" * 78)
    print("6. PAIRED SANITY CHECK ON THE PASS-A UNPAIRED FLOOR")
    print("=" * 78)
    print("  For each objective: take the policies whose MEAN D_i over SOWs is")
    print("  closest to 0 on that axis (near-ties with the incumbent), and take")
    print("  the SD across SOWs of their per-SOW D_i. That SD is the PAIRED")
    print("  null spread - it already nets out the shared inflow sequence -")
    print("  and is still conservative (it retains real policy x SOW response).")
    z = sc.RTOL_FALSE_HARM_Z
    ft_rows = []
    for i, n in enumerate(names):
        Dall = np.concatenate([cubes[d].D[:, :, i] for d in DESIGNS], axis=0)
        mean_d = np.nanmean(Dall, axis=1)
        rank = np.argsort(np.abs(mean_d))
        for frac, tag in ((0.02, "nearest 2%"), (0.05, "nearest 5%"),
                          (0.10, "nearest 10%")):
            sel = rank[:max(3, int(frac * rank.size))]
            sds = np.nanstd(Dall[sel, :], axis=1, ddof=1)
            sd = float(np.median(sds))
            ft_rows.append({"objective": n, "near_tie_set": tag,
                            "n_policies": int(sel.size),
                            "median_abs_mean_D": float(np.median(np.abs(mean_d[sel]))),
                            "paired_sd_median": sd,
                            "paired_floor_z": z * sd,
                            "unpaired_floor_passA": flo[n],
                            "ratio_unpaired_over_paired":
                                float(flo[n] / (z * sd)) if sd > 0 else np.nan})
    ft = pd.DataFrame(ft_rows)
    ft.to_csv(OUT / "rtolB_paired_floor_check.csv", index=False)
    print(ft.to_string(index=False))

    ###########################################################################
    # 7. ROUND candidate tolerance vectors
    ###########################################################################
    print("\n" + "=" * 78)
    print("7. ROUND CANDIDATE TOLERANCE VECTORS")
    print("=" * 78)
    rel_grid = (0.005, 0.01, 0.02, 0.05, 0.10)
    def_grid = (0.5, 1.0, 2.0, 5.0, 10.0)
    fld_grid = (0.05, 0.1, 0.25, 0.5, 1.0)
    sto_grid = (0.5, 1.0, 2.0, 5.0)

    #: One family swept, the others held at the middle round value.
    HOLD = {"rel": 0.02, "def": 2.0, "fld": 0.25, "sto": 2.0}

    def build(rel, dfc, fld, sto) -> dict:
        t = {}
        for n in names:
            if n in RELIABILITY:
                t[n] = rel
            elif n in DEFICIT:
                t[n] = dfc
            elif n in FLOOD:
                t[n] = fld
            else:
                t[n] = sto
        return t

    print("\n  (a) ONE FAMILY AT A TIME (others held at "
          f"rel={HOLD['rel']}, def={HOLD['def']}, fld={HOLD['fld']}, "
          f"sto={HOLD['sto']}):")
    fam_rows = []
    for fam, grid in (("reliability", rel_grid), ("deficit_p99_pct", def_grid),
                      ("flood_ftd", fld_grid), ("storage_p1_pct", sto_grid)):
        for v in grid:
            kw = dict(rel=HOLD["rel"], dfc=HOLD["def"], fld=HOLD["fld"],
                      sto=HOLD["sto"])
            kw[{"reliability": "rel", "deficit_p99_pct": "dfc",
                "flood_ftd": "fld", "storage_p1_pct": "sto"}[fam]] = v
            tau = build(**kw)
            r = {"family": fam, "value": v}
            r.update(score_vector(cubes, tau))
            rc = score_vector(cubes, tau, keep_comp)
            for kk, vv in rc.items():
                r[f"comp_{kk}"] = vv
            fam_rows.append(r)
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(OUT / "rtolB_candidate_families.csv", index=False)
    cols = ["family", "value", "best__historic", "best__monte_carlo",
            "best__hazard_filling_stationary", "verdict_best", "spread_best",
            "median__historic", "median__monte_carlo",
            "median__hazard_filling_stationary", "verdict_median",
            "spread_median"]
    print(fam[cols].to_string(index=False))
    print("\n  (a2) same, restricted to the COMPROMISE 3 axes:")
    ccols = ["family", "value", "comp_best__historic",
             "comp_best__monte_carlo",
             "comp_best__hazard_filling_stationary", "comp_verdict_best",
             "comp_median__historic", "comp_median__monte_carlo",
             "comp_median__hazard_filling_stationary", "comp_verdict_median"]
    print(fam[ccols].to_string(index=False))

    print("\n  (b) FULL ROUND GRID (all family combinations):")
    grid_rows = []
    for rel, dfc, fld, sto in itertools.product(
            (0.01, 0.02, 0.05, 0.10), (1.0, 2.0, 5.0, 10.0),
            (0.1, 0.25, 0.5, 1.0), (1.0, 2.0, 5.0)):
        tau = build(rel, dfc, fld, sto)
        r = {"rel": rel, "def": dfc, "fld": fld, "sto": sto}
        r.update(score_vector(cubes, tau))
        rc = score_vector(cubes, tau, keep_comp)
        for kk, vv in rc.items():
            r[f"comp_{kk}"] = vv
        grid_rows.append(r)
    gr = pd.DataFrame(grid_rows)
    gr.to_csv(OUT / "rtolB_candidate_grid.csv", index=False)

    info = gr[(gr["verdict_best"] == "informative")
              & (gr["verdict_median"] == "informative")]
    print(f"    {len(gr)} vectors; informative on BOTH best and median: "
          f"{len(info)}")
    show = ["rel", "def", "fld", "sto", "best__historic",
            "best__monte_carlo", "best__hazard_filling_stationary",
            "spread_best", "median__historic",
            "median__monte_carlo",
            "median__hazard_filling_stationary", "spread_median"]
    if len(info):
        print(info.sort_values("spread_median", ascending=False)[show]
              .head(40).to_string(index=False))
    print("\n    top-20 by |assay gap| on the MEDIAN policy "
          "(hazfill minus historic), informative-on-median only:")
    im = gr[gr["verdict_median"] == "informative"].copy()
    im["assay_min_gap_median"] = im[[
        f"assay_gap_median__{d}" for d in MATCHED]].min(axis=1)
    print(im.sort_values("assay_min_gap_median", ascending=False)[
        show + ["assay_gap_median__monte_carlo",
                "assay_gap_median__hazard_filling_stationary"]]
        .head(20).to_string(index=False))

    print("\n  (c) COMPROMISE-3 SUBSET GRID (rel x flood only):")
    comp_rows = []
    for rel, fld in itertools.product((0.005, 0.01, 0.02, 0.05, 0.10),
                                      (0.05, 0.1, 0.25, 0.5, 1.0)):
        tau = build(rel, 2.0, fld, 2.0)
        r = {"rel": rel, "fld": fld}
        rc = score_vector(cubes, tau, keep_comp)
        r.update(rc)
        comp_rows.append(r)
    cg = pd.DataFrame(comp_rows)
    cg.to_csv(OUT / "rtolB_candidate_grid_compromise.csv", index=False)
    print(cg[["rel", "fld", "best__historic", "best__monte_carlo",
              "best__hazard_filling_stationary", "verdict_best",
              "median__historic", "median__monte_carlo",
              "median__hazard_filling_stationary", "verdict_median",
              "spread_median"]].to_string(index=False))

    ###########################################################################
    # 8. Paired bootstrap for the shortlisted round vectors
    ###########################################################################
    print("\n" + "=" * 78)
    print("8. PAIRED BOOTSTRAP FOR SHORTLISTED ROUND VECTORS")
    print("=" * 78)
    shortlist = {
        "round_002": build(0.02, 2.0, 0.25, 2.0),
        "round_001": build(0.01, 1.0, 0.1, 1.0),
        "round_005": build(0.05, 5.0, 0.5, 5.0),
        "round_010": build(0.10, 10.0, 1.0, 5.0),
        "round_0005": build(0.005, 0.5, 0.05, 0.5),
    }
    srows = []
    for label, tau in shortlist.items():
        for a, b in itertools.combinations(DESIGNS, 2):
            for stat in ("best", "median"):
                for ax, keep in (("all8", None), ("compromise3", keep_comp)):
                    r = paired_bootstrap(cubes, tau, a, b, keep, stat=stat,
                                         n_boot=1000)
                    r["candidate"] = label
                    r["axes"] = ax
                    srows.append(r)
    sb = pd.DataFrame(srows)
    sb.to_csv(OUT / "rtolB_shortlist_bootstrap.csv", index=False)
    print(sb[["candidate", "axes", "stat", "design_a", "design_b", "diff",
              "se", "ci_lo", "ci_hi"]].to_string(index=False))

    print("\n[rtolB] tables ->", OUT)


if __name__ == "__main__":
    main()
