"""regret_tolerance_passb_rank.py - rank the ROUND candidate tau vectors.

Companion to ``regret_tolerance_passb_candidates.py``. MEASUREMENT ONLY.

Scoring rule: on the all-8 conjunction (a one-binding-objective metric) the
informative endpoint is the BEST policy; on the ``compromise`` 3-axis subset
it is the MEDIAN policy. Each candidate is scored on that endpoint, the whole
``Pi_tau`` distribution is reported alongside, and candidates are ranked on
assay separation from ``historic``.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts" / "supplemental"))

import supplemental_config as sc                                    # noqa: E402
from regret_tolerance_passb_candidates import (                     # noqa: E402
    COMPROMISE_AXES, CONTROL, DESIGNS, DesignCube, MATCHED, DEFICIT,
    FLOOD, RELIABILITY, eps_vector, floor_vector, paired_bootstrap,
    tau_vec_of)

OUT = sc.RTOL_TABLES_DIR
QS = (0.10, 0.25, 0.50, 0.75, 0.90)


def build(names, rel, dfc, fld, sto) -> dict:
    return {n: (rel if n in RELIABILITY else
                dfc if n in DEFICIT else
                fld if n in FLOOD else sto) for n in names}


def profile_row(cubes, names, tau, keep=None) -> dict:
    """Whole Pi_tau distribution per design, not two order statistics."""
    r = {}
    for d in DESIGNS:
        v = cubes[d].pi(tau_vec_of(names, tau), keep)
        r[f"best__{d}"] = float(v.max())
        r[f"frac_at_1__{d}"] = float((v >= 1.0).mean())
        for q in QS:
            r[f"q{int(q * 100):02d}__{d}"] = float(np.quantile(v, q))
        r[f"mean__{d}"] = float(v.mean())
    for stat in ("best", "q50", "mean", "q90"):
        vals = [r[f"{stat}__{d}"] for d in DESIGNS]
        r[f"spread_{stat}"] = float(max(vals) - min(vals))
        r[f"verdict_{stat}"] = ("saturated" if min(vals) > sc.RTOL_SATURATION_HI
                                else "starved" if max(vals) < sc.RTOL_SATURATION_LO
                                else "informative")
        for d in MATCHED:
            r[f"assay_{stat}__{d}"] = r[f"{stat}__{d}"] - r[f"{stat}__{CONTROL}"]
        r[f"assay_min_{stat}"] = min(r[f"assay_{stat}__{d}"] for d in MATCHED)
    return r


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 800)
    pd.set_option("display.max_columns", 80)

    cubes = {d: DesignCube(d) for d in DESIGNS}
    names = cubes[DESIGNS[0]].names
    keep_comp = [i for i, n in enumerate(names) if n in COMPROMISE_AXES]
    eps = eps_vector(names)
    flo = floor_vector()
    umax = {n: max(eps[n], flo[n]) for n in names}

    ###########################################################################
    print("=" * 78)
    print("A. REFERENCE RUNGS, full Pi_tau distribution")
    print("=" * 78)
    ref = {
        "tau = 0 (strict weak-Pareto)": {n: 0.0 for n in names},
        "eps-shape k=1 (code default)": dict(eps),
        "ADOPTED max-shape k=1": dict(umax),
        "ADOPTED / 2": {n: v / 2 for n, v in umax.items()},
        "round 0.02 / 2 / 0.1 / 2": build(names, 0.02, 2.0, 0.1, 2.0),
        "round 0.02 / 2 / 0.25 / 2": build(names, 0.02, 2.0, 0.25, 2.0),
    }
    rows = []
    for label, tau in ref.items():
        for ax, keep in (("all8", None), ("compromise3", keep_comp)):
            r = {"vector": label, "axes": ax}
            r.update(profile_row(cubes, names, tau, keep))
            rows.append(r)
    rf = pd.DataFrame(rows)
    rf.to_csv(OUT / "rtolB_reference_rungs.csv", index=False)
    for ax in ("all8", "compromise3"):
        print(f"\n  --- {ax} ---")
        g = rf[rf["axes"] == ax]
        cols = (["vector"]
                + [f"best__{d}" for d in DESIGNS]
                + [f"q50__{d}" for d in DESIGNS]
                + ["verdict_best", "verdict_q50", "assay_min_best",
                   "assay_min_q50"])
        print(g[cols].to_string(index=False))
        print("   frac of policies at Pi_tau == 1:")
        print(g[["vector"] + [f"frac_at_1__{d}" for d in DESIGNS]]
              .to_string(index=False))

    ###########################################################################
    print("\n" + "=" * 78)
    print("B. FULL ROUND GRID ranked on the NON-DEGENERATE endpoint")
    print("=" * 78)
    grid = []
    for rel, dfc, fld, sto in itertools.product(
            (0.005, 0.01, 0.02, 0.05, 0.10), (0.5, 1.0, 2.0, 5.0, 10.0),
            (0.05, 0.1, 0.25, 0.5, 1.0), (1.0, 2.0, 5.0)):
        tau = build(names, rel, dfc, fld, sto)
        r8 = profile_row(cubes, names, tau)
        rc = profile_row(cubes, names, tau, keep_comp)
        row = {"rel": rel, "def": dfc, "fld": fld, "sto": sto}
        row.update({f"a8_{k}": v for k, v in r8.items()})
        row.update({f"c3_{k}": v for k, v in rc.items()})
        grid.append(row)
    g = pd.DataFrame(grid)
    g.to_csv(OUT / "rtolB_candidate_grid_ranked.csv", index=False)
    print(f"  {len(g)} candidate vectors.")

    a8cols = ["rel", "def", "fld", "sto", "a8_best__historic",
              "a8_best__fixed_probabilistic",
              "a8_best__hazard_filling_stationary", "a8_verdict_best",
              "a8_spread_best", "a8_assay_min_best"]
    ok8 = g[g["a8_verdict_best"] == "informative"]
    print(f"\n  (i) ALL-8, BEST endpoint: informative on {len(ok8)}/{len(g)} "
          f"vectors. Top 25 by min assay gap vs historic:")
    print(ok8.sort_values("a8_assay_min_best", ascending=False)[a8cols]
          .head(25).to_string(index=False))

    c3cols = ["rel", "def", "fld", "sto", "c3_q50__historic",
              "c3_q50__fixed_probabilistic",
              "c3_q50__hazard_filling_stationary", "c3_verdict_q50",
              "c3_spread_q50", "c3_assay_min_q50", "c3_verdict_best"]
    okc = g[(g["c3_verdict_q50"] == "informative")].drop_duplicates(
        subset=["rel", "fld"])
    print(f"\n  (ii) COMPROMISE-3, MEDIAN endpoint (only rel & fld matter; "
          f"deduplicated). Sorted by min assay gap:")
    print(okc.sort_values("c3_assay_min_q50", ascending=False)[c3cols]
          .to_string(index=False))

    print("\n  (iii) Vectors informative on the all-8 BEST endpoint AND on the "
          "compromise-3 MEDIAN endpoint, ranked by the smaller of the two "
          "assay gaps:")
    both = g[(g["a8_verdict_best"] == "informative")
             & (g["c3_verdict_q50"] == "informative")].copy()
    both["assay_worst"] = both[["a8_assay_min_best",
                                "c3_assay_min_q50"]].min(axis=1)
    print(f"    {len(both)} qualify.")
    print(both.sort_values("assay_worst", ascending=False)[
        ["rel", "def", "fld", "sto", "a8_best__historic",
         "a8_best__fixed_probabilistic",
         "a8_best__hazard_filling_stationary", "a8_assay_min_best",
         "c3_q50__historic", "c3_q50__fixed_probabilistic",
         "c3_q50__hazard_filling_stationary", "c3_assay_min_q50",
         "assay_worst"]].head(30).to_string(index=False))

    ###########################################################################
    print("\n" + "=" * 78)
    print("C. SENSITIVITY OF THE VERDICT TO EACH FAMILY, around the "
          "recommendation")
    print("=" * 78)
    base = dict(rel=0.02, dfc=2.0, fld=0.1, sto=2.0)
    for fam, key, grid_v in (("reliability", "rel", (0.005, 0.01, 0.02, 0.05, 0.10)),
                             ("deficit_p99", "dfc", (0.5, 1.0, 2.0, 5.0, 10.0)),
                             ("flood", "fld", (0.05, 0.1, 0.25, 0.5, 1.0)),
                             ("storage", "sto", (1.0, 2.0, 5.0))):
        print(f"\n  --- {fam} (others at rel={base['rel']}, def={base['dfc']}, "
              f"fld={base['fld']}, sto={base['sto']}) ---")
        sub = g.copy()
        for k2, v2 in base.items():
            col = {"rel": "rel", "dfc": "def", "fld": "fld", "sto": "sto"}[k2]
            if k2 != key:
                sub = sub[sub[col] == v2]
        col = {"rel": "rel", "dfc": "def", "fld": "fld", "sto": "sto"}[key]
        sub = sub[sub[col].isin(grid_v)].sort_values(col)
        print(sub[[col, "a8_best__historic", "a8_best__fixed_probabilistic",
                   "a8_best__hazard_filling_stationary", "a8_verdict_best",
                   "a8_assay_min_best", "c3_q50__historic",
                   "c3_q50__fixed_probabilistic",
                   "c3_q50__hazard_filling_stationary", "c3_verdict_q50",
                   "c3_assay_min_q50"]].to_string(index=False))

    ###########################################################################
    print("\n" + "=" * 78)
    print("D. PAIRED BOOTSTRAP AT THE RECOMMENDED VECTOR(S)")
    print("=" * 78)
    finals = {
        "REC round 0.02/2/0.1/2": build(names, 0.02, 2.0, 0.1, 2.0),
        "alt round 0.02/2/0.25/2": build(names, 0.02, 2.0, 0.25, 2.0),
        "alt round 0.05/5/0.25/5": build(names, 0.05, 5.0, 0.25, 5.0),
        "ADOPTED max-shape k=1": dict(umax),
    }
    brows = []
    for label, tau in finals.items():
        for a, b in itertools.combinations(DESIGNS, 2):
            for ax, keep, stat in (("all8", None, "best"),
                                   ("compromise3", keep_comp, "median")):
                r = paired_bootstrap(cubes, tau, a, b, keep, stat=stat,
                                     n_boot=1000)
                r["vector"] = label
                r["axes"] = ax
                brows.append(r)
    bt = pd.DataFrame(brows)
    bt.to_csv(OUT / "rtolB_recommended_bootstrap.csv", index=False)
    print(bt[["vector", "axes", "stat", "design_a", "design_b", "diff", "se",
              "ci_lo", "ci_hi"]].to_string(index=False))
    print("\n  delta (2 x max paired SE; DRAW TERM NOT ESTIMABLE at K=1):")
    for label in finals:
        for ax in ("all8", "compromise3"):
            s = bt[(bt["vector"] == label) & (bt["axes"] == ax)]["se"].max()
            print(f"    {label:26s} {ax:12s} delta_lower = {2 * s:.4f}")

    ###########################################################################
    print("\n" + "=" * 78)
    print("E. TOLERANCE MAGNITUDE IN CONTEXT")
    print("=" * 78)
    print("  tau as a fraction of the incumbent's own per-SOW q10-q90 spread")
    print("  (E_test 1000-SOW incumbent cube), and as a multiple of the")
    print("  paired near-tie floor from part 6 of the companion script.")
    spread = {"nyc_delivery_reliability_annual": 0.3225306,
              "nyc_delivery_deficit_p99_pct": 0.7538743,
              "montague_flow_reliability_annual": 0.3072653,
              "montague_flow_deficit_p99_pct": 10.36104,
              "trenton_flow_reliability_annual": 0.3216327,
              "downstream_flood_exceedance_annual": 2.524967,
              "nyc_storage_min_p01_pct": 22.78376,
              "nj_delivery_reliability_annual": 0.3502857}
    paired = {"nyc_delivery_reliability_annual": 0.019722,
              "nyc_delivery_deficit_p99_pct": 18.424124,
              "montague_flow_reliability_annual": 0.017605,
              "montague_flow_deficit_p99_pct": 4.848150,
              "trenton_flow_reliability_annual": 0.017092,
              "downstream_flood_exceedance_annual": 0.042579,
              "nyc_storage_min_p01_pct": 3.020688,
              "nj_delivery_reliability_annual": 0.024032}
    ctx = []
    for label, tau in (("ADOPTED max-shape k=1", umax),
                       ("eps-shape k=1", eps),
                       ("REC round 0.02/2/0.1/2",
                        build(names, 0.02, 2.0, 0.1, 2.0))):
        for n in names:
            ctx.append({"vector": label, "objective": n, "tau": tau[n],
                        "frac_of_incumbent_q10q90": tau[n] / spread[n],
                        "multiple_of_paired_floor": tau[n] / paired[n],
                        "unpaired_passA_floor": flo[n],
                        "paired_nearty_floor": paired[n]})
    cf = pd.DataFrame(ctx)
    cf.to_csv(OUT / "rtolB_tau_in_context.csv", index=False)
    print(cf.to_string(index=False))
    print("\n[rtolB-rank] tables ->", OUT)


if __name__ == "__main__":
    main()
