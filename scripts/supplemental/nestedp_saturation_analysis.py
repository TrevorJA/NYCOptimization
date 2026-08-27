"""nestedp_saturation_analysis.py - Nested-P saturation: cross-rung tables, fits, figure.

Consumes the per-rung outputs of ``diagnose_hazard_selectors.py`` run in prefix mode
(``NYCOPT_SELDIAG_PREFIX_P``) against ONE staged stream-only candidate pool, and
records how the hazard-filling selector's tail enrichment — minimum per-axis tail
share above the pool P90 at N = 100, seed-mean convention (within-seed minimum,
averaged over selector seeds; ``hazard_selector_diagnostics.md`` §4 block D) —
saturates with pool size P on the campaign and full retained axis sets.

Each rung P' is scored from the first P' rows of the one staged image. Because
realizations are keyed to a global index with per-realization child streams, a prefix
is bit-identical to a standalone i.i.d. pool of size P' from the same seed domain, so
the ladder measures honest pools while generating only once.

Fits (both vs log P):

  * snap-concentration ratio (full m): log-log power law — the mechanism trend; the
    expected anchor-to-nearest-member distance scales as P^(-1/m_eff), so the fitted
    exponent is read against the P^(-1/8)..P^(-1/5) bracket from the intrinsic
    dimension (~5.1) of the 8-axis image.
  * min per-axis tail share (full m): saturating form  tail(P) = A - B * P^(-beta);
    the asymptote A, the exponent beta, and the P at which the fit comes within
    ``SATURATION_TOL`` of A state where enrichment saturates.

All configuration is via environment variables (no CLI value flags):

    NYCOPT_NESTEDP_POOL_SLUG  staged pool slug the prefixes were cut from (required)
    NYCOPT_NESTEDP_RUNGS      space-separated prefix sizes (required)
    NYCOPT_NESTEDP_GEN_SU     SU spent on pool generation, written verbatim (optional)

Outputs -> ``outputs/supplemental/hazard_selector_diagnostics/nested_P_saturation.md``
plus ``figures/nested_P_saturation.png`` in the same directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

import config  # noqa: E402
from src.plotting.style import apply_style, save_figure  # noqa: E402

POOL_SLUG = os.environ["NYCOPT_NESTEDP_POOL_SLUG"]
RUNGS = [int(p) for p in os.environ["NYCOPT_NESTEDP_RUNGS"].split()]
GEN_SU = os.environ.get("NYCOPT_NESTEDP_GEN_SU", "(not recorded)")

DIAG_DIR = config.OUTPUTS_DIR / "supplemental" / "hazard_selector_diagnostics"

#: Share of an i.i.d. selection above the pool P90 (the reference line).
TAIL_NULL_SHARE = 0.10

#: Tail-share distance from the fitted asymptote at which the fit is read as
#: saturated (the seed-level spread of the statistic is ~0.02).
SATURATION_TOL = 0.01

#: Theoretical scaling bracket for the improvement exponent: P^(-1/m) with m the
#: nominal (8) and intrinsic (~5.1 -> 1/5 used as the round bracket edge) dimension.
EXPONENT_BRACKET = (1.0 / 8.0, 1.0 / 5.0)

#: Axis sets scored across rungs: the campaign selection set and the full
#: retained set (the sets the live battery emits).
_MSETS = ("campaign", "full")
_MSET_COLORS = {"campaign": "#7d3f9b", "full": "#1f6fb4"}


def _rung_dir(p: int) -> Path:
    return DIAG_DIR / f"{POOL_SLUG}_prefix{p}"


def _load_rung(p: int) -> dict:
    """Extract one rung's tail-share metrics from its diagnostics outputs.

    Full-battery rungs (no ``saturation_mode`` in summary.json) carry the full-set
    per-axis table in ``per_axis_coverage.csv`` (no ``m_set`` column) and the
    campaign-set numbers in ``n_sweep.csv`` at N = 100; saturation rungs carry
    every axis set in ``per_axis_coverage.csv`` keyed by ``m_set``. Both are
    reduced with the same block-D convention (within-seed min, averaged over
    seeds).
    """
    out = _rung_dir(p)
    summary = json.loads((out / "summary.json").read_text())
    per_axis = pd.read_csv(out / "per_axis_coverage.csv")
    dim = pd.read_csv(out / "dimension_sweep.csv")
    saturation = bool(summary.get("saturation_mode", False))

    screen = summary["axis_screen"]
    rho = np.abs(np.asarray(screen["spearman_rho"], dtype=float))
    np.fill_diagonal(rho, 0.0)

    rung: dict = {
        "P": p,
        "saturation_mode": saturation,
        "n_retained": len(screen["retained"]),
        "retained": screen["retained"],
        "dropped": screen["dropped"],
        "max_abs_rho": float(rho.max()),
        "msets": {},
    }

    if saturation:
        for mset, adq in summary["adequacy"].items():
            rung["msets"][mset] = {
                "m": adq["m"],
                "tail_min": adq["tail_share_min"],
                "tail_mean": adq["tail_share_mean"],
                "worst_axis": adq["worst_axis"],
                "worst_axis_seed_mean": adq["worst_axis_seed_mean"],
                "per_axis": adq["per_axis_seed_mean"],
                "conc_ratio": adq["concentration_ratio"],
                "snap_mean": adq["snap_mean"],
            }
    else:
        nsw = pd.read_csv(out / "n_sweep.csv")
        n_sel = int(summary["n_select"])
        for mset in dict.fromkeys(dim["m_set"]):
            m = int(dim.loc[dim.m_set == mset, "m"].iloc[0])
            conc = float(dim.loc[dim.m_set == mset, "concentration_ratio"].mean())
            snap = float(dim.loc[dim.m_set == mset, "snap_mean"].mean())
            if mset == "full":
                sel = per_axis.loc[per_axis.selector == "lhs_nn"]
                by_seed = sel.groupby("seed")["tail_share_p90"]
                axis_means = sel.groupby("axis")["tail_share_p90"].mean()
                rung["msets"][mset] = {
                    "m": m,
                    "tail_min": float(by_seed.min().mean()),
                    "tail_mean": float(by_seed.mean().mean()),
                    "worst_axis": str(axis_means.idxmin()),
                    "worst_axis_seed_mean": float(axis_means.min()),
                    "per_axis": {str(a): float(v) for a, v in axis_means.items()},
                    "conc_ratio": conc,
                    "snap_mean": snap,
                }
            else:
                sel = nsw.loc[(nsw.m_set == mset) & (nsw.selector == "lhs_nn")
                              & (nsw.n == n_sel)]
                rung["msets"][mset] = {
                    "m": m,
                    "tail_min": float(sel["tail_share_min"].mean()),
                    "tail_mean": float(sel["tail_share_mean"].mean()),
                    "worst_axis": None,
                    "worst_axis_seed_mean": None,
                    "per_axis": None,
                    "conc_ratio": conc,
                    "snap_mean": snap,
                }
    return rung


def _fit_powerlaw(P: np.ndarray, y: np.ndarray) -> dict:
    """OLS of log y on log P: y ~ c * P^(-beta). Returns beta and the fit."""
    slope, intercept = np.polyfit(np.log(P), np.log(y), 1)
    return {"beta": float(-slope), "c": float(np.exp(intercept)),
            "yhat": np.exp(intercept + slope * np.log(P))}


def _fit_saturating(P: np.ndarray, y: np.ndarray) -> dict | None:
    """Fit tail(P) = A - B * P^(-beta); return params + the saturation size.

    ``P_saturated`` is the pool size at which the fit is within ``SATURATION_TOL``
    of its asymptote A, i.e. B * P^(-beta) = SATURATION_TOL.
    """
    def f(p, A, B, beta):
        return A - B * np.power(p, -beta)

    try:
        (A, B, beta), _ = curve_fit(
            f, P.astype(float), y, p0=(y.max(), 1.0, 0.15),
            bounds=([0.0, 0.0, 0.01], [1.0, 100.0, 1.0]), maxfev=20000,
        )
    except RuntimeError:
        return None
    p_sat = float((B / SATURATION_TOL) ** (1.0 / beta)) if B > 0 else float(P.min())
    return {"A": float(A), "B": float(B), "beta": float(beta),
            "P_saturated": p_sat, "yhat": f(P.astype(float), A, B, beta)}


def _figure(rungs: list[dict], fits: dict, out_stub: Path) -> None:
    """Tail share + snap concentration vs P: the saturation record in one look."""
    P = np.array([r["P"] for r in rungs], dtype=float)
    fig, (a, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    for mset in [m for m in _MSETS if m in rungs[0]["msets"]]:
        tail = [r["msets"][mset]["tail_min"] for r in rungs]
        m = rungs[0]["msets"][mset]["m"]
        a.plot(P, tail, "o-", color=_MSET_COLORS[mset], label=f"{mset} (m={m})")
    fit = fits.get("tail_full")
    if fit is not None:
        Pg = np.geomspace(P.min(), P.max(), 200)
        a.plot(Pg, fit["A"] - fit["B"] * Pg ** (-fit["beta"]), "--", lw=1,
               color=_MSET_COLORS["full"], alpha=0.7,
               label=f"fit: {fit['A']:.2f} − {fit['B']:.2f}·P^(−{fit['beta']:.3f})")
    a.axhline(TAIL_NULL_SHARE, color="0.5", lw=1, ls="--",
              label=f"i.i.d. reference ({TAIL_NULL_SHARE:g})")
    a.set_xscale("log")
    a.set_xlabel("candidate pool size P")
    a.set_ylabel("min per-axis tail share (seed mean)")
    a.set_ylim(bottom=0)
    a.set_title("Tail enrichment vs pool size (N = 100)")
    a.legend(fontsize=6.5)

    for mset in [m for m in _MSETS if m in rungs[0]["msets"]]:
        conc = [r["msets"][mset]["conc_ratio"] for r in rungs]
        m = rungs[0]["msets"][mset]["m"]
        a2.plot(P, conc, "o-", color=_MSET_COLORS[mset], label=f"{mset} (m={m})")
    fit = fits.get("conc_full")
    if fit is not None:
        Pg = np.geomspace(P.min(), P.max(), 200)
        a2.plot(Pg, fit["c"] * Pg ** (-fit["beta"]), "--", lw=1,
                color=_MSET_COLORS["full"], alpha=0.7,
                label=f"fit ∝ P^(−{fit['beta']:.3f}) "
                      f"(bracket {EXPONENT_BRACKET[0]:.3f}–{EXPONENT_BRACKET[1]:.3f})")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("candidate pool size P")
    a2.set_ylabel("snap / random-pair distance ratio")
    a2.set_title("Snap concentration vs pool size")
    a2.legend(fontsize=6.5)

    fig.suptitle(f"Nested-P saturation diagnostic ({POOL_SLUG}, prefix rungs)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out_stub)
    plt.close(fig)


def _markdown(rungs: list[dict], fits: dict) -> str:
    """Render the concise results file."""
    lines = [
        "# Nested-P saturation diagnostic",
        "",
        f"*Pool `{POOL_SLUG}` (stream-only, draw d0); every rung P' is the first P'",
        "rows of that one image — an honest i.i.d. pool of its size (global-index",
        "child streams). Robust p1/p99 bounds, pool P90s, and the axis screen are",
        "recomputed per prefix. Selector: campaign `lhs_nn` at N = 100, 10 seeds",
        "(+50-seed random null). Statistic: within-seed minimum per-axis tail",
        "share above the pool P90, averaged over seeds (block-D convention),",
        f"reported against the {TAIL_NULL_SHARE:g} share of an i.i.d. selection.*",
        "",
        "## Per-rung results (full retained set, m = 8)",
        "",
        "| P | screen retained | max abs rho_S | tail share min | tail share mean |"
        " worst axis | conc. ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rungs:
        f = r["msets"]["full"]
        lines.append(
            f"| {r['P']:,} | {r['n_retained']}/8 | {r['max_abs_rho']:.2f} "
            f"| {f['tail_min']:.3f} | {f['tail_mean']:.3f} "
            f"| {f['worst_axis']} ({f['worst_axis_seed_mean']:.3f}) "
            f"| {f['conc_ratio']:.3f} |"
        )

    if all("campaign" in r["msets"] for r in rungs):
        lines += [
            "",
            "## Tail share at the campaign selection set",
            "",
            "| P | campaign tail min | campaign tail mean |",
            "|---|---|---|",
        ]
        for r in rungs:
            c = r["msets"]["campaign"]
            lines.append(
                f"| {r['P']:,} | {c['tail_min']:.3f} | {c['tail_mean']:.3f} |"
            )

    lines += ["", "## Fitted scaling (full set)", ""]
    fit = fits.get("conc_full")
    if fit is not None:
        lo, hi = EXPONENT_BRACKET
        verdict = "inside" if lo <= fit["beta"] <= hi else "outside"
        lines.append(
            f"- Snap-concentration ratio: c · P^(−beta) with beta = "
            f"{fit['beta']:.3f} ({verdict} the P^(−1/8)..P^(−1/5) bracket "
            f"[{lo:.3f}, {hi:.3f}])."
        )
    fit = fits.get("tail_full")
    if fit is not None:
        lines.append(
            f"- Min per-axis tail share: A − B·P^(−beta) with A = "
            f"{fit['A']:.3f}, B = {fit['B']:.3f}, beta = {fit['beta']:.3f}; "
            f"the fit is within {SATURATION_TOL:g} of A from "
            f"P ≈ {fit['P_saturated']:,.0f}."
        )

    top = rungs[-1]
    lines += ["", "## Saturation", ""]
    lines.append(
        f"- Full retained set (m = 8, N = 100) at the largest rung P = {top['P']:,}: "
        f"min per-axis tail share {top['msets']['full']['tail_min']:.3f} "
        f"(i.i.d. reference {TAIL_NULL_SHARE:g}; worst axis "
        f"{top['msets']['full']['worst_axis']})."
    )
    if "campaign" in top["msets"]:
        c = top["msets"]["campaign"]
        lines.append(
            f"- Campaign selection set (m = {c['m']}, N = 100) at P = {top['P']:,}: "
            f"min per-axis tail share {c['tail_min']:.3f}, mean {c['tail_mean']:.3f}."
        )
    lines += [
        "",
        f"- SU spent on pool generation: {GEN_SU}.",
        "",
        "![nested-P saturation](figures/nested_P_saturation.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Assemble rungs, fit trends, write the results file and figure."""
    apply_style()
    rungs = [_load_rung(p) for p in sorted(RUNGS)]
    P = np.array([r["P"] for r in rungs], dtype=float)

    fits = {
        "conc_full": _fit_powerlaw(
            P, np.array([r["msets"]["full"]["conc_ratio"] for r in rungs])
        ),
        "tail_full": _fit_saturating(
            P, np.array([r["msets"]["full"]["tail_min"] for r in rungs])
        ),
    }

    (DIAG_DIR / "figures").mkdir(parents=True, exist_ok=True)
    _figure(rungs, fits, DIAG_DIR / "figures" / "nested_P_saturation")
    md = _markdown(rungs, fits)
    (DIAG_DIR / "nested_P_saturation.md").write_text(md)

    payload = {
        "pool_slug": POOL_SLUG, "rungs": rungs,
        "tail_null_share": TAIL_NULL_SHARE, "saturation_tol": SATURATION_TOL,
        "fits": {k: (None if v is None else {kk: vv for kk, vv in v.items() if kk != "yhat"})
                 for k, v in fits.items()},
        "generation_su": GEN_SU,
    }
    (DIAG_DIR / "nested_P_saturation.json").write_text(json.dumps(payload, indent=2))
    print(f"[nestedp] wrote nested_P_saturation.md/.json + figure -> {DIAG_DIR}")


if __name__ == "__main__":
    main()
