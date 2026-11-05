"""
policy_profile.py - Performance and robustness profile of ONE policy on the
held-out E_test re-evaluation.

The cross-design results figures (``src.figures.registry``) read every
design's cube at once. This module profiles a single policy from a single
persisted per-SOW cube, so it renders for the FFMP incumbent alone (the
baseline cube written by workflow step 10 with ``NYCOPT_CHUNK_POLICIES=
baseline``) and, unchanged, for any Pareto policy of any design once that
design's cube exists. The profiled policy is selected by the environment
(repo rule: identity from the environment, no CLI value flags):

    NYCOPT_PROFILE_POLICY   ``baseline`` (default): the incumbent cube
                            ``outputs/baseline/reeval/{tag}/``, solution 0;
                            ``<design>:<solution_id>``: that solution of
                            ``outputs/<design>/<slug>/reeval/{tag}/``, with the
                            incumbent overlaid as the status-quo reference.

Output stems carry the target key (``baseline`` or ``<design>_s<id>``) so
profiles of different policies coexist. Figures:

- distributions: per-objective empirical CDFs of the per-SOW annual-unit
  values with the adopted threshold and the failing side marked;
- satisficing: SOW satisficing fractions per objective (adopted thresholds)
  and per named criterion set (``src.satisficing_criteria``);
- failure maps: per-objective pass/fail of every SOW at its forcing-space
  coordinates, on the (volume multiplier, utilization) and (volume
  multiplier, seasonal amplitude) planes, with the focal joint criterion as
  the last panel;
- value maps: the per-SOW objective values themselves over the (volume
  multiplier, utilization) plane, with the count of failed adopted axes per
  SOW as the last panel.

Every exact number goes to a companion CSV under ``table_dir``; panels carry
no in-panel annotations beyond the pass fraction in a title.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

import config
from src import results_data as rd
from src import robustness as rob
from src.reeval_core import _CUBE_FILES, find_incumbent_cube, incumbent_cube_dir
from src.satisficing_criteria import ALL_SETS, focal_criterion
from src.plotting.factor_maps import FAIL_COLOR, PASS_COLOR
from src.plotting.forcing_space import load_etest_sample
from src.plotting.layout import INCUMBENT_LABEL, add_colorbar
from src.plotting.style import (
    ETEST,
    INCUMBENT_COLOR,
    THRESHOLD_COLOR,
    design_color,
    design_label,
    label_for,
    save_figure,
    short_label_for,
)

#: Env var naming the profiled policy (see the module docstring).
PROFILE_POLICY_ENV = "NYCOPT_PROFILE_POLICY"

#: Panel letters for multi-panel grids.
_LETTERS = "abcdefghijklmnop"

#: Axis labels of the forcing-space planes, keyed by coordinate name.
_COORD_LABELS = {
    "em": "Water-year volume multiplier, $e^{m}$",
    "r1": "Seasonal amplitude, $r_1$",
    "r2": "Semiannual amplitude, $r_2$",
    "u": "Nominal entitlement utilization, $u$",
}


@dataclass
class PolicyProfile:
    """One policy's per-SOW objective matrix with its references.

    Attributes:
        key: Output-stem key (``baseline`` or ``<design>_s<id>``).
        label: Display name of the profiled policy.
        color: Line/marker color of the profiled policy.
        design: Scenario design the cube belongs to (``baseline`` for the
            incumbent cube).
        solution_id: The profiled ``solution_id``.
        path: The re-eval leaf directory the cube was read from.
        raw: The loaded cube (all solutions; ``row`` indexes the profiled one).
        row: Row index of the profiled policy in ``raw.cube``.
        values: ``(G, M)`` per-SOW objective values of the profiled policy.
        incumbent: ``(G, M)`` incumbent values aligned to ``raw.sow_labels``,
            or None when the profiled policy IS the incumbent.
        coords: ``{name: (G,)}`` forcing-space coordinates of the SOWs
            (``em``, ``r1``, ``r2`` and, when staged, ``u``).
    """

    key: str
    label: str
    color: str
    design: str
    solution_id: int
    path: Path
    raw: rob.RawCube
    row: int
    values: np.ndarray
    incumbent: Optional[np.ndarray]
    coords: dict

    @property
    def obj_names(self) -> list:
        return list(self.raw.obj_names)

    @property
    def n_sow(self) -> int:
        return int(self.values.shape[0])


###############################################################################
# Target resolution
###############################################################################

def profile_target(tag: Optional[str] = None,
                   slug: Optional[str] = None) -> dict:
    """Resolve ``NYCOPT_PROFILE_POLICY`` to a cube path and a solution id.

    Returns:
        ``{"key", "design", "solution_id", "path"}``; the path is the re-eval
        leaf directory (it need not exist yet -- callers probe it).
    """
    from src.figures.registry import default_reeval_tag

    tag = tag or default_reeval_tag()
    slug = slug or config.CAMPAIGN_RESULTS_SLUG
    spec = os.environ.get(PROFILE_POLICY_ENV, "baseline").strip()
    if spec in ("", "baseline"):
        leaf = incumbent_cube_dir(config.OUTPUTS_DIR / "baseline" / "reeval" / tag)
        return {"key": "baseline", "design": "baseline", "solution_id": 0,
                "path": leaf}
    design, sep, sid = spec.partition(":")
    if not sep or not sid.strip().lstrip("-").isdigit():
        raise ValueError(
            f"{PROFILE_POLICY_ENV}={spec!r}: expected 'baseline' or "
            f"'<design>:<solution_id>' (e.g. historic:12)")
    sid = int(sid)
    leaf = config.OUTPUTS_DIR / design / slug / "reeval" / tag
    return {"key": f"{design}_s{sid:04d}", "design": design,
            "solution_id": sid, "path": leaf}


def target_available(tag: Optional[str] = None,
                     slug: Optional[str] = None) -> bool:
    """Cheap existence probe for the figure driver's need check."""
    try:
        leaf = profile_target(tag, slug)["path"]
    except ValueError:
        return False
    return any((leaf / f).exists() for f in _CUBE_FILES)


def load_profile(ctx) -> PolicyProfile:
    """Load the profiled policy, its incumbent reference and SOW coordinates."""
    target = profile_target(ctx.tag, ctx.slug)
    raw = rob.load_raw(target["path"])
    ids = list(raw.solution_ids)
    if target["solution_id"] not in ids:
        raise KeyError(f"solution_id {target['solution_id']} not in the cube "
                       f"at {target['path']} (ids {ids[:5]}...)")
    row = ids.index(target["solution_id"])
    values = raw.cube[row]

    incumbent = None
    if target["design"] == "baseline":
        label, color = INCUMBENT_LABEL, INCUMBENT_COLOR
    else:
        short = design_label(target["design"]).split(" (")[0]
        label = f"{short} policy #{target['solution_id']}"
        color = design_color(target["design"])
        base_dir = find_incumbent_cube(target["path"])
        if base_dir is not None:
            incumbent = rob._aligned_baseline(raw, rob.load_raw(base_dir))

    sample = load_etest_sample(config.STAGED_ENSEMBLE_DIR / ctx.tag)
    labels = np.asarray(raw.sow_labels, dtype=int)
    if labels.max() >= sample["n_sow"]:
        raise ValueError(f"cube SOW label {labels.max()} exceeds the staged "
                         f"sample of {sample['n_sow']} SOWs (tag {ctx.tag})")
    theta, names = sample["theta"], sample["theta_names"]
    coords = {
        "em": np.exp(theta[labels, names.index("m")]),
        "r1": theta[labels, names.index("r1")],
        "r2": theta[labels, names.index("r2")],
    }
    if "u" in sample:
        coords["u"] = np.asarray(sample["u"], dtype=float)[labels]

    return PolicyProfile(
        key=target["key"], label=label, color=color, design=target["design"],
        solution_id=target["solution_id"], path=Path(target["path"]),
        raw=raw, row=row, values=values, incumbent=incumbent, coords=coords)


###############################################################################
# Shared reductions
###############################################################################

def _pass_matrix(prof: PolicyProfile, values: np.ndarray,
                 thresholds: Optional[dict] = None) -> np.ndarray:
    """Per-cell pass/fail ``(G, M)`` of a ``(G, M)`` matrix under a vector."""
    thr = thresholds if thresholds is not None else prof.raw.thresholds
    return rob._satisfy(values[np.newaxis], prof.obj_names, thr,
                        prof.raw.kinds)[0]


def _criterion_vector(prof: PolicyProfile, cset) -> dict:
    return cset.thresholds(prof.raw.thresholds, prof.raw.kinds)


def _sow_table(prof: PolicyProfile) -> pd.DataFrame:
    """Tidy per-SOW table: coordinates, each objective's value and pass flag."""
    ok = _pass_matrix(prof, prof.values)
    df = pd.DataFrame({"sow_id": prof.raw.sow_labels})
    for c, arr in prof.coords.items():
        df[c] = arr
    for k, name in enumerate(prof.obj_names):
        df[name] = prof.values[:, k]
        df[f"pass__{name}"] = ok[:, k]
    df["n_failed_axes"] = (~ok).sum(axis=1)
    return df


def _provenance(prof: PolicyProfile) -> dict:
    return {"policy": prof.key, "label": prof.label, "design": prof.design,
            "solution_id": prof.solution_id, "cube": str(prof.path),
            "n_sow": prof.n_sow,
            "realizations_per_sow": prof.raw.realizations_per_sow}


def _ecdf(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF over the finite entries, as a fraction of ALL SOWs."""
    finite = v[np.isfinite(v)]
    xs = np.sort(finite)
    ys = np.arange(1, len(xs) + 1) / len(v)
    return xs, ys


###############################################################################
# Figure 1: per-objective distributions across SOWs
###############################################################################

def fig_policy_profile_distributions(ctx, out_stub: Path,
                                     table_dir: Path) -> dict:
    """Empirical CDFs of the per-SOW values, one panel per objective.

    The adopted threshold (crimson dashed) splits each panel; the failing
    side is lightly shaded, so the height of the curve at the threshold reads
    directly as the failing SOW fraction (for a ``ge`` axis) or its
    complement (``le``). A design policy carries the incumbent's CDF in
    firebrick as the status-quo reference.
    """
    prof = load_profile(ctx)
    names = prof.obj_names
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.6), constrained_layout=True)
    rows = []
    for k, (name, ax) in enumerate(zip(names, axes.ravel())):
        v = prof.values[:, k]
        thr, kind = prof.raw.thresholds[name], prof.raw.kinds[name]
        xs, ys = _ecdf(v)
        ax.step(xs, ys, where="post", color=prof.color, lw=2.0, zorder=4)
        if prof.incumbent is not None:
            bx, by = _ecdf(prof.incumbent[:, k])
            ax.step(bx, by, where="post", color=INCUMBENT_COLOR, lw=1.5,
                    ls=(0, (4, 2)), zorder=3)
        lo = min(xs.min(), thr) if xs.size else thr
        hi = max(xs.max(), thr) if xs.size else thr
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.axvline(thr, color=THRESHOLD_COLOR, ls="--", lw=1.4, zorder=5)
        fail_lo, fail_hi = ((lo - pad, thr) if kind == "ge" else (thr, hi + pad))
        ax.axvspan(fail_lo, fail_hi, color="0.5", alpha=0.10, lw=0, zorder=1)
        ax.set_ylim(0, 1.02)
        ax.grid(color="0.92", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_title(f"({_LETTERS[k]}) " + textwrap.fill(label_for(name), 34),
                     loc="left")
        ax.set_xlabel(short_label_for(name))
        if k % 4 == 0:
            ax.set_ylabel("Fraction of SOWs at or below the value")
        frac = float(_pass_matrix(prof, prof.values)[:, k].mean())
        q = np.nanquantile(v, [0.05, 0.25, 0.50, 0.75, 0.95])
        row = {"objective": name, "threshold": thr, "kind": kind,
               "sat_fraction": frac, "q05": q[0], "q25": q[1], "q50": q[2],
               "q75": q[3], "q95": q[4], "n_missing": int((~np.isfinite(v)).sum())}
        if prof.incumbent is not None:
            bi = prof.incumbent[:, k]
            row["incumbent_sat_fraction"] = float(
                _pass_matrix(prof, prof.incumbent)[:, k].mean())
            row["incumbent_q50"] = float(np.nanmedian(bi))
        rows.append(row)

    handles = [Line2D([], [], color=prof.color, lw=2.0, label=prof.label)]
    if prof.incumbent is not None:
        handles.append(Line2D([], [], color=INCUMBENT_COLOR, lw=1.5,
                              ls=(0, (4, 2)), label=INCUMBENT_LABEL))
    handles += [
        Line2D([], [], color=THRESHOLD_COLOR, ls="--", lw=1.4,
               label="Adopted satisficing threshold"),
        plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.18,
                      label="Failing side of the threshold"),
    ]
    fig.legend(handles=handles, loc="outside lower center",
               ncol=len(handles), frameon=False)
    save_figure(fig, out_stub)
    plt.close(fig)

    table = pd.DataFrame(rows)
    for key, val in _provenance(prof).items():
        table[key] = val
    table.to_csv(table_dir / f"policy_profile_distributions_{prof.key}.csv",
                 index=False)
    _sow_table(prof).to_csv(table_dir / f"policy_profile_sow_values_{prof.key}.csv",
                            index=False)
    return {"policy": prof.key, "n_sow": prof.n_sow}


###############################################################################
# Figure 2: satisficing fractions per objective and per criterion set
###############################################################################

def fig_policy_profile_satisficing(ctx, out_stub: Path,
                                   table_dir: Path) -> dict:
    """SOW satisficing fractions: per adopted axis (left) and per criterion
    set (right), as horizontal bars.

    Left: the univariate fraction of SOWs meeting each objective's adopted
    threshold -- the decomposition of the all-axes reference criterion. Right:
    the joint (Starr) fraction under each named criterion set, reference set
    last. A design policy shows the incumbent as paired open bars.
    """
    prof = load_profile(ctx)
    names = prof.obj_names
    ok = _pass_matrix(prof, prof.values)
    uni = ok.mean(axis=0)
    inc_uni = (_pass_matrix(prof, prof.incumbent).mean(axis=0)
               if prof.incumbent is not None else None)

    sets = list(ALL_SETS)
    joint, inc_joint = [], []
    for cset in sets:
        thr = _criterion_vector(prof, cset)
        joint.append(float(_pass_matrix(prof, prof.values, thr).all(axis=1).mean()))
        if prof.incumbent is not None:
            inc_joint.append(float(
                _pass_matrix(prof, prof.incumbent, thr).all(axis=1).mean()))

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(14.0, 5.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 0.85]})
    paired = prof.incumbent is not None
    h = 0.38 if paired else 0.62

    def _bars(ax, labels, vals, inc_vals):
        y = np.arange(len(labels))[::-1]
        off = h / 2 if paired else 0.0
        ax.barh(y + off, vals, height=h, color=prof.color, zorder=3)
        if paired:
            ax.barh(y - off, inc_vals, height=h, facecolor="white",
                    edgecolor=INCUMBENT_COLOR, lw=1.4, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(f"Fraction of {ETEST} SOWs satisficing")
        ax.grid(axis="x", color="0.92", lw=0.7)
        ax.set_axisbelow(True)

    _bars(axl, [short_label_for(n) for n in names], uni, inc_uni)
    axl.set_title("(a) Per objective, adopted thresholds", loc="left")
    _bars(axr, [c.label for c in sets], joint,
          inc_joint if paired else None)
    axr.set_title("(b) Per criterion set, jointly", loc="left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=prof.color, label=prof.label)]
    if paired:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor=INCUMBENT_COLOR, lw=1.4,
                                     label=INCUMBENT_LABEL))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False)
    save_figure(fig, out_stub)
    plt.close(fig)

    rows = [{"scope": "objective", "name": n, "threshold": prof.raw.thresholds[n],
             "kind": prof.raw.kinds[n], "sat_fraction": float(f),
             "incumbent_sat_fraction": (float(inc_uni[k]) if paired else np.nan)}
            for k, (n, f) in enumerate(zip(names, uni))]
    rows += [{"scope": "criterion_set", "name": c.key,
              "threshold": np.nan, "kind": "joint", "sat_fraction": j,
              "incumbent_sat_fraction": (inc_joint[i] if paired else np.nan),
              "axes": ";".join(c.axes if not c.reference else names)}
             for i, (c, j) in enumerate(zip(sets, joint))]
    table = pd.DataFrame(rows)
    for key, val in _provenance(prof).items():
        table[key] = val
    table.to_csv(table_dir / f"policy_profile_satisficing_{prof.key}.csv",
                 index=False)
    return {"policy": prof.key,
            "joint": dict(zip([c.key for c in sets], joint))}


###############################################################################
# Figures 3-4: pass/fail maps over the forcing space
###############################################################################

def _plane(prof: PolicyProfile, xname: str, yname: str):
    for c in (xname, yname):
        if c not in prof.coords:
            raise ValueError(f"the staged E_test carries no '{c}' coordinate "
                             f"(utilization not appended?)")
    return prof.coords[xname], prof.coords[yname]


def _failure_map(ctx, out_stub: Path, table_dir: Path,
                 xname: str, yname: str) -> dict:
    """3 x 3 grid: pass/fail per objective at the SOW coordinates, plus the
    focal joint criterion as the last panel."""
    prof = load_profile(ctx)
    x, y = _plane(prof, xname, yname)
    names = prof.obj_names
    ok = _pass_matrix(prof, prof.values)
    focal = focal_criterion()
    joint_ok = _pass_matrix(prof, prof.values,
                            _criterion_vector(prof, focal)).all(axis=1)

    panels = [(short_label_for(n), ok[:, k]) for k, n in enumerate(names)]
    panels.append((f"{focal.label}, jointly", joint_ok))
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.9 * nrow),
                             sharex=True, sharey=True, constrained_layout=True)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(panels):
            ax.axis("off")
            continue
        title, p = panels[i]
        ax.scatter(x[~p], y[~p], s=14, color=FAIL_COLOR, lw=0, alpha=0.85,
                   zorder=3)
        ax.scatter(x[p], y[p], s=20, color=PASS_COLOR, lw=0, zorder=4)
        ax.grid(color="0.93", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"({_LETTERS[i]}) {title}\n{p.mean():.2f} of SOWs pass",
                     loc="left")
        if i % ncol == 0:
            ax.set_ylabel(_COORD_LABELS[yname])
        if i >= len(panels) - ncol:
            ax.set_xlabel(_COORD_LABELS[xname])
    if xname == "em":
        for ax in axes.ravel():
            ax.axvline(1.0, color="0.55", lw=0.8, ls=":", zorder=0)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=7, color=PASS_COLOR,
               label=f"SOW meets the criterion under {prof.label}"),
        Line2D([], [], marker="o", ls="none", ms=6, color=FAIL_COLOR,
               label="SOW fails the criterion"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2,
               frameon=False)
    save_figure(fig, out_stub)
    plt.close(fig)

    table = _sow_table(prof)
    table[f"pass__joint_{focal.key}"] = joint_ok
    table.to_csv(table_dir / f"policy_profile_failure_map_{xname}_{yname}_{prof.key}.csv",
                 index=False)
    return {"policy": prof.key, "plane": (xname, yname),
            "focal": focal.key, "joint_pass": float(joint_ok.mean())}


def fig_policy_profile_failure_map_volume_utilization(ctx, out_stub: Path,
                                                      table_dir: Path) -> dict:
    """Pass/fail per objective over the (volume multiplier, utilization) plane."""
    return _failure_map(ctx, out_stub, table_dir, "em", "u")


def fig_policy_profile_failure_map_volume_seasonality(ctx, out_stub: Path,
                                                      table_dir: Path) -> dict:
    """Pass/fail per objective over the (volume multiplier, seasonal amplitude) plane."""
    return _failure_map(ctx, out_stub, table_dir, "em", "r1")


###############################################################################
# Figure 5: objective values over the forcing space
###############################################################################

def fig_policy_profile_value_map_volume_utilization(ctx, out_stub: Path,
                                                    table_dir: Path) -> dict:
    """Per-SOW objective values over the (volume multiplier, utilization)
    plane, one colorbar per objective; last panel: the number of adopted
    axes each SOW fails (0 = meets the all-axes reference criterion)."""
    prof = load_profile(ctx)
    x, y = _plane(prof, "em", "u")
    names = prof.obj_names
    ok = _pass_matrix(prof, prof.values)
    n_fail = (~ok).sum(axis=1)

    ncol = 3
    nrow = int(np.ceil((len(names) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow),
                             sharex=True, sharey=True, constrained_layout=True)
    flat = axes.ravel()
    for k, name in enumerate(names):
        ax = flat[k]
        v = prof.values[:, k]
        finite = np.isfinite(v)
        sc = ax.scatter(x[finite], y[finite], c=v[finite], s=18, cmap="viridis",
                        lw=0, zorder=3)
        if (~finite).any():
            ax.scatter(x[~finite], y[~finite], s=18, marker="x", color="0.3",
                       lw=0.8, zorder=4)
        add_colorbar(fig, sc, ax, label=short_label_for(name), shrink=0.85)
        ax.set_title(f"({_LETTERS[k]}) {short_label_for(name)}", loc="left")
        ax.grid(color="0.93", lw=0.6)
        ax.set_axisbelow(True)

    k = len(names)
    ax = flat[k]
    m = len(names)
    cmap = ListedColormap(plt.get_cmap("magma_r")(np.linspace(0.05, 0.95, m + 1)))
    norm = BoundaryNorm(np.arange(-0.5, m + 1.5), cmap.N)
    sc = ax.scatter(x, y, c=n_fail, s=18, cmap=cmap, norm=norm, lw=0, zorder=3)
    cb = add_colorbar(fig, sc, ax, label="Adopted axes failed", shrink=0.85,
                      ticks=np.arange(0, m + 1))
    ax.set_title(f"({_LETTERS[k]}) Failed adopted axes per SOW", loc="left")
    ax.grid(color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    for i in range(k + 1, len(flat)):
        flat[i].axis("off")
    for i in range(k + 1):
        if i % ncol == 0:
            flat[i].set_ylabel(_COORD_LABELS["u"])
        if i >= k + 1 - ncol:
            flat[i].set_xlabel(_COORD_LABELS["em"])
        flat[i].axvline(1.0, color="0.55", lw=0.8, ls=":", zorder=0)

    save_figure(fig, out_stub)
    plt.close(fig)
    _sow_table(prof).to_csv(table_dir / f"policy_profile_sow_values_{prof.key}.csv",
                            index=False)
    return {"policy": prof.key,
            "n_failed_axes_histogram": {int(i): int((n_fail == i).sum())
                                        for i in range(m + 1)}}
