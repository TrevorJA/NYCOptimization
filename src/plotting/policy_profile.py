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
- performance CDFs: per objective, the SOW fraction performing at least as
  well as a value (every axis oriented so performance degrades to the
  right), for the policy and the incumbent over every policy of the same
  re-evaluated Pareto set;
- satisficing: SOW satisficing fractions per objective (adopted thresholds)
  and per named criterion set (``src.satisficing_criteria``);
- failure maps: per-objective pass/fail of every SOW at its forcing-space
  coordinates, on the (volume multiplier, utilization) and (volume
  multiplier, seasonal amplitude) planes, with the focal joint criterion as
  the last panel;
- regret maps (design policies only): the same grids with each SOW labelled
  low/high regret against the FFMP incumbent, per objective at the scored
  tolerance, with the focal set's joint low-regret label as the last panel
  (the per-SOW decomposition of ``no_harm_freq_tau__{focal}``);
- factor-mapped variants of both map families (``surface_model="logit"``,
  used by ``scripts/supplemental/robust_policy_figures.py``): each panel's
  label is fitted on the SOW coordinates with ``src.factor_mapping`` and the
  panel shows the fitted probability field and its P = 0.5 boundary under
  the SOW labels; fit skill goes to a companion CSV, never into the figure;
- joint factor map: the focal joint criterion alone, factor-mapped on both
  planes side by side, with its member conditions written out in full above
  the panels (the last panel of the two failure-map grids, enlarged);
- value maps: the per-SOW objective values themselves over the (volume
  multiplier, utilization) plane, with the count of failed adopted axes per
  SOW as the last panel.

Every exact number goes to a companion CSV under ``table_dir``; panels carry
no in-panel annotations beyond the pass fraction in a title.
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex
from matplotlib.lines import Line2D

import config
from src import factor_mapping as fm
from src import results_data as rd
from src import robustness as rob
from src.reeval_core import _CUBE_FILES, find_incumbent_cube, incumbent_cube_dir
from src.satisficing_criteria import ALL_SETS, focal_criterion
from src.plotting.factor_map_surfaces import REGRET_CMAP
from src.plotting.factor_maps import FAIL_COLOR, PASS_COLOR
from src.plotting.forcing_space import load_etest_sample
from src.plotting.layout import INCUMBENT_LABEL, add_colorbar
from src.plotting.style import (
    DU_FACTOR_LABELS,
    ETEST,
    FACTOR_MAP_CMAP,
    FACTOR_MAP_MARKS,
    INCUMBENT_COLOR,
    THRESHOLD_COLOR,
    criterion_condition,
    design_color,
    design_label,
    factor_label,
    factor_symbol,
    font_rc,
    label_for,
    save_figure,
    short_label_for,
)

#: Env var naming the profiled policy (see the module docstring).
PROFILE_POLICY_ENV = "NYCOPT_PROFILE_POLICY"

#: Env var overriding the display name of a profiled searched policy (legend
#: text); unset, the name is ``"{design} policy #{solution_id}"``.
PROFILE_LABEL_ENV = "NYCOPT_PROFILE_LABEL"

#: Low/high-regret mark colours: the two ends of the regret colormap of the
#: surface maps (purple = low regret), so a regret label is never read as a
#: satisficing one.
LOW_REGRET_COLOR, HIGH_REGRET_COLOR = (
    to_hex(plt.get_cmap(REGRET_CMAP)(v)) for v in (0.85, 0.25))

#: Panel letters for multi-panel grids.
_LETTERS = "abcdefghijklmnop"

#: Axis labels of the forcing-space planes, keyed by coordinate name (the
#: project's DU-factor labeling key, in its coordinate order).
_COORD_LABELS = {c: factor_label(c) for c in DU_FACTOR_LABELS}

#: The no-change reference line (``e^m = 1``) of the factor-mapped grids.
_REFERENCE_LINE = {"color": "0.05", "lw": 1.3, "ls": (0, (4, 3))}

#: Longest line, in characters, of a factor label on a panel's vertical axis;
#: a longer label is wrapped (``style.factor_label``).
_YLABEL_WIDTH = 36

#: Math symbols of the same coordinates, for legend prose.
_COORD_SYMBOLS = {c: factor_symbol(c) for c in DU_FACTOR_LABELS}


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
        label = (os.environ.get(PROFILE_LABEL_ENV, "").strip()
                 or f"{short} policy #{target['solution_id']}")
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
# Performance CDFs: the policy against its Pareto set and the incumbent
###############################################################################

#: Line color of the profiled policy in the performance-CDF figure. Not the
#: design color: the Pareto-set lines behind it are grey, as some designs are.
_CDF_POLICY_COLOR = "#0072B2"

#: Height on each highlighted curve at which its identifying marker sits.
_CDF_MARK_LEVELS = {"policy": 0.62, "incumbent": 0.38}


def _attainment_curve(v: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Step curve of the SOW fraction performing at least as well as a value.

    Args:
        v: ``(G,)`` per-SOW values of one objective.
        kind: ``"ge"`` (higher is better) or ``"le"`` (lower is better).

    Returns:
        ``(x, y)`` with ``x`` ordered best to worst over the finite entries and
        ``y`` the fraction of ALL SOWs at least as good as ``x`` (a leading
        ``y = 0`` point closes the curve at the best value).
    """
    finite = np.sort(v[np.isfinite(v)])
    xs = finite[::-1] if kind == "ge" else finite
    ys = np.arange(1, len(xs) + 1) / len(v)
    if xs.size:
        xs, ys = np.concatenate([xs[:1], xs]), np.concatenate([[0.0], ys])
    return xs, ys


@plt.rc_context(font_rc(16))
def fig_policy_profile_performance_cdfs(ctx, out_stub: Path, table_dir: Path,
                                        axes: str = "focal",
                                        set_label: Optional[str] = None) -> dict:
    """Performance across the E_test SOWs as cumulative curves, one panel per
    objective: the profiled policy and the FFMP incumbent over every policy
    of the same re-evaluated Pareto set.

    Every horizontal axis is oriented so performance degrades to the right,
    and each curve is the fraction of SOWs performing at least as well as the
    value: a curve further up and to the left is better, and its height at
    the threshold line is the fraction of SOWs meeting that threshold.

    Args:
        ctx: Figure context (tag, slug).
        out_stub: Output stub for ``save_figure``.
        table_dir: Directory of the companion CSV.
        axes: ``"focal"`` for the member axes of the focal criterion set at
            its thresholds (one row of panels), ``"all"`` for every objective
            at the adopted thresholds.
        set_label: Legend text of the Pareto-set curves, with ``{n}`` for the
            set size; unset, the set is named after its design.

    Returns:
        The profiled policy, the plotted objectives and the set size.
    """
    if axes not in ("focal", "all"):
        raise ValueError(f"axes={axes!r}: expected 'focal' or 'all'")
    prof = load_profile(ctx)
    kinds = prof.raw.kinds
    if axes == "focal":
        thr = _criterion_vector(prof, focal_criterion())
        shown = [n for n in prof.obj_names if np.isfinite(thr[n])]
    else:
        thr, shown = prof.raw.thresholds, prof.obj_names
    cols = [prof.obj_names.index(n) for n in shown]
    cube = prof.raw.cube                                   # (S, G, M)
    show_set = cube.shape[0] > 1
    ok_set = rob._satisfy(cube, prof.obj_names, thr, kinds)            # (S, G, M)

    ncol = min(4, len(shown))
    nrow = int(np.ceil(len(shown) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 4.5 * nrow + 1.3),
                            sharey=True, constrained_layout=True, squeeze=False)
    highlighted = [("policy", prof.values, _CDF_POLICY_COLOR, "o", "-")]
    if prof.incumbent is not None:
        highlighted.append(("incumbent", prof.incumbent, INCUMBENT_COLOR, "^",
                            (0, (5, 1.5))))

    rows = []
    for i, ax in enumerate(axs.ravel()):
        if i >= len(shown):
            ax.axis("off")
            continue
        name, k = shown[i], cols[i]
        kind = kinds[name]
        if show_set:
            for s in range(cube.shape[0]):
                ax.step(*_attainment_curve(cube[s, :, k], kind), where="post",
                        color="0.70", lw=0.6, alpha=0.55, zorder=2,
                        rasterized=True)
        for key, values, color, marker, ls in highlighted:
            xs, ys = _attainment_curve(values[:, k], kind)
            ax.step(xs, ys, where="post", color=color, lw=2.8, ls=ls, zorder=5)
            j = int(np.searchsorted(ys, _CDF_MARK_LEVELS[key]))
            ax.plot(xs[min(j, len(xs) - 1)], ys[min(j, len(ys) - 1)],
                    marker=marker, ms=12, color=color, markerfacecolor="white",
                    markeredgewidth=2.6, ls="none", zorder=6)
        ax.axvline(thr[name], color="0.1", ls=(0, (1.5, 1.5)), lw=1.8, zorder=4)

        pooled = np.concatenate([cube[:, :, k].ravel(), [thr[name]]])
        lo, hi = np.nanmin(pooled), np.nanmax(pooled)
        pad = 0.03 * (hi - lo if hi > lo else 1.0)
        # Best value on the left: a higher-is-better axis runs high to low.
        ax.set_xlim((hi + pad, lo - pad) if kind == "ge" else (lo - pad, hi + pad))
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(color="0.92", lw=0.7)
        ax.set_axisbelow(True)
        # The condition without its objective name (the x label carries it).
        condition = criterion_condition(name, thr[name], kind)
        condition = condition[max(condition.find("≥"), condition.find("≤")):]
        ax.set_title(f"({_LETTERS[i]}) Threshold: {condition}", loc="left")
        ax.set_xlabel(textwrap.fill(label_for(name), 26))
        if i % ncol == 0:
            ax.set_ylabel("Fraction of SOWs performing\nat least this well")

        row = {"objective": name, "threshold": thr[name], "kind": kind,
               "sat_fraction": float(ok_set[prof.row, :, k].mean()),
               "q50": float(np.nanmedian(prof.values[:, k]))}
        if prof.incumbent is not None:
            row["incumbent_sat_fraction"] = float(
                _pass_matrix(prof, prof.incumbent, thr)[:, k].mean())
            row["incumbent_q50"] = float(np.nanmedian(prof.incumbent[:, k]))
        if show_set:
            frac = ok_set[:, :, k].mean(axis=1)
            row.update({"set_size": int(cube.shape[0]),
                        "set_sat_fraction_min": float(frac.min()),
                        "set_sat_fraction_median": float(np.median(frac)),
                        "set_sat_fraction_max": float(frac.max())})
        rows.append(row)

    handles = [Line2D([], [], color=color, lw=2.8, ls=ls, marker=marker, ms=12,
                      markerfacecolor="white", markeredgewidth=2.6,
                      label=prof.label if key == "policy" else INCUMBENT_LABEL)
               for key, _, color, marker, ls in highlighted]
    if show_set:
        short = design_label(prof.design).split(" (")[0]
        set_label = set_label or f"All {{n}} {short} Pareto-approximate policies"
        handles.append(Line2D([], [], color="0.70", lw=1.2,
                              label=set_label.format(n=cube.shape[0])))
    handles.append(Line2D([], [], color="0.1", ls=(0, (1.5, 1.5)), lw=1.8,
                          label="Satisficing threshold"))
    fig.legend(handles=handles, loc="outside upper center", ncol=2,
               frameon=False)
    fig.supxlabel("Degrading performance  →")
    save_figure(fig, out_stub)
    plt.close(fig)

    table = pd.DataFrame(rows)
    for key, val in _provenance(prof).items():
        table[key] = val
    table.to_csv(table_dir / f"policy_profile_performance_cdfs_{axes}_"
                             f"{prof.key}.csv", index=False)
    return {"policy": prof.key, "objectives": shown,
            "set_size": int(cube.shape[0])}


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


def _pack_lines(items: list, width: int = 40, sep: str = ", ") -> str:
    """Join short items into lines of at most ``width`` characters, never
    breaking inside an item (keeps each criterion condition on one line)."""
    lines, cur = [], ""
    for it in items:
        cand = it if not cur else f"{cur}{sep}{it}"
        if cur and len(cand) > width:
            lines.append(cur + sep.rstrip())
            cur = it
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _failure_map(ctx, out_stub: Path, table_dir: Path,
                 xname: str, yname: str,
                 surface_model: Optional[str] = None) -> dict:
    """3 x 3 grid: pass/fail per objective at the SOW coordinates, plus the
    focal joint criterion as the last panel. Each panel title states its
    satisficing condition (the joint panel lists its member conditions) and
    the passing SOW fraction. With ``surface_model`` (``"logit"``), each
    panel also carries that classifier's fitted probability of passing
    (:func:`_draw_surface_grid`)."""
    prof = load_profile(ctx)
    x, y = _plane(prof, xname, yname)
    names = prof.obj_names
    ok = _pass_matrix(prof, prof.values)
    focal = focal_criterion()
    joint_ok = _pass_matrix(prof, prof.values,
                            _criterion_vector(prof, focal)).all(axis=1)

    thr, kinds = prof.raw.thresholds, prof.raw.kinds
    panels = [(criterion_condition(n, thr[n], kinds[n]), ok[:, k])
              for k, n in enumerate(names)]
    focal_thr = _criterion_vector(prof, focal)
    members = [names.index(n) for n in names if np.isfinite(focal_thr[n])]
    if all(focal_thr[names[k]] == thr[names[k]] for k in members):
        # Member conditions are exactly the panels above: cite them by letter.
        joint_title = f"{focal.label}:\n{_cite_panels(members)} jointly"
    else:
        conds = [criterion_condition(names[k], focal_thr[names[k]], kinds[names[k]])
                 for k in members]
        joint_title = f"{focal.label}, all of:\n" + _pack_lines(conds)
    panels.append((joint_title, joint_ok))
    good_label = f"SOW meets the criterion under {prof.label}"
    if surface_model is None:
        _draw_label_grid(
            panels, x, y, xname, yname, out_stub, count_phrase="pass",
            good=(PASS_COLOR, good_label),
            bad=(FAIL_COLOR, "SOW fails the criterion"))
    else:
        _draw_surface_grid(
            prof, panels, xname, yname, out_stub, model=surface_model,
            count_phrase="pass", cmap=FACTOR_MAP_CMAP, good_label=good_label,
            bad_label="SOW fails the criterion",
            quantity="Probability that a SOW meets the criterion",
        ).to_csv(table_dir / f"policy_profile_failure_map_{surface_model}"
                             f"_fits_{xname}_{yname}_{prof.key}.csv",
                 index=False)

    table = _sow_table(prof)
    table[f"pass__joint_{focal.key}"] = joint_ok
    table.to_csv(table_dir / f"policy_profile_failure_map_{xname}_{yname}_{prof.key}.csv",
                 index=False)
    return {"policy": prof.key, "plane": (xname, yname), "focal": focal.key,
            "joint_pass": float(joint_ok.mean()),
            "surface_model": surface_model}


def _cite_panels(members: list) -> str:
    """Panel letters of the member axes as prose, e.g. ``(a), (c) and (f)``."""
    refs = [f"({_LETTERS[k]})" for k in members]
    return ", ".join(refs[:-1]) + f" and {refs[-1]}" if len(refs) > 1 else refs[0]


def _label_grid(panels: list, xname: str, yname: str, count_phrase: str,
                panel_size: tuple = (4.2, 3.9)):
    """Three-column, shared-axis panel grid with titles and outer axis labels.

    Args:
        panels: ``(title, good_mask)`` per panel; each title gains the good
            class's SOW fraction as ``"{fraction} of SOWs {count_phrase}"``.
        xname, yname: Coordinate names of the plotted plane.
        count_phrase: Words naming the good class in the panel titles.
        panel_size: ``(width, height)`` of one panel, inches.

    Returns:
        ``(fig, axes)``; panels beyond ``len(panels)`` are switched off.
    """
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(panel_size[0] * ncol, panel_size[1] * nrow),
        sharex=True, sharey=True, constrained_layout=True)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(panels):
            ax.axis("off")
            continue
        title, p = panels[i]
        ax.set_title(f"({_LETTERS[i]}) {title}\n"
                     f"{p.mean():.2f} of SOWs {count_phrase}", loc="left")
        if i % ncol == 0:
            ax.set_ylabel(factor_label(yname, width=_YLABEL_WIDTH))
        if i >= len(panels) - ncol:
            ax.set_xlabel(_COORD_LABELS[xname])
    return fig, axes


def _draw_label_grid(panels: list, x: np.ndarray, y: np.ndarray, xname: str,
                     yname: str, out_stub: Path, *, count_phrase: str,
                     good: tuple, bad: tuple, front: str = "good") -> None:
    """Three-column grid of two-class SOW scatters, one panel per label vector.

    Args:
        panels: ``(title, good_mask)`` per panel (see :func:`_label_grid`).
        x, y: SOW coordinates on the ``(xname, yname)`` plane.
        out_stub: Output stub for ``save_figure``.
        count_phrase: Words naming the good class in the panel titles.
        good, bad: ``(color, legend label)`` of each class.
        front: The class drawn on top, with the larger mark.
    """
    fig, axes = _label_grid(panels, xname, yname, count_phrase)
    for ax, (_, p) in zip(axes.ravel(), panels):
        layers = [(~p, bad[0]), (p, good[0])]
        (m_back, c_back), (m_front, c_front) = (
            layers if front == "good" else layers[::-1])
        ax.scatter(x[m_back], y[m_back], s=14, color=c_back, lw=0, alpha=0.85,
                   zorder=3)
        ax.scatter(x[m_front], y[m_front], s=20, color=c_front, lw=0, zorder=4)
        ax.grid(color="0.93", lw=0.6)
        ax.set_axisbelow(True)
    if xname == "em":
        for ax in axes.ravel():
            ax.axvline(1.0, color="0.55", lw=0.8, ls=":", zorder=0)

    handles = [Line2D([], [], marker="o", ls="none", ms=7, color=color,
                      label=label) for color, label in (good, bad)]
    fig.legend(handles=handles, loc="outside lower center", ncol=2,
               frameon=False)
    save_figure(fig, out_stub)
    plt.close(fig)


def _factor_matrix(prof: PolicyProfile) -> tuple[list, np.ndarray]:
    """The staged DU coordinates of the SOWs as ``(factor names, (G, m))``."""
    factors = [c for c in _COORD_LABELS if c in prof.coords]
    return factors, np.column_stack([prof.coords[c] for c in factors])


def _draw_surface_panel(ax, fit, X: np.ndarray, p: np.ndarray, i1: int,
                        i2: int, cmap: str, *, reference: bool,
                        mark_sizes: tuple = (20, 30)):
    """One factor-map panel: the fitted probability field of ``fit`` on the
    plane of features ``(i1, i2)``, its P = 0.5 boundary and the SOW labels.

    Args:
        ax: Target axes.
        fit: The :class:`src.factor_mapping.FactorMapFit` of the label ``p``
            on ``X``. A single-class fit draws the constant field of its class.
        X: ``(G, m)`` SOW coordinates the fit was trained on.
        p: ``(G,)`` good-class mask of the SOWs.
        i1, i2: Feature indices on the horizontal and vertical axis.
        cmap: Diverging colormap of the field (high = the good class).
        reference: Draw the no-change reference line at 1 on the horizontal
            axis (the volume multiplier).
        mark_sizes: Scatter areas of the (good, bad) SOW marks.

    Returns:
        The field's mappable, for the shared colorbar.
    """
    x, y = X[:, i1], X[:, i2]
    extent = (x.min(), x.max(), y.min(), y.max())
    m_ok, m_no = FACTOR_MAP_MARKS["success"], FACTOR_MAP_MARKS["failure"]
    if fit.backend == "single_class":
        P = np.full((2, 2), float(p.all()))
    else:
        g1, g2, P = fm.probability_surface(fit, X, i1, i2)
        ax.contour(g1, g2, P, levels=[0.5], colors="0.05", linewidths=2.4,
                   zorder=3)
    # The surface grid spans exactly the sampled range of each factor.
    mappable = ax.imshow(P, extent=extent, origin="lower", aspect="auto",
                         interpolation="bilinear", cmap=cmap, vmin=0.0,
                         vmax=1.0, alpha=0.85, zorder=1)
    if reference:
        # Dark dashes on a white bed: legible over both ends of the field.
        ax.axvline(1.0, color="white", lw=2.6, zorder=2)
        ax.axvline(1.0, zorder=2, **_REFERENCE_LINE)
    ax.scatter(x[p], y[p], s=mark_sizes[0], marker=m_ok["marker"],
               facecolors=m_ok["facecolor"], edgecolors=m_ok["edgecolor"],
               linewidths=0.6, zorder=4, clip_on=False)
    ax.scatter(x[~p], y[~p], s=mark_sizes[1], marker=m_no["marker"],
               facecolors=m_no["facecolor"], edgecolors=m_no["edgecolor"],
               linewidths=0.5, zorder=5, clip_on=False)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    return mappable


def _surface_fit_row(fit, factors: list, panel: str, title: str, xname: str,
                     yname: str) -> dict:
    """One companion-table row of a factor-map panel: class counts, CV skill,
    factor importances and the family's own diagnostics."""
    return {
        "panel": panel, "label": " ".join(title.split()),
        "plane": f"{xname}:{yname}", "backend": fit.backend,
        "n_good": fit.n_pos, "n_bad": fit.n_neg,
        "train_accuracy": fit.train_accuracy, "cv_auc": fit.cv_auc,
        "cv_auc_std": fit.cv_auc_std, "cv_accuracy": fit.cv_accuracy,
        "cv_note": fit.cv_note,
        **{f"imp__{n}": float(v) for n, v in
           zip(factors, np.atleast_1d(fit.importances))},
        **fm.fit_diagnostics(fit),
    }


def _surface_legend_handles(good_label: str, bad_label: str, *,
                            reference: bool,
                            boundary_label: str = "Fitted boundary: "
                                                  "probability = 0.5",
                            reference_label: Optional[str] = None) -> list:
    """Legend handles of a factor map: the two SOW marks, the P = 0.5
    boundary and (``reference``) the no-change volume line."""
    m_ok, m_no = FACTOR_MAP_MARKS["success"], FACTOR_MAP_MARKS["failure"]
    handles = [
        Line2D([], [], ls="none", marker=m_ok["marker"], ms=10,
               markerfacecolor=m_ok["facecolor"],
               markeredgecolor=m_ok["edgecolor"], label=good_label),
        Line2D([], [], ls="none", marker=m_no["marker"], ms=11,
               markerfacecolor=m_no["facecolor"],
               markeredgecolor=m_no["edgecolor"], label=bad_label),
        Line2D([], [], color="0.05", lw=2.4, label=boundary_label),
    ]
    if reference:
        handles.append(Line2D(
            [], [], **_REFERENCE_LINE,
            label=reference_label or ("No change in annual mean flow "
                                      f"({_COORD_SYMBOLS['em']} = 1)")))
    return handles


@plt.rc_context(font_rc(14))
def _draw_surface_grid(prof: PolicyProfile, panels: list, xname: str,
                       yname: str, out_stub: Path, *, model: str,
                       count_phrase: str, cmap: str, good_label: str,
                       bad_label: str, quantity: str) -> pd.DataFrame:
    """The label grid as factor maps: fitted probability field under the SOWs.

    Each panel's label is fitted on ALL of the SOWs' DU coordinates with
    :func:`src.factor_mapping.fit_classifier` and its probability of the good
    class is drawn on the ``(xname, yname)`` plane with the other coordinates
    held at their median (:func:`src.factor_mapping.probability_surface`),
    with the P = 0.5 boundary and the raw SOW labels on top. A label with one
    class has no fit: its panel is the constant field of that class. Fit
    skill is returned for the companion table and never drawn.

    Args:
        prof: The profiled policy (supplies the SOW coordinates).
        panels: ``(title, good_mask)`` per panel (see :func:`_label_grid`).
        xname, yname: Coordinate names of the plotted plane.
        out_stub: Output stub for ``save_figure``.
        model: Classifier family (``src.factor_mapping.MODELS``).
        count_phrase: Words naming the good class in the panel titles.
        cmap: Diverging colormap of the field (high = the good class).
        good_label, bad_label: Legend text of each SOW mark.
        quantity: What the field's probability is of, for the colorbar.

    Returns:
        One row per panel: class counts, CV skill, factor importances and the
        family's own diagnostics (``src.factor_mapping.fit_diagnostics``).
    """
    factors, X = _factor_matrix(prof)
    i1, i2 = factors.index(xname), factors.index(yname)

    fig, axes = _label_grid(panels, xname, yname, count_phrase,
                            panel_size=(4.7, 4.3))
    mappable, rows = None, []
    for i, (ax, (title, p)) in enumerate(zip(axes.ravel(), panels)):
        fit = fm.fit_classifier(X, p, factors, space="theta", model=model)
        mappable = _draw_surface_panel(ax, fit, X, p, i1, i2, cmap,
                                       reference=xname == "em")
        rows.append(_surface_fit_row(fit, factors, _LETTERS[i], title,
                                     xname, yname))

    method = fm.MODEL_BACKENDS[fm.resolve_model(model)].replace("_", " ")
    fitted_on = ", ".join(_COORD_SYMBOLS[c] for c in factors)
    held = ", ".join(f"{_COORD_SYMBOLS[c]} = {np.nanmedian(X[:, k]):.2f}"
                     for k, c in enumerate(factors) if k not in (i1, i2))
    cbar = add_colorbar(
        fig, mappable, list(axes.ravel()), location="bottom", shrink=0.55,
        aspect=42, pad=0.025, ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        label=(f"{quantity}: {method} on {fitted_on}\n"
               f"(surface drawn with the other factors at their median: "
               f"{held})"))
    cbar.ax.axvline(0.5, color="0.05", lw=2.4)

    fig.legend(handles=_surface_legend_handles(good_label, bad_label,
                                               reference=xname == "em"),
               loc="outside lower center", ncol=2, frameon=False)
    save_figure(fig, out_stub)
    plt.close(fig)
    return pd.DataFrame(rows)


###############################################################################
# Joint-criterion factor map: the focal criterion alone, on both planes
###############################################################################

#: The two planes of the joint-criterion figure: (horizontal, vertical).
_JOINT_PLANES = (("em", "u"), ("em", "r1"))


@plt.rc_context(font_rc(18))
def _draw_joint_surface_pair(prof: PolicyProfile, joint_ok: np.ndarray,
                             criterion_label: str, conditions: list,
                             out_stub: Path, *, model: str,
                             cmap: str) -> pd.DataFrame:
    """The joint criterion's factor map on both planes, side by side.

    One classifier is fitted to the joint pass/fail label on all of the SOWs'
    DU coordinates; each panel is the same fit drawn on one plane of
    :data:`_JOINT_PLANES` with the off-plane factors at their median (named
    in the panel title). The criterion's member conditions are written out
    above the panels, and the legend carries the pass and fail SOW counts.

    Args:
        prof: The profiled policy (supplies the SOW coordinates).
        joint_ok: ``(G,)`` mask of the SOWs meeting every member condition.
        criterion_label: Display name of the joint criterion.
        conditions: The member conditions as text, one per entry.
        out_stub: Output stub for ``save_figure``.
        model: Classifier family (``src.factor_mapping.MODELS``).
        cmap: Diverging colormap of the field (high = passing).

    Returns:
        One row per panel (:func:`_surface_fit_row`).
    """
    factors, X = _factor_matrix(prof)
    fit = fm.fit_classifier(X, joint_ok, factors, space="theta", model=model)
    n_cond, n_pass = len(conditions), int(joint_ok.sum())

    # Figure height: the header (a title line over the conditions in two
    # columns), the panels, and the colorbar and legend below them.
    fs = plt.rcParams["font.size"]
    pitch = 1.5 * fs                                   # header line pitch, pt
    n_rows = int(np.ceil(n_cond / 2))
    head_h = (n_rows + 1.85) * pitch / 72.0
    panel_h, foot_h = 6.6, 8.0 * pitch / 72.0
    fig = plt.figure(figsize=(15.5, head_h + panel_h + foot_h),
                     constrained_layout=True)
    gs = fig.add_gridspec(2, len(_JOINT_PLANES),
                          height_ratios=[head_h, panel_h])

    # Header text is stacked upward from the bottom of its row in points, so
    # it sits just above the panel titles whatever height the layout gives
    # the row; each column starts where the widest entry before it ends.
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    renderer = fig.canvas.get_renderer()

    def _line(text, x_pt, row_from_bottom, **kwargs):
        return head.annotate(
            text, xy=(0.0, 0.0), xycoords="axes fraction",
            xytext=(x_pt, (row_from_bottom + 0.45) * pitch),
            textcoords="offset points",
            ha="left", va="bottom", annotation_clip=False, **kwargs)

    x_pt = 0.8 * fs
    for col in range(0, n_cond, n_rows):
        entries = [_line(f"•  {cond}", x_pt, n_rows - 1 - r)
                   for r, cond in enumerate(conditions[col:col + n_rows])]
        widest = max(t.get_window_extent(renderer).width for t in entries)
        x_pt += widest * 72.0 / fig.dpi + 2.5 * fs
    _line(f"{criterion_label} satisficing criteria: a SOW passes only if "
          f"all {n_cond} hold", 0.0, n_rows + 0.25,
          fontsize=plt.rcParams["figure.titlesize"])

    axes, mappable, rows = [], None, []
    for i, (xname, yname) in enumerate(_JOINT_PLANES):
        _plane(prof, xname, yname)
        ax = fig.add_subplot(gs[1, i])
        i1, i2 = factors.index(xname), factors.index(yname)
        mappable = _draw_surface_panel(ax, fit, X, joint_ok, i1, i2, cmap,
                                       reference=xname == "em",
                                       mark_sizes=(46, 64))
        held = ", ".join(f"{_COORD_SYMBOLS[c]} = {np.nanmedian(X[:, k]):.2f}"
                         for k, c in enumerate(factors) if k not in (i1, i2))
        ax.set_title(f"({_LETTERS[i]}) Other factors at median: {held}",
                     loc="left")
        # The two factor names are the figure's main reading aid: set them
        # above the tick and legend type.
        axis_label_size = plt.rcParams["axes.labelsize"] + 4
        ax.set_xlabel(_COORD_LABELS[xname], fontsize=axis_label_size)
        ax.set_ylabel(factor_label(yname, width=_YLABEL_WIDTH),
                      fontsize=axis_label_size)
        axes.append(ax)
        rows.append(_surface_fit_row(fit, factors, _LETTERS[i],
                                     f"{criterion_label} (joint)",
                                     xname, yname))

    method = fm.MODEL_BACKENDS[fm.resolve_model(model)].replace("_", " ")
    fitted_on = ", ".join(_COORD_SYMBOLS[c] for c in factors)
    cbar = add_colorbar(
        fig, mappable, axes, location="bottom", shrink=0.5, aspect=28,
        pad=0.03, ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        label=(f"Probability that a SOW meets all {n_cond} criteria\n"
               f"({method} on {fitted_on})"))
    cbar.ax.axvline(0.5, color="0.05", lw=2.4)
    # Name the two ends, so the field reads without decoding the hue.
    cbar.ax.text(-0.03, 0.5, "Fails", ha="right", va="center",
                 transform=cbar.ax.transAxes)
    cbar.ax.text(1.03, 0.5, "Passes", ha="left", va="center",
                 transform=cbar.ax.transAxes)

    fig.legend(
        handles=_surface_legend_handles(
            f"SOW meets all {n_cond} criteria under {prof.label}: "
            f"{n_pass} of {joint_ok.size} SOWs",
            f"SOW fails at least one criterion: "
            f"{joint_ok.size - n_pass} of {joint_ok.size} SOWs",
            reference=True),
        loc="outside lower center", ncol=2, frameon=False)
    save_figure(fig, out_stub)
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_policy_profile_joint_failure_map(
        ctx, out_stub: Path, table_dir: Path,
        surface_model: str = "logit") -> dict:
    """The focal joint criterion alone, factor-mapped on the (volume
    multiplier, utilization) and (volume multiplier, seasonal amplitude)
    planes: the last panel of the two failure-map grids, enlarged, with the
    member conditions written out in full above the panels."""
    prof = load_profile(ctx)
    focal = focal_criterion()
    kinds = prof.raw.kinds
    focal_thr = _criterion_vector(prof, focal)
    joint_ok = _pass_matrix(prof, prof.values, focal_thr).all(axis=1)
    conditions = [criterion_condition(n, focal_thr[n], kinds[n], long=True)
                  for n in prof.obj_names if np.isfinite(focal_thr[n])]

    _draw_joint_surface_pair(
        prof, joint_ok, focal.label, conditions, out_stub,
        model=surface_model, cmap=FACTOR_MAP_CMAP,
    ).to_csv(table_dir / f"policy_profile_joint_failure_map_{surface_model}"
                         f"_fits_{prof.key}.csv", index=False)
    return {"policy": prof.key, "planes": _JOINT_PLANES, "focal": focal.key,
            "joint_pass": float(joint_ok.mean()),
            "surface_model": surface_model}


def fig_policy_profile_joint_failure_map_gbt(ctx, out_stub: Path,
                                             table_dir: Path) -> dict:
    """:func:`fig_policy_profile_joint_failure_map` with the gradient-boosted
    classifier in place of the logistic regression (the registry's builders
    take no model argument)."""
    return fig_policy_profile_joint_failure_map(ctx, out_stub, table_dir,
                                                surface_model="gbt")


def fig_policy_profile_failure_map_volume_utilization(
        ctx, out_stub: Path, table_dir: Path,
        surface_model: Optional[str] = None) -> dict:
    """Pass/fail per objective over the (volume multiplier, utilization) plane."""
    return _failure_map(ctx, out_stub, table_dir, "em", "u", surface_model)


def fig_policy_profile_failure_map_volume_seasonality(
        ctx, out_stub: Path, table_dir: Path,
        surface_model: Optional[str] = None) -> dict:
    """Pass/fail per objective over the (volume multiplier, seasonal amplitude) plane."""
    return _failure_map(ctx, out_stub, table_dir, "em", "r1", surface_model)


###############################################################################
# Regret maps: low/high regret against the incumbent over the forcing space
###############################################################################

def _regret_tolerance(prof: PolicyProfile) -> dict:
    """The per-objective regret tolerance the cube was SCORED with.

    Read from the run's ``robustness_meta.json`` so the labels decompose the
    scorecard's ``no_harm_freq_tau`` columns whatever the live environment;
    ``robustness.tau_ladder`` only when the run has not been scored.
    """
    meta = prof.path / "robustness_meta.json"
    if meta.exists():
        tau = json.loads(meta.read_text()).get("regret_tau")
        if tau:
            return {n: float(tau[n]) for n in prof.obj_names}
    return rob.tau_ladder(prof.obj_names)


def _tolerance_text(name: str, tau: float) -> str:
    """One regret tolerance as compact text, e.g. ``NYC Rel. (ann): τ = 0.02``."""
    unit = (" pp" if name.endswith("_pct")
            else " ft·d/yr" if "flood_exceedance" in name else "")
    return f"{short_label_for(name).removesuffix(' %')}: τ = {tau:g}{unit}"


def _regret_map(ctx, out_stub: Path, table_dir: Path,
                xname: str, yname: str,
                surface_model: Optional[str] = None) -> dict:
    """3 x 3 grid: low/high regret per objective at the SOW coordinates, plus
    the focal set's joint low-regret label as the last panel.

    A SOW is HIGH regret on an objective when the policy is worse than the
    FFMP incumbent there, in that same SOW, by more than the tolerance (a
    non-finite difference counts as regret, as in
    ``robustness.regret_frequencies``). The joint panel requires low regret
    on every member axis of the focal criterion set. With ``surface_model``
    (``"logit"``), each panel also carries that classifier's fitted
    probability of low regret (:func:`_draw_surface_grid`).
    """
    prof = load_profile(ctx)
    if prof.incumbent is None:
        raise ValueError(
            f"regret is measured against the FFMP incumbent, so it needs a "
            f"design policy with an incumbent cube on the same tag; set "
            f"{PROFILE_POLICY_ENV}=<design>:<solution_id> (got '{prof.key}').")
    x, y = _plane(prof, xname, yname)
    names = prof.obj_names
    tau = _regret_tolerance(prof)
    # Signed advantage over the incumbent, positive = better
    # (robustness.incumbent_advantage, for the profiled policy alone).
    adv = prof.raw.direction_signs()[None, :] * (prof.values - prof.incumbent)
    low = np.isfinite(adv) & (adv >= -np.array([tau[n] for n in names]))

    focal = focal_criterion()
    members = [k for k, n in enumerate(names) if n in set(focal.axes)]
    joint_low = low[:, members].all(axis=1)
    panels = [(_tolerance_text(n, tau[n]), low[:, k])
              for k, n in enumerate(names)]
    panels.append((f"{focal.label}:\n{_cite_panels(members)} jointly",
                   joint_low))
    good_label = (f"{prof.label}: low regret (within τ of the FFMP "
                  f"incumbent, or better)")
    bad_label = "High regret (worse than the FFMP incumbent by more than τ)"
    if surface_model is None:
        _draw_label_grid(
            panels, x, y, xname, yname, out_stub, count_phrase="low regret",
            good=(LOW_REGRET_COLOR, good_label),
            bad=(HIGH_REGRET_COLOR, bad_label), front="bad")
    else:
        _draw_surface_grid(
            prof, panels, xname, yname, out_stub, model=surface_model,
            count_phrase="low regret", cmap=REGRET_CMAP,
            good_label=good_label, bad_label=bad_label,
            quantity="Probability that a SOW is low regret",
        ).to_csv(table_dir / f"policy_profile_regret_map_{surface_model}"
                             f"_fits_{xname}_{yname}_{prof.key}.csv",
                 index=False)

    table = pd.DataFrame({"sow_id": prof.raw.sow_labels})
    for c, arr in prof.coords.items():
        table[c] = arr
    for k, name in enumerate(names):
        table[f"advantage__{name}"] = adv[:, k]
        table[f"low_regret__{name}"] = low[:, k]
    table[f"low_regret__joint_{focal.key}"] = joint_low
    table.to_csv(table_dir / f"policy_profile_regret_map_{xname}_{yname}_{prof.key}.csv",
                 index=False)
    return {"policy": prof.key, "plane": (xname, yname), "focal": focal.key,
            "tau": tau, "joint_low_regret": float(joint_low.mean()),
            "surface_model": surface_model}


def fig_policy_profile_regret_map_volume_utilization(
        ctx, out_stub: Path, table_dir: Path,
        surface_model: Optional[str] = None) -> dict:
    """Low/high regret per objective over the (volume multiplier, utilization) plane."""
    return _regret_map(ctx, out_stub, table_dir, "em", "u", surface_model)


def fig_policy_profile_regret_map_volume_seasonality(
        ctx, out_stub: Path, table_dir: Path,
        surface_model: Optional[str] = None) -> dict:
    """Low/high regret per objective over the (volume multiplier, seasonal amplitude) plane."""
    return _regret_map(ctx, out_stub, table_dir, "em", "r1", surface_model)


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
