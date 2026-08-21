"""
factor_map_surfaces.py - Manuscript Figures 8 and 9: DU-space outcome surfaces.

Two figures in the Hadjimichael et al. (2020) / Lau et al. (2023) factor-map
idiom, drawn by ONE routine so they are read panel-for-panel:

* **Figure 8, success/failure** -- for each design's focal-criterion
  compromise policy (plus the FFMP incumbent), the predicted probability that
  a state of the world MEETS the focal satisficing set.
* **Figure 9, regret** -- the same policies, relabelled by whether the policy
  harms the FFMP incumbent beyond tolerance on the focal set's member axes.
  This is the per-SOW decomposition of the ``no_harm_freq_tau__{key}``
  scorecard column: where in the DU space does reoptimizing actually cost the
  Decree parties something relative to current operations?

Both draw the boosted-tree probability as a diverging field with the same
orientation -- **blue = the good outcome** (meets the set / low regret), red =
the bad one, neutral at the P = 0.5 contour, which is also drawn -- under the
raw E_test SOW labels (white circles = good, dark X = bad). A reader who has
parsed Figure 8 can read Figure 9 without re-learning the grammar; only the
LABEL changes.

Figure 9 has no incumbent panel: regret is measured against the incumbent, so
its own regret is zero in every SOW by construction and the panel would carry
no information.

Data: the ``factor_mapping/{criterion}/`` artifacts written by
``scripts/main/factor_mapping_run.py`` (surfaces NPZ + labels/fits CSVs) --
figures render anywhere, no cubes needed. CV skill per fit goes to the
companion CSV, never into the panels.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.plotting.layout import (WIDTH_DOUBLE_COL, add_colorbar,
                                 criteria_footer, shared_legend)
from src.plotting.pareto_parallel import DESIGN_TITLES
from src.plotting.style import (DESIGN_ORDER, ETEST, FACTOR_MAP_CMAP,
                                FACTOR_MAP_MARKS, add_figure_footer,
                                design_label, save_manuscript_figure,
                                short_label_for)
from src.satisficing_criteria import focal_criterion

#: Axis display names for the theta features.
_THETA_LABELS = {"em": "Water-year volume multiplier, $e^{m}$",
                 "r1": "Seasonal amplitude, $r_1$",
                 "r2": "Semiannual amplitude, $r_2$"}


@dataclass(frozen=True)
class _MapSpec:
    """What distinguishes one factor-map figure from the other.

    Attributes:
        prefix: Artifact filename prefix (``factor_map`` | ``regret_map``).
        cbar_label: Colorbar label -- always the probability of the GOOD class.
        pos_legend: Legend text for the good-outcome marker.
        neg_legend: Legend text for the bad-outcome marker.
        count_word: Title suffix for the good-class count, e.g. ``"pass"``.
        table_stem: Companion-CSV stem (the focal key is appended).
        include_incumbent: Draw the incumbent panel last.
    """

    prefix: str
    cbar_label: str
    pos_legend: str
    neg_legend: str
    count_word: str
    table_stem: str
    include_incumbent: bool
    #: Value of the artifacts' ``view`` column to keep (None = no filter).
    view: str | None = None
    #: Extra sentence for the footer describing how the policy was chosen.
    selection: str = ""


SUCCESS_MAP = _MapSpec(
    prefix="factor_map",
    cbar_label="P(SOW meets focal set)",
    pos_legend="SOW meets the focal set",
    neg_legend="SOW fails at least one criterion",
    count_word="pass",
    table_stem="success_failure_surfaces",
    include_incumbent=True,
)

_REGRET_LEGEND = dict(
    pos_legend="Low regret: no focal axis harmed beyond tolerance",
    neg_legend="High regret: at least one focal axis harmed",
    count_word="low regret",
    cbar_label="P(SOW is low-regret)",
    include_incumbent=False,
    prefix="regret_map",
)

#: Variant A of the regret map: the SAME policy Figure 8 shows.
REGRET_MAP = _MapSpec(
    table_stem="regret_surfaces", view="compromise",
    selection="Policy per design: the focal-set compromise policy, i.e. the "
              "one Figure 8 shows.",
    **_REGRET_LEGEND,
)

#: Variant B: the worst case each design's front actually contains.
REGRET_WORST_MAP = _MapSpec(
    table_stem="regret_surfaces_worst", view="worst",
    selection="Policy per design: the Pareto-set policy that harms the "
              "incumbent in the MOST SOWs -- the worst case the front "
              "contains, not the policy anyone would select.",
    **_REGRET_LEGEND,
)


def _load_artifacts(ctx, criterion: str, prefix: str) -> dict:
    """Load one map's persisted fits, labels, surfaces, and shared metadata."""
    root = ctx.comparison_dir() / "factor_mapping" / criterion
    fits_path = root / f"{prefix}_fits.csv"
    if not fits_path.exists():
        raise FileNotFoundError(
            f"{prefix} artifacts not found at {fits_path} -- run "
            f"scripts/main/factor_mapping_run.py first (Anvil)."
        )
    out = {
        "fits": pd.read_csv(fits_path),
        "labels": pd.read_csv(root / f"{prefix}_labels.csv"),
        "meta": json.loads((root / "factor_mapping_meta.json").read_text()),
        "surfaces": {},
    }
    npz = root / f"{prefix}_surfaces.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            out["surfaces"] = {k: z[k] for k in z.files}
    return out


def _ordered_panels(fits: pd.DataFrame, include_incumbent: bool,
                    view: str | None = None) -> list:
    """Theta-space panels in series order, incumbent last (when included).

    "...and here is the status quo for contrast" reads best last, and the
    designs keep the order used by every other figure so a reader comparing
    panels across figures never has to re-learn the arrangement.
    """
    theta = fits[fits["space"] == "theta"]
    if view is not None and "view" in theta.columns:
        theta = theta[theta["view"] == view]
    panels, incumbent_row = [], None
    for _, row in theta.iterrows():
        if row["policy"] == "incumbent":
            if incumbent_row is None:
                incumbent_row = row
            continue
        panels.append(row)
    panels.sort(key=lambda r: (DESIGN_ORDER.index(r["design"])
                               if r["design"] in DESIGN_ORDER
                               else len(DESIGN_ORDER),
                               str(r["policy"])))
    if include_incumbent and incumbent_row is not None:
        panels.append(incumbent_row)
    return panels


def _draw_surface_grid(ctx, out_stub: Path, table_dir: Path,
                       spec: _MapSpec) -> dict:
    """Render one row of DU-space probability panels for ``spec``."""
    focal = focal_criterion()
    art = _load_artifacts(ctx, focal.key, spec.prefix)
    panels = _ordered_panels(art["fits"], spec.include_incumbent, spec.view)
    if not panels:
        raise FileNotFoundError(
            f"no theta-space {spec.prefix} fits in the artifacts")

    n = len(panels)
    # sharey: the panels differ only in the policy, so one y-axis serves all
    # and the inner tick labels are pure repetition.
    fig, axes = plt.subplots(
        1, n, figsize=(WIDTH_DOUBLE_COL, WIDTH_DOUBLE_COL / n * 1.55),
        constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    mappable = None
    csv_rows = []
    shared_axis_names = None
    for i, (ax, row) in enumerate(zip(axes, panels)):
        design, policy = row["design"], str(row["policy"])
        # Regret artifacts namespace their surfaces by view, so one NPZ can
        # hold several policy selections without key collisions.
        key = (f"{design}__{policy}__theta" if spec.view is None
               else f"{spec.view}__{design}__{policy}__theta")
        ax_names = None

        if f"{key}__P" in art["surfaces"]:
            g1 = art["surfaces"][f"{key}__g1"]
            g2 = art["surfaces"][f"{key}__g2"]
            P = art["surfaces"][f"{key}__P"]
            ax_names = [str(a) for a in art["surfaces"][f"{key}__axes"]]
            mappable = ax.pcolormesh(g1, g2, P, cmap=FACTOR_MAP_CMAP, vmin=0.0,
                                     vmax=1.0, alpha=0.85, shading="auto",
                                     rasterized=True)
            ax.contour(g1, g2, P, levels=[0.5], colors="0.15", linewidths=1.0)
        else:
            # A single-class fit has no probability surface to draw -- e.g. a
            # policy that never regrets ANY SOW. That is a RESULT, not a
            # missing artifact, so the panel still plots its SOW labels (all
            # one class) instead of being blanked. The fits row records the
            # axes even when the fit degenerates.
            axis_pair = [row.get("top_axis_1"), row.get("top_axis_2")]
            if all(isinstance(a, str) for a in axis_pair):
                ax_names = [str(a) for a in axis_pair]
            elif shared_axis_names is not None:
                ax_names = list(shared_axis_names)

        n_good = -1
        n_sow = 0
        lab = art["labels"]
        # Compare policy ids AS STRINGS on both sides. Solution ids are
        # integers, so a labels file that happens to contain no "incumbent"
        # row reads the column back as int64 and an ==-against-str silently
        # matches nothing, dropping the whole marker layer.
        sub = lab[(lab["design"] == design)
                  & (lab["policy"].astype(str) == policy)
                  & (lab["criterion"] == focal.key)]
        if spec.view is not None and "view" in sub.columns:
            sub = sub[sub["view"] == spec.view]
        if ax_names is not None and not sub.empty \
                and all(a in sub.columns for a in ax_names):
            ok = sub["pass"].astype(bool).to_numpy()
            x, y = sub[ax_names[0]].to_numpy(), sub[ax_names[1]].to_numpy()
            m_ok = FACTOR_MAP_MARKS["success"]
            ax.scatter(x[ok], y[ok], s=12, marker=m_ok["marker"],
                       facecolors=m_ok["facecolor"],
                       edgecolors=m_ok["edgecolor"], linewidths=0.5, zorder=4)
            m_no = FACTOR_MAP_MARKS["failure"]
            ax.scatter(x[~ok], y[~ok], s=16, marker=m_no["marker"],
                       facecolors=m_no["facecolor"],
                       edgecolors=m_no["edgecolor"], linewidths=0.4, zorder=4)
            n_good, n_sow = int(ok.sum()), len(sub)
            shared_axis_names = ax_names
        elif ax_names is None:
            ax.axis("off")
            continue

        # The panel letter rides IN the title. As a separate text it either
        # sits on the dark-red field (unreadable) or above the axes, where it
        # collides with the title on a ~1.6-inch-wide panel.
        letter = chr(ord("a") + i)
        who = ("FFMP incumbent" if policy == "incumbent"
               else f"{design_label(design).split(' (')[0]}")
        detail = "" if policy == "incumbent" else f"#{policy}, "
        title = (f"({letter}) {who}" if n_good < 0
                 else f"({letter}) {who}\n"
                      f"{detail}{n_good}/{n_sow} {spec.count_word}")
        ax.set_title(title, fontsize=9)
        csv_rows.append({
            "design": design, "policy": policy, "criterion": focal.key,
            "axis_1": ax_names[0], "axis_2": ax_names[1],
            "n_good": n_good, "n_sow": n_sow,
            "cv_auc": row.get("cv_auc"), "cv_auc_std": row.get("cv_auc_std"),
            "train_accuracy": row.get("train_accuracy"),
            "backend": row.get("backend"),
        })

    # ONE x-label for the row instead of one per panel: at four panels across
    # a double-column width each panel is ~1.6 in wide, narrower than the
    # label itself, so per-panel labels overprint their neighbours.
    if shared_axis_names is not None:
        fig.supxlabel(_THETA_LABELS.get(shared_axis_names[0],
                                        shared_axis_names[0]))
        fig.supylabel(_THETA_LABELS.get(shared_axis_names[1],
                                        shared_axis_names[1]))

    if mappable is not None:
        add_colorbar(fig, mappable, list(axes), label=spec.cbar_label)

    m_ok, m_no = FACTOR_MAP_MARKS["success"], FACTOR_MAP_MARKS["failure"]
    shared_legend(fig, [
        Line2D([], [], ls="none", marker=m_ok["marker"],
               markerfacecolor=m_ok["facecolor"],
               markeredgecolor=m_ok["edgecolor"], markersize=7,
               label=spec.pos_legend),
        Line2D([], [], ls="none", marker=m_no["marker"],
               markerfacecolor=m_no["facecolor"],
               markeredgecolor=m_no["edgecolor"], markersize=7,
               label=spec.neg_legend),
    ], ncol=2, y=-0.04)

    _footer(fig, art["meta"], focal, spec)
    save_manuscript_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(csv_rows).to_csv(
        table_dir / f"{spec.table_stem}_{focal.key}.csv", index=False)
    return {"criterion": focal.key, "n_panels": n}


#: Footer wrap width (characters). A single long footer line widens the
#: tight bounding box past the figure itself, which squeezes the panels.
_FOOTER_WRAP = 118


def _wrap(lines: list) -> list:
    """Wrap footer prose, leaving the bulleted threshold lines intact."""
    out = []
    for line in lines:
        if line.strip().startswith("•") or not line.strip():
            out.append(line)
        else:
            out.extend(textwrap.wrap(line, _FOOTER_WRAP) or [""])
    return out


def _footer(fig, meta: dict, focal, spec: _MapSpec) -> None:
    """The provenance + criteria block, stating what LABELS the SOWs."""
    n_sow = meta.get("n_sow", "?")
    if spec.prefix == "factor_map":
        if "criterion_thresholds" in meta and "criterion_kinds" in meta:
            criteria_footer(
                fig, focal, dict(meta["criterion_thresholds"]),
                dict(meta["criterion_kinds"]),
                obj_order=list(meta["criterion_thresholds"]), y=-0.16,
                provenance=(f"Per-set compromise policies + incumbent on "
                            f"{n_sow} {ETEST} SOWs; boosted-tree surfaces "
                            f"(CV skill in the companion table)."))
        return

    # Regret: the criteria that matter are the TOLERANCES, not the satisficing
    # thresholds, so the footer states tau per member axis in its own units.
    tau = meta.get("regret_tau") or {}
    lines = [
        f"One policy per design on {n_sow} {ETEST} SOWs; boosted-tree "
        f"surfaces (CV skill in the companion table). No incumbent panel: "
        f"regret is measured against it.",
        "",
        f"A SOW is HIGH REGRET when the policy is worse than the FFMP "
        f"incumbent, in that same SOW, by more than the tolerance on any "
        f"{focal.label} axis:",
    ]
    lines += [f"  •  {short_label_for(n)}:  τ = {v:.3g}" for n, v in tau.items()]
    if spec.selection:
        lines += ["", spec.selection]
    add_figure_footer(fig, _wrap(lines), y=-0.16)


def fig_success_failure_surfaces(ctx, out_stub: Path, table_dir: Path) -> dict:
    """Theta-space GBM success surfaces per design policy + incumbent."""
    return _draw_surface_grid(ctx, out_stub, table_dir, SUCCESS_MAP)


def fig_regret_surfaces(ctx, out_stub: Path, table_dir: Path) -> dict:
    """Theta-space GBM low-regret surfaces per design compromise policy."""
    return _draw_surface_grid(ctx, out_stub, table_dir, REGRET_MAP)


def fig_regret_surfaces_worst(ctx, out_stub: Path, table_dir: Path) -> dict:
    """Theta-space GBM low-regret surfaces for each design's WORST policy."""
    return _draw_surface_grid(ctx, out_stub, table_dir, REGRET_WORST_MAP)


###############################################################################
# Variant C -- front-wide regret exposure (no single policy selected)
###############################################################################

def fig_regret_exposure(ctx, out_stub: Path, table_dir: Path) -> dict:
    """Per-SOW share of each design's Pareto set that stays low-regret.

    The single-policy maps answer "does THIS policy regret here?", which makes
    them hostage to the selection rule -- the compromise policy never regrets,
    the worst one may regret everywhere. This panel asks the question of the
    WHOLE front instead: in each state of the world, what share of the design's
    Pareto policies avoid harming the incumbent? It is a frequency, so it needs
    no cross-objective normalization, and it cannot be made degenerate by
    picking a policy.

    Drawn in the same DU coordinates as Figures 8 and 9, one panel per design,
    with the same blue = good orientation.
    """
    focal = focal_criterion()
    root = ctx.comparison_dir() / "factor_mapping" / focal.key
    path = root / "regret_exposure.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"regret exposure table not found at {path} -- run "
            f"scripts/main/factor_mapping_run.py first (Anvil).")
    df = pd.read_csv(path)
    df = df[df["criterion"] == focal.key]
    meta = json.loads((root / "factor_mapping_meta.json").read_text())

    designs = [d for d in DESIGN_ORDER if d in set(df["design"])]
    if not designs:
        raise FileNotFoundError("no campaign designs in the exposure table")

    ax_names = [n for n in ("em", "r1") if n in df.columns]
    if len(ax_names) < 2:
        raise FileNotFoundError(
            "exposure table carries no theta coordinates to plot against")

    n = len(designs)
    fig, axes = plt.subplots(
        1, n, figsize=(WIDTH_DOUBLE_COL, WIDTH_DOUBLE_COL / n * 1.55),
        constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    sc = None
    rows = []
    for i, (ax, design) in enumerate(zip(axes, designs)):
        sub = df[df["design"] == design]
        sc = ax.scatter(sub[ax_names[0]], sub[ax_names[1]],
                        c=sub["share_low_regret"], cmap=FACTOR_MAP_CMAP,
                        vmin=0.0, vmax=1.0, s=26, linewidths=0.4,
                        edgecolors="0.25", zorder=3)
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        share = sub["share_low_regret"]
        ax.set_title(f"({chr(ord('a') + i)}) "
                     f"{design_label(design).split(' (')[0]}\n"
                     f"median {share.median():.2f} "
                     f"(n = {int(sub['n_policies'].iloc[0])})", fontsize=9)
        rows += sub.assign(design=design).to_dict("records")

    fig.supxlabel(_THETA_LABELS.get(ax_names[0], ax_names[0]))
    fig.supylabel(_THETA_LABELS.get(ax_names[1], ax_names[1]))
    if sc is not None:
        add_colorbar(fig, sc, list(axes),
                     label="Share of policies that are low-regret")

    tau = meta.get("regret_tau") or {}
    lines = [
        f"Every Pareto-set policy per design, on {meta.get('n_sow', '?')} "
        f"{ETEST} SOWs. One marker per SOW, coloured by the SHARE of that "
        f"design's policies that do NOT harm the incumbent there; no single "
        f"policy is selected, so the panel cannot be degenerate by choice.",
        "",
        f"A policy harms the incumbent in a SOW when it is worse than the "
        f"FFMP incumbent there by more than the tolerance on any "
        f"{focal.label} axis:",
    ]
    lines += [f"  •  {short_label_for(k)}:  τ = {v:.3g}" for k, v in tau.items()]
    add_figure_footer(fig, _wrap(lines), y=-0.10)

    save_manuscript_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(
        table_dir / f"regret_exposure_{focal.key}.csv", index=False)
    return {"criterion": focal.key, "designs": designs}


###############################################################################
# Manuscript Figure 8 -- combined robustness + regret surfaces (fig05 style)
###############################################################################

#: Every character on the figure is drawn at >= this size (style guide).
FONTSIZE = 14

#: Canvas width in inches, matching figure 5's review-round sizing.
FIG_WIDTH = 13.5

#: Diverging colormap for the REGRET row. Deliberately different from the
#: success row's :data:`FACTOR_MAP_CMAP` (RdBu) so the two labels cannot be
#: conflated; orientation matches (high P = the good outcome = purple).
REGRET_CMAP = "PuOr"

#: SOW marker sizing at the 13.5-inch canvas (the module's older 7.48-inch
#: figures use smaller marks).
_MARK_SIZES = {"good": 34, "bad": 42}


def _panel_field(ax, art: dict, key: str, fit_row, cmap: str,
                 shared_axes: list | None):
    """Draw one panel's probability field; return (mappable, axes, names).

    A single-class fit has no fitted surface -- the classifier's probability
    is the constant 0 or 1 -- so the panel is filled with that constant
    rather than left blank: "low regret everywhere" is a RESULT the reader
    should see as a solid good-class field, not as a missing panel.
    """
    surf = art["surfaces"]
    if f"{key}__P" in surf:
        g1, g2 = surf[f"{key}__g1"], surf[f"{key}__g2"]
        P = surf[f"{key}__P"]
        names = [str(a) for a in surf[f"{key}__axes"]]
        if shared_axes is not None and names == list(reversed(shared_axes)):
            g1, g2, P = g2.T, g1.T, P.T
            names = list(reversed(names))
        m = ax.pcolormesh(g1, g2, P, cmap=cmap, vmin=0.0, vmax=1.0,
                          alpha=0.85, shading="auto", rasterized=True)
        ax.contour(g1, g2, P, levels=[0.5], colors="0.15", linewidths=1.2)
        return m, names
    # Degenerate (single-class) fit: constant field at the class value.
    names = shared_axes
    axis_pair = [fit_row.get("top_axis_1"), fit_row.get("top_axis_2")]
    if all(isinstance(a, str) for a in axis_pair):
        names = axis_pair if shared_axes is None else shared_axes
    return None, names


def _constant_field(ax, value: float, xy: tuple, cmap: str):
    """Fill the panel with the constant probability ``value`` (0 or 1)."""
    x, y = xy
    g1, g2 = np.meshgrid(np.linspace(np.min(x), np.max(x), 8),
                         np.linspace(np.min(y), np.max(y), 8))
    return ax.pcolormesh(g1, g2, np.full_like(g1, value), cmap=cmap,
                         vmin=0.0, vmax=1.0, alpha=0.85, shading="auto",
                         rasterized=True)


def _sow_marks(ax, sub: pd.DataFrame, ax_names: list) -> tuple:
    """Scatter the raw SOW labels; returns ``(n_good, n_sow, x, y)``."""
    ok = sub["pass"].astype(bool).to_numpy()
    x = sub[ax_names[0]].to_numpy(dtype=float)
    y = sub[ax_names[1]].to_numpy(dtype=float)
    m_ok, m_no = FACTOR_MAP_MARKS["success"], FACTOR_MAP_MARKS["failure"]
    ax.scatter(x[ok], y[ok], s=_MARK_SIZES["good"], marker=m_ok["marker"],
               facecolors=m_ok["facecolor"], edgecolors=m_ok["edgecolor"],
               linewidths=0.7, zorder=4)
    ax.scatter(x[~ok], y[~ok], s=_MARK_SIZES["bad"], marker=m_no["marker"],
               facecolors=m_no["facecolor"], edgecolors=m_no["edgecolor"],
               linewidths=0.6, zorder=4)
    return int(ok.sum()), len(sub), x, y


def _policy_labels(art: dict, design: str, policy: str, criterion: str,
                   view: str | None = None) -> pd.DataFrame:
    """The per-SOW label rows for one (design, policy) panel."""
    lab = art["labels"]
    sub = lab[(lab["design"] == design)
              & (lab["policy"].astype(str) == str(policy))
              & (lab["criterion"] == criterion)]
    if view is not None and "view" in sub.columns:
        sub = sub[sub["view"] == view]
    return sub


def fig_robustness_regret_surfaces(ctx, out_stub: Path,
                                   table_dir: Path) -> dict:
    """Manuscript figure 8: All-Parties robustness AND regret surfaces.

    Two rows over the DU forcing space (theta), one policy per scenario
    design -- the design's max-robustness / min-regret selection on the
    figure-7 (robustness, regret) frontier:

    * **Top row (a-d)**: for that policy in each scenario design, plus the
      FFMP incumbent, the boosted-tree probability that a SOW meets the
      All-Parties compromise criterion set (the robustness metric of
      figures 6 and 7). Blue = meets the set.
    * **Bottom row (e-g)**: the SAME policies, each SOW relabelled
      high/low regret -- whether the policy leaves the FFMP incumbent
      worse off beyond tolerance on any member axis in that SOW -- with
      the same classifier on a different colormap. No incumbent panel:
      its regret is zero by construction.

    Style follows figure 5 (14 pt minimum, no bold, no footers; exact
    numbers in the companion CSVs).
    """
    focal = focal_criterion()
    art_s = _load_artifacts(ctx, focal.key, "factor_map")
    art_r = _load_artifacts(ctx, focal.key, "regret_map")
    meta = art_s["meta"]
    selection = (meta.get("policy_selection") or {}).get("per_design", {})

    top = _ordered_panels(art_s["fits"], include_incumbent=True)
    if not top:
        raise FileNotFoundError("no theta-space factor_map fits in the "
                                "artifacts")
    regret_fits = art_r["fits"]
    regret_fits = regret_fits[(regret_fits["space"] == "theta")
                              & (regret_fits["view"] == "compromise")]

    fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH, 7.4),
                             sharex=True, sharey=True)
    axes[1, 3].axis("off")

    map_s = map_r = None
    shared_axes: list | None = None
    csv_rows = []

    def _draw(ax, art, fit_row, key, cmap, view, letter, count_word):
        nonlocal shared_axes
        design, policy = fit_row["design"], str(fit_row["policy"])
        mappable, ax_names = _panel_field(ax, art, key, fit_row, cmap,
                                          shared_axes)
        if ax_names is None:
            ax_names = shared_axes
        sub = _policy_labels(art, design, policy, focal.key, view)
        n_good = n_sow = 0
        if ax_names is not None and not sub.empty \
                and all(a in sub.columns for a in ax_names):
            n_good, n_sow, x, y = _sow_marks(ax, sub, ax_names)
            if mappable is None:
                mappable = _constant_field(
                    ax, 1.0 if n_good == n_sow else 0.0, (x, y), cmap)
            shared_axes = list(ax_names)
        who = ("Current FFMP policy" if policy == "incumbent"
               else DESIGN_TITLES.get(design, design))
        ax.set_title(f"({letter}) {who}\n{count_word} in "
                     f"{n_good}/{n_sow} SOWs",
                     loc="left", fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE)
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_linewidth(1.2)
        sel = selection.get(design, {}) if policy != "incumbent" else {}
        csv_rows.append({
            "row": "robustness" if view is None else "regret",
            "panel": letter, "design": design, "policy": policy,
            "criterion": focal.key,
            "axis_1": ax_names[0] if ax_names else None,
            "axis_2": ax_names[1] if ax_names else None,
            "n_good": n_good, "n_sow": n_sow,
            "selection_rule": sel.get("rule"),
            "selection_robustness": sel.get("robustness"),
            "selection_regret_freq": sel.get("regret_freq"),
            "cv_auc": fit_row.get("cv_auc"),
            "cv_auc_std": fit_row.get("cv_auc_std"),
            "train_accuracy": fit_row.get("train_accuracy"),
            "backend": fit_row.get("backend"),
        })
        return mappable

    # Top row: scenario designs then the incumbent, letters a-d.
    designs_top = []
    for i, fit_row in enumerate(top[:4]):
        design, policy = fit_row["design"], str(fit_row["policy"])
        if policy != "incumbent":
            designs_top.append((design, policy))
        m = _draw(axes[0, i], art_s, fit_row,
                  f"{design}__{policy}__theta", FACTOR_MAP_CMAP, None,
                  chr(ord("a") + i), "meets criteria")
        map_s = m if m is not None else map_s

    # Bottom row: the SAME policies, regret label, letters e-g.
    for j, (design, policy) in enumerate(designs_top[:3]):
        match = regret_fits[
            (regret_fits["design"] == design)
            & (regret_fits["policy"].astype(str) == policy)]
        if match.empty:
            axes[1, j].axis("off")
            continue
        m = _draw(axes[1, j], art_r, match.iloc[0],
                  f"compromise__{design}__{policy}__theta", REGRET_CMAP,
                  "compromise", chr(ord("e") + j), "low regret")
        map_r = m if m is not None else map_r

    if shared_axes is not None:
        fig.supxlabel(_THETA_LABELS.get(shared_axes[0], shared_axes[0]),
                      fontsize=FONTSIZE + 1)
        fig.supylabel(_THETA_LABELS.get(shared_axes[1], shared_axes[1]),
                      fontsize=FONTSIZE + 1)
    fig.tight_layout(rect=(0.025, 0.03, 1.0, 1.0))

    # Two figure-level horizontal colorbars below the panels (one per
    # encoded quantity), then the shared frameless legend -- the fig05
    # stacking; savefig bbox='tight' keeps them in frame.
    for mappable, x0, label in (
            (map_s, 0.10, "Probability the SOW meets all criteria"),
            (map_r, 0.58, "Probability the SOW is low-regret")):
        if mappable is None:
            continue
        cax = fig.add_axes([x0, -0.050, 0.30, 0.014])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=FONTSIZE)
        cb.ax.tick_params(labelsize=FONTSIZE)
        cb.outline.set_visible(False)

    m_ok, m_no = FACTOR_MAP_MARKS["success"], FACTOR_MAP_MARKS["failure"]
    shared_legend(fig, [
        Line2D([], [], ls="none", marker=m_ok["marker"],
               markerfacecolor=m_ok["facecolor"],
               markeredgecolor=m_ok["edgecolor"], markersize=9,
               label="SOW meets all criteria (top row) / "
                     "low-regret (bottom row)"),
        Line2D([], [], ls="none", marker=m_no["marker"],
               markerfacecolor=m_no["facecolor"],
               markeredgecolor=m_no["edgecolor"], markersize=9,
               label="SOW fails at least one criterion (top row) / "
                     "high-regret (bottom row)"),
    ], ncol=2, y=-0.115, fontsize=FONTSIZE)

    save_manuscript_figure(fig, out_stub)
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(
        table_dir / f"robustness_regret_surfaces_{focal.key}.csv",
        index=False)
    tau = meta.get("regret_tau") or {}
    pd.DataFrame(
        [{"kind": "criterion", "objective": n, "value": v,
          "direction": meta.get("criterion_kinds", {}).get(n)}
         for n, v in (meta.get("criterion_thresholds") or {}).items()]
        + [{"kind": "regret_tau", "objective": n, "value": v,
            "direction": None} for n, v in tau.items()]).to_csv(
        table_dir / f"robustness_regret_surfaces_{focal.key}_criteria.csv",
        index=False)
    return {"criterion": focal.key,
            "policies": {d: p for d, p in designs_top}}
