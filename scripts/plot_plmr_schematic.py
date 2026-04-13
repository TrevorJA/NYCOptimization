"""
plot_plmr_schematic.py - Flowchart schematic for technical presentations.

Shows how external policy parameters (RBF / Tree / ANN) integrate with
the Pywr-DRB simulation model via Post-Load Model Replacement (PLMR).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colors ──────────────────────────────────────────────────────────────
C_BORG    = "#2C3E50"   # dark blue-grey  — optimizer
C_POLICY  = "#8E44AD"   # purple           — policy / DV space
C_PYWR    = "#2980B9"   # blue             — pywr model components
C_STATE   = "#27AE60"   # green            — state extraction
C_OBJ     = "#E74C3C"   # red              — objectives
C_FFMP    = "#E67E22"   # amber            — FFMP (replaced)
C_LIGHT   = "#EBF5FB"   # very light blue  — pywr zone bg
C_RED     = "#C0392B"   # strike-through red


def _box(ax, xy, w, h, text, fc, ec="#555555", fontsize=9, lw=1.2,
         fontweight="normal", alpha=1.0, text_color="white",
         boxstyle="round,pad=0.15", zorder=3, fontstyle="normal"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                         facecolor=fc, edgecolor=ec, linewidth=lw,
                         alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=fontweight,
            fontstyle=fontstyle, zorder=zorder + 1)


def _arrow(ax, xy_from, xy_to, color="#34495E", lw=1.5, style="-|>",
           connectionstyle="arc3,rad=0", zorder=2):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, color=color,
        linewidth=lw, connectionstyle=connectionstyle,
        zorder=zorder, mutation_scale=14))


def _label(ax, xy, text, fontsize=7.5, color="#555", ha="center",
           va="center", fontweight="normal", fontstyle="normal",
           zorder=5, rotation=0):
    ax.text(xy[0], xy[1], text, ha=ha, va=va, fontsize=fontsize,
            color=color, fontweight=fontweight, fontstyle=fontstyle,
            zorder=zorder, rotation=rotation)


def make_figure():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10.5))
    ax.set_xlim(-1.5, 13.0)
    ax.set_ylim(-2.0, 10.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Title ───────────────────────────────────────────────────────────
    ax.text(5.5, 9.5, "External Policy Integration via\nPost-Load Model Replacement (PLMR)",
            ha="center", va="center", fontsize=15, fontweight="bold", color=C_BORG)
    ax.text(5.5, 8.7, "How RBF / Oblique Tree / ANN policies replace FFMP release logic in Pywr-DRB",
            ha="center", va="center", fontsize=10, color="#666", fontstyle="italic")

    # ═══════════════════════════════════════════════════════════════════
    # BACKGROUND: Pywr-DRB model zone (large dashed box)
    # ═══════════════════════════════════════════════════════════════════
    pywr_bg = FancyBboxPatch((3.0, -0.2), 8.8, 7.6,
                             boxstyle="round,pad=0.25",
                             facecolor=C_LIGHT, edgecolor=C_PYWR,
                             linewidth=1.8, alpha=0.25, linestyle="--",
                             zorder=0)
    ax.add_patch(pywr_bg)
    _label(ax, (11.2, 7.15), "Pywr-DRB Simulation",
           fontsize=9, color=C_PYWR, fontweight="bold", ha="right")

    # ═══════════════════════════════════════════════════════════════════
    # LEFT COLUMN: Optimizer + Policy (outside pywr zone)
    # ═══════════════════════════════════════════════════════════════════

    # -- Borg MOEA --
    _box(ax, (-0.5, 7.5), 2.8, 1.0,
         "Borg MOEA\n(multi-objective optimizer)",
         fc=C_BORG, fontsize=10, fontweight="bold")

    # -- Decision Variables --
    _box(ax, (-0.5, 5.8), 2.8, 1.0,
         "Decision Variables\n(flat vector, 100-210 DVs)",
         fc=C_POLICY, fontsize=9)

    # -- Policy Function --
    _box(ax, (-0.5, 3.6), 2.8, 1.5,
         "Policy Function\n\nRBF  |  Tree  |  ANN\npolicy(state) -> action",
         fc=C_POLICY, fontsize=9, alpha=0.85)

    # Arrows: Borg -> DVs -> Policy
    _arrow(ax, (0.9, 7.5), (0.9, 6.8), color=C_BORG)
    _label(ax, (1.55, 7.15), "propose DVs", fontsize=8, color=C_BORG)

    _arrow(ax, (0.9, 5.8), (0.9, 5.1), color=C_POLICY)
    _label(ax, (1.7, 5.45), "set_params()", fontsize=8, color=C_POLICY)

    # ═══════════════════════════════════════════════════════════════════
    # TOP-CENTER: PLMR injection (before model.run)
    # ═══════════════════════════════════════════════════════════════════

    # -- PLMR box --
    _box(ax, (3.5, 6.2), 3.5, 1.0,
         "PLMR: Replace FFMP Parameters\non outflow nodes with\nExternalPolicyParameter",
         fc=C_PYWR, fontsize=8.5, fontweight="bold", ec=C_PYWR)

    # Arrow: Policy -> PLMR (inject policy_fn)
    _arrow(ax, (2.3, 4.35), (3.5, 6.4), color=C_POLICY,
           connectionstyle="arc3,rad=-0.2")
    _label(ax, (2.4, 5.6), "inject\npolicy_fn", fontsize=8,
           color=C_POLICY, ha="left")

    # -- Crossed-out FFMP box (to the right of PLMR) --
    _box(ax, (7.8, 6.3), 3.2, 0.8,
         "FFMP Release Rules\n(MRF targets, drought zones, flood limits)",
         fc=C_FFMP, fontsize=8, alpha=0.35, text_color="#777",
         ec="#999", lw=1.0)
    # Strike-through
    ax.plot([7.8, 11.0], [6.7, 6.7], color=C_RED, lw=3, zorder=5, alpha=0.75)

    # Arrow: PLMR -> FFMP (replaces)
    _arrow(ax, (7.0, 6.7), (7.8, 6.7), color=C_RED, lw=1.8)
    _label(ax, (7.4, 7.0), "replaces", fontsize=8, color=C_RED,
           fontstyle="italic")

    # ═══════════════════════════════════════════════════════════════════
    # TIMESTEP LOOP: dotted boundary
    # ═══════════════════════════════════════════════════════════════════
    loop_bg = FancyBboxPatch((0.3, 1.3), 11.0, 4.5,
                             boxstyle="round,pad=0.15",
                             facecolor="none", edgecolor="#888",
                             linewidth=1.5, linestyle=":",
                             zorder=1)
    ax.add_patch(loop_bg)
    _label(ax, (5.8, 5.6), "repeated each timestep  (t = 1, 2, ... , T)",
           fontsize=8.5, color="#777", fontstyle="italic", fontweight="bold")

    # ═══════════════════════════════════════════════════════════════════
    # RIGHT COLUMN: State extraction (inside timestep loop)
    # ═══════════════════════════════════════════════════════════════════

    _box(ax, (7.5, 3.5), 3.5, 1.5,
         "State Extraction\n\n3 storage fractions (NYC reservoirs)\n"
         "6 flow predictions (Montague/Trenton)\n"
         "4 NJ demand predictions\n"
         "2 temporal features (sin/cos DOY)",
         fc=C_STATE, fontsize=7.5, ec=C_STATE)

    # Arrow: State -> Policy (feedback left)
    _arrow(ax, (7.5, 4.25), (2.3, 4.25), color=C_STATE)
    _label(ax, (5.0, 3.55), "15-dim state vector",
           fontsize=9, color=C_STATE, fontweight="bold")

    # ═══════════════════════════════════════════════════════════════════
    # CENTER-BOTTOM: Volume balancing + reservoir nodes
    # ═══════════════════════════════════════════════════════════════════

    # -- Volume Balancing --
    _box(ax, (7.5, 1.8), 3.5, 1.1,
         "Volume Balancing\n\nDistribute total release across\n"
         "Cannonsville | Pepacton | Neversink\n"
         "(proportional to storage excess)",
         fc=C_PYWR, fontsize=7.5, alpha=0.85)

    # Arrow: Policy -> Volume Balancing (total release)
    _arrow(ax, (0.9, 3.6), (0.9, 2.35), color=C_POLICY)
    _arrow(ax, (0.9, 2.35), (7.5, 2.35), color=C_POLICY)
    _label(ax, (4.2, 2.65), "total release (MGD)",
           fontsize=9, color=C_POLICY, fontweight="bold")

    # -- Reservoir Nodes --
    _box(ax, (4.0, 0.1), 3.0, 0.8,
         "Pywr Reservoir Nodes\noutflow_can | outflow_pep | outflow_nev",
         fc=C_PYWR, fontsize=7.5, alpha=0.7)

    # Arrow: Balancing -> Reservoir Nodes
    _arrow(ax, (9.25, 1.8), (6.5, 0.9), color=C_PYWR,
           connectionstyle="arc3,rad=0.15")
    _label(ax, (8.3, 1.35), "3 individual\nreleases (MGD)", fontsize=7.5,
           color=C_PYWR)

    # Arrow: Reservoir Nodes -> State (loop back up)
    _arrow(ax, (7.0, 0.5), (10.8, 0.5), color=C_STATE,
           connectionstyle="arc3,rad=0.0")
    _arrow(ax, (10.8, 0.5), (10.8, 3.5), color=C_STATE)
    _label(ax, (11.1, 2.0), "live model\nstate",
           fontsize=7.5, color=C_STATE, ha="left")

    # ═══════════════════════════════════════════════════════════════════
    # BOTTOM: Simulation results + Objectives -> Borg
    # ═══════════════════════════════════════════════════════════════════

    # -- Simulation Results --
    _box(ax, (4.2, -1.1), 3.2, 0.8,
         "Simulation Results\n(storage, flows, diversions)",
         fc=C_PYWR, fontsize=8, alpha=0.7)

    _arrow(ax, (5.8, 0.1), (5.8, -0.3), color=C_PYWR)
    _label(ax, (6.5, -0.1), "after T steps", fontsize=7, color=C_PYWR,
           fontstyle="italic", ha="left")

    # -- 7 Objectives --
    _box(ax, (0.3, -1.1), 3.4, 0.8,
         "7 Objectives\n(reliability, vulnerability, flood risk, ...)",
         fc=C_OBJ, fontsize=8.5, fontweight="bold")

    # Arrow: Results -> Objectives
    _arrow(ax, (4.2, -0.7), (3.7, -0.7), color=C_OBJ)
    _label(ax, (3.95, -0.45), "compute", fontsize=7.5, color=C_OBJ,
           fontstyle="italic")

    # Arrow: Objectives -> Borg (feedback loop up the left side)
    _arrow(ax, (0.2, -0.3), (0.2, 7.5), color=C_OBJ,
           connectionstyle="arc3,rad=0.2")
    _label(ax, (-0.4, 3.5), "minimize /\nmaximize",
           fontsize=8, color=C_OBJ, fontweight="bold", rotation=90)

    # ── Save ────────────────────────────────────────────────────────────
    out = r"C:\Users\tjame\Desktop\Research\DRB\Pywr-DRB\NYCOptimization\outputs\figures\plmr_schematic.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_figure()
