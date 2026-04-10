"""
figures/fig02_architecture_schematic.py - Figure 2: Policy architecture spectrum.

Five-panel figure showing all architectures (A through E) arranged left-to-right
by expressiveness. Each panel shows a stylized representation:
  A: Zone lookup table (FFMP)
  B: Variable-resolution zone curves
  C: Gaussian RBF basis functions
  D: Oblique decision tree
  E: Neural network diagram

Annotated with DV counts and interpretability ranking.

Data: conceptual, no simulation needed.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.plotting.style import apply_style

OUTPUT_PATH = Path("outputs/manuscript_figures/fig02_architecture_schematic.png")

# Architecture metadata
ARCHS = [
    {"key": "ffmp", "label": "A. FFMP\n(6 zones)", "dvs": 24, "interp": "High"},
    {"key": "ffmp_vr", "label": "B. Var-Res FFMP\n(3\u201315 zones)", "dvs": "19\u201339", "interp": "High"},
    {"key": "rbf", "label": "C. RBF Network\n(6 centers)", "dvs": 102, "interp": "Low"},
    {"key": "tree", "label": "D. Oblique Tree\n(depth 3)", "dvs": 120, "interp": "Medium"},
    {"key": "ann", "label": "E. ANN\n(2\u00d78 hidden)", "dvs": 209, "interp": "Very Low"},
]


def _draw_ffmp_panel(ax):
    """Zone-based lookup: horizontal bands with drought level labels."""
    zones = [0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0]
    colors_z = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(zones) - 1))
    labels = ["L5", "L4", "L3", "L2", "L1c", "Normal"]
    for i in range(len(zones) - 1):
        ax.axhspan(zones[i], zones[i + 1], color=colors_z[i], alpha=0.6)
        ax.text(0.5, (zones[i] + zones[i + 1]) / 2, labels[i],
                ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_ylabel("Storage fraction", fontsize=8)
    ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)


def _draw_vr_ffmp_panel(ax):
    """Variable-resolution: more zone boundaries."""
    n_zones = 10
    zones = np.linspace(0, 1, n_zones + 1)
    colors_z = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_zones))
    for i in range(n_zones):
        ax.axhspan(zones[i], zones[i + 1], color=colors_z[i], alpha=0.6)
    ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.text(0.5, 0.5, f"N={n_zones}\nzones", ha="center", va="center",
            fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3",
            facecolor="white", alpha=0.8))


def _draw_rbf_panel(ax):
    """Gaussian basis functions."""
    x = np.linspace(-1, 1, 200)
    centers = np.linspace(-0.8, 0.8, 6)
    rng = np.random.default_rng(42)
    widths = rng.uniform(0.15, 0.4, 6)
    weights = rng.uniform(-1, 1, 6)
    total = np.zeros_like(x)
    for c, w, wt in zip(centers, widths, weights):
        basis = np.exp(-((x - c) ** 2) / (2 * w ** 2))
        ax.plot(x, basis * 0.8, alpha=0.3, color="steelblue", linewidth=1)
        total += wt * basis
    output = 1.0 / (1.0 + np.exp(-2 * total))
    ax.plot(x, output, color="firebrick", linewidth=2, label="Output")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("State input", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_tree_panel(ax):
    """Oblique decision tree (depth 3)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    nodes = {
        "root": (0.5, 0.9),
        "l1": (0.25, 0.6), "r1": (0.75, 0.6),
        "l2": (0.12, 0.3), "l3": (0.38, 0.3),
        "r2": (0.62, 0.3), "r3": (0.88, 0.3),
    }
    edges = [("root", "l1"), ("root", "r1"), ("l1", "l2"), ("l1", "l3"),
             ("r1", "r2"), ("r1", "r3")]
    for p, c in edges:
        ax.plot([nodes[p][0], nodes[c][0]], [nodes[p][1], nodes[c][1]],
                "k-", linewidth=1, alpha=0.5)

    for key in ["root", "l1", "r1"]:
        circle = plt.Circle(nodes[key], 0.06, color="steelblue", alpha=0.7, zorder=5)
        ax.add_patch(circle)
        ax.text(*nodes[key], r"$\mathbf{w}^\top\mathbf{x}$", ha="center",
                va="center", fontsize=6, color="white", zorder=6)

    for key in ["l2", "l3", "r2", "r3"]:
        rect = mpatches.FancyBboxPatch(
            (nodes[key][0] - 0.05, nodes[key][1] - 0.04), 0.10, 0.08,
            boxstyle="round,pad=0.02", facecolor="forestgreen", alpha=0.6, zorder=5)
        ax.add_patch(rect)
        ax.text(*nodes[key], "$q$", ha="center", va="center", fontsize=7,
                color="white", fontweight="bold", zorder=6)


def _draw_ann_panel(ax):
    """Feedforward ANN diagram."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    layers = [3, 6, 6, 1]
    layer_x = np.linspace(0.15, 0.85, len(layers))
    positions = {}

    for li, (n, x) in enumerate(zip(layers, layer_x)):
        ys = np.linspace(0.2, 0.8, n) if n > 1 else [0.5]
        for ni, y in enumerate(ys):
            positions[(li, ni)] = (x, y)
            color = "steelblue" if li < len(layers) - 1 else "firebrick"
            circle = plt.Circle((x, y), 0.03, color=color, alpha=0.7, zorder=5)
            ax.add_patch(circle)

    for li in range(len(layers) - 1):
        for ni in range(layers[li]):
            for nj in range(layers[li + 1]):
                p1 = positions[(li, ni)]
                p2 = positions[(li + 1, nj)]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        "k-", linewidth=0.3, alpha=0.2)

    ax.text(layer_x[0], 0.05, "State", ha="center", fontsize=6)
    ax.text(layer_x[-1], 0.05, "Release", ha="center", fontsize=6)


PANEL_DRAWERS = [_draw_ffmp_panel, _draw_vr_ffmp_panel, _draw_rbf_panel,
                 _draw_tree_panel, _draw_ann_panel]


def make_figure() -> None:
    """Generate Figure 2 and save to OUTPUT_PATH."""
    apply_style()
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))

    for ax, arch, drawer in zip(axes, ARCHS, PANEL_DRAWERS):
        drawer(ax)
        ax.set_title(arch["label"], fontsize=9, fontweight="bold", pad=8)
        dvs_str = str(arch["dvs"])
        ax.text(0.5, -0.12, f"{dvs_str} DVs | Interp: {arch['interp']}",
                ha="center", va="top", fontsize=7, color="0.4",
                transform=ax.transAxes)

    fig.text(0.15, 0.01, r"$\longleftarrow$ Interpretable", fontsize=9,
             ha="left", va="bottom", color="0.4")
    fig.text(0.85, 0.01, r"Expressive $\longrightarrow$", fontsize=9,
             ha="right", va="bottom", color="0.4")

    fig.suptitle("Policy Architecture Spectrum", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    make_figure()
