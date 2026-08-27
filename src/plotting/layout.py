"""
layout.py - Figure-level composition helpers shared by the figure sequence.

``style.py`` remains the entity/label/rcParams authority; this module owns the
COMPOSITION conventions every manuscript-tier figure shares: column-true
widths, panel lettering, one shared-legend placement, one colorbar helper,
the design legend, and the mandatory criteria footer. Builders that use these
helpers agree on geometry by construction instead of by copy-paste.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotting.style import (INCUMBENT_COLOR, add_figure_footer,
                                criteria_lines, design_color, design_label)

#: AGU column widths in inches (95 mm / 190 mm) -- every manuscript figure is
#: built at one of these two widths so fonts land at true print size.
WIDTH_DOUBLE_COL = 7.48

#: Display label for the status-quo policy, shared across figures.
INCUMBENT_LABEL = "FFMP incumbent (status quo)"


def panel_grid(nrows: int, ncols: int, *, width: float = WIDTH_DOUBLE_COL,
               panel_aspect: float = 1.0, **kwargs):
    """A constrained-layout grid at a column-true figure width.

    Args:
        nrows: Grid rows.
        ncols: Grid columns.
        width: Figure width in inches (default :data:`WIDTH_DOUBLE_COL`).
        panel_aspect: Height/width ratio of one panel; sets figure height.
        **kwargs: Forwarded to ``plt.subplots``.

    Returns:
        ``(fig, axes)`` as from ``plt.subplots``.
    """
    height = width * (nrows / ncols) * panel_aspect
    kwargs.setdefault("constrained_layout", True)
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)


def panel_label(ax, letter: str, *, x: float = 0.02, y: float = 0.98,
                fontsize: Optional[float] = None, loc: str = "inside") -> None:
    """Uniform "(a)"-style panel lettering.

    Args:
        ax: Target axes.
        letter: The letter, without parentheses.
        x, y: Position in axes fraction.
        fontsize: Override; None keeps the rcParams size.
        loc: ``"inside"`` (default) places it inside the upper-left corner;
            ``"above"`` places it just outside the top-left. Use ``"above"``
            whenever the corner holds data -- a dense scatter or a dark
            heatmap cell renders an inside label unreadable.
    """
    va = "bottom" if loc == "above" else "top"
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="left", va=va,
            fontsize=fontsize, fontweight="bold")


def shared_legend(fig, handles: Sequence, *, ncol: Optional[int] = None,
                  y: Optional[float] = None, **kwargs) -> None:
    """The single legend convention: frameless, centered below the figure.

    Args:
        fig: The figure.
        handles: Legend handles.
        ncol: Columns; defaults to one row of up to three entries.
        y: Figure-fraction y of the legend's TOP. Pass this on any figure that
            also carries a footer: the default ``"outside lower center"`` is
            positioned by constrained-layout and can land on top of the
            footer's text box, which is anchored in figure coordinates.
            Anchoring both the same way makes the stack deterministic --
            legend above, footer below.
        **kwargs: Forwarded to ``fig.legend``.
    """
    ncol = ncol if ncol is not None else min(len(handles), 3)
    kwargs.setdefault("frameon", False)
    if y is None:
        fig.legend(handles=handles, loc="outside lower center", ncol=ncol,
                   **kwargs)
        return
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol, **kwargs)


def add_colorbar(fig, mappable, axes, *, label: str, **kwargs):
    """One colorbar per encoded quantity, labelled metric + statistic."""
    kwargs.setdefault("shrink", 0.9)
    cbar = fig.colorbar(mappable, ax=axes, label=label, **kwargs)
    cbar.outline.set_visible(False)
    return cbar


def criteria_footer(fig, cset, thresholds: dict, kinds: dict,
                    obj_order: Sequence[str], *, y: float = -0.06,
                    provenance: Optional[str] = None) -> None:
    """The mandatory criterion footer, from a criterion set.

    Every criterion-dependent figure (all tiers) carries the exact thresholds
    as one bullet per condition; subset sets collapse their unconstrained
    axes into a single trailing line (see ``style.criteria_lines``).

    Args:
        fig: The figure.
        cset: The :class:`~src.satisficing_criteria.CriterionSet` shown.
        thresholds: The set's FULL resolved vector
            (``results_data.criterion_thresholds``).
        kinds: ``{objective: "ge"|"le"}``.
        obj_order: Axis display order (usually the cube's ``obj_names``).
        y: Figure-fraction y of the footer top.
        provenance: Optional leading provenance line (policies, ensemble tag).
    """
    lines = ([provenance, ""] if provenance else [])
    lines += criteria_lines(
        thresholds, kinds, obj_order=obj_order,
        header=f"Satisficing criteria — {cset.label} (all must hold):")
    add_figure_footer(fig, lines, y=y)
