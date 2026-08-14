"""
registry.py - The single figure registry: every figure, tiered and ordered.

One registry drives all figure rendering (``scripts/main/figures.py``). Each
:class:`FigureSpec` names its tier, manuscript/SI number, manuscript section,
data needs, and builder; tuple order IS render order (no more alphabetical
accidents). The tiers:

- ``manuscript``: the numbered main-text sequence (``figures/manuscript/``,
  manuscript style, git-tracked). The sequence builds §4's narrative:
  ensemble composition -> robustness under criteria sets (RQ1) -> where
  robustness lives in objective space -> regret vs the incumbent (RQ2) ->
  success/failure surfaces (the mechanism). Every cross-design panel shows
  E_test re-evaluated quantities ONLY -- search-time values are computed
  under each design's own ensemble and are never compared across designs.
- ``si``: supporting-information candidates (``figures/si/``,
  manuscript style, git-tracked).
- ``exploratory``: internal diagnostics
  (``outputs/figures/_exploratory/...``, PNG, dense style, gitignored).

Builders receive ``(ctx, out_stub, table_dir)`` where ``ctx`` is a
:class:`FigureContext`; legacy results builders taking
``(results, out_dir, table_dir)`` are adapted by :func:`legacy`. Every
builder keeps the companion-CSV contract: exact numbers go to ``table_dir``,
never into panel annotations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import config

#: Valid tiers, in display order.
TIERS = ("manuscript", "si", "exploratory")


class FigureContext:
    """Lazy accessors for everything figure builders consume.

    Cube-backed data (``results()``) loads on first use and raises a clear
    error when the raw per-SOW cubes are absent (they live on Anvil; local
    checkouts hold only scored CSVs) -- scorecard/table-only builders render
    locally without ever touching them.
    """

    def __init__(self, reeval_tag: Optional[str] = None,
                 slug: Optional[str] = None):
        from src.reeval_core import reeval_tag as tag_of

        self.tag = reeval_tag or os.environ.get(
            "NYCOPT_REEVAL_TAG", tag_of(config.REEVAL_ENSEMBLE_SPEC))
        self.slug = slug or os.environ.get("NYCOPT_RESULTS_SLUG", "ffmp_obj8")
        self._results = None

    def results(self) -> dict:
        """Every design's loaded re-eval cube + scorecard (+ incumbent)."""
        if self._results is None:
            from src import results_data as rd
            try:
                self._results = rd.load_design_results(self.tag, slug=self.slug)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"raw re-eval cubes for tag '{self.tag}' are not on this "
                    f"machine (they are Anvil-side); run this figure's compute "
                    f"stage there, or render only table-backed figures here. "
                    f"Missing: {exc}"
                ) from exc
        return self._results

    def comparison_dir(self) -> Path:
        """The cross-design table root for this slug + tag."""
        return config.OUTPUTS_DIR / "comparison" / self.slug / self.tag

    def tables(self, kind: str) -> Path:
        """The companion-CSV dir for one figure kind (created)."""
        p = self.comparison_dir() / "figure_tables" / kind
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass(frozen=True)
class FigureSpec:
    """One figure's identity in the sequence.

    Attributes:
        name: Stable slug (output stem for si/exploratory tiers).
        builder: ``(ctx, out_stub, table_dir) -> dict | list | None``.
        tier: ``"manuscript" | "si" | "exploratory"``.
        number: Manuscript/SI figure number (None for exploratory).
        section: Manuscript anchor, e.g. ``"4.3"`` ("" if none).
        kind: Companion-table subdir under ``figure_tables/``.
        needs: Data requirements, subset of
            {"cube", "scorecard", "figure_tables", "ensemble", "refset",
            "fdc_cache", "factor_mapping"} -- drives skip-with-message.
        caption: One-line description for ``--list`` and the contact sheet.
    """

    name: str
    builder: Callable
    tier: str
    number: Optional[int]
    section: str
    kind: str
    needs: frozenset = field(default_factory=frozenset)
    caption: str = ""

    @property
    def stem(self) -> str:
        """Output filename stem, carrying the number on numbered tiers."""
        if self.tier == "manuscript" and self.number is not None:
            return f"fig{self.number:02d}_{self.name}"
        if self.tier == "si" and self.number is not None:
            return f"figS{self.number:02d}_{self.name}"
        return self.name

    def out_dir(self) -> Path:
        """The tier's output directory (created)."""
        if self.tier == "manuscript":
            p = config.MANUSCRIPT_FIG_DIR
        elif self.tier == "si":
            p = config.SI_FIG_DIR
        else:
            p = (config.FIG_EXPLORATORY_DIR / "comparison" / "figures"
                 / self.kind)
        p.mkdir(parents=True, exist_ok=True)
        return p


def legacy(builder: Callable) -> Callable:
    """Adapt a legacy results builder ``(results, out_dir, table_dir)``."""
    def _run(ctx: FigureContext, out_stub: Path, table_dir: Path):
        return builder(ctx.results(), out_stub.parent, table_dir)
    _run.__doc__ = builder.__doc__
    _run.__name__ = getattr(builder, "__name__", "legacy_builder")
    return _run


def _lazy(module: str, attr: str, adapt: bool = False) -> Callable:
    """Import a builder at call time (keeps registry import light)."""
    def _run(ctx, out_stub, table_dir):
        import importlib
        fn = getattr(importlib.import_module(module), attr)
        fn = legacy(fn) if adapt else fn
        return fn(ctx, out_stub, table_dir)
    _run.__name__ = attr
    return _run


#: The figure sequence. Tuple order = render order. Manuscript numbers pick
#: up after the methods figures (1 basin map, 2 experimental design, 3
#: forcing space).
FIGURES: tuple[FigureSpec, ...] = (
    # ------------------------------------------------------------- manuscript
    FigureSpec(
        name="forcing_space",
        builder=_lazy("src.plotting.forcing_figure", "build_forcing_space"),
        tier="manuscript", number=3, section="2",
        kind="forcing_space", needs=frozenset({"ensemble", "fdc_cache"}),
        caption="Construction of the deeply uncertain forcing space "
                "(harmonic model, CMIP6 fits, sampled box, FDCs).",
    ),
    FigureSpec(
        name="ensemble_composition",
        builder=_lazy("src.plotting.ensemble_composition",
                      "fig_ensemble_composition"),
        tier="manuscript", number=4, section="4.1",
        kind="ensemble_composition", needs=frozenset({"ensemble"}),
        caption="Realized hazard-space composition of the search ensembles "
                "vs the candidate pool and E_test.",
    ),
    FigureSpec(
        name="criteria_robustness",
        builder=_lazy("src.plotting.criteria_rank_curves",
                      "fig_criteria_rank_curves"),
        tier="manuscript", number=5, section="4.3",
        kind="criteria", needs=frozenset({"criteria_scorecard"}),
        caption="Sorted robustness-rank curves per criterion set (RQ1 "
                "headline) with cross-set ranking stability.",
    ),
    FigureSpec(
        name="robust_tradeoffs",
        builder=_lazy("src.plotting.robustness_comparison",
                      "fig_parallel_coords_focal", adapt=True),
        tier="manuscript", number=6, section="4.3",
        kind="parallel_coords", needs=frozenset({"cube", "scorecard"}),
        caption="Re-evaluated objective trade-offs recolored by focal-set "
                "robustness, per design.",
    ),
    FigureSpec(
        name="regret_vs_incumbent",
        builder=_lazy("src.plotting.regret_headline",
                      "fig_regret_vs_incumbent"),
        tier="manuscript", number=7, section="4.4",
        kind="robustness", needs=frozenset({"criteria_scorecard",
                                            "figure_tables"}),
        caption="Robustness vs no-harm against the FFMP incumbent (RQ2 "
                "headline) with the tolerance sweep.",
    ),
    FigureSpec(
        name="success_failure_surfaces",
        builder=_lazy("src.plotting.factor_map_surfaces",
                      "fig_success_failure_surfaces"),
        tier="manuscript", number=8, section="4.3",
        kind="factor_maps", needs=frozenset({"factor_mapping"}),
        caption="Boosted-tree success/failure probability surfaces over the "
                "DU forcing space, per design policy and the incumbent.",
    ),
    # --------------------------------------------------------------------- si
    FigureSpec(
        name="satisficing_decomposition",
        builder=_lazy("src.plotting.satisficing_diagnostics",
                      "fig_satisficing_decomposition", adapt=True),
        tier="si", number=1, section="4.3", kind="satisficing",
        needs=frozenset({"cube", "scorecard"}),
        caption="Univariate satisficing decomposition per objective axis.",
    ),
    FigureSpec(
        name="conjunction_collapse",
        builder=_lazy("src.plotting.satisficing_diagnostics",
                      "fig_conjunction_collapse", adapt=True),
        tier="si", number=2, section="4.3", kind="satisficing",
        needs=frozenset({"cube", "scorecard"}),
        caption="Joint satisficing collapse as axes are conjoined.",
    ),
    FigureSpec(
        name="threshold_response",
        builder=_lazy("src.plotting.satisficing_diagnostics",
                      "fig_threshold_response", adapt=True),
        tier="si", number=3, section="4.3", kind="satisficing",
        needs=frozenset({"cube", "scorecard"}),
        caption="Satisficing response to threshold placement per axis.",
    ),
    FigureSpec(
        name="attainability_blockers",
        builder=_lazy("src.plotting.satisficing_diagnostics",
                      "fig_attainability_blockers", adapt=True),
        tier="si", number=4, section="4.3", kind="satisficing",
        needs=frozenset({"cube", "scorecard"}),
        caption="Which axes block attainability across E_test SOWs.",
    ),
    FigureSpec(
        name="pairwise_cosatisficing",
        builder=_lazy("src.plotting.satisficing_diagnostics",
                      "fig_pairwise_cosatisficing", adapt=True),
        tier="si", number=5, section="4.3", kind="satisficing",
        needs=frozenset({"cube", "scorecard"}),
        caption="Pairwise co-satisficing structure between axes.",
    ),
    FigureSpec(
        name="criterion_robustness_matrix",
        builder=_lazy("src.plotting.criteria_comparison",
                      "fig_criterion_robustness_matrix", adapt=True),
        tier="si", number=6, section="4.3", kind="criteria",
        needs=frozenset({"cube", "scorecard"}),
        caption="Best/median robustness per criterion set and design.",
    ),
    FigureSpec(
        name="criterion_collapse",
        builder=_lazy("src.plotting.criteria_comparison",
                      "fig_criterion_collapse", adapt=True),
        tier="si", number=7, section="4.3", kind="criteria",
        needs=frozenset({"cube", "scorecard"}),
        caption="Collapse curves under each criterion set.",
    ),
    FigureSpec(
        name="drought_flood_split",
        builder=_lazy("src.plotting.criteria_comparison",
                      "fig_drought_flood_split", adapt=True),
        tier="si", number=8, section="4.3", kind="criteria",
        needs=frozenset({"cube", "scorecard"}),
        caption="Dry-axes vs flood-axis satisficing pincer.",
    ),
    FigureSpec(
        name="robustness_cdf",
        builder=_lazy("src.plotting.robustness_comparison",
                      "fig_robustness_cdf_focal", adapt=True),
        tier="si", number=9, section="4.3", kind="robustness_cdf",
        needs=frozenset({"cube", "scorecard"}),
        caption="Exceedance curves of joint focal-set robustness per design.",
    ),
    FigureSpec(
        name="factor_maps_theta",
        builder=_lazy("src.plotting.factor_maps",
                      "fig_factor_maps_theta_focal", adapt=True),
        tier="si", number=10, section="4.3", kind="factor_maps",
        needs=frozenset({"cube", "scorecard", "ensemble"}),
        caption="Raw pass/fail labels over the theta forcing space "
                "(the unfitted companion of the surface maps).",
    ),
    FigureSpec(
        name="search_convergence",
        builder=_lazy("src.plotting.search_outcomes", "fig_search_outcomes"),
        tier="si", number=11, section="4.2",
        kind="search_outcomes", needs=frozenset(),
        caption="Per-design MOEA runtime convergence (per-design scales; "
                "search-time values are never compared across designs).",
    ),
)


def by_tier(tier: str) -> tuple[FigureSpec, ...]:
    """The registry restricted to one tier, in render order."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    return tuple(s for s in FIGURES if s.tier == tier)


def by_name(name: str) -> FigureSpec:
    """Look up a spec by its slug or its numbered stem."""
    for s in FIGURES:
        if name in (s.name, s.stem):
            return s
    raise KeyError(f"unknown figure '{name}'; see --list")
