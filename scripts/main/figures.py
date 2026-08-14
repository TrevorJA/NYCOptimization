"""figures.py - Render the figure sequence from the single registry.

The one entrypoint for every figure tier (``src/figures/registry.py``):

    python -m scripts.main.figures --list
    python -m scripts.main.figures --tier manuscript
    python -m scripts.main.figures --figure regret_vs_incumbent
    python -m scripts.main.figures --tier manuscript --contact-sheet

Tiers route themselves: manuscript -> ``figures/manuscript/`` (PNG + PDF,
manuscript style, tracked), si -> ``figures/si/`` (same), exploratory ->
``outputs/figures/_exploratory/`` (PNG, dense style). A figure whose data
needs are not present on this machine is SKIPPED with a message naming the
missing need (raw cubes are Anvil-side; scored CSVs and figure tables are
synced), so a local render pass completes with whatever is renderable.

Identity comes from the environment (repo rule -- no CLI value flags):
``NYCOPT_REEVAL_TAG`` (default: the configured E_test tag),
``NYCOPT_RESULTS_SLUG`` (default ``ffmp_obj8``), ``NYCOPT_FOCAL_CRITERION``.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from src.figures.registry import (FIGURES, TIERS, FigureContext, FigureSpec,  # noqa: E402
                                  by_name, by_tier)
from src.plotting import style  # noqa: E402


def _need_available(need: str, ctx: FigureContext) -> bool:
    """Probe one data need on this machine (cheap existence checks only)."""
    designs = _campaign_designs()
    reeval = lambda d: (config.OUTPUTS_DIR / d / ctx.slug / "reeval" / ctx.tag)  # noqa: E731
    if need == "cube":
        return all(any((reeval(d) / f).exists()
                       for f in ("reeval_raw.parquet", "reeval_raw.csv.gz"))
                   for d in designs)
    if need == "scorecard":
        return all((reeval(d) / "robustness_scorecard.csv").exists()
                   for d in designs)
    if need == "criteria_scorecard":
        return all((reeval(d) / "robustness_scorecard_criteria.csv").exists()
                   for d in designs)
    if need == "figure_tables":
        return ctx.comparison_dir().is_dir()
    if need == "ensemble":
        return (config.STAGED_ENSEMBLE_DIR / ctx.tag).is_dir()
    if need == "refset":
        return all(any((config.OUTPUTS_DIR / d / ctx.slug / "sets").glob("*.set"))
                   for d in designs
                   if (config.OUTPUTS_DIR / d / ctx.slug / "sets").is_dir())
    if need == "fdc_cache":
        from scripts.main.forcing_fdc_cache import DEFAULT_CACHE
        return DEFAULT_CACHE.is_file()
    if need == "factor_mapping":
        return (ctx.comparison_dir() / "factor_mapping").is_dir()
    return True


def _campaign_designs() -> list[str]:
    from src.scenario_designs import campaign_designs
    return list(campaign_designs())


def _render(spec: FigureSpec, ctx: FigureContext) -> bool:
    """Render one figure; returns True on success."""
    missing = [n for n in sorted(spec.needs) if not _need_available(n, ctx)]
    if missing:
        print(f"[figures] SKIP {spec.stem}: missing {', '.join(missing)} "
              f"(tag={ctx.tag})")
        return False

    # PNG only during iteration rounds on every tier; the vector copies come
    # from style.MANUSCRIPT_FIGURE_FORMATS at the manuscript-final pass.
    if spec.tier in ("manuscript", "si"):
        style.apply_manuscript_style()
    else:
        style.apply_style()
    style.FIGURE_FORMATS = ("png",)

    out_stub = spec.out_dir() / spec.stem
    table_dir = ctx.tables(spec.kind)
    print(f"[figures] building {spec.stem} ({spec.tier}, section {spec.section})")
    try:
        spec.builder(ctx, out_stub, table_dir)
    except Exception:  # noqa: BLE001 - one bad figure must not kill the pass
        print(f"[figures] FAILED {spec.stem}:")
        traceback.print_exc()
        return False
    print(f"[figures]   -> {out_stub}.png")
    return True


def _contact_sheet(specs: list[FigureSpec], ctx: FigureContext) -> None:
    """A thumbnail grid of the rendered sequence, tag-stamped."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[figures] contact sheet skipped: Pillow not installed")
        return

    thumbs = []
    for spec in specs:
        png = spec.out_dir() / f"{spec.stem}.png"
        if png.exists():
            img = Image.open(png)
            img.thumbnail((640, 640))
            thumbs.append((spec.stem, img))
    if not thumbs:
        print("[figures] contact sheet skipped: nothing rendered")
        return

    cols = min(3, len(thumbs))
    rows = -(-len(thumbs) // cols)
    cell_w = max(t.width for _, t in thumbs) + 16
    cell_h = max(t.height for _, t in thumbs) + 40
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), f"reeval tag: {ctx.tag}   slug: {ctx.slug}", fill="black")
    for i, (name, img) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x, y = c * cell_w + 8, r * cell_h + 30
        sheet.paste(img, (x, y))
        draw.text((x, y + img.height + 4), name, fill="black")
    out = config.MANUSCRIPT_FIG_DIR / "contact_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"[figures] contact sheet -> {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--figure", action="append",
                   help="render one figure by name or stem (repeatable)")
    p.add_argument("--tier", choices=TIERS,
                   help="render every figure of one tier")
    p.add_argument("--list", action="store_true",
                   help="list the sequence and exit")
    p.add_argument("--contact-sheet", action="store_true",
                   help="assemble a thumbnail sheet of the rendered figures")
    args = p.parse_args(argv)

    ctx = FigureContext()

    if args.list:
        print(f"reeval tag: {ctx.tag}   slug: {ctx.slug}")
        for tier in TIERS:
            specs = by_tier(tier)
            if not specs:
                continue
            print(f"\n{tier}:")
            for s in specs:
                sec = f"§{s.section}" if s.section else ""
                print(f"  {s.stem:32s} {sec:6s} {s.caption}")
        return 0

    if args.figure:
        specs = [by_name(n) for n in args.figure]
    elif args.tier:
        specs = list(by_tier(args.tier))
    else:
        specs = [s for s in FIGURES if s.tier in ("manuscript", "si")]

    ok = sum(_render(s, ctx) for s in specs)
    print(f"[figures] rendered {ok}/{len(specs)}")
    if args.contact_sheet:
        _contact_sheet(specs, ctx)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
