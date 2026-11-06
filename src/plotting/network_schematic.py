"""
network_schematic.py - Manuscript Figure 1 (schematic panel): the Pywr-DRB
components the NYC re-optimization acts on.

Drawn in the visual language of Quinn et al. (2018, Figure 1b): catchments as
pentagons, reservoirs as triangles, the estuary as a trapezoid, point
locations where outcomes are computed as filled markers, and every flow as a
labelled right-angled arrow whose symbol carries a site superscript.

The figure is written directly as SVG (no matplotlib) so every node, line and
label is an editable element grouped by role (``<g id="...">``). The SVG user
unit is one typographic point, so a ``font-size`` of 14 is 14 pt at 100 %
zoom; nothing is set below :data:`FS_MIN`. A PNG copy is rasterized through a
headless Chromium (Edge / Chrome) when one is found, which is what the figure
registry (``src.figures.registry``) reports on.

Emphasis follows the problem formulation: the three NYC reservoirs, the NYC
diversion, the Montague and Trenton flow targets, the NJ diversion and the
three reservoir-tail flood gauges are drawn in full; the non-NYC tributaries,
whose releases are policy-independent boundary inflows, are aggregated into
two grey catchments.

Run::

    python -m src.plotting.network_schematic            # -> figures/manuscript/
    python -m src.plotting.network_schematic --out my_stub
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import config

# ---------------------------------------------------------------------------
# Canvas and style (user unit = 1 pt)
# ---------------------------------------------------------------------------

WIDTH: int = 640
HEIGHT: int = 604

FONT: str = "Helvetica Neue, Helvetica, Arial, sans-serif"
FS_MIN: int = 14         # smallest type anywhere (superscripts included)
FS_NODE: int = 14        # names inside / beside nodes, legend
FS_SYMBOL: int = 17      # flow / state symbols (q, s, r, d, Q)
FS_SUP: int = 14         # their site superscripts

STROKE: float = 1.3
BLACK: str = "#000000"
NODE_FILL: str = "#FFFFFF"
RESERVOIR_FILL: str = "#D9D9D9"
GREY: str = "#8A8A8A"        # de-emphasized (non-NYC) lines and outlines
GREY_TEXT: str = "#6B6B6B"
TARGET_RED: str = "#E3211B"  # flow-target gauges
TARGET_TEXT: str = "#B51D18"
FLOOD_BLACK: str = "#1A1A1A"

#: Per-reservoir colour (Okabe-Ito; the storage symbol inside each triangle and
#: the name beside it share the colour, as in the reference figure).
RESERVOIR_COLORS: dict[str, str] = {
    "cannonsville": "#0072B2",
    "pepacton": "#D55E00",
    "neversink": "#009E73",
}

# ---------------------------------------------------------------------------
# Layout (pt). Columns are x-centres; rows are y-centres or edges as named.
# ---------------------------------------------------------------------------

X_CAN, X_PEP, X_NEV = 140, 296, 432   # NYC reservoir columns
X_MAIN = (X_CAN + X_PEP) // 2         # Delaware mainstem below the confluence
X_TRIB = 335                          # aggregated non-NYC catchments
X_NJ = 335                            # NJ demand box
X_NYC = 576                           # NYC demand box

Y_CATCH = 45          # catchment pentagon centre
PENT_W, PENT_H = 100, 70
Y_RES_TOP = 120       # reservoir triangle apex
RES_W, RES_H = 56, 50
Y_FLOOD = 212         # flood-gauge square centre
FLOOD_SZ = 10
Y_CONFL = 250         # West / East Branch confluence bar
Y_TRIB_UP = 295       # Lackawaxen-Mongaup inflow joins the mainstem
Y_NEV_JOIN = 340      # Neversink joins the mainstem
Y_MONTAGUE = 376
Y_TRIB_LOW = 420      # Lehigh and lower-basin inflow joins the mainstem
Y_NJ = 482            # NJ diversion take-off
Y_TRENTON = 520
Y_EST_TOP = 548       # estuary trapezoid top edge
EST_W_TOP, EST_W_BOT, EST_H = 70, 110, 45
TARGET_R = 6

# NYC system enclosure (dashed) and the diversion leaving it.
ENC_X0, ENC_X1 = 12, 468
ENC_Y0, ENC_Y1 = 113, 178
Y_DIVERT = Y_RES_TOP + RES_H // 2
BOX_W, BOX_H = 108, 34

# Legend box.
LEG_X0, LEG_Y0, LEG_W = 410, 376, 220
LEG_ROW = 24


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _pts(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in points)


def pentagon(cx: float, cy: float, w: float = PENT_W, h: float = PENT_H
             ) -> list[tuple[float, float]]:
    """House-shaped pentagon (apex up) centred at ``(cx, cy)``."""
    hw, hh = w / 2, h / 2
    shoulder = cy - hh + 0.38 * h
    return [(cx, cy - hh), (cx + hw, shoulder), (cx + hw, cy + hh),
            (cx - hw, cy + hh), (cx - hw, shoulder)]


def triangle(cx: float, y_top: float, w: float = RES_W, h: float = RES_H
             ) -> list[tuple[float, float]]:
    """Isosceles triangle, apex up at ``(cx, y_top)``."""
    return [(cx, y_top), (cx + w / 2, y_top + h), (cx - w / 2, y_top + h)]


def trapezoid(cx: float, y_top: float, w_top: float, w_bot: float, h: float
              ) -> list[tuple[float, float]]:
    """Trapezoid widening downward from ``y_top``."""
    return [(cx - w_top / 2, y_top), (cx + w_top / 2, y_top),
            (cx + w_bot / 2, y_top + h), (cx - w_bot / 2, y_top + h)]


def polygon(points: Sequence[tuple[float, float]], fill: str = NODE_FILL,
            stroke: str = BLACK, width: float = STROKE) -> str:
    return (f'<polygon points="{_pts(points)}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{_fmt(width)}" stroke-linejoin="round"/>')


def rect(x: float, y: float, w: float, h: float, fill: str = NODE_FILL,
         stroke: str = BLACK, width: float = STROKE, rx: float = 0,
         dashed: bool = False) -> str:
    dash = ' stroke-dasharray="4,3"' if dashed else ""
    return (f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" '
            f'height="{_fmt(h)}" rx="{_fmt(rx)}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{_fmt(width)}"{dash}/>')


def path(points: Sequence[tuple[float, float]], stroke: str = BLACK,
         width: float = STROKE, arrow: bool = True, dashed: bool = False) -> str:
    """Open polyline with an optional arrowhead at its last point."""
    key = "grey" if stroke == GREY else "black"
    marker = f' marker-end="url(#arrow-{key})"' if arrow else ""
    dash = ' stroke-dasharray="5,3.5"' if dashed else ""
    return (f'<polyline points="{_pts(points)}" fill="none" stroke="{stroke}" '
            f'stroke-width="{_fmt(width)}" stroke-linejoin="round" '
            f'stroke-linecap="butt"{dash}{marker}/>')


def text(x: float, y: float, s: str, size: int = FS_NODE, anchor: str = "start",
         color: str = BLACK) -> str:
    """One text element; ``^{...}`` in ``s`` renders as a raised superscript
    of :data:`FS_SUP` points."""
    assert size >= FS_MIN, f"type below {FS_MIN} pt: {s!r}"
    return (f'<text x="{_fmt(x)}" y="{_fmt(y)}" font-family="{FONT}" '
            f'font-size="{size}" text-anchor="{anchor}" fill="{color}">'
            f'{_rich(s, size)}</text>')


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _rich(s: str, size: int) -> str:
    """Expand ``base^{sup}`` runs into tspans."""
    shift = 0.33 * size
    out, i, raised = [], 0, False
    while i < len(s):
        j = s.find("^{", i)
        if j < 0:
            chunk = s[i:]
            out.append(f'<tspan dy="{_fmt(shift)}">{_esc(chunk)}</tspan>'
                       if raised and chunk else _esc(chunk))
            break
        chunk = s[i:j]
        if chunk:
            out.append(f'<tspan dy="{_fmt(shift)}">{_esc(chunk)}</tspan>'
                       if raised else _esc(chunk))
        k = s.index("}", j)
        out.append(f'<tspan dy="{_fmt(-shift)}" font-size="{FS_SUP}">'
                   f'{_esc(s[j + 2:k])}</tspan>')
        raised = True
        i = k + 1
    return "".join(out)


def multiline(cx: float, cy: float, lines: Sequence[str], size: int = FS_NODE,
              color: str = BLACK, anchor: str = "middle", lead: float = 1.18
              ) -> str:
    """Vertically centred block of text lines."""
    step = lead * size
    y0 = cy - step * (len(lines) - 1) / 2 + 0.36 * size
    return "".join(text(cx, y0 + i * step, ln, size=size, anchor=anchor,
                        color=color) for i, ln in enumerate(lines))


def group(gid: str, parts: Iterable[str]) -> str:
    body = "\n    ".join(p for p in parts if p)
    return f'  <g id="{gid}">\n    {body}\n  </g>'


# ---------------------------------------------------------------------------
# Figure content
# ---------------------------------------------------------------------------

def _defs() -> str:
    def marker(key: str, color: str) -> str:
        return (f'<marker id="arrow-{key}" viewBox="0 0 10 10" refX="10" '
                f'refY="5" markerWidth="6.5" markerHeight="6.5" '
                f'orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')
    return ("  <defs>\n    " + marker("black", BLACK) + "\n    "
            + marker("grey", GREY) + "\n  </defs>")


def _square(cx: float, cy: float, size: float = FLOOD_SZ) -> str:
    return rect(cx - size / 2, cy - size / 2, size, size, fill=FLOOD_BLACK,
                stroke="none", width=0)


def _circle(cx: float, cy: float) -> str:
    return (f'<circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{TARGET_R}" '
            f'fill="{TARGET_RED}" stroke="{BLACK}" stroke-width="0.9"/>')


def _reservoir_column(key: str, x: float, catchment: Sequence[str],
                      sym: str, name: str, gauge_name: str
                      ) -> tuple[list[str], list[str], list[str]]:
    """Catchment -> reservoir -> flood gauge for one NYC reservoir.

    Returns the parts for the catchment, reservoir and gauge groups.
    """
    color = RESERVOIR_COLORS[key]
    pent_bot = Y_CATCH + PENT_H / 2
    res_bot = Y_RES_TOP + RES_H
    catch = [
        polygon(pentagon(x, Y_CATCH)),
        multiline(x, Y_CATCH + 5, catchment),
        path([(x, pent_bot), (x, Y_RES_TOP - 1)]),
        text(x - 5, (pent_bot + Y_RES_TOP) / 2 + 6, f"q^{{{sym}}}",
             size=FS_SYMBOL, anchor="end"),
    ]
    res = [
        polygon(triangle(x, Y_RES_TOP), fill=RESERVOIR_FILL),
        text(x, res_bot - 6, f"s^{{{sym}}}", size=FS_SYMBOL, anchor="middle",
             color=color),
        text(x - RES_W / 2 - 5, Y_RES_TOP + RES_H * 0.78, name, anchor="end",
             color=color),
    ]
    gauge = [
        path([(x, res_bot), (x, Y_FLOOD - FLOOD_SZ / 2 - 1)]),
        text(x - 5, (res_bot + Y_FLOOD) / 2 + 6, f"r^{{{sym}}}",
             size=FS_SYMBOL, anchor="end"),
        _square(x, Y_FLOOD),
        text(x + FLOOD_SZ / 2 + 4, Y_FLOOD + 5, gauge_name),
    ]
    return catch, res, gauge


def _target(y: float, sym: str, name: str) -> list[str]:
    """A flow-target gauge on the mainstem: red point, symbol left, name right."""
    return [
        _circle(X_MAIN, y),
        text(X_MAIN - TARGET_R - 5, y + 6, f"Q^{{{sym}}}", size=FS_SYMBOL,
             anchor="end", color=TARGET_TEXT),
        text(X_MAIN + TARGET_R + 5, y + 5, name, color=TARGET_TEXT),
    ]


def _grey_catchment(y: float, lines: Sequence[str], sym: str) -> list[str]:
    """An aggregated non-NYC catchment joining the mainstem from the right."""
    w = PENT_W + 10
    left = X_TRIB - w / 2
    return [
        polygon(pentagon(X_TRIB, y, w), stroke=GREY),
        multiline(X_TRIB, y + 5, lines, color=GREY_TEXT),
        path([(left, y), (X_MAIN + 1, y)], stroke=GREY),
        text((left + X_MAIN) / 2, y - 5, f"q^{{{sym}}}", size=FS_SYMBOL,
             anchor="middle", color=GREY_TEXT),
    ]


def _demand_box(cx: float, cy: float, label: str) -> str:
    return (rect(cx - BOX_W / 2, cy - BOX_H / 2, BOX_W, BOX_H, rx=3)
            + text(cx, cy + 5, label, anchor="middle"))


def _legend() -> list[str]:
    sx = LEG_X0 + 24          # symbol centre
    tx = LEG_X0 + 46          # label start

    def row(i: int) -> float:
        return LEG_Y0 + 18 + LEG_ROW * i

    y = row(0); r0 = (polygon(pentagon(sx, y, 24, 17)), "catchment")
    y = row(1); r1 = (polygon(triangle(sx, y - 8, 20, 16), fill=RESERVOIR_FILL),
                      "NYC reservoir")
    y = row(2); r2 = (rect(sx - 13, y - 7, 26, 14, fill="none", stroke=GREY,
                           width=1.0, rx=2, dashed=True), "NYC system")
    y = row(3); r3 = (polygon(pentagon(sx, y, 24, 17), stroke=GREY),
                      "non-NYC inflow (fixed)")
    y = row(4); r4 = (_circle(sx, y), "flow-target gauge")
    y = row(5); r5 = (_square(sx, y), "flood-stage gauge")
    y = row(6); r6 = (rect(sx - 13, y - 7, 26, 14, rx=2), "water demand")
    y = row(7); r7 = (polygon(trapezoid(sx, y - 7, 16, 26, 14), stroke=GREY),
                      "estuary")
    rows = [r0, r1, r2, r3, r4, r5, r6, r7]

    height = LEG_ROW * len(rows) + 8
    parts = [rect(LEG_X0, LEG_Y0, LEG_W, height)]
    for i, (symbol, label) in enumerate(rows):
        parts.append(symbol)
        parts.append(text(tx, row(i) + 5, label))
    return parts


def build_svg(title: str = "") -> str:
    """Return the complete SVG document as text.

    Args:
        title: Optional panel label drawn at the top-left (e.g. ``"b)"``);
            empty for none.
    """
    catch_parts, res_parts, gauge_parts = [], [], []
    for key, x, catchment, sym, name, gname in (
        ("cannonsville", X_CAN, ("West Branch", "Delaware"), "C",
         "Cannonsville", "Hale Eddy"),
        ("pepacton", X_PEP, ("East Branch", "Delaware"), "P",
         "Pepacton", "Fishs Eddy"),
        ("neversink", X_NEV, ("Neversink", "River"), "N",
         "Neversink", "Bridgeville"),
    ):
        c, r, g = _reservoir_column(key, x, catchment, sym, name, gname)
        catch_parts += c
        res_parts += r
        gauge_parts += g

    # NYC system enclosure + the aggregated diversion to the city.
    box_left = X_NYC - BOX_W / 2
    nyc_parts = [
        rect(ENC_X0, ENC_Y0, ENC_X1 - ENC_X0, ENC_Y1 - ENC_Y0, fill="none",
             stroke=GREY, width=1.0, rx=6, dashed=True),
        path([(ENC_X1, Y_DIVERT), (box_left - 1, Y_DIVERT)], dashed=True),
        text((ENC_X1 + box_left) / 2, Y_DIVERT - 5, "d^{NYC}", size=FS_SYMBOL,
             anchor="middle"),
        _demand_box(X_NYC, Y_DIVERT, "New York City"),
    ]

    # Mainstem: confluence bar, Neversink join, flow targets, estuary.
    flood_bot = Y_FLOOD + FLOOD_SZ / 2
    main_parts = [
        path([(X_CAN, flood_bot), (X_CAN, Y_CONFL), (X_PEP, Y_CONFL),
              (X_PEP, flood_bot)], arrow=False),
        path([(X_MAIN, Y_CONFL), (X_MAIN, Y_MONTAGUE - TARGET_R - 1)]),
        path([(X_NEV, flood_bot), (X_NEV, Y_NEV_JOIN), (X_MAIN + 1, Y_NEV_JOIN)]),
        *_target(Y_MONTAGUE, "Mon", "Montague"),
        path([(X_MAIN, Y_MONTAGUE + TARGET_R), (X_MAIN, Y_TRENTON - TARGET_R - 1)]),
        *_target(Y_TRENTON, "Tre", "Trenton"),
        path([(X_MAIN, Y_TRENTON + TARGET_R), (X_MAIN, Y_EST_TOP - 1)],
             stroke=GREY),
        polygon(trapezoid(X_MAIN, Y_EST_TOP, EST_W_TOP, EST_W_BOT, EST_H),
                stroke=GREY),
        multiline(X_MAIN, Y_EST_TOP + EST_H / 2, ("Delaware", "Estuary"),
                  color=GREY_TEXT),
    ]

    trib_parts = (_grey_catchment(Y_TRIB_UP, ("Lackawaxen &", "Mongaup"), "LM")
                  + _grey_catchment(Y_TRIB_LOW, ("Lehigh &", "lower basin"), "LB"))

    nj_left = X_NJ - BOX_W / 2
    nj_parts = [
        path([(X_MAIN, Y_NJ), (nj_left - 1, Y_NJ)], dashed=True),
        text((X_MAIN + nj_left) / 2, Y_NJ - 5, "d^{NJ}", size=FS_SYMBOL,
             anchor="middle"),
        _demand_box(X_NJ, Y_NJ, "New Jersey"),
    ]

    title_part = [text(10, 22, title, size=FS_SYMBOL)] if title else []

    body = "\n".join([
        _defs(),
        f'  <rect id="background" width="{WIDTH}" height="{HEIGHT}" fill="#FFFFFF"/>',
        group("title", title_part),
        group("catchments", catch_parts),
        group("nyc-system", nyc_parts),
        group("reservoirs", res_parts),
        group("flood-gauges", gauge_parts),
        group("mainstem", main_parts),
        group("non-nyc-tributaries", trib_parts),
        group("nj-diversion", nj_parts),
        group("legend", _legend()),
    ])
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}pt" '
            f'height="{HEIGHT}pt" viewBox="0 0 {WIDTH} {HEIGHT}">\n'
            f'{body}\n</svg>\n')


# ---------------------------------------------------------------------------
# Rasterization and entry points
# ---------------------------------------------------------------------------

_CHROMIUM_CANDIDATES: tuple[str, ...] = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)

#: CSS pixels per point: the SVG is sized in pt, the browser window in px.
_PX_PER_PT: float = 96 / 72


def find_chromium() -> Optional[str]:
    """A headless-capable Chromium binary: ``NYCOPT_CHROMIUM``, then PATH,
    then the standard install locations; ``None`` when there is none."""
    explicit = os.environ.get("NYCOPT_CHROMIUM", "").strip()
    if explicit:
        return explicit if Path(explicit).is_file() else None
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome",
                 "msedge"):
        hit = shutil.which(name)
        if hit:
            return hit
    for cand in _CHROMIUM_CANDIDATES:
        if Path(cand).is_file():
            return cand
    return None


def rasterize(svg_path: Path, png_path: Path, scale: int = 3,
              timeout_s: float = 60.0) -> bool:
    """Screenshot ``svg_path`` to ``png_path`` at ``scale`` x its 100 % size.

    Returns False when no Chromium binary is available or no screenshot
    appeared within ``timeout_s`` seconds.
    """
    exe = find_chromium()
    if exe is None:
        return False
    png_path = Path(png_path)
    if png_path.exists():
        png_path.unlink()
    win_w = math.ceil(WIDTH * _PX_PER_PT)
    win_h = math.ceil(HEIGHT * _PX_PER_PT)
    # The launcher process can return before the detached browser has written
    # the screenshot (Edge on Windows does), so wait for the file to appear and
    # stop growing, and only then discard the throwaway profile.
    profile = tempfile.mkdtemp(prefix="nycopt-chromium-")
    cmd = [
        exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--no-first-run", "--disable-extensions",
        f"--user-data-dir={profile}",
        f"--force-device-scale-factor={scale}",
        f"--window-size={win_w},{win_h}",
        f"--screenshot={png_path}",
        svg_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=False, timeout=timeout_s,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + timeout_s
    last_size = -1
    while time.monotonic() < deadline:
        if png_path.is_file():
            size = png_path.stat().st_size
            if size > 0 and size == last_size:
                break
            last_size = size
        time.sleep(0.25)
    time.sleep(0.5)
    shutil.rmtree(profile, ignore_errors=True)
    return png_path.is_file() and png_path.stat().st_size > 0


def write_schematic(out_stub: Path, title: str = "") -> list[Path]:
    """Write ``{out_stub}.svg`` and, when a rasterizer exists, ``{out_stub}.png``.

    Returns:
        The written paths (SVG first).
    """
    out_stub = Path(out_stub)
    out_stub.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_stub.with_suffix(".svg")
    svg_path.write_text(build_svg(title), encoding="utf-8")
    written = [svg_path]
    png_path = out_stub.with_suffix(".png")
    if rasterize(svg_path, png_path):
        written.append(png_path)
    else:
        print(f"[network_schematic] no PNG: no Chromium found or the screenshot "
              f"did not appear; wrote {svg_path.name} only (set NYCOPT_CHROMIUM "
              f"to point at a browser)", file=sys.stderr)
    return written


def build_network_schematic(ctx, out_stub: Path, table_dir: Path) -> list[Path]:
    """Figure 1 (schematic panel), registry builder: no data inputs."""
    return write_schematic(out_stub)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", type=Path,
                   default=config.MANUSCRIPT_FIG_DIR / "fig01_network_schematic",
                   help="output stub (extension added)")
    p.add_argument("--title", default="", help="optional panel label")
    args = p.parse_args(argv)
    for path_ in write_schematic(args.out, args.title):
        print(path_)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
