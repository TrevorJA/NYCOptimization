"""The project figure style is defined once, in ``src/plotting/style.py``.

Source-level contract, checked on every figure module and script:

1. No call passes a literal (or downward-derived) font size. Text inherits the
   rcParams of ``style.apply_style``; a figure that needs larger type raises
   the floor with ``plt.rc_context(style.font_rc(n))``, and a size that must
   differ within a figure is derived upward from ``style.MIN_FONTSIZE`` or the
   active rcParams.
2. Font-size rcParams are written only by ``style.py``.
3. Figures are written only by ``style.save_figure`` and
   ``style.save_manuscript_figure``, which enforce the type floor.

and the behaviour of the style itself (the 12 pt floor).
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.plotting import style

REPO = Path(__file__).resolve().parents[1]
STYLE_FILE = REPO / "src" / "plotting" / "style.py"

#: Keyword arguments that set a text size.
SIZE_KWARGS = {"fontsize", "labelsize", "title_fontsize", "titlesize"}

#: Methods that set a text size on an artist.
SIZE_SETTERS = {"set_fontsize", "set_size"}

#: rcParams keys that set a text size.
FONT_RC_KEYS = {
    "font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
    "ytick.labelsize", "legend.fontsize", "legend.title_fontsize",
    "figure.titlesize", "figure.labelsize",
}


def _figure_sources() -> list[Path]:
    """Every source file that may draw a figure, ``style.py`` excepted."""
    files = [p for root in ("src", "scripts") for p in (REPO / root).rglob("*.py")]
    return sorted(p for p in files
                  if p != STYLE_FILE and "temporary" not in p.parts
                  and "__pycache__" not in p.parts)


def _is_fixed_size(node: ast.AST) -> bool:
    """A literal size, or one derived downward from another size."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, str)) and not isinstance(node.value, bool)
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, (ast.Sub, ast.Div, ast.FloorDiv)) or (
            _is_fixed_size(node.left) and _is_fixed_size(node.right))
    return False


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(REPO).as_posix()
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in SIZE_KWARGS and _is_fixed_size(kw.value):
                    found.append(f"{rel}:{kw.value.lineno} fixed {kw.arg}=")
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in SIZE_SETTERS and node.args and _is_fixed_size(node.args[0]):
                    found.append(f"{rel}:{node.lineno} fixed .{func.attr}()")
                if func.attr == "savefig":
                    found.append(f"{rel}:{node.lineno} direct .savefig(); use style.save_figure")
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value in FONT_RC_KEYS:
                    found.append(f"{rel}:{key.lineno} font rcParam {key.value!r} set "
                                 "outside style.py; use style.font_rc")
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            key = node.slice
            if isinstance(key, ast.Constant) and key.value in FONT_RC_KEYS:
                found.append(f"{rel}:{node.lineno} font rcParam {key.value!r} assigned "
                             "outside style.py; use style.font_rc")
    return found


def test_no_figure_code_sets_its_own_type_size_or_saves_directly():
    found = [v for path in _figure_sources() for v in _violations(path)]
    assert not found, "figure-style violations:\n  " + "\n  ".join(found)


def test_style_helpers_take_no_font_size():
    """The drawing helpers of ``style.py`` inherit the rcParams like any figure code."""
    tree = ast.parse(STYLE_FILE.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found += [f"{node.name}({a.arg}=...)" for a in node.args.args + node.args.kwonlyargs
                      if a.arg in SIZE_KWARGS]
        elif isinstance(node, ast.Call):
            found += [f"line {kw.value.lineno}: {kw.arg}=" for kw in node.keywords
                      if kw.arg in SIZE_KWARGS and _is_fixed_size(kw.value)]
    assert not found, f"style.py helpers set a type size: {found}"


def test_every_type_size_of_the_style_is_at_or_above_the_floor():
    assert style.MIN_FONTSIZE == 12
    assert set(style.font_rc()) >= FONT_RC_KEYS - {"figure.labelsize"}
    assert min(style.font_rc().values()) == style.MIN_FONTSIZE
    with plt.rc_context():
        style.apply_style()
        for key in style.font_rc():
            assert plt.rcParams[key] >= style.MIN_FONTSIZE, key


def test_the_floor_can_be_raised_but_not_lowered():
    assert min(style.font_rc(14).values()) == 14
    with pytest.raises(ValueError):
        style.font_rc(style.MIN_FONTSIZE - 1)


def test_saving_enforces_the_floor_on_text_built_outside_the_style(tmp_path):
    with plt.rc_context({"font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        note = ax.text(0.5, 0.5, "note", fontsize=6)  # noqa: the case under test
        with pytest.warns(UserWarning, match="raised to the floor"):
            written = style.save_figure(fig, tmp_path / "small")
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text()]
    plt.close(fig)
    assert note.get_fontsize() == style.MIN_FONTSIZE
    assert min(sizes) >= style.MIN_FONTSIZE
    assert [p.name for p in written] == ["small.png"] and written[0].exists()


def test_a_figure_built_under_the_style_saves_without_a_warning(tmp_path):
    with plt.rc_context():
        style.apply_style()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_title("title")
        ax.set_xlabel("x")
        ax.legend(["series"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            style.save_manuscript_figure(fig, tmp_path / "styled")
        plt.close(fig)
