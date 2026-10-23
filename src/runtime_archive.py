"""
runtime_archive.py - Read Borg runtime files and build equal-NFE archives.

MM Borg writes one runtime file per island (``seed_{SS}_{slug}_{i}.runtime``)
holding the island's epsilon archive at every ``runtime_frequency`` NFE::

    //NFE=2500
    //ElapsedTime=...
    //<operator probabilities, Improvements, Restarts, ...>
    <var_1> ... <var_n> <obj_1> ... <obj_m>      (feasible solutions only)
    ...
    #

The controller (coordinator) node writes nothing: its global epsilon archive
is only returned at shutdown as the seed's ``.set`` file. Step 07 therefore
reconstructs the coordinator archive at every snapshot as the epsilon-box
merge of the island archives recorded at that per-island NFE
(``coordinator_series``). Every merge in the project (this series, the
diagnostics reference set, the equal-NFE campaign set) uses the one
epsilon-box archive implementation ``src.sensitivity_common
.epsilon_nondominated``, validated against Borg's own archive membership;
MOEAFramework is used only to score the series (MetricsEvaluator). Its
mergers are not used: in v5.0 ``ResultFileMerger`` ignores ``--epsilon`` and
``ResultFileSeedMerger`` retains boxes Borg's rule rejects. The pure
functions here parse Borg runtime and MOEAFramework result files, check that
the island snapshot grids agree, build and write the series, and compare two
archives box by box; they know nothing about config so they are testable on
synthetic text.

The campaign runs seed 1 to a longer budget than the later seeds
(``MOEAConfig.max_evaluations_by_seed``) and reports every seed at the shorter
budget, so seed 1's equal-NFE archive is the union of its islands' snapshots
at that NFE, epsilon-box filtered under the campaign vector
(``scripts/main/extract_runtime_archive.py``).

Objective values in runtime files and ``.set`` files are in Borg's internal
convention (maximize objectives negated), the same convention
``src.sensitivity_common.epsilon_nondominated`` expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_result_file(text: str) -> list[tuple[dict[str, str], np.ndarray]]:
    """Parse a Borg runtime file or a MOEAFramework result file into entries.

    Both layouts share one grammar: optional ``# key=value`` header comment
    lines, then entries made of ``//key=value`` metadata lines followed by
    whitespace-delimited solution rows and closed by a line holding only
    ``#``. A trailing entry without its terminator is kept.

    Args:
        text: Full file contents.

    Returns:
        Entries in file order as ``(metadata, rows)``; ``rows`` has shape
        ``(n_solutions, n_columns)`` or ``(0, 0)`` for an empty archive.
    """
    entries: list[tuple[dict[str, str], np.ndarray]] = []
    meta: dict[str, str] = {}
    rows: list[list[float]] = []
    in_entry = False

    def _close() -> None:
        entries.append((meta, np.asarray(rows, dtype=float) if rows
                        else np.empty((0, 0))))

    for raw in text.splitlines():
        line = raw.strip()
        if line == "#":
            if in_entry:
                _close()
            else:
                entries.append(({}, np.empty((0, 0))))
            meta, rows, in_entry = {}, [], False
        elif not line:
            continue
        elif line.startswith("//"):
            in_entry = True
            key, _, value = line[2:].partition("=")
            meta[key.strip()] = value.strip()
        elif line.startswith("#"):
            if in_entry:
                raise ValueError(f"header comment inside an entry: {line!r}")
            continue
        else:
            in_entry = True
            rows.append([float(x) for x in line.split()])
    if in_entry:
        _close()
    return entries


def parse_runtime_snapshots(text: str) -> dict[int, np.ndarray]:
    """Parse a Borg runtime file into ``{nfe: rows}``.

    Args:
        text: Full runtime-file contents.

    Returns:
        Mapping from the ``//NFE=`` marker of each snapshot to a float array of
        shape ``(n_solutions, n_vars + n_objs)``; an empty archive yields a
        ``(0, 0)`` array. Later duplicate markers overwrite earlier ones;
        entries without an NFE marker are ignored.
    """
    return {int(meta["NFE"]): rows
            for meta, rows in parse_result_file(text) if "NFE" in meta}


def nfe_markers(text: str) -> list[int]:
    """The ``//NFE=`` values of a file's entries, in file order."""
    return [int(meta["NFE"]) for meta, _ in parse_result_file(text)
            if "NFE" in meta]


def island_nfe_grid(files: Iterable[Path]) -> list[int]:
    """The snapshot NFE grid shared by a seed's island runtime files.

    The per-snapshot coordinator archive merges the islands' k-th entries,
    which is only meaningful when every island recorded the same NFE
    sequence.

    Args:
        files: The seed's per-island runtime files.

    Returns:
        The common ``//NFE=`` sequence.

    Raises:
        ValueError: If no files are given, a file has no snapshots, or the
            sequences differ (the message names the first differing file).
    """
    files = [Path(f) for f in files]
    if not files:
        raise ValueError("no island runtime files")
    grids = {f: nfe_markers(f.read_text()) for f in files}
    ref_file, ref = files[0], grids[files[0]]
    if not ref:
        raise ValueError(f"{ref_file.name}: no runtime snapshots")
    for f in files[1:]:
        if grids[f] != ref:
            raise ValueError(
                f"island snapshot grids differ: {ref_file.name} has "
                f"{len(ref)} snapshots ({ref[0]}..{ref[-1]}), {f.name} has "
                f"{len(grids[f])} ({grids[f][0] if grids[f] else '-'}.."
                f"{grids[f][-1] if grids[f] else '-'})")
    return ref


def snapshot_at(files: Iterable[Path], nfe: int) -> np.ndarray:
    """Union of the island archives recorded exactly at ``nfe``.

    Args:
        files: The seed's per-island runtime files.
        nfe: Per-island NFE of the snapshot to extract.

    Returns:
        Stacked rows ``(n_solutions, n_vars + n_objs)`` across islands.

    Raises:
        KeyError: If any file lacks a snapshot at ``nfe`` (the message lists
            that file's recorded NFE values).
    """
    blocks = []
    for f in files:
        snaps = parse_runtime_snapshots(Path(f).read_text())
        if nfe not in snaps:
            raise KeyError(
                f"{Path(f).name}: no snapshot at NFE={nfe}; recorded NFE values "
                f"run {min(snaps) if snaps else '-'}..{max(snaps) if snaps else '-'} "
                f"({len(snaps)} snapshots)")
        if snaps[nfe].size:
            blocks.append(snaps[nfe])
    if not blocks:
        return np.empty((0, 0))
    return np.vstack(blocks)


# ---------------------------------------------------------------------------
# Epsilon-box archives
# ---------------------------------------------------------------------------

def epsilon_merge(rows: np.ndarray, n_vars: int,
                  epsilons: Sequence[float]) -> np.ndarray:
    """Epsilon-box nondominated subset of ``rows`` (Borg archive convention).

    Args:
        rows: ``(n, n_vars + n_objs)`` array, objectives minimized.
        n_vars: Number of decision-variable columns.
        epsilons: Campaign epsilon vector, one per objective.

    Returns:
        The retained rows, in their original order.
    """
    from src.sensitivity_common import epsilon_nondominated

    if rows.size == 0:
        return rows
    keep = np.sort(epsilon_nondominated(rows[:, n_vars:], epsilons))
    return rows[keep]


def epsilon_boxes(rows: np.ndarray, n_vars: int,
                  epsilons: Sequence[float]) -> set[tuple[int, ...]]:
    """The set of epsilon boxes ``floor(f / eps)`` occupied by ``rows``."""
    if rows.size == 0:
        return set()
    objs = np.asarray(rows, dtype=float)[:, n_vars:]
    eps = np.asarray(epsilons, dtype=float)
    boxes = np.floor(objs / eps).astype(int)
    return {tuple(b) for b in boxes.tolist()}


def compare_archives(rows_a: np.ndarray, rows_b: np.ndarray, n_vars: int,
                     epsilons: Sequence[float]) -> dict:
    """Compare two epsilon archives of the same solutions box by box.

    Two implementations of epsilon-box archiving over the same solutions must
    occupy the same nondominated boxes; they may differ in which solution
    represents a box only when two members tie in distance to the box corner.

    Args:
        rows_a: ``(n, n_vars + n_objs)`` archive (e.g. the MOEAFramework
            merge).
        rows_b: ``(m, n_vars + n_objs)`` archive (e.g. the Python merge).
        n_vars: Number of decision-variable columns.
        epsilons: Epsilon vector, one per objective.

    Returns:
        ``{"n_a", "n_b", "boxes_only_a", "boxes_only_b",
        "representatives_differ"}``: sizes, the boxes present in one archive
        but not the other, and the number of shared boxes whose representative
        rows are not numerically equal.
    """
    boxes_a = epsilon_boxes(rows_a, n_vars, epsilons)
    boxes_b = epsilon_boxes(rows_b, n_vars, epsilons)
    shared = boxes_a & boxes_b

    def _by_box(rows):
        out = {}
        if rows.size:
            eps = np.asarray(epsilons, dtype=float)
            for row in np.asarray(rows, dtype=float):
                out[tuple(np.floor(row[n_vars:] / eps).astype(int).tolist())] = row
        return out

    rep_a, rep_b = _by_box(rows_a), _by_box(rows_b)
    differ = sum(1 for b in shared if not np.allclose(rep_a[b], rep_b[b],
                                                      rtol=1e-9, atol=0.0))
    return {
        "n_a": int(len(rows_a)) if rows_a.size else 0,
        "n_b": int(len(rows_b)) if rows_b.size else 0,
        "boxes_only_a": sorted(boxes_a - boxes_b),
        "boxes_only_b": sorted(boxes_b - boxes_a),
        "representatives_differ": int(differ),
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def coordinator_series(files: Iterable[Path], n_vars: int,
                       epsilons: Sequence[float]) -> list[tuple[int, np.ndarray]]:
    """The coordinator archive at every snapshot of one seed.

    Entry k is the epsilon-box merge of every island's archive recorded at
    the k-th per-island NFE of the common grid: the controller's global
    archive at that NFE, free of the controller's update lag.

    Args:
        files: The seed's per-island runtime files.
        n_vars: Number of decision-variable columns.
        epsilons: Campaign epsilon vector, one per objective.

    Returns:
        ``[(island_nfe, rows)]`` in grid order; ``rows`` is ``(0, 0)`` when
        no island had a feasible solution yet.
    """
    files = [Path(f) for f in files]
    grid = island_nfe_grid(files)
    snaps = [parse_runtime_snapshots(f.read_text()) for f in files]
    series = []
    for nfe in grid:
        blocks = [s[nfe] for s in snaps if s[nfe].size]
        union = np.vstack(blocks) if blocks else np.empty((0, 0))
        series.append((nfe, epsilon_merge(union, n_vars, epsilons)))
    return series


def moea_result_header(problem: str, n_vars: int, n_objs: int,
                       bounds: tuple[float, float] = (-1000000.0, 1000000.0)) -> list[str]:
    """The MOEAFramework v5 result-file header for one of the ``drb_*`` problems.

    Mirrors what MOEAFramework writes for the problem JARs built by
    ``workflow/00_setup_borg_jars.sh`` (real variables on ``bounds``, all
    objectives minimized in Borg's sign convention, no constraint columns).
    """
    lo, hi = bounds
    lines = [f"# Version=5", f"# Problem={problem}",
             f"# NumberOfVariables={n_vars}", f"# NumberOfObjectives={n_objs}",
             "# NumberOfConstraints=0"]
    lines += [f"# Variable.{i + 1}.Definition=RealVariable({lo},{hi})" for i in range(n_vars)]
    lines += [f"# Objective.{i + 1}.Definition=Minimize" for i in range(n_objs)]
    return lines


def write_result_file(path: Path, entries: Sequence[tuple[dict, np.ndarray]],
                      header_lines: Sequence[str] = ()) -> None:
    """Write entries in the MOEAFramework result-file layout.

    Each entry is written as its ``//key=value`` metadata lines, its rows at
    full double precision, and the ``#`` terminator, after the ``#`` header
    comment lines (``moea_result_header``). Readable by ``parse_result_file``
    and by the MOEAFramework CLI.

    Args:
        path: Output file.
        entries: ``[(metadata, rows)]``; an empty ``rows`` writes an entry
            with no solutions.
        header_lines: Lines written verbatim before the first entry.
    """
    with open(path, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        for meta, rows in entries:
            for k, v in meta.items():
                f.write(f"//{k}={v}\n")
            if rows.size:
                for row in np.asarray(rows, dtype=float):
                    f.write(" ".join(f"{v:.17g}" for v in row) + "\n")
            f.write("#\n")


def write_set_file(path: Path, rows: np.ndarray, var_names: Sequence[str],
                   obj_names: Sequence[str], header_lines: Sequence[str] = ()) -> None:
    """Write rows in the ``.set`` layout of ``src.mmborg._write_set_file``.

    Args:
        path: Output ``.set`` file.
        rows: ``(n, n_vars + n_objs)`` array.
        var_names: Decision-variable names (header only).
        obj_names: Objective names (header only).
        header_lines: Extra ``#``-prefixed provenance lines.
    """
    n_vars = len(var_names)
    with open(path, "w") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        f.write(f"# Variables: {','.join(var_names)}\n")
        f.write(f"# Objectives: {','.join(obj_names)}\n")
        for row in rows:
            f.write(" ".join(f"{v:.6e}" for v in row[:n_vars])
                    + " " + " ".join(f"{o:.6e}" for o in row[n_vars:]) + "\n")


# ---------------------------------------------------------------------------
# Per-seed equal-NFE sets
# ---------------------------------------------------------------------------

def equal_nfe_seed_set(sets_dir: Path, slug: str, seed: int, budget: int,
                       nfe: int) -> Path:
    """The file holding a seed's archive at the reporting NFE.

    A seed whose per-island budget equals the reporting NFE is reported by
    its final controller archive (``seed_SS_{slug}.set``); a seed run beyond
    it is reported by the extraction of its runtime snapshot at the reporting
    NFE (``seed_SS_{slug}_nfe{nfe}.set``). Both the equal-NFE merge and the
    per-seed figures resolve through this one function.

    Args:
        sets_dir: ``outputs/{scenario}/{slug}/sets``.
        slug: The moea slug.
        seed: Seed number (1-indexed).
        budget: The seed's per-island NFE budget.
        nfe: The reporting per-island NFE.

    Raises:
        ValueError: If the budget is below the reporting NFE.
    """
    if budget < nfe:
        raise ValueError(f"seed {seed} budget {budget} is below the reporting NFE {nfe}")
    if budget == nfe:
        return Path(sets_dir) / f"seed_{seed:02d}_{slug}.set"
    return Path(sets_dir) / f"seed_{seed:02d}_{slug}_nfe{nfe}.set"
