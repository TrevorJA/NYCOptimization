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

The campaign runs seed 1 to a longer budget than the later seeds
(``MOEAConfig.max_evaluations_by_seed``) and reports every seed at the shorter
budget, so seed 1's equal-NFE archive is the union of its islands' snapshots
at that NFE, epsilon-box filtered under the campaign vector. These are the
pure functions behind ``scripts/main/extract_runtime_archive.py``; they know
nothing about config so they are testable on synthetic text.

Objective values in runtime files and ``.set`` files are in Borg's internal
convention (maximize objectives negated), the same convention
``src.sensitivity_common.epsilon_nondominated`` expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def parse_runtime_snapshots(text: str) -> dict[int, np.ndarray]:
    """Parse a Borg runtime file into ``{nfe: rows}``.

    Args:
        text: Full runtime-file contents.

    Returns:
        Mapping from the ``//NFE=`` marker of each snapshot to a float array of
        shape ``(n_solutions, n_vars + n_objs)``; an empty archive yields a
        ``(0, 0)`` array. Later duplicate markers overwrite earlier ones.
    """
    snapshots: dict[int, np.ndarray] = {}
    nfe: int | None = None
    rows: list[list[float]] = []

    def _close() -> None:
        if nfe is not None:
            snapshots[nfe] = (np.asarray(rows, dtype=float)
                              if rows else np.empty((0, 0)))

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("//NFE="):
            _close()
            nfe = int(line.split("=", 1)[1])
            rows = []
        elif line.startswith("//") or not line:
            continue
        elif line == "#":
            _close()
            nfe, rows = None, []
        else:
            rows.append([float(x) for x in line.split()])
    _close()
    return snapshots


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
