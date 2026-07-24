"""
reference_set.py - Load Borg/MOEAFramework reference and solution set files.
"""

import numpy as np
from pathlib import Path


def load_reference_set(ref_file: Path, n_vars: int, n_objs: int = None) -> tuple:
    """Load a reference set file (.ref) and split into DVs and objectives.

    Reference set files contain whitespace-delimited lines of:
        var1 var2 ... varN obj1 obj2 ... objM

    Rows are feasible solutions only, with no constraint columns — both the
    MM Borg C writer and mmborg._write_set_file strip constraint violators
    and emit variables + objectives.

    Args:
        ref_file: Path to .ref file.
        n_vars: Number of decision variables (to split columns).
        n_objs: Optional expected objective count. When given, raises if the
            file's column count is not exactly ``n_vars + n_objs`` — a
            fail-loud guard against format drift (e.g. a constraint-declaring
            problem definition producing merged sets with extra columns).

    Returns:
        Tuple of (dv_array, obj_array) where:
            dv_array: shape (n_solutions, n_vars)
            obj_array: shape (n_solutions, n_objs)
    """
    data = _parse_set_file(ref_file)
    if data.shape[0] == 0:
        return np.empty((0, n_vars)), np.empty((0, 0))
    if n_objs is not None and data.shape[1] != n_vars + n_objs:
        raise ValueError(
            f"{ref_file}: expected {n_vars} vars + {n_objs} objs = "
            f"{n_vars + n_objs} columns, found {data.shape[1]}."
        )
    return data[:, :n_vars], data[:, n_vars:]


def load_set_file(set_file: Path) -> np.ndarray:
    """Load a .set file as a raw numpy array (all columns)."""
    return _parse_set_file(set_file)


def _parse_set_file(filepath: Path) -> np.ndarray:
    """Parse a whitespace-delimited set/ref file, skipping comment lines."""
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            try:
                values = [float(x) for x in line.split()]
                rows.append(values)
            except ValueError:
                continue
    return np.array(rows) if rows else np.empty((0, 0))
