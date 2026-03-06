"""
reference_set.py - Load Borg/MOEAFramework reference and solution set files.
"""

import numpy as np
from pathlib import Path


def load_reference_set(ref_file: Path, n_vars: int) -> tuple:
    """Load a reference set file (.ref) and split into DVs and objectives.

    Reference set files contain whitespace-delimited lines of:
        var1 var2 ... varN obj1 obj2 ... objM

    Args:
        ref_file: Path to .ref file.
        n_vars: Number of decision variables (to split columns).

    Returns:
        Tuple of (dv_array, obj_array) where:
            dv_array: shape (n_solutions, n_vars)
            obj_array: shape (n_solutions, n_objs)
    """
    data = _parse_set_file(ref_file)
    if data.shape[0] == 0:
        return np.empty((0, n_vars)), np.empty((0, 0))
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
