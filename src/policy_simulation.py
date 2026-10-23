"""policy_simulation.py - Full-output simulations of named policy sets.

The single place where a collection of policies (decision vectors) is
simulated on the historic trace with complete HDF5 output, so that system
dynamics can be plotted later without re-running the model. Three concerns are
kept separate so each can be reused on its own:

1. ``PolicySet``: a named, ordered collection of decision vectors with labels
   and provenance. Built from a reference set plus a selection table
   (``from_selection_csv``), from explicit row indices
   (``from_reference_set``), or by hand (``PolicySet({"name": dv_vector})``).
   Construction validates the DV layout (count, bounds) against the active
   formulation, so a set produced under another DV design fails fast.
2. ``simulate_or_load`` / ``simulate_policy_set``: run one policy, or every
   member of a set, to ``{out_dir}/{name}.hdf5``, reusing runs that already
   exist, and write a manifest, the decision variables, and the ACTIVE
   objective set recomputed from each run.
3. ``load_policy_runs``: load a finished directory back into memory for
   plotting (manifest + one results dict per policy).

Consumers: ``scripts/supplemental/simulate_selected_policies.py`` (batch
driver), ``scripts/main/explore_results.py`` (timeseries figures),
``scripts/supplemental/objective_dynamics_figures.py`` (cached contrasts).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import config
from config import OUTPUTS_DIR
from src.formulations import (
    get_baseline_values, get_bounds, get_n_vars, get_var_names,
)

#: Files written next to the HDF5 runs; ``load_policy_runs`` reads the manifest.
MANIFEST_NAME = "manifest.csv"
DV_TABLE_NAME = "decision_variables.csv"
OBJECTIVES_TABLE_NAME = "objectives_current.csv"
#: Policy name of the formulation's baseline (incumbent FFMP) policy.
BASELINE_NAME = "baseline"
#: Selection rules of ``scripts.main.explore_results`` whose rows form the
#: highlighted (plotted) subset; used when a selection table carries no
#: ``highlighted`` column.
HIGHLIGHTED_RULES = ("mean_scaled_dominating", "mean_scaled_balanced")
HIGHLIGHTED_RULE_PREFIX = "diverse_"


# --------------------------------------------------------------------------- #
# Reference sets and selection tables
# --------------------------------------------------------------------------- #
def read_reference_set(path: Path, formulation: str = "ffmp",
                       bound_tol: float = 1e-9) -> np.ndarray:
    """Load a Borg/MOEAFramework ``.set`` file and validate its DV layout.

    The first ``n_vars`` columns are decision variables; the remaining columns
    are the (Borg-signed) objectives of the search that produced the set and
    are not interpreted here, so the set may come from any objective set.

    Raises:
        ValueError: If the set has no objective column after the decision
            variables, or any decision variable lies outside the active
            bounds (a set produced under a different DV design).
    """
    n_vars = get_n_vars(formulation)
    arr = np.loadtxt(path, comments="#", ndmin=2)
    if arr.shape[1] <= n_vars:
        raise ValueError(
            f"{path}: {arr.shape[1]} columns, expected {n_vars} decision "
            f"variables followed by at least one objective column")
    lower, upper = (np.asarray(b, dtype=float) for b in get_bounds(formulation))
    dv = arr[:, :n_vars]
    out = (dv < lower - bound_tol) | (dv > upper + bound_tol)
    if out.any():
        bad = np.flatnonzero(out.any(axis=1))
        raise ValueError(
            f"{path}: {len(bad)} rows have decision variables outside the "
            f"active bounds (first rows: {bad[:5].tolist()}); the set was "
            f"produced under a different decision-variable design")
    return arr


def is_highlighted_rule(rule: str) -> bool:
    """True for the selection rules that form the highlighted subset."""
    return rule in HIGHLIGHTED_RULES or rule.startswith(HIGHLIGHTED_RULE_PREFIX)


def read_selection_labels(csv: Path, highlighted_only: bool = True
                          ) -> "OrderedDict[int, list]":
    """Map each unique ``row_index`` of a selection table to its labels.

    Accepts the ``selected_solutions.csv`` written by
    ``scripts.main.explore_results`` or any table with ``row_index`` and
    ``label`` columns. Rows chosen by several rules keep every label, in
    file order.

    Args:
        csv: Selection table.
        highlighted_only: Keep only the plotted subset: rows flagged in a
            ``highlighted`` column when the table has one, else rows whose
            ``rule`` is one of :data:`HIGHLIGHTED_RULES` or a diverse pick.
            False keeps every row (single-objective optima included).
    """
    df = pd.read_csv(csv)
    if highlighted_only:
        if "highlighted" in df.columns:
            df = df[df["highlighted"].astype(bool)]
        elif "rule" in df.columns:
            df = df[df["rule"].astype(str).map(is_highlighted_rule)]
    labels: "OrderedDict[int, list]" = OrderedDict()
    for idx, label in zip(df["row_index"].astype(int), df["label"].astype(str)):
        labels.setdefault(idx, [])
        if label not in labels[idx]:
            labels[idx].append(label)
    return labels


# --------------------------------------------------------------------------- #
# Policy sets
# --------------------------------------------------------------------------- #
class PolicySet:
    """A named, ordered collection of decision vectors to simulate.

    Args:
        policies: Mapping ``name -> decision vector``; names become HDF5
            stems, so they must be filesystem-safe and unique.
        labels: Optional mapping ``name -> list of descriptive labels``.
        formulation: Formulation that decodes the vectors.
        source: Free-text provenance recorded in the manifest.
    """

    def __init__(self, policies: "dict[str, np.ndarray]",
                 labels: "dict[str, list] | None" = None,
                 formulation: str = "ffmp", source: str = ""):
        n_vars = get_n_vars(formulation)
        self.policies: "OrderedDict[str, np.ndarray]" = OrderedDict()
        for name, dv in policies.items():
            vec = np.asarray(dv, dtype=float).ravel()
            if vec.size != n_vars:
                raise ValueError(
                    f"policy '{name}' has {vec.size} decision variables, "
                    f"formulation '{formulation}' expects {n_vars}")
            self.policies[name] = vec
        self.labels = {name: list((labels or {}).get(name, [name]))
                       for name in self.policies}
        self.formulation = formulation
        self.source = source

    @staticmethod
    def row_name(index: int) -> str:
        """Policy name of a reference-set row (``row_{index}``)."""
        return f"row_{int(index)}"

    @classmethod
    def from_reference_set(cls, set_file: Path, rows: "list[int]",
                           labels: "dict[int, list] | None" = None,
                           formulation: str = "ffmp") -> "PolicySet":
        """Policies given by 0-based row indices of a reference set.

        Args:
            set_file: Reference ``.set`` file (validated by
                :func:`read_reference_set`).
            rows: Row indices, as in ``selected_solutions.csv``.
            labels: Optional ``row_index -> labels`` mapping, e.g. from
                :func:`read_selection_labels`.
            formulation: Formulation that decodes the vectors.
        """
        arr = read_reference_set(set_file, formulation)
        n_vars = get_n_vars(formulation)
        rows = [int(r) for r in rows]
        missing = [r for r in rows if r < 0 or r >= len(arr)]
        if missing:
            raise IndexError(
                f"rows {missing} outside the {len(arr)} rows of {set_file}")
        names = [cls.row_name(r) for r in rows]
        policies = OrderedDict((n, arr[r, :n_vars]) for n, r in zip(names, rows))
        lab = {n: (labels or {}).get(r, [n]) for n, r in zip(names, rows)}
        return cls(policies, lab, formulation, source=f"{set_file} rows {rows}")

    @classmethod
    def from_selection_csv(cls, set_file: Path, selection_csv: Path,
                           formulation: str = "ffmp",
                           highlighted_only: bool = True) -> "PolicySet":
        """Rows of an ``explore_results`` selection table.

        Args:
            highlighted_only: Only the plotted subset (default), or every
                rule's row; see :func:`read_selection_labels`.
        """
        labels = read_selection_labels(selection_csv, highlighted_only)
        ps = cls.from_reference_set(set_file, list(labels), labels, formulation)
        ps.source = f"{set_file} via {selection_csv}"
        return ps

    def with_baseline(self, name: str = BASELINE_NAME,
                      label: str = "Incumbent FFMP") -> "PolicySet":
        """Return a copy with the formulation's baseline policy prepended."""
        policies = OrderedDict()
        policies[name] = np.asarray(get_baseline_values(self.formulation), float)
        policies.update(self.policies)
        return PolicySet(policies, {name: [label], **self.labels},
                         self.formulation, self.source)

    def __len__(self) -> int:
        return len(self.policies)

    def __iter__(self):
        return iter(self.policies.items())

    def dv_table(self) -> pd.DataFrame:
        """Decision variables as a table (one row per policy)."""
        names = list(get_var_names(self.formulation))
        return pd.DataFrame(
            [{"policy": n, **dict(zip(names, dv))} for n, dv in self])


# --------------------------------------------------------------------------- #
# Simulation and persistence
# --------------------------------------------------------------------------- #
def default_run_dir(scenario: str = "historic", slug: str = "ffmp_obj8",
                    tag: str = "selected_policies") -> Path:
    """Conventional output directory of a policy-set simulation.

    Full-model HDF5 runs are ~140 MB each; on Anvil the directory should be a
    symlink into projects space (see ``ANVIL_NOTES.md``).
    """
    return OUTPUTS_DIR / scenario / slug / "timeseries" / tag


def simulate_or_load(dv: np.ndarray, hdf5: Path, formulation: str = "ffmp", *,
                     use_trimmed: bool = False, overwrite: bool = False,
                     load_existing: bool = True, results_sets: "list | None" = None,
                     log=print) -> "tuple[dict | None, bool]":
    """Simulate one policy to ``hdf5``, or reuse the file if it exists.

    Args:
        dv: Decision vector.
        hdf5: Output file; its parent is created if missing.
        formulation: Formulation that decodes ``dv``.
        use_trimmed: Trimmed model instead of the full model.
        overwrite: Re-simulate even if ``hdf5`` exists.
        load_existing: When the file is reused, load it (False returns None
            as data, which avoids reading large files that are not needed).
        results_sets: pywrdrb results sets to load from an existing file;
            default ``config.RESULTS_SETS``.
        log: Callable for progress messages.

    Returns:
        ``(data, simulated)``: the results dict (results-set name -> daily
        DataFrame, or None when an existing file was not loaded) and whether
        a simulation ran.
    """
    from src.load.results import load_simulation_results
    from src.simulation import dvs_to_config, run_simulation_to_disk

    hdf5 = Path(hdf5)
    if hdf5.exists() and not overwrite:
        log(f"reusing {hdf5.name}")
        if not load_existing:
            return None, False
        return load_simulation_results(hdf5, results_sets), False
    hdf5.parent.mkdir(parents=True, exist_ok=True)
    cfg = dvs_to_config(np.asarray(dv, dtype=float), formulation_name=formulation)
    t0 = time.perf_counter()
    data = run_simulation_to_disk(cfg, hdf5, use_trimmed=use_trimmed)
    log(f"{'trimmed' if use_trimmed else 'full'} model, "
        f"{time.perf_counter() - t0:.0f} s -> {hdf5.name}")
    return data, True


def simulate_policy_set(policy_set: PolicySet, out_dir: Path, *,
                        use_trimmed: bool = False, overwrite: bool = False,
                        compute_objectives: bool = True,
                        log=print) -> pd.DataFrame:
    """Simulate every policy to ``{out_dir}/{name}.hdf5`` and write the tables.

    Existing runs are reused unless ``overwrite`` is set, so an interrupted
    job can be resubmitted. Objective values (the ACTIVE objective set, which
    need not be the set the policies were searched under) are written for the
    runs simulated in this call and merged into an existing objectives table.

    Returns:
        The manifest written to ``out_dir`` (one row per policy).
    """
    from config import get_objective_set

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_mode = "trimmed" if use_trimmed else "full"
    objective_set = get_objective_set() if compute_objectives else None
    log(f"policies : {len(policy_set)} ({policy_set.source})")
    log(f"model    : {model_mode}, {config.START_DATE} -> {config.END_DATE}")
    log(f"out dir  : {out_dir}")

    manifest, obj_rows = [], []
    for name, dv in policy_set:
        hdf5 = out_dir / f"{name}.hdf5"
        data, simulated = simulate_or_load(
            dv, hdf5, policy_set.formulation, use_trimmed=use_trimmed,
            overwrite=overwrite, load_existing=False,
            log=lambda msg, n=name: log(f"[{n}] {msg}"))
        if simulated and objective_set is not None:
            obj_rows.append({"policy": name, **dict(zip(
                objective_set.names, objective_set.compute([data])))})
        manifest.append({
            "policy": name, "labels": "; ".join(policy_set.labels[name]),
            "hdf5": hdf5.name, "model_mode": model_mode,
            "formulation": policy_set.formulation,
            "start_date": config.START_DATE, "end_date": config.END_DATE,
            "source": policy_set.source,
        })

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(out_dir / MANIFEST_NAME, index=False)
    policy_set.dv_table().to_csv(out_dir / DV_TABLE_NAME, index=False)
    if obj_rows:
        _merge_table(out_dir / OBJECTIVES_TABLE_NAME, pd.DataFrame(obj_rows))
    return manifest_df


def _merge_table(path: Path, new: pd.DataFrame, key: str = "policy") -> None:
    """Write ``new`` to ``path``, replacing rows that share ``key``."""
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old[key].isin(new[key])], new], ignore_index=True)
    new.to_csv(path, index=False)


def load_policy_runs(run_dir: Path, results_sets: "list | None" = None,
                     policies: "list[str] | None" = None
                     ) -> "tuple[pd.DataFrame, OrderedDict]":
    """Load a finished policy-set directory for plotting.

    Args:
        run_dir: Directory written by :func:`simulate_policy_set`.
        results_sets: pywrdrb results sets to load; default
            ``config.RESULTS_SETS``.
        policies: Subset of policy names to load; default all. Names absent
            from the manifest are skipped (check the returned manifest).

    Returns:
        ``(manifest, runs)``: the manifest rows loaded and ``runs[name]``,
        the results dict (results-set name -> daily DataFrame) per policy.
    """
    from src.load.results import load_simulation_results

    run_dir = Path(run_dir)
    manifest = pd.read_csv(run_dir / MANIFEST_NAME)
    if policies is not None:
        manifest = manifest[manifest["policy"].isin(policies)]
    runs = OrderedDict()
    for name, hdf5 in zip(manifest["policy"], manifest["hdf5"]):
        runs[name] = load_simulation_results(run_dir / hdf5, results_sets)
    return manifest.reset_index(drop=True), runs
