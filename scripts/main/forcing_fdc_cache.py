"""Build the flow-duration-curve cache backing panel (d) of the forcing-space figure.

Panel (d) compares three sources of daily aggregate NYC reservoir inflow on one
flow duration curve (FDC) axis:

    E_test    the re-evaluation ensemble's realizations (Kirsch-Nowak generator
              driven by the sampled harmonic forcing) - 25,000 x 50 years
    CMIP6     the raw DBCCA-downscaled, PRMS/VIC5-simulated daily flows of the
              54 CMIP6 future runs - 40 years each, each paired with its OWN
              1980-2019 historical sibling run so the figure can show a
              model-specific change rather than a change plus model bias
    historic  the reconstructed historical record

Reading the E_test daily flows costs ~11 GB of HDF5 I/O across 50 chunk
directories, so this is a separate compute step in the repo's usual
compute-then-plot split. It stores ONE FDC ROW PER REALIZATION rather than a
pre-reduced band, so the envelope definition (min-max, 5th-95th, ...) stays a
plot-time choice and re-rendering never re-reads the ensemble.

Written to ``outputs/diagnostics/forcing_parameterization/fdc_cache.npz``.

Run through srun/sbatch, never on a login node::

    sbatch workflow/13_main_figures.sh                    # builds if missing
    python3 -m scripts.main.forcing_fdc_cache --stride 250 --out <path>   # smoke
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import config

#: Exceedance-probability grid (percent) every FDC is interpolated onto. Dense
#: enough to resolve both tails of a log-scaled flow axis without bloating the
#: cache (500 x 25,000 float32 = 50 MB).
N_GRID: int = 500

#: Cache location. Under ``outputs/diagnostics/`` alongside the SI forcing
#: figures; gitignored and regenerable.
DEFAULT_CACHE = (config.OUTPUTS_DIR / "diagnostics" / "forcing_parameterization"
                 / "fdc_cache.npz")

#: Directory of per-run CMIP6 pywrdrb inputs (sibling repo). Same daily
#: catchment inflows, in MGD, with the same node names as the E_test HDF5s.
CMIP6_INPUTS_DIR = (config.PROJECT_DIR.parent / "CMIP6_multimodel_streamflow"
                    / "pywrdrb" / "inputs")

#: Token marking a CMIP6 run's HISTORICAL sibling. Excluded so the cache holds
#: exactly the 54 future runs that ``load_cmip6_envelope`` fits.
CMIP6_BASELINE_TOKEN: str = "1980_2019"


def exceedance_grid() -> np.ndarray:
    """Exceedance probabilities (percent) at which every FDC is evaluated."""
    return np.linspace(0.0, 100.0, N_GRID)


def fdc_on_grid(values: np.ndarray, grid_pct: np.ndarray) -> np.ndarray:
    """Flow duration curve of ``values``, interpolated onto ``grid_pct``.

    Standard Weibull plotting positions: sort descending, assign exceedance
    ``i / (n + 1)``. Flows are clipped to the observed range outside the
    empirical support rather than extrapolated.

    Args:
        values: 1-D daily flows (NaNs dropped).
        grid_pct: Exceedance probabilities in percent, ascending.

    Returns:
        Flow at each grid exceedance probability.
    """
    v = np.asarray(values, dtype=float)
    v = np.sort(v[~np.isnan(v)])[::-1]
    if v.size == 0:
        return np.full(grid_pct.shape, np.nan)
    ep_pct = 100.0 * np.arange(1, v.size + 1) / (v.size + 1)
    return np.interp(grid_pct, ep_pct, v, left=v[0], right=v[-1])


def nyc_inflow_nodes() -> tuple[str, ...]:
    """Catchments summed into the aggregate NYC reservoir inflow."""
    from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES

    return tuple(DEFAULT_NYC_INFLOW_NODES)


# ---------------------------------------------------------------------------
# The three sources
# ---------------------------------------------------------------------------

def historic_fdc(grid_pct: np.ndarray, nodes: tuple[str, ...]) -> np.ndarray:
    """FDC of the reconstructed historical aggregate NYC inflow."""
    from src.load.historical_flows import load_historical_flows

    flows = load_historical_flows(gage=False, period="full")
    return fdc_on_grid(flows.loc[:, list(nodes)].sum(axis=1).to_numpy(), grid_pct)


def cmip6_future_runs() -> list[Path]:
    """The 54 CMIP6 future-run input directories, excluding historical siblings."""
    if not CMIP6_INPUTS_DIR.is_dir():
        raise FileNotFoundError(
            f"CMIP6 pywrdrb inputs not found at {CMIP6_INPUTS_DIR}. Panel (d) needs the "
            "sibling CMIP6_multimodel_streamflow repo checked out alongside this one."
        )
    runs = [d for d in sorted(CMIP6_INPUTS_DIR.iterdir())
            if d.is_dir()
            and "ssp" in d.name
            and CMIP6_BASELINE_TOKEN not in d.name
            and (d / "catchment_inflow_mgd.csv").is_file()]
    if not runs:
        raise FileNotFoundError(f"no CMIP6 future runs under {CMIP6_INPUTS_DIR}")
    return runs


def cmip6_sibling_key(name: str) -> str:
    """``{hydro}_RAPID_{GCM}`` key shared by a future run and its 1980-2019 sibling.

    Mirrors ``scengen.forcing_space._gcm_key``, the same pairing the monthly
    change-factor table was built on (``diff_relative_to_dataset_baseline``), so
    the FDC change and the fitted change factors use one baseline convention.
    """
    return re.sub(r"_ssp\d+_.*", "", name, flags=re.IGNORECASE)


def cmip6_baseline_runs() -> dict[str, Path]:
    """Historical (1980-2019) sibling directory of each CMIP6 hydro-model/GCM pair.

    Keyed by :func:`cmip6_sibling_key`. Purely observation-forced runs
    (Daymet2019, Livneh2018) carry no ``ssp`` token and are excluded: they are
    not a GCM's own baseline.
    """
    return {
        cmip6_sibling_key(d.name): d
        for d in sorted(CMIP6_INPUTS_DIR.iterdir())
        if d.is_dir()
        and "ssp" in d.name
        and CMIP6_BASELINE_TOKEN in d.name
        and (d / "catchment_inflow_mgd.csv").is_file()
    }


def _aggregate_inflow(run: Path, nodes: tuple[str, ...]) -> np.ndarray:
    """Daily aggregate NYC reservoir inflow (MGD) of one pywrdrb input directory."""
    flows = pd.read_csv(run / "catchment_inflow_mgd.csv",
                        index_col=0, parse_dates=True)
    return flows.loc[:, list(nodes)].sum(axis=1).to_numpy()


def cmip6_fdcs(grid_pct: np.ndarray, nodes: tuple[str, ...]
               ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """FDCs of the raw CMIP6-driven daily flows, one row per future run.

    Each future run is returned alongside the FDC of its OWN 1980-2019
    historical sibling. Panel (d) divides the two, which cancels the hydrologic
    model's bias against the reconstructed record and leaves the model's
    projected change.

    Returns:
        ``(future (K, n_grid), baseline (K, n_grid), future_labels,
        baseline_labels)`` with the two label lists aligned row-for-row.
    """
    runs = cmip6_future_runs()
    baselines = cmip6_baseline_runs()
    curves = np.empty((len(runs), grid_pct.size), dtype=np.float32)
    base_curves = np.empty_like(curves)
    base_labels: list[str] = []
    seen: dict[str, np.ndarray] = {}

    for i, run in enumerate(runs):
        curves[i] = fdc_on_grid(_aggregate_inflow(run, nodes), grid_pct)
        key = cmip6_sibling_key(run.name)
        sibling = baselines.get(key)
        if sibling is None:
            raise FileNotFoundError(
                f"no {CMIP6_BASELINE_TOKEN} historical sibling for {run.name} "
                f"(key {key!r}) under {CMIP6_INPUTS_DIR}. Panel (d) needs each "
                "future run's own baseline to show a model-specific change."
            )
        if key not in seen:
            seen[key] = fdc_on_grid(_aggregate_inflow(sibling, nodes),
                                    grid_pct).astype(np.float32)
        base_curves[i] = seen[key]
        base_labels.append(sibling.name)

    return curves, base_curves, [r.name for r in runs], base_labels


def etest_chunk_dirs(ensemble_dir: Path) -> list[tuple[Path, int]]:
    """Staged daily-flow chunks as ``(directory, first global realization id)``."""
    index_file = ensemble_dir / "chunk_index.json"
    if not index_file.is_file():
        raise FileNotFoundError(
            f"{index_file} not found - the ensemble is not staged in chunked form."
        )
    index = json.loads(index_file.read_text())
    return [(ensemble_dir.parent / c["slug"], int(c["global_start"]))
            for c in index["chunks"]]


def etest_fdcs(ensemble_dir: Path, grid_pct: np.ndarray, nodes: tuple[str, ...],
               *, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """FDCs of the E_test realizations, one row per (kept) realization.

    Reads one chunk at a time so peak memory stays at a single chunk's NYC
    columns rather than the whole 11 GB ensemble.

    Args:
        ensemble_dir: Staged ensemble root (holding ``chunk_index.json``).
        grid_pct: Exceedance grid in percent.
        nodes: Catchments summed into the aggregate inflow.
        stride: Keep every ``stride``-th realization (1 = all). For smoke runs.

    Returns:
        ``(curves (n_kept, n_grid) float32, realization_ids (n_kept,))`` with
        ids global to the ensemble.
    """
    curves: list[np.ndarray] = []
    ids: list[int] = []
    chunks = etest_chunk_dirs(ensemble_dir)

    for c, (chunk, offset) in enumerate(chunks):
        path = chunk / "catchment_inflow_mgd.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found (chunk {c} of {len(chunks)})")
        with h5py.File(path, "r") as f:
            missing = [n for n in nodes if n not in f]
            if missing:
                raise KeyError(f"{path} is missing NYC inflow nodes {missing}")
            # Realization datasets are keyed by chunk-local index as strings;
            # each site group also carries a non-numeric 'date' dataset.
            realizations = sorted(int(k) for k in f[nodes[0]].keys() if k.isdigit())
            for r in realizations:
                gid = offset + r
                if gid % stride:
                    continue
                total = np.zeros(f[nodes[0]][str(r)].shape, dtype=np.float64)
                for node in nodes:
                    total += f[node][str(r)][:]
                curves.append(fdc_on_grid(total, grid_pct).astype(np.float32))
                ids.append(gid)
        print(f"[fdc] chunk {c + 1}/{len(chunks)}: {len(curves)} realizations",
              flush=True)

    return np.asarray(curves, dtype=np.float32), np.asarray(ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_cache(ensemble_dir: Path, out_file: Path, *, stride: int = 1) -> dict:
    """Compute all three FDC sets and write the cache npz."""
    grid = exceedance_grid()
    nodes = nyc_inflow_nodes()
    meta = json.loads((ensemble_dir / "_meta.json").read_text())

    print(f"[fdc] historic ({len(nodes)} nodes: {', '.join(nodes)})", flush=True)
    hist = historic_fdc(grid, nodes)

    print("[fdc] CMIP6 raw daily runs (+ 1980-2019 siblings)", flush=True)
    cmip6, cmip6_base, cmip6_labels, cmip6_base_labels = cmip6_fdcs(grid, nodes)
    print(f"[fdc] {len(cmip6_labels)} CMIP6 future runs, "
          f"{len(set(cmip6_base_labels))} historical siblings", flush=True)

    print(f"[fdc] E_test from {ensemble_dir.name} (stride={stride})", flush=True)
    etest, etest_ids = etest_fdcs(ensemble_dir, grid, nodes, stride=stride)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        exceedance=grid,
        etest_fdc=etest,
        etest_realization_ids=etest_ids,
        cmip6_fdc=cmip6,
        cmip6_labels=np.array(cmip6_labels),
        cmip6_baseline_fdc=cmip6_base,
        cmip6_baseline_labels=np.array(cmip6_base_labels),
        historic_fdc=hist,
        nyc_nodes=np.array(nodes),
        ensemble_slug=meta.get("slug", ensemble_dir.name),
        flowtype=meta.get("flowtype", ""),
        forcing_hash=meta.get("forcing_hash", ""),
        stride=stride,
    )
    print(f"[fdc] wrote {out_file} "
          f"(E_test {etest.shape}, CMIP6 {cmip6.shape}, grid {grid.size})", flush=True)
    return {"n_etest": int(etest.shape[0]), "n_cmip6": int(cmip6.shape[0])}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ensemble", default=None,
                   help="staged ensemble slug (default: the campaign E_test)")
    p.add_argument("--out", type=Path, default=DEFAULT_CACHE,
                   help=f"cache path (default: {DEFAULT_CACHE})")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every Nth realization; >1 is for smoke runs only")
    args = p.parse_args(argv)

    if args.ensemble is None:
        from src import etest as etest_mod
        slug = etest_mod.E_TEST_VARIANTS[etest_mod.E_TEST_VARIANT].slug
    else:
        slug = args.ensemble

    ensemble_dir = config.STAGED_ENSEMBLE_DIR / slug
    if not ensemble_dir.is_dir():
        raise FileNotFoundError(f"staged ensemble not found: {ensemble_dir}")

    build_cache(ensemble_dir, args.out, stride=args.stride)
    return 0


if __name__ == "__main__":
    sys.exit(main())
