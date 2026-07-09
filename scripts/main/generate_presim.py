"""
generate_presim.py - Generate pre-simulated releases for the trimmed model.

This is a one-time setup step that must be run before Borg optimization.
It computes the independent STARFIT reservoir releases directly from the
catchment inflows using pywrdrb's offline STARFIT simulator
(``STARFITOfflineSimulator``), which replicates STARFITReservoirRelease's
release arithmetic without running a full Pywr model. The trimmed model
replaces these reservoirs with simple input nodes that read the pre-simulated
values, reducing per-evaluation runtime during optimization.

What the trimmed model skips:
    The DRB contains ~11 independent STARFIT reservoirs (Beltzville, Fewalter,
    etc.) whose operations are unaffected by NYC decisions. Simulating them
    repeatedly is pure overhead during optimization. The trimmed model uses
    pre-computed releases for these reservoirs, keeping only the NYC-coupled
    nodes active.

    The offline simulator is the single-trace counterpart of the ensemble
    ``STARFITReleaseEnsemblePreprocessor`` used by workflow step 04, so the
    historic and ensemble trimmed-model paths share one release engine.

Runtime:
    Offline STARFIT over 1945-2022 runs in seconds (no full Pywr simulation).
    This only needs to be run once per inflow_type / date range combination.

Usage:
    python scripts/main/generate_presim.py

Outputs:
    outputs/presim/presimulated_releases_mgd.csv
    outputs/presim/presimulated_releases_mgd_metadata.json
"""

import sys
import json
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    START_DATE,
    END_DATE,
    INFLOW_TYPE,
    INITIAL_VOLUME_FRAC,
    PRESIM_DIR,
    PRESIM_FILE,
)


def generate_presim():
    """Compute presimulated STARFIT releases offline (no full Pywr run)."""
    import pandas as pd
    from pywrdrb.path_manager import get_pn_object
    from pywrdrb.pre.generate_presimulated_releases import STARFITOfflineSimulator

    print("=" * 60)
    print("  Generate Pre-Simulated Releases (trimmed model setup)")
    print("=" * 60)
    print(f"\n  Inflow type : {INFLOW_TYPE}")
    print(f"  Period      : {START_DATE} to {END_DATE}")
    print(f"  Output dir  : {PRESIM_DIR}")

    if PRESIM_FILE.exists():
        print(f"\n  Overwriting existing file: {PRESIM_FILE}")

    PRESIM_DIR.mkdir(parents=True, exist_ok=True)

    metadata_file = PRESIM_DIR / "presimulated_releases_mgd_metadata.json"

    # --- Step 1: Load catchment inflows for this inflow_type ---
    print("\n--- Step 1: Loading catchment inflows ---")
    t0 = time.perf_counter()

    pn = get_pn_object()
    inflow_file = Path(pn.sc.get(f"flows/{INFLOW_TYPE}")) / "catchment_inflow_mgd.csv"
    if not inflow_file.exists():
        raise FileNotFoundError(
            f"Catchment inflow file not found: {inflow_file}\n"
            f"Expected pywrdrb flows data for inflow_type '{INFLOW_TYPE}'."
        )
    catchment_inflows = pd.read_csv(str(inflow_file), index_col=0, parse_dates=True)
    catchment_inflows.index = pd.DatetimeIndex(catchment_inflows.index)
    # Restrict to the configured simulation window (matches the historic period
    # the trimmed model is scored over).
    catchment_inflows = catchment_inflows.loc[START_DATE:END_DATE]

    elapsed = time.perf_counter() - t0
    print(f"  Rows:  {len(catchment_inflows)} "
          f"({catchment_inflows.index[0].date()} to {catchment_inflows.index[-1].date()})")
    print(f"  Loaded in {elapsed:.1f}s")

    # --- Step 2: Offline STARFIT simulation of independent reservoirs ---
    print("\n--- Step 2: Simulating STARFIT reservoir releases (offline) ---")
    t0 = time.perf_counter()

    sim = STARFITOfflineSimulator(initial_volume_frac=INITIAL_VOLUME_FRAC)
    sim.load_parameters()
    releases_df = sim.simulate_all(catchment_inflows)

    elapsed = time.perf_counter() - t0
    reservoirs = list(releases_df.columns)
    print(f"  Simulation complete in {elapsed:.1f}s")
    print(f"\n  Reservoirs ({len(reservoirs)}):")
    for r in reservoirs:
        print(f"    - {r}")

    # --- Step 3: Write presimulated releases + metadata ---
    print("\n--- Step 3: Writing presimulated releases ---")

    releases_out = releases_df.copy()
    releases_out.index.name = "datetime"
    releases_out.index = pd.to_datetime(releases_out.index).strftime("%Y-%m-%d")
    releases_out.to_csv(PRESIM_FILE, float_format="%.10f")

    metadata = {
        "inflow_type": INFLOW_TYPE,
        "start_date": str(releases_out.index[0]),
        "end_date": str(releases_out.index[-1]),
        "reservoirs": reservoirs,
        "initial_volume_frac": INITIAL_VOLUME_FRAC,
        "source": "STARFITOfflineSimulator",
        "output_file": str(PRESIM_FILE),
        "metadata_file": str(metadata_file),
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  CSV:      {PRESIM_FILE}")
    print(f"  Metadata: {metadata_file}")

    print("\n" + "=" * 60)
    print("  Setup complete. You can now run Borg optimization.")
    print("=" * 60)


if __name__ == "__main__":
    generate_presim()
