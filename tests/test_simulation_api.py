"""
test_simulation_api.py - Test pywrdrb model build/run/extract pipeline.

Verifies:
1. ModelBuilder.make_model() produces model_dict
2. Model.load() from JSON
3. model.run() return value and .to_dataframe()
4. OutputRecorder in-memory data access (recorder_dict[key].data)
5. Name mapping from raw pywr keys to standard pywrdrb node names
6. Whether objectives can be computed from extracted data

Run: python tests/test_simulation_api.py
"""

import sys
import time
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    START_DATE, END_DATE, INFLOW_TYPE, USE_TRIMMED_MODEL,
    INITIAL_VOLUME_FRAC, RESULTS_SETS, NYC_RESERVOIRS,
    get_baseline_values, get_var_names,
)

###############################################################################
# Test 1: Build model with ModelBuilder
###############################################################################
print("=" * 60)
print("TEST 1: ModelBuilder.make_model()")
print("=" * 60)

import pywrdrb
from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig

config = NYCOperationsConfig.from_defaults()
print(f"  NYCOperationsConfig created: {type(config)}")

# Use shorter date range for quick test
test_start = "2000-01-01"
test_end = "2001-12-31"

mb = pywrdrb.ModelBuilder(
    inflow_type=INFLOW_TYPE,
    start_date=test_start,
    end_date=test_end,
    options={
        "nyc_nj_demand_source": "historical",
        "use_trimmed_model": False,   # Full model: no presim file required for test
        "initial_volume_frac": INITIAL_VOLUME_FRAC,
    },
    nyc_operations_config=config,
)
mb.make_model()

print(f"  mb.model_dict type: {type(mb.model_dict)}")
print(f"  mb.model_dict keys: {list(mb.model_dict.keys())}")
print(f"  Has 'model' attr: {hasattr(mb, 'model')}")

# Check if mb.model exists (it shouldn't based on source)
if hasattr(mb, 'model'):
    print(f"  mb.model type: {type(mb.model)}")
else:
    print(f"  CONFIRMED: mb.model does NOT exist. Use mb.model_dict.")


###############################################################################
# Test 2: Write JSON and load model
###############################################################################
print("\n" + "=" * 60)
print("TEST 2: Write JSON and Model.load()")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    model_json = Path(tmpdir) / "test_model.json"
    output_hdf5 = Path(tmpdir) / "test_output.hdf5"

    mb.write_model(str(model_json))
    print(f"  Model JSON written: {model_json.stat().st_size / 1e6:.1f} MB")

    model = pywrdrb.Model.load(str(model_json))
    print(f"  Model loaded: {type(model)}")
    print(f"  Model nodes: {len(list(model.nodes))}")
    print(f"  Model parameters: {len(list(model.parameters))}")

    ###########################################################################
    # Test 3: Run model WITH OutputRecorder and check return value
    ###########################################################################
    print("\n" + "=" * 60)
    print("TEST 3: model.run() with OutputRecorder")
    print("=" * 60)

    recorder = pywrdrb.OutputRecorder(
        model=model,
        output_filename=str(output_hdf5),
    )
    print(f"  OutputRecorder created: {type(recorder)}")
    print(f"  recorder_dict keys (sample): {list(recorder.recorder_dict.keys())[:10]}")
    print(f"  Total recorder keys: {len(recorder.recorder_dict)}")

    t0 = time.time()
    stats = model.run()
    elapsed = time.time() - t0
    print(f"\n  model.run() returned: {type(stats)}")
    print(f"  model.run() value: {stats}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # Check if stats has to_dataframe
    if stats is not None:
        print(f"  stats attributes: {[a for a in dir(stats) if not a.startswith('_')]}")
        if hasattr(stats, 'to_dataframe'):
            stats_df = stats.to_dataframe()
            print(f"  stats.to_dataframe() type: {type(stats_df)}")
            print(f"  stats_df shape: {stats_df.shape}")
            print(f"  stats_df columns: {list(stats_df.columns)}")
            print(f"  stats_df head:\n{stats_df.head()}")

    ###########################################################################
    # Test 4: Access OutputRecorder data IN-MEMORY
    ###########################################################################
    print("\n" + "=" * 60)
    print("TEST 4: In-memory data access from OutputRecorder")
    print("=" * 60)

    # Check which keys exist for our results_sets
    all_keys = list(recorder.recorder_dict.keys())

    # Check reservoir storage keys
    res_keys = [k for k in all_keys if k.startswith("reservoir_")]
    print(f"  Reservoir keys: {res_keys[:5]}")

    # Check major flow keys
    flow_keys = [k for k in all_keys if k.startswith("link_")]
    print(f"  Link keys (sample): {flow_keys[:5]}")

    # Check demand/delivery keys
    demand_keys = [k for k in all_keys if "demand" in k.lower()]
    delivery_keys = [k for k in all_keys if "delivery" in k.lower()]
    print(f"  Demand keys: {demand_keys}")
    print(f"  Delivery keys: {delivery_keys}")

    # Access raw numpy data from a recorder
    sample_key = res_keys[0] if res_keys else all_keys[0]
    sample_rec = recorder.recorder_dict[sample_key]
    print(f"\n  Sample recorder ({sample_key}):")
    print(f"    type: {type(sample_rec)}")
    print(f"    data type: {type(sample_rec.data)}")
    print(f"    data shape: {sample_rec.data.shape}")
    print(f"    data[0:3]: {sample_rec.data[0:3]}")

    # Get datetime index (convert PeriodIndex to DatetimeIndex)
    datetime_index = model.timestepper.datetime_index.to_timestamp()
    print(f"\n  Datetime index: {len(datetime_index)} timesteps")
    print(f"    First: {datetime_index[0]}, Last: {datetime_index[-1]}")

    ###########################################################################
    # Test 5: Build DataFrames matching pywrdrb Data() format
    ###########################################################################
    print("\n" + "=" * 60)
    print("TEST 5: Build DataFrames with standard node names")
    print("=" * 60)

    from pywrdrb.utils.lists import reservoir_list, majorflow_list

    # Build res_storage DataFrame
    res_storage_data = {}
    for k in all_keys:
        parts = k.split("_", 1)
        if parts[0] == "reservoir" and len(parts) > 1:
            node_name = parts[1]
            if node_name in reservoir_list:
                rec = recorder.recorder_dict[k]
                # Shape is [timesteps, scenarios]. Take scenario 0.
                res_storage_data[node_name] = rec.data[:, 0]

    res_storage_df = pd.DataFrame(
        res_storage_data,
        index=pd.DatetimeIndex(datetime_index),
    )
    print(f"  res_storage: {res_storage_df.shape}, cols={list(res_storage_df.columns)[:5]}")
    print(f"  NYC reservoirs present: {[r for r in NYC_RESERVOIRS if r in res_storage_df.columns]}")

    # Build major_flow DataFrame
    major_flow_data = {}
    for k in all_keys:
        parts = k.split("_", 1)
        if parts[0] == "link" and len(parts) > 1:
            node_name = parts[1]
            if node_name in majorflow_list:
                rec = recorder.recorder_dict[k]
                major_flow_data[node_name] = rec.data[:, 0]

    major_flow_df = pd.DataFrame(
        major_flow_data,
        index=pd.DatetimeIndex(datetime_index),
    )
    print(f"  major_flow: {major_flow_df.shape}, cols={list(major_flow_df.columns)[:5]}")
    print(f"  delMontague present: {'delMontague' in major_flow_df.columns}")
    print(f"  delTrenton present: {'delTrenton' in major_flow_df.columns}")

    # Build ibt_demands DataFrame
    ibt_demands_data = {}
    for k in ["demand_nyc", "demand_nj"]:
        if k in all_keys:
            rec = recorder.recorder_dict[k]
            ibt_demands_data[k] = rec.data[:, 0]

    ibt_demands_df = pd.DataFrame(
        ibt_demands_data,
        index=pd.DatetimeIndex(datetime_index),
    )
    print(f"  ibt_demands: {ibt_demands_df.shape}, cols={list(ibt_demands_df.columns)}")

    # Build ibt_diversions DataFrame
    ibt_diversions_data = {}
    for k in ["delivery_nyc", "delivery_nj"]:
        if k in all_keys:
            rec = recorder.recorder_dict[k]
            ibt_diversions_data[k] = rec.data[:, 0]

    ibt_diversions_df = pd.DataFrame(
        ibt_diversions_data,
        index=pd.DatetimeIndex(datetime_index),
    )
    print(f"  ibt_diversions: {ibt_diversions_df.shape}, cols={list(ibt_diversions_df.columns)}")

    ###########################################################################
    # Test 6: Compute objectives from extracted data
    ###########################################################################
    print("\n" + "=" * 60)
    print("TEST 6: Compute objectives from in-memory data")
    print("=" * 60)

    data = {
        "major_flow": major_flow_df,
        "res_storage": res_storage_df,
        "ibt_demands": ibt_demands_df,
        "ibt_diversions": ibt_diversions_df,
    }

    from src.objectives import DEFAULT_OBJECTIVES

    try:
        obj_values = DEFAULT_OBJECTIVES.compute(data)
        obj_names = DEFAULT_OBJECTIVES.names
        print("  Objectives computed successfully!")
        for name, val in zip(obj_names, obj_values):
            print(f"    {name} = {val:.6f}")
    except Exception as e:
        print(f"  ERROR computing objectives: {e}")
        import traceback
        traceback.print_exc()

    ###########################################################################
    # Test 7: Compare with HDF5/Data() path
    ###########################################################################
    print("\n" + "=" * 60)
    print("TEST 7: Compare in-memory vs HDF5/Data() loading")
    print("=" * 60)

    # The OutputRecorder should have saved the HDF5 on finish()
    if output_hdf5.exists():
        print(f"  HDF5 exists: {output_hdf5.stat().st_size / 1e6:.1f} MB")

        data_loader = pywrdrb.Data()
        data_loader.load_output(
            output_filenames=[str(output_hdf5)],
            results_sets=["major_flow", "res_storage", "ibt_demands", "ibt_diversions"],
        )

        # Access the loaded data
        label = output_hdf5.stem

        # Compare major_flow
        hdf5_flow = data_loader.major_flow[label][0]
        print(f"\n  HDF5 major_flow shape: {hdf5_flow.shape}")
        print(f"  HDF5 major_flow cols: {list(hdf5_flow.columns)[:5]}")

        if "delMontague" in hdf5_flow.columns and "delMontague" in major_flow_df.columns:
            inmem = major_flow_df["delMontague"].values
            fromdisk = hdf5_flow["delMontague"].values
            max_diff = np.max(np.abs(inmem - fromdisk))
            print(f"  delMontague max diff (inmem vs disk): {max_diff:.2e}")

        # Compare res_storage
        hdf5_storage = data_loader.res_storage[label][0]
        if "cannonsville" in hdf5_storage.columns and "cannonsville" in res_storage_df.columns:
            inmem = res_storage_df["cannonsville"].values
            fromdisk = hdf5_storage["cannonsville"].values
            max_diff = np.max(np.abs(inmem - fromdisk))
            print(f"  cannonsville storage max diff: {max_diff:.2e}")
    else:
        print("  HDF5 not created (OutputRecorder may not have called finish)")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
