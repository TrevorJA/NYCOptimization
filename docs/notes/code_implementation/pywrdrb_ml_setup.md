# PywrDRB-ML Environment Setup

The salinity LSTM (and the deferred temperature LSTM) require PywrDRB-ML to
be importable from the NYCOptimization venv. Neither LSTM is used in the
manuscript (both default off; the coupling machinery is dormant —
`config.py` §Temperature & Salinity LSTM Coupling), but the setup below keeps
them one config flag from re-enable.

## `xarray` dependency

`PywrDRB-ML/src/prep_data_utils.py` imports `xarray`; it must be installed in
the NYCOpt venv (`pip install xarray`).

## `lstm_model.py` portable root-dir resolution

`../PywrDRB-ML/src/lstm_model.py` carries a local patch that resolves
`root_dir` from the `PYWRDRB_ML_DIR` env var (falling back to the repo
layout) instead of hardcoded developer paths, and avoids
`pathnavigator.expanduser()` (removed in pathnavigator >= 0.6). If the
upstream repo is re-pulled and overwrites this file, re-apply the patch:

```python
import os
from pathlib import Path
import pathnavigator

root_dir = os.environ.get(
    "PYWRDRB_ML_DIR",
    str(Path(__file__).resolve().parent.parent),
)
pn = pathnavigator.create(root_dir)
pn.add_to_sys_path()
pn.chdir()
```

(`src/model_builder.py` has a similar hardcoded block but is not imported by
the salinity-LSTM code path.)

## `src/` namespace collision

PywrDRB-ML uses absolute `from src.X import Y` imports inside its flat
`src/` directory, and NYCOptimization also has a `src/` package.
`src/ts_options.py::_bootstrap_pywrdrb_ml_namespace()` temporarily swaps
`sys.modules['src']` for a fresh PywrDRB-ML view, imports the `lstm_model` +
`torch_bmi` chain, then restores NYCOpt's `src` while keeping the PywrDRB-ML
leaf modules cached in `sys.modules` (no leaf names collide). The bootstrap
is idempotent and runs lazily (only when `INCLUDE_TEMPERATURE_MODEL or
INCLUDE_SALINITY_MODEL`). Long-term fix: PywrDRB-ML should rename `src/` to
a real package name; until then, the bootstrap stays.

## SalinityLSTM database range

The SalinityLSTM database
(`models/SalinityLSTM/SalinityLSTM/data_SalinityLSTM.npz` + the upstream
`SalinityLSTM_database.csv`) spans 1945-01-01 onward, matching the Amestoy
et al. (2026) reconstructed inflow record, so the LSTM runs over the full
simulation window. The trained weights are date-agnostic (no retraining;
only the inference-time `dates_all` array extends back). NYCOpt's
`LSTM_START_DATE` defaults to `START_DATE` (`"1945-10-01"`); the override
knob remains for clipped-window experiments.
