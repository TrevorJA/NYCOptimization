# PywrDRB-ML Environment Setup

The salinity LSTM (and the deferred temperature LSTM) require PywrDRB-ML to be importable from the NYCOptimization venv. Three issues had to be resolved in the 2026-04-29 working session — all are now fixed in this checkout.

## Resolved: `xarray` dependency

`PywrDRB-ML/src/prep_data_utils.py` imports `xarray`. Installed into the NYCOpt venv:

```bash
source venv/bin/activate
pip install xarray
```

(xarray 2026.4.0 lives in `venv/lib/python3.11/site-packages/`.)

## Resolved: `lstm_model.py` developer-path patch

`PywrDRB-ML/src/lstm_model.py` previously contained hardcoded developer paths and used `pathnavigator.expanduser()` (removed in pathnavigator >= 0.6). Patched in place to portably resolve `root_dir`:

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

This is a local edit to `../PywrDRB-ML/src/lstm_model.py`. It should be upstreamed as a PR. If the upstream repo gets re-pulled and overwrites this file, re-apply the patch.

`src/model_builder.py` has the same hardcoded developer-path block at L9-14 but is **not imported by the salinity-LSTM code path** (only `src.lstm_model` and its `from .torch_bmi` chain are exercised). Leave it for now; patch when needed.

## Resolved: `src/` namespace collision

PywrDRB-ML uses absolute `from src.X import Y` imports inside its flat `src/` directory. NYCOptimization also has a `src/` package. After NYCOpt's modules load, `sys.modules['src']` is NYCOpt's package, so `from src.lstm_model import SalinityLSTMModel` (which Pywr-DRB's salinity parameter does at load time) raises `ImportError`.

**Fix in NYCOpt code**: `src/ts_options.py::_bootstrap_pywrdrb_ml_namespace()` temporarily swaps `sys.modules['src']` for a fresh PywrDRB-ML view, imports the `lstm_model` + `torch_bmi` chain, then restores NYCOpt's `src` while keeping `src.lstm_model`, `src.torch_bmi`, etc. cached in `sys.modules`. Python's `from X.Y import Z` consults `sys.modules['X.Y']` first, so the cached PywrDRB-ML modules win without affecting NYCOpt's own `from src.X import Y` lookups (no PywrDRB-ML leaf names collide with NYCOpt's `src/*` layout).

The bootstrap is idempotent and runs lazily (only when `INCLUDE_TEMPERATURE_MODEL or INCLUDE_SALINITY_MODEL`).

**Long-term fix**: PywrDRB-ML should rename its `src/` directory to a real package name (e.g. `pywrdrb_ml/`). Until then, the bootstrap stays.

## Resolved (2026-04-30): SalinityLSTM database extended to 1945

Originally the SalinityLSTM internal `dates_all` array began at 1979-01-01 (the LSTM training start), and `LSTM_START_DATE` defaulted to that. After extending the PywrDRB-ML database (`models/SalinityLSTM/SalinityLSTM/data_SalinityLSTM.npz` + the upstream `data/database/SalinityLSTM_database.csv`) back to 1945-01-01, the LSTM now runs over the full simulation window. **No retraining** — the trained weights are date-agnostic; only the inference-time `dates_all` array needed to extend back. See [decisions/2026-04-30_inflow_and_du_search.md](../decisions/2026-04-30_inflow_and_du_search.md) (Salinity LSTM appendix) for the full change list and rationale.

PywrDRB-ML changes (additive, backward-compatible — defaults preserve old behavior with explicit args):
- `src/form_SalinityLSTM_db/create_SalinityLSTM_database.py`: parameterized `flow_input_filename` (default `pywrdrb_pub_nhmv10_BC_withObsScaled_flow_and_storage_1945.csv`), `start='1945-01-01'`, root-dir resolution honors `PYWRDRB_ML_DIR` env var.
- `models/SalinityLSTM/SalinityLSTM.yml`: `min_date: '1945-01-01'`. Training window unchanged.
- `models/SalinityLSTM/SalinityLSTM/data_SalinityLSTM.npz`: regenerated via `data_prep`.

NYCOpt change: `LSTM_START_DATE` default flipped from `"1979-01-01"` to `START_DATE` (i.e. `"1945-10-01"`).

## Verification

Smoke baseline runs cleanly with salinity enabled (5-year debug window):

```bash
source venv/bin/activate
NYCOPT_SALINITY_ON=1 \
PYWRDRB_SIM_START_DATE=2018-01-01 PYWRDRB_SIM_END_DATE=2022-12-31 \
NYCOPT_OBJECTIVES="nyc_reliability_weekly,nyc_vulnerability,nj_reliability_weekly,montague_reliability_weekly_fixed,trenton_reliability_weekly_fixed,flood_risk_downstream_flow_days,storage_min_combined_pct,salt_front_max_rm" \
python scripts/run_baseline.py --use-trimmed
```

Produces 8 finite objective values including a finite `salt_front_max_rm`. Full-period (1945-10-01 → 2022-09-30) baseline runs in ~107 s with the LSTM extended to 1945; `salt_front_location_mu` is finite for 28,123/28,124 sim days (one NaN at the gate-skipped first day, dropped by the objective).

The salinity LSTM is verified responsive to NYC operational decisions in sync mode — see [decisions/2026-04-29_salinity_lstm_responsive_sync.md](../decisions/2026-04-29_salinity_lstm_responsive_sync.md). The 2026-04-30 MM-Borg smoke (500 NFE × 2 islands, debug window) yielded 92 Pareto solutions with `salt_front_max_rm` varying across the set, confirming the objective drives optimization.
