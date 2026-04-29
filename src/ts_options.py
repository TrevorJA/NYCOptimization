"""
ts_options.py - Build the temperature/salinity LSTM options dicts for
pywrdrb.ModelBuilder.

The ModelBuilder accepts an `options` dict with `temperature_model` and/or
`salinity_model` sub-dicts. Each sub-dict points the corresponding pywrdrb
Parameter at YAML/JSON artifacts on disk; the parameter does its own loading.

Design choices for NYCOptimization (see local_notes/decisions/):

- **Salinity is the active manuscript path.** It depends only on simulated
  Q_Trenton_bc and Q_Schuylkill_bc, which are available in any pywrdrb run
  (including stochastic re-eval ensembles).

- **Temperature is deferred** because the TempLSTM consumes multivariate
  meteorology (tmmn, tmmx, pr, srad) which we cannot supply for synthetic
  / climate-perturbed scenarios. The options builder remains here so a
  future re-enable is one config flag away.

- **Salinity defaults to async-update mode** (`asycronized_update=True`),
  which keeps the LSTM observe-only — the LSTM still updates per-timestep
  and publishes `salt_front_location_mu`, but the salt-front-driven rewrite
  of `mrf_target_{Montague,Trenton}` is skipped. This preserves the meaning
  of the existing flow-target objectives.

- All paths and toggles come from `config.py`, which honors `NYCOPT_*` env
  overrides. The functions below take the active config as their single
  source of truth.
"""

from __future__ import annotations

from typing import Any

from config import (
    INCLUDE_SALINITY_MODEL,
    INCLUDE_TEMPERATURE_MODEL,
    LSTM_START_DATE,
    PYWRDRB_ML_DIR,
    SALINITY_ASYNC_UPDATE,
    SALINITY_LSTM_MODEL,
    TEMPERATURE_LSTM_MODEL1,
    TEMPERATURE_LSTM_MODEL2,
    TEMPERATURE_LSTM_TAVG2TMAX,
    END_DATE,
)


def build_salinity_options(*, debug: bool = False) -> dict[str, Any]:
    """Return the `salinity_model` options dict for `pywrdrb.ModelBuilder`.

    Inputs to the salinity LSTM are pywrdrb-simulated flows
    (`Q_Trenton_bc`, `Q_Schuylkill_bc`) which are always available, so this
    LSTM is robust to scenario sources (deterministic, stochastic, climate-
    perturbed). The published parameter is `salt_front_location_mu` (river
    mile, 7-day average).

    Returns:
        Dict suitable for `options={"salinity_model": <this>, ...}`. The
        ModelBuilder consumes it at `add_parameter_salinity_model()`.
    """
    return {
        "ml_model_type": "lstm",
        "PywrDRB_ML_plugin_path": str(PYWRDRB_ML_DIR),
        "model_salinity": str(SALINITY_LSTM_MODEL),
        "start_date": LSTM_START_DATE,
        "end_date": END_DATE,
        "Q_Trenton_lstm_var_name": "Q_Trenton_bc",
        "Q_Schuylkill_lstm_var_name": "Q_Schuylkill_bc",
        "asycronized_update": bool(SALINITY_ASYNC_UPDATE),
        "debug": debug,
    }


def build_temperature_options(*, debug: bool = False) -> dict[str, Any]:
    """Return the `temperature_model` options dict for `pywrdrb.ModelBuilder`.

    NOT WIRED IN BY DEFAULT. The TempLSTM requires multivariate meteorology
    (`tmmn`, `tmmx`, `pr`, `srad`) that is not available for synthetic /
    climate-perturbed re-eval scenarios. Kept here for future re-enable.
    """
    return {
        "ml_model_type": "lstm",
        "PywrDRB_ML_plugin_path": str(PYWRDRB_ML_DIR),
        "model1": str(TEMPERATURE_LSTM_MODEL1),
        "model2": str(TEMPERATURE_LSTM_MODEL2),
        "Tavg2Tmax_coefs": str(TEMPERATURE_LSTM_TAVG2TMAX),
        "start_date": LSTM_START_DATE,
        "end_date": END_DATE,
        "activate_thermal_control": False,
        "Q_C_lstm_var_name": "QbcTavg_Q_C",
        "Q_i_lstm_var_name": "QbcTavg_Q_i",
        "cannonsville_storage_pct_lstm_var_name": "bc_cannonsville_storage_pct",
        "thermal_mitigation_bank_size": 1620,  # MGD
        "asycronized_update": False,
        "debug": debug,
    }


def build_lstm_options_block(*, debug: bool = False) -> dict[str, Any]:
    """Return the LSTM portion of the ModelBuilder `options` dict.

    Includes only the sub-dicts whose `INCLUDE_*_MODEL` toggle is True. If
    neither is on, returns an empty dict — the caller can merge it into a
    larger options dict without conditional logic.
    """
    block: dict[str, Any] = {}
    if INCLUDE_TEMPERATURE_MODEL:
        block["temperature_model"] = build_temperature_options(debug=debug)
    if INCLUDE_SALINITY_MODEL:
        block["salinity_model"] = build_salinity_options(debug=debug)
    return block
