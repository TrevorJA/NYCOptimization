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

import importlib
import os
import sys
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


###############################################################################
# Namespace bootstrap (works around upstream PywrDRB-ML / NYCOpt src/ collision)
###############################################################################
# PywrDRB-ML uses absolute `from src.lstm_model import X` style imports inside
# its flat src/ directory. NYCOptimization also has a src/ package. After
# NYCOpt's modules are imported, sys.modules['src'] is NYCOpt's package, so
# `from src.lstm_model import X` (which Pywr-DRB's salinity parameter does at
# load time) raises ImportError.
#
# Workaround: temporarily swap sys.modules['src'] for a fresh PywrDRB-ML view,
# import its lstm_model + torch_bmi chain, then restore NYCOpt's src while
# keeping `src.lstm_model`, `src.torch_bmi`, etc. cached. Python's `from X.Y
# import Z` consults sys.modules['X.Y'] first, so the cached modules win
# without affecting NYCOpt's own `from src.X import Y` lookups (none of the
# PywrDRB-ML leaf names collide with NYCOpt's src/* layout).
#
# A long-term fix is for PywrDRB-ML to rename `src/` to a proper package name
# (e.g. `pywrdrb_ml/`); see local_notes/configuration/pywrdrb_ml_setup.md.

_ML_LEAF_MODULES = (
    "src.lstm_model",
    "src.torch_bmi",
    "src.torch_models",
    "src.training_utils",
    "src.sampling_utils",
    "src.prep_data",
    "src.prep_data_utils",
    "src.rf_model",
    "src.crossval_utils",
)


def _bootstrap_pywrdrb_ml_namespace() -> None:
    """Idempotently pre-cache PywrDRB-ML's `src.lstm_model` chain in sys.modules.

    Safe to call multiple times. No-op when the chain is already cached and
    looks like PywrDRB-ML's (rather than a stale NYCOpt re-import).
    """
    cached_lstm_model = sys.modules.get("src.lstm_model")
    if cached_lstm_model is not None and hasattr(cached_lstm_model, "SalinityLSTMModel"):
        # Even if PywrDRB-ML modules are cached, ensure our parameter
        # subclass is also registered (this re-import is cheap and idempotent
        # since `Parameter.register()` is keyed by class name).
        _register_nycopt_parameters()
        return  # Already bootstrapped.

    pywrdrb_ml_str = str(PYWRDRB_ML_DIR)
    os.environ.setdefault("PYWRDRB_ML_DIR", pywrdrb_ml_str)

    # Snapshot NYCOpt's src state so we can restore it.
    saved_src = sys.modules.pop("src", None)
    saved_src_submods: dict[str, Any] = {
        k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("src.")
    }

    # Ensure PywrDRB-ML is at sys.path[0] so a fresh `import src` resolves there.
    inserted = False
    if pywrdrb_ml_str not in sys.path:
        sys.path.insert(0, pywrdrb_ml_str)
        inserted = True

    ml_modules_to_keep: dict[str, Any] = {}
    try:
        importlib.import_module("src")               # PywrDRB-ML's src package
        importlib.import_module("src.lstm_model")    # chains in torch_bmi etc.
        for name in _ML_LEAF_MODULES:
            if name in sys.modules:
                ml_modules_to_keep[name] = sys.modules[name]
    finally:
        # Drop every src* entry the bootstrap created.
        for k in list(sys.modules):
            if k == "src" or k.startswith("src."):
                del sys.modules[k]
        # Restore NYCOpt's src + submodules.
        if saved_src is not None:
            sys.modules["src"] = saved_src
        for k, v in saved_src_submods.items():
            sys.modules[k] = v
        # Insert PywrDRB-ML's leaf modules under their src.* names. These
        # names don't collide with NYCOpt's src/* layout, and Python's
        # `from src.lstm_model import X` will find them via sys.modules.
        for k, v in ml_modules_to_keep.items():
            sys.modules[k] = v
        if inserted and pywrdrb_ml_str in sys.path:
            sys.path.remove(pywrdrb_ml_str)

    # After the upstream pywrdrb modules are loaded and pywr's parameter
    # registry is populated, register our NYCOpt-side subclasses. Doing this
    # AFTER the bootstrap ensures the parent class (FlowTargetSaltFrontAdjustmentRatio)
    # is importable.
    _register_nycopt_parameters()


def _register_nycopt_parameters() -> None:
    """Import + register NYCOpt's custom Pywr parameter subclasses.

    Called from the LSTM bootstrap so that the model JSON's `type` strings
    (e.g. ``"NYCOptParameterizedSaltFrontAdjustmentRatio"``) resolve when
    `pywrdrb.Model.load()` parses the model.
    """
    # Import inside the function so importing this module without LSTMs on
    # doesn't drag in the parameter subclass machinery.
    try:
        from src.parameters import parameterized_salt_front_adjustment  # noqa: F401
    except Exception as e:
        # Failing to import a custom parameter is a real bug; surface it
        # rather than silently skipping.
        raise ImportError(
            f"Failed to register NYCOpt parameter subclasses: {type(e).__name__}: {e}"
        ) from e


def _resolve_lstm_end_date() -> str:
    """LSTM end_date should track the simulation end_date.

    `simulation.py` honors `PYWRDRB_SIM_END_DATE` env override (e.g. for
    DEBUG_SIM=true 5-yr smoke runs). Mirror that here so the LSTM's
    internal time axis covers the same window the model actually runs.
    """
    return os.environ.get("PYWRDRB_SIM_END_DATE") or END_DATE


def build_salinity_options() -> dict[str, Any]:
    """Return the `salinity_model` options dict for `pywrdrb.ModelBuilder`.

    Inputs to the salinity LSTM are pywrdrb-simulated flows
    (`Q_Trenton_bc`, `Q_Schuylkill_bc`) which are always available, so this
    LSTM is robust to scenario sources (deterministic, stochastic, climate-
    perturbed). The published parameter is `salt_front_location_mu` (river
    mile, 7-day average).

    `debug=True` is hardcoded — in PywrDRB-ML the `debug` flag toggles
    whether `ml_model.records` (per-timestep `sf_mu`/`sf_sd` series) is
    populated, NOT verbose logging. The async-update post-process in
    `simulation.py::_postprocess_async_salinity()` relies on those records
    to extract a time-series objective. The memory cost is ~5×T floats
    (trivial).

    Returns:
        Dict suitable for `options={"salinity_model": <this>, ...}`. The
        ModelBuilder consumes it at `add_parameter_salinity_model()`.
    """
    return {
        "ml_model_type": "lstm",
        "PywrDRB_ML_plugin_path": str(PYWRDRB_ML_DIR),
        "model_salinity": str(SALINITY_LSTM_MODEL),
        "start_date": LSTM_START_DATE,
        "end_date": _resolve_lstm_end_date(),
        "Q_Trenton_lstm_var_name": "Q_Trenton_bc",
        "Q_Schuylkill_lstm_var_name": "Q_Schuylkill_bc",
        "asycronized_update": bool(SALINITY_ASYNC_UPDATE),
        "debug": True,
    }


def build_temperature_options() -> dict[str, Any]:
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
        "end_date": _resolve_lstm_end_date(),
        "activate_thermal_control": False,
        "Q_C_lstm_var_name": "QbcTavg_Q_C",
        "Q_i_lstm_var_name": "QbcTavg_Q_i",
        "cannonsville_storage_pct_lstm_var_name": "bc_cannonsville_storage_pct",
        "thermal_mitigation_bank_size": 1620,  # MGD
        "asycronized_update": False,
        "debug": debug,
    }


def build_lstm_options_block() -> dict[str, Any]:
    """Return the LSTM portion of the ModelBuilder `options` dict.

    Includes only the sub-dicts whose `INCLUDE_*_MODEL` toggle is True. If
    neither is on, returns an empty dict — the caller can merge it into a
    larger options dict without conditional logic. When at least one LSTM
    is requested, runs `_bootstrap_pywrdrb_ml_namespace()` once to make
    `from src.lstm_model import ...` resolve to PywrDRB-ML's module.
    """
    block: dict[str, Any] = {}
    if not (INCLUDE_TEMPERATURE_MODEL or INCLUDE_SALINITY_MODEL):
        return block
    _bootstrap_pywrdrb_ml_namespace()
    if INCLUDE_TEMPERATURE_MODEL:
        block["temperature_model"] = build_temperature_options()
    if INCLUDE_SALINITY_MODEL:
        block["salinity_model"] = build_salinity_options()
    return block
