"""
external_policy.py - Post-Load Model Replacement (PLMR) for external policy evaluation.

Replaces FFMP release parameters on NYC reservoir outflow nodes with an
ExternalPolicyParameter that delegates release decisions to an arbitrary
callable. This enables optimization of RBF, ANN, and tree-based policies.

PLMR operates between Model.load() and model.run():
  1. Load the standard Pywr-DRB model (with FFMP parameters)
  2. Replace outflow_{res}.max_flow with ExternalPolicyParameter
  3. Run the model — external policy controls releases

Two action modes:
  - "individual": policy outputs [release_can, release_pep, release_nev] in MGD
  - "aggregate": policy outputs [total_release] in MGD, then a built-in
    volume-balancing rule distributes across reservoirs proportional to
    relative storage (mimicking FFMP VolBalanceNYCDownstreamMRF logic)
"""

import os
import sys
import copy
import json
import numpy as np
from pathlib import Path

from pywr.parameters import Parameter

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import pywrdrb at module level so the geopandas/shapely import error (NumPy 2.x
# incompatibility in dataretrieval) is handled once by pywrdrb/__init__.py's
# try/except wrapper rather than propagating through deferred imports mid-eval.
import pywrdrb  # noqa: F401

from config import (
    NYC_RESERVOIRS,
    NYC_RESERVOIR_CAPACITIES,
    NYC_TOTAL_CAPACITY,
)

###############################################################################
# State feature registry
###############################################################################
# Single source of truth for every feature a policy can observe. Users
# select features by listing names (strings) in config.STATE_FEATURES, or
# by passing dict entries directly to build_state_config() for ad-hoc use.
#
# Each registry entry is a dict with the same schema consumed by
# _resolve_state_sources / _extract_state_vector:
#   - one of: 'combined_nodes' (list), 'node' (str), 'parameter' (str)
#   - 'attribute' (for nodes)
#   - 'normalize_by' (scalar divisor; omit for no normalization)
#
# sin/cos(DOY) are always appended automatically by _extract_state_vector
# and are NOT part of this registry.

# Total NYC reservoir capacity (MG) used for combined-storage normalization.
_TOTAL_NYC_CAPACITY = NYC_TOTAL_CAPACITY  # 270,837.0 MG

# Normalization constants — 99th percentile of historical perfect-foresight
# predictions (pub_nhmv10_BC_withObsScaled, 1945-2022):
_MONTAGUE_FLOW_NORM = 18300.0
_TRENTON_FLOW_NORM = 26000.0
_NJ_DEMAND_NORM = 105.0

# Number of temporal features appended automatically by _extract_state_vector.
N_STATE_TEMPORAL = 2  # sin(DOY), cos(DOY)


def _make_feature_registry() -> dict:
    reg: dict = {}

    # --- Storage features ---
    reg["combined_nyc_storage_frac"] = {
        "combined_nodes": [f"reservoir_{res}" for res in NYC_RESERVOIRS],
        "attribute": "volume",
        "normalize_by": _TOTAL_NYC_CAPACITY,
    }
    for res in NYC_RESERVOIRS:
        reg[f"{res}_storage_frac"] = {
            "node": f"reservoir_{res}",
            "attribute": "volume",
            "normalize_by": NYC_RESERVOIR_CAPACITIES[res],
        }

    # --- Montague non-NYC flow forecasts (lags 1-2) ---
    for lag in (1, 2):
        reg[f"montague_flow_lag{lag}"] = {
            "parameter": f"predicted_nonnyc_gage_flow_delMontague_lag{lag}",
            "normalize_by": _MONTAGUE_FLOW_NORM,
        }

    # --- Trenton non-NYC flow forecasts (lags 1-4) ---
    for lag in (1, 2, 3, 4):
        reg[f"trenton_flow_lag{lag}"] = {
            "parameter": f"predicted_nonnyc_gage_flow_delTrenton_lag{lag}",
            "normalize_by": _TRENTON_FLOW_NORM,
        }

    # --- NJ demand forecasts (lags 1-4) ---
    for lag in (1, 2, 3, 4):
        reg[f"nj_demand_lag{lag}"] = {
            "parameter": f"predicted_demand_nj_lag{lag}",
            "normalize_by": _NJ_DEMAND_NORM,
        }

    return reg


STATE_FEATURE_REGISTRY: dict = _make_feature_registry()


def list_available_features() -> str:
    """Return a formatted listing of all registered state features."""
    lines = [f"Available state features ({len(STATE_FEATURE_REGISTRY)}):"]
    for name in STATE_FEATURE_REGISTRY:
        lines.append(f"  {name}")
    lines.append("  (sin/cos of DOY are appended automatically.)")
    return "\n".join(lines)


###############################################################################
# Policy cache (shared across ExternalPolicyParameter instances)
###############################################################################

_POLICY_CACHE = {}


def _clear_policy_cache():
    _POLICY_CACHE.clear()


###############################################################################
# Shared state-extraction helpers
###############################################################################

def _resolve_state_sources(model, state_config: list) -> list:
    """Resolve state config entries to live model objects.

    Returns a list of (source_type, source_object) tuples parallel to
    state_config. source_type is one of 'node', 'combined_nodes', 'parameter'.
    """
    resolved = []
    for cfg in state_config:
        if 'combined_nodes' in cfg:
            nodes = [model.nodes[n] for n in cfg['combined_nodes']]
            resolved.append(('combined_nodes', nodes))
        elif 'node' in cfg:
            resolved.append(('node', model.nodes[cfg['node']]))
        elif 'parameter' in cfg:
            resolved.append(('parameter', model.parameters[cfg['parameter']]))
    return resolved


def _extract_state_vector(state_config: list, resolved_sources: list,
                          timestep, scenario_index) -> np.ndarray:
    """Extract a normalized state vector from the current model state.

    Iterates over state_config / resolved_sources, applies normalize_by
    scaling, then appends sin(DOY) and cos(DOY) at the end.

    Args:
        state_config: List of state spec dicts (from build_state_config).
        resolved_sources: Parallel list of (source_type, source_object) from
            _resolve_state_sources.
        timestep: Pywr timestep object.
        scenario_index: Pywr scenario index object.

    Returns:
        1-D numpy array of length len(state_config) + N_STATE_TEMPORAL.
    """
    sid = scenario_index.global_id
    state = []
    for cfg, (source_type, source) in zip(state_config, resolved_sources):
        if source_type == 'combined_nodes':
            # Sum volumes across all nodes, then normalize.
            val = sum(float(getattr(n, cfg['attribute'])[sid]) for n in source)
        elif source_type == 'node':
            raw = getattr(source, cfg['attribute'])
            if hasattr(raw, '__getitem__'):
                raw = raw[sid]
            val = float(raw)
        else:  # 'parameter'
            val = float(source.get_value(scenario_index))
        norm = cfg.get('normalize_by')
        if norm:
            val /= norm
        state.append(val)
    doy = timestep.datetime.timetuple().tm_yday
    state.append(np.sin(2.0 * np.pi * doy / 366.0))
    state.append(np.cos(2.0 * np.pi * doy / 366.0))
    return np.array(state, dtype=np.float64)


###############################################################################
# ExternalPolicyParameter (individual mode — 3 outputs)
###############################################################################

class ExternalPolicyParameter(Parameter):
    """Pywr Parameter delegating release decisions to an external callable.

    At each timestep, extracts system state (storage fractions + temporal
    features), passes through a user-supplied policy function, and returns
    one element of the action vector as this parameter's value.

    Multiple instances share the same policy_fn via a timestep-keyed cache,
    so the policy is called exactly once per timestep regardless of how many
    reservoirs it controls.
    """

    def __init__(self, model, policy_fn, state_config, output_index,
                 output_min=0.0, output_max=None, name=None, **kwargs):
        super().__init__(model, name=name, **kwargs)
        self.policy_fn = policy_fn
        self.state_config = state_config
        self.output_index = output_index
        self.output_min = output_min
        self.output_max = output_max
        self._resolved_sources = None

    def setup(self):
        super().setup()
        self._resolved_sources = _resolve_state_sources(self.model, self.state_config)

    def value(self, timestep, scenario_index):
        cache_key = (timestep.index, scenario_index.global_id)
        if cache_key not in _POLICY_CACHE:
            state = _extract_state_vector(
                self.state_config, self._resolved_sources, timestep, scenario_index
            )
            _POLICY_CACHE[cache_key] = self.policy_fn(state)
        action = _POLICY_CACHE[cache_key]
        val = float(action[self.output_index])
        if self.output_max is not None:
            val = min(val, self.output_max)
        return max(val, self.output_min)


###############################################################################
# AggregateBalancedReleaseParameter (aggregate mode — 1 output, volume-balanced)
###############################################################################

class AggregateBalancedReleaseParameter(Parameter):
    """Distributes an aggregate release target across NYC reservoirs by storage.

    The external policy outputs a single total release (MGD). This parameter
    distributes it across the 3 NYC reservoirs proportional to their current
    storage fraction relative to their capacity share of the system.

    Balancing logic (simplified FFMP VolBalanceNYCDownstreamMRF):
      For each reservoir i:
        excess_i = vol_i - (cap_i / cap_total) * (vol_total - total_release)
      Then: release_i = clip(excess_i, 0, max_release_i)
      Rescale so sum(release_i) == total_release.
    """

    def __init__(self, model, policy_fn, state_config, reservoir_index,
                 capacity, total_capacity, capacities_all, max_release,
                 name=None, **kwargs):
        super().__init__(model, name=name, **kwargs)
        self.policy_fn = policy_fn
        self.state_config = state_config
        self.reservoir_index = reservoir_index
        self.capacity = capacity
        self.total_capacity = total_capacity
        self.capacities_all = capacities_all  # list of 3 capacities
        self.max_release = max_release
        self._resolved_sources = None
        self._reservoir_nodes = None

    def setup(self):
        super().setup()
        self._resolved_sources = _resolve_state_sources(self.model, self.state_config)
        self._reservoir_nodes = [
            self.model.nodes[f'reservoir_{res}'] for res in NYC_RESERVOIRS
        ]

    def _balance_releases(self, total_release, scenario_index):
        """Distribute total_release across 3 reservoirs by relative storage."""
        sid = scenario_index.global_id
        n = len(self._reservoir_nodes)

        volumes = np.array([
            float(self._reservoir_nodes[i].volume[sid]) for i in range(n)
        ])
        caps = np.array(self.capacities_all)
        vol_total = volumes.sum()
        cap_total = caps.sum()

        target_remaining = max(vol_total - total_release, 0.0)
        vol_targets = (caps / cap_total) * target_remaining

        releases = volumes - vol_targets
        releases = np.clip(releases, 0.0, None)

        max_releases = np.array([
            _CONTROLLED_MAX_RELEASE_MGD[res] for res in NYC_RESERVOIRS
        ])
        releases = np.minimum(releases, max_releases)

        release_sum = releases.sum()
        if release_sum > 1e-6:
            releases *= min(total_release, release_sum) / release_sum
        elif total_release > 0:
            releases = np.full(n, total_release / n)
            releases = np.minimum(releases, max_releases)

        return releases

    def value(self, timestep, scenario_index):
        cache_key = (timestep.index, scenario_index.global_id)
        balance_cache_key = ("balanced", timestep.index, scenario_index.global_id)

        if balance_cache_key not in _POLICY_CACHE:
            if cache_key not in _POLICY_CACHE:
                state = _extract_state_vector(
                    self.state_config, self._resolved_sources, timestep, scenario_index
                )
                _POLICY_CACHE[cache_key] = self.policy_fn(state)
            action = _POLICY_CACHE[cache_key]
            total_release = max(float(action[0]), 0.0)
            _POLICY_CACHE[balance_cache_key] = self._balance_releases(
                total_release, scenario_index
            )

        balanced = _POLICY_CACHE[balance_cache_key]
        return float(balanced[self.reservoir_index])


###############################################################################
# CacheClearParameter
###############################################################################

class CacheClearParameter(Parameter):
    """Clears the policy cache once per timestep."""

    def value(self, timestep, scenario_index):
        _clear_policy_cache()
        return 0.0


###############################################################################
# PLMR: Apply External Policy to Model
###############################################################################

# Controlled max release limits (MGD) — from FFMP flood release caps
_CONTROLLED_MAX_RELEASE_MGD = {
    "cannonsville": 4200.0 * 0.645932368556,   # ~2713 MGD
    "pepacton": 2400.0 * 0.645932368556,        # ~1550 MGD
    "neversink": 3400.0 * 0.645932368556,       # ~2196 MGD
}


def build_state_config(features=None) -> list:
    """Assemble a state_config list from feature names and/or inline dicts.

    The default feature list in `config.STATE_FEATURES` mirrors the
    information used by FFMP's decision logic: combined NYC storage, Montague
    non-NYC flow at lag 2d, Trenton non-NYC flow at lag 4d, and NJ demand at
    lag 4d. sin/cos(DOY) are appended automatically by the state extractor.

    Items may be:
      - str:  look up in STATE_FEATURE_REGISTRY
      - dict: use directly (must have one of combined_nodes/node/parameter,
              plus optional 'attribute' and 'normalize_by' keys)

    Args:
        features: Iterable of str | dict, or None to use config.STATE_FEATURES.

    Returns:
        List of state_config dicts (sin/cos NOT included — appended at extract).

    Raises:
        KeyError: A string feature name is not in the registry.
        TypeError: An item is neither a string nor a dict.
    """
    if features is None:
        from config import STATE_FEATURES
        features = STATE_FEATURES

    cfg: list = []
    for item in features:
        if isinstance(item, dict):
            cfg.append(dict(item))  # shallow copy to decouple from caller
        elif isinstance(item, str):
            if item not in STATE_FEATURE_REGISTRY:
                raise KeyError(
                    f"Unknown state feature '{item}'. "
                    f"Available: {sorted(STATE_FEATURE_REGISTRY)}"
                )
            cfg.append(dict(STATE_FEATURE_REGISTRY[item]))
        else:
            raise TypeError(
                f"build_state_config items must be str or dict; "
                f"got {type(item).__name__}"
            )
    return cfg


def apply_external_policy(model, policy_fn, mode="aggregate",
                          state_features=None,
                          reservoir_capacities=None, max_releases=None):
    """Replace FFMP release parameters with external policy via PLMR.

    Must be called between Model.load() and model.run().

    Replaces:
      1. outflow_{res}.max_flow — controls actual downstream releases
      2. VolBalanceNYCDemand's downstream_release_target references —
         so delivery balancing sees the policy's releases, not FFMP targets
      3. VolBalanceNYCDemand's mrf_target_individual and flood_release
         references — zeroed since the external policy subsumes these

    Args:
        model: Loaded pywr.Model instance.
        policy_fn: Callable(np.ndarray) -> np.ndarray.
            Input: state vector (see build_state_config for layout)
            Output depends on mode:
              "individual": [release_can, release_pep, release_nev] in MGD
              "aggregate":  [total_release] in MGD (balanced internally)
        mode: "individual" or "aggregate" (default: "aggregate").
        state_features: List of feature names (or dicts) passed to
            build_state_config(). None uses config.STATE_FEATURES.
        reservoir_capacities: Dict {name: capacity_MG}.
        max_releases: Dict {name: max_release_MGD}.

    Returns:
        model (modified in-place).
    """
    from pywr.parameters import ConstantParameter

    if reservoir_capacities is None:
        reservoir_capacities = NYC_RESERVOIR_CAPACITIES
    if max_releases is None:
        max_releases = _CONTROLLED_MAX_RELEASE_MGD

    CacheClearParameter(model, name="policy_cache_clear")

    reservoirs = list(NYC_RESERVOIRS)
    state_config = build_state_config(features=state_features)
    capacities_list = [reservoir_capacities[res] for res in reservoirs]

    ext_params = []
    if mode == "individual":
        for i, res in enumerate(reservoirs):
            ext_param = ExternalPolicyParameter(
                model, policy_fn=policy_fn, state_config=state_config,
                output_index=i, output_min=0.0,
                output_max=max_releases.get(res),
                name=f"external_policy_release_{res}",
            )
            ext_params.append(ext_param)

    elif mode == "aggregate":
        for i, res in enumerate(reservoirs):
            ext_param = AggregateBalancedReleaseParameter(
                model, policy_fn=policy_fn, state_config=state_config,
                reservoir_index=i, capacity=reservoir_capacities[res],
                total_capacity=sum(capacities_list),
                capacities_all=capacities_list,
                max_release=max_releases.get(res),
                name=f"external_policy_balanced_release_{res}",
            )
            ext_params.append(ext_param)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'individual' or 'aggregate'.")

    # 1. Replace outflow node max_flow
    for i, res in enumerate(reservoirs):
        model.nodes[f"outflow_{res}"].max_flow = ext_params[i]

    # 2. Patch VolBalanceNYCDemand to see external policy releases
    zero_param = ConstantParameter(model, value=0.0, name="plmr_zero")
    _patch_vol_balance_nyc_demand(model, ext_params, zero_param, reservoirs)

    return model


def _patch_vol_balance_nyc_demand(model, ext_params, zero_param, reservoirs):
    """Patch VolBalanceNYCDemand parameter references to see external policy releases."""
    for p in model.parameters:
        if type(p).__name__ != "VolBalanceNYCDemand":
            continue

        if hasattr(p, 'downstream_release_target_reservoirs'):
            old_refs = p.downstream_release_target_reservoirs
            for i in range(min(len(reservoirs), len(old_refs))):
                try:
                    p.children.remove(old_refs[i])
                except (ValueError, KeyError):
                    pass
                p.downstream_release_target_reservoirs[i] = ext_params[i]
                p.children.add(ext_params[i])

        if hasattr(p, 'mrf_target_individual_reservoirs'):
            old_refs = p.mrf_target_individual_reservoirs
            for i in range(min(len(reservoirs), len(old_refs))):
                try:
                    p.children.remove(old_refs[i])
                except (ValueError, KeyError):
                    pass
                p.mrf_target_individual_reservoirs[i] = zero_param
                p.children.add(zero_param)

        if hasattr(p, 'flood_release_reservoirs'):
            old_refs = p.flood_release_reservoirs
            for i in range(min(len(reservoirs), len(old_refs))):
                try:
                    p.children.remove(old_refs[i])
                except (ValueError, KeyError):
                    pass
                p.flood_release_reservoirs[i] = zero_param
                p.children.add(zero_param)


###############################################################################
# Pre-Load Dict Stripping
###############################################################################

_STRIPPABLE_PARAMS = [
    "downstream_release_target_cannonsville",
    "downstream_release_target_pepacton",
    "downstream_release_target_neversink",
    "flood_release_cannonsville",
    "flood_release_pepacton",
    "flood_release_neversink",
    "mrf_baseline_cannonsville",
    "mrf_baseline_pepacton",
    "mrf_baseline_neversink",
    "mrf_drought_factor_combined_final_cannonsville",
    "mrf_drought_factor_combined_final_pepacton",
    "mrf_drought_factor_combined_final_neversink",
    "mrf_montagueTrenton_cannonsville",
    "mrf_montagueTrenton_pepacton",
    "mrf_montagueTrenton_neversink",
    "mrf_target_individual_cannonsville",
    "mrf_target_individual_pepacton",
    "mrf_target_individual_neversink",
]


def _strip_ffmp_from_dict(model_dict):
    """Replace NYC FFMP release parameters with constants in model_dict."""
    params = model_dict["parameters"]
    for pname in _STRIPPABLE_PARAMS:
        if pname in params:
            params[pname] = {"type": "constant", "value": 0.0}

    for node in model_dict["nodes"]:
        if node["name"] in (
            "outflow_cannonsville", "outflow_pepacton", "outflow_neversink"
        ):
            node["max_flow"] = 0.0


###############################################################################
# Evaluation Functions
###############################################################################

def _build_and_load_model(use_trimmed=None, strip_ffmp=True):
    """Build model dict, write JSON, load pywr.Model."""
    from src.simulation import (
        _get_cached_model_dict, _patch_model_dict,
        _get_cached_defaults, _get_temp_dir,
    )

    base_dict = _get_cached_model_dict(use_trimmed=use_trimmed)
    model_dict = copy.deepcopy(base_dict)
    defaults = _get_cached_defaults()
    _patch_model_dict(model_dict, defaults)

    if strip_ffmp:
        _strip_ffmp_from_dict(model_dict)

    tmp_dir = _get_temp_dir()
    model_json = str(Path(tmp_dir) / "plmr_model.json")
    with open(model_json, "w") as f:
        json.dump(model_dict, f)
    return pywrdrb.Model.load(model_json)


def evaluate_with_policy(policy_fn, mode="aggregate",
                         state_features=None,
                         objective_set=None, use_trimmed=None):
    """Evaluate an external policy through the Pywr-DRB model.

    Args:
        policy_fn: Callable(np.ndarray) -> np.ndarray.
        mode: "individual" or "aggregate" (default: "aggregate").
        state_features: Feature names / dicts for build_state_config.
            If None, uses config.STATE_FEATURES.
        objective_set: ObjectiveSet instance. If None, uses active set.
        use_trimmed: Whether to use the trimmed model.

    Returns:
        List of objective values (Borg-compatible, all minimized).
    """
    from src.simulation import InMemoryRecorder, _extract_results_from_recorder
    from config import get_objective_set as _get_objective_set

    if objective_set is None:
        objective_set = _get_objective_set()

    _clear_policy_cache()
    model = _build_and_load_model(use_trimmed=use_trimmed)
    apply_external_policy(model, policy_fn, mode=mode, state_features=state_features)

    mem_recorder = InMemoryRecorder(model)
    model.run()

    datetime_index = model.timestepper.datetime_index.to_timestamp()
    data = _extract_results_from_recorder(mem_recorder.recorder_dict, datetime_index)
    objs = objective_set.compute_for_borg(data)

    del model, mem_recorder
    return objs


def run_policy_simulation(policy_fn, mode="aggregate",
                          state_features=None, use_trimmed=None):
    """Run a policy simulation and return raw results (no objectives).

    Returns:
        Dict of DataFrames keyed by results set name.
    """
    from src.simulation import InMemoryRecorder, _extract_results_from_recorder

    _clear_policy_cache()
    model = _build_and_load_model(use_trimmed=use_trimmed)
    apply_external_policy(model, policy_fn, mode=mode, state_features=state_features)

    mem_recorder = InMemoryRecorder(model)
    model.run()

    datetime_index = model.timestepper.datetime_index.to_timestamp()
    data = _extract_results_from_recorder(mem_recorder.recorder_dict, datetime_index)

    del model, mem_recorder
    return data
