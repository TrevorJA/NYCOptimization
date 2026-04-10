# Architecture Design: External Policy Integration with pywrdrb

**Date:** 2026-04-09  
**Status:** DESIGN (pre-implementation)  
**Scope:** Minimal edits to support Architectures A, B, C/D/E without modifying existing pywrdrb files

---

## Executive Summary

This document specifies the MINIMAL set of NEW files and classes required to add external policy control to the pywrdrb/NYCOptimization workflow, WITHOUT modifying any existing pywrdrb source code.

**Key Design Decisions:**

1. **No pywrdrb source modifications** — All new functionality lives in NEW Python files in pywrdrb, or in the NYCOptimization `simulation.py` wrapper.

2. **Strategy: Post-Load Model Replacement (PLMR)** — After `Model.load(model_dict_json)`, programmatically replace release parameters with custom Parameter subclasses that delegate to external policy functions. This requires NO changes to existing pywrdrb Parameter classes.

3. **Fixed MRF Targets for Fair Comparison** — All architectures (A, B, C/D/E) are evaluated against the SAME fixed downstream flow targets (Montague=1,131 MGD, Trenton=1,939 MGD), eliminating the FFMP's measurement advantage from dynamic target relaxation.

4. **Two Execution Paths in `simulation.py`:**
   - **Path 1 (FFMP, A/B):** Standard caching + patching of model_dict, no changes to model structure
   - **Path 2 (Policy, C/D/E):** Load cached model, then replace release parameters with ExternalPolicyParameter instances before calling model.run()

5. **ExternalPolicyParameter in NEW file** — Minimal custom Parameter class that (a) reads state from Pywr nodes, (b) calls an external policy function, (c) caches results per timestep to handle multi-output policies.

---

## Part I: Problem Decomposition

### 1.1 Current FFMP Control Chain (Architecture A)

```
Input: NYCOperationsConfig with parameterized FFMP rules
         ↓
ModelBuilder.make_model() → model_dict (JSON serializable)
         ↓
model_dict["parameters"]: contains 386 entries including:
  - controlcurveindex: parameter class="ScenarioDependentParameter"
  - level1a, level1b, ..., level5: 366-day zone thresholds
  - level1a_factor_mrf_delMontague, etc.: monthly MRF factors (12 values each)
         ↓
Model.load(model_json) → Pywr Model instance
         ↓
Pywr LP Solver each timestep:
  1. Evaluate controlcurveindex → drought_level (0-6)
  2. Use drought_level to index into monthly profile params → mrf_target_montague, mrf_target_trenton
  3. Enforce MRF constraints in LP solver
  4. Set release link max_flow from reservoir nodes
         ↓
Output: Time series flows and storages
```

### 1.2 The Challenge: External Policy Override

For Architectures C/D/E (RBF, Tree, ANN), we need:

```
Input: Policy function (pre-trained, from optimization DVs)
         ↓
At each timestep:
  1. Read current storage (aggregate or per-reservoir)
  2. Read current inflows
  3. Compute day-of-year (for seasonal features)
  4. Call policy(state) → action vector (release fractions or absolute releases)
  5. Set release link max_flow from policy output
         ↓
Pywr LP Solver:
  Allocates water subject to fixed MRF targets (Montague=1,131 MGD, Trenton=1,939 MGD)
  Respects policy's release guidance IF feasible; may adjust to meet MRF constraints
         ↓
Output: Time series flows and storages
```

### 1.3 Why Post-Load Model Replacement (PLMR)?

**Option A: Modify model_dict before Model.load()**
- Pro: Avoids runtime Parameter replacement
- Con: Requires serializing the policy function to JSON (impossible for RBF, Tree, ANN)
- Con: Requires modifying ModelBuilder's parameter-generation logic

**Option B: Create a custom Model subclass**
- Pro: Encapsulates the replacement logic
- Con: Requires monkey-patching Pywr's load() or creating a wrapper, still fragile

**Option C: Post-Load Model Replacement (CHOSEN)**
- Pro: Load standard model, then programmatically replace Parameter instances
- Pro: No changes to existing pywrdrb classes
- Pro: Clean separation: standard model structure is built normally, then enhanced
- Con: Requires understanding which parameters to replace (identified via model inspection)

---

## Part II: Architecture Overview

### 2.1 New Files (No Existing Files Modified)

**pywrdrb/parameters/external_policy.py** (NEW)
- `ExternalPolicyParameter`: Custom Pywr Parameter that reads state and calls policy_fn
- `_PolicyCache`: Thread-safe cache for multi-output policies
- Helper functions for state extraction and normalization

**pywrdrb/parameters/fixed_target.py** (NEW, OPTIONAL)
- `FixedMRFParameter`: Simple wrapper that returns a constant (for fixed-target mode)
- Used when policy evaluation uses fixed targets instead of FFMP's dynamic targets

**NYCOptimization/optimization/policies/\_\_init\_\_.py** (NEW)
- Import all policy classes

**NYCOptimization/optimization/policies/base.py** (NEW)
- `PolicyBase` ABC: common interface for all policy architectures

**NYCOptimization/optimization/policies/rbf_policy.py** (NEW)
- `RBFPolicy`: thin-plate spline radial basis function policy

**NYCOptimization/optimization/policies/tree_policy.py** (NEW)
- `ObliqueTreePolicy`: gradient-boosted tree with oblique splits

**NYCOptimization/optimization/policies/ann_policy.py** (NEW)
- `ANNPolicy`: artificial neural network policy

**NYCOptimization/optimization/formulations.py** (NEW)
- `generate_ffmp_formulation(n_zones)`: variable-resolution FFMP generator
- `register_formulation(name, formulation_dict)`: central registry for formulations
- Updated `config.py` to use the new registry

**NYCOptimization/src/model_builder_utils.py** (NEW)
- `replace_release_parameters(model, policy_fn, state_config)`: Post-load model replacement
- `set_fixed_mrf_targets(model, montague_mgd, trenton_mgd)`: Replace FFMP dynamic targets with fixed constants
- `get_parameters_to_replace()`: Introspect model_dict to find release parameters

**NYCOptimization/src/simulation_policy.py** (NEW)
- `evaluate_with_policy()`: Borg-compatible evaluation function for C/D/E architectures
- Wraps Policy instance, calls model replacement, runs simulation, computes objectives

---

### 2.2 Modified Files (Minimal Changes)

**NYCOptimization/src/simulation.py**
- Add import: `from src.simulation_policy import evaluate_with_policy`
- Modify `evaluate()` function to dispatch based on formulation_name:
  ```python
  def evaluate(dv_vector, formulation_name="ffmp", objective_set=None):
      if formulation_name in ("ffmp", "ffmp_vr_3", "ffmp_vr_6", "ffmp_vr_10"):
          # Existing Path 1: FFMP (no changes to core logic)
          return evaluate_ffmp(dv_vector, formulation_name, objective_set)
      elif formulation_name in ("rbf", "tree", "ann"):
          # New Path 2: External Policy
          return evaluate_with_policy(dv_vector, formulation_name, objective_set)
      else:
          raise ValueError(f"Unknown formulation: {formulation_name}")
  ```

**NYCOptimization/config.py**
- Import `register_formulation` from `optimization/formulations.py`
- Replace hardcoded `FORMULATIONS = {"ffmp": {...}}` with:
  ```python
  from optimization.formulations import register_formulation, generate_ffmp_formulation
  
  FORMULATIONS = {}
  
  # Register Formulation A: Standard FFMP (6 zones)
  register_formulation("ffmp", generate_ffmp_formulation(n_zones=6, base_formulation="ffmp"))
  
  # Register Formulation B: Variable-resolution FFMP
  register_formulation("ffmp_vr_3", generate_ffmp_formulation(n_zones=3))
  register_formulation("ffmp_vr_10", generate_ffmp_formulation(n_zones=10))
  
  # Policy formulations will be registered by the policy code itself
  ```

---

## Part III: Detailed Design

### 3.1 ExternalPolicyParameter (pywrdrb/parameters/external_policy.py)

```python
import numpy as np
from typing import Callable, List, Dict, Optional
from pywr.parameters import Parameter
import logging

logger = logging.getLogger(__name__)

# Global cache for multi-output policies (cleared at start of each timestep)
_POLICY_CACHE: Dict[tuple, np.ndarray] = {}

class ExternalPolicyParameter(Parameter):
    """Pywr Parameter that delegates decisions to an external policy function.
    
    Reads system state (storage, inflows, time of year) from Pywr nodes at each
    timestep, calls an external policy function, and returns one element of the
    policy's action vector.
    
    Used for Architectures C, D, E (RBF, Tree, ANN policies).
    
    Attributes:
        model: Pywr Model instance
        policy_fn: Callable(state: np.ndarray) -> action: np.ndarray
        state_config: List of dicts specifying which node attributes to read
        output_index: Which element of the action vector to return (0-indexed)
        output_min: Minimum clipping value (default 0.0)
        output_max: Maximum clipping value (default None, no clip)
        name: Parameter name
    """
    
    def __init__(
        self,
        model,
        policy_fn: Callable,
        state_config: List[Dict],
        output_index: int,
        output_min: float = 0.0,
        output_max: Optional[float] = None,
        name: str = None,
        **kwargs
    ):
        """Initialize ExternalPolicyParameter.
        
        Args:
            model: Pywr Model instance
            policy_fn: Callable that takes np.ndarray of state and returns action vector
            state_config: List of dicts, each containing:
                {
                    'node': str,  # Name of Pywr node to read
                    'attribute': str,  # 'volume', 'flow', etc.
                    'normalize_by': float or None,  # Divisor for normalization
                }
            output_index: Integer index of action vector to return
            output_min: Lower clipping bound (default 0.0)
            output_max: Upper clipping bound (default None = no clip)
            name: Parameter name (for debugging)
        """
        super().__init__(model, name=name, **kwargs)
        self.policy_fn = policy_fn
        self.state_config = state_config
        self.output_index = output_index
        self.output_min = output_min
        self.output_max = output_max
        
    def _extract_state(self, timestep, scenario_index) -> np.ndarray:
        """Extract state vector from Pywr nodes.
        
        Returns:
            np.ndarray of length len(state_config) + 2 (temporal features added)
        """
        state = []
        
        for cfg in self.state_config:
            node_name = cfg['node']
            attribute = cfg['attribute']
            normalize_by = cfg.get('normalize_by')
            
            # Find the node
            if node_name not in self.model.nodes:
                raise KeyError(f"Node '{node_name}' not found in model. "
                             f"Available: {list(self.model.nodes.keys())[:5]}...")
            
            node = self.model.nodes[node_name]
            
            # Get attribute value (handle scenario-indexed arrays)
            raw_val = getattr(node, attribute)
            if isinstance(raw_val, np.ndarray):
                # Scenario-indexed array; use global_id to index
                val = float(raw_val[scenario_index.global_id])
            else:
                val = float(raw_val)
            
            # Normalize if specified
            if normalize_by is not None and normalize_by != 0:
                val /= normalize_by
            
            state.append(val)
        
        # Add temporal features (day of year, encoded as sin/cos)
        doy = timestep.datetime.timetuple().tm_yday
        state.append(np.sin(2.0 * np.pi * doy / 366.0))
        state.append(np.cos(2.0 * np.pi * doy / 366.0))
        
        return np.array(state, dtype=np.float64)
    
    def value(self, timestep, scenario_index):
        """Return the policy output value for this parameter.
        
        Caches the policy action vector so that multi-output policies
        (controlling 3 reservoirs) only call policy_fn ONCE per timestep.
        """
        global _POLICY_CACHE
        
        cache_key = (timestep.index, scenario_index.global_id)
        
        if cache_key not in _POLICY_CACHE:
            # First call for this timestep: extract state and call policy
            state = self._extract_state(timestep, scenario_index)
            try:
                action = self.policy_fn(state)
            except Exception as e:
                logger.error(f"Policy function failed at timestep {timestep.index}: {e}")
                raise
            
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            
            _POLICY_CACHE[cache_key] = action
        
        # Retrieve from cache
        action = _POLICY_CACHE[cache_key]
        
        # Extract the requested output and clip
        if self.output_index >= len(action):
            raise IndexError(
                f"Policy output_index {self.output_index} out of range for action "
                f"vector of length {len(action)}"
            )
        
        val = float(action[self.output_index])
        
        # Clip to bounds
        if self.output_max is not None:
            val = np.clip(val, self.output_min, self.output_max)
        else:
            val = max(val, self.output_min)
        
        return val


def clear_policy_cache():
    """Clear the policy cache (called at start of each timestep by CacheClearParameter)."""
    global _POLICY_CACHE
    _POLICY_CACHE.clear()


class CacheClearParameter(Parameter):
    """Dummy parameter that clears the policy cache at the start of each timestep.
    
    Pywr evaluates parameters in definition order at each timestep, so by placing
    this parameter first, we ensure the cache is cleared before any
    ExternalPolicyParameter is evaluated.
    """
    
    def __init__(self, model, name="cache_clear", **kwargs):
        super().__init__(model, name=name, **kwargs)
    
    def value(self, timestep, scenario_index):
        clear_policy_cache()
        return 0.0  # Dummy return value
```

**Key Design Points:**

1. **State Extraction:** Reads from any Pywr node attribute (volume, flow, etc.). Normalizes by a divisor (e.g., divide volume by capacity to get fraction). Adds temporal features (sin/cos of DOY).

2. **Caching:** Multi-output policies call `policy_fn` ONCE per (timestep, scenario), cache the result. Three ExternalPolicyParameter instances (for cannonsville, pepacton, neversink) all read from the same cache.

3. **Error Handling:** Clear logging if policy_fn fails or output_index is out of range.

4. **CacheClearParameter:** Dummy parameter inserted at the START of the parameter evaluation order to clear the cache. Ensures fresh state at each timestep.

---

### 3.2 PolicyBase ABC (NYCOptimization/optimization/policies/base.py)

```python
from abc import ABC, abstractmethod
import numpy as np

class PolicyBase(ABC):
    """Abstract base class for all policy architectures.
    
    Provides a common interface for policies to be optimized by Borg MOEA.
    Each policy takes a state vector and returns an action vector (releases for 3 reservoirs).
    """
    
    @abstractmethod
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Evaluate policy: state -> action.
        
        Args:
            state: np.ndarray of length >= 4 (3 storage values + time features)
        
        Returns:
            np.ndarray of length >= 3 (one release per reservoir, in MGD)
        """
        pass
    
    @abstractmethod
    def set_params(self, flat_vector: np.ndarray) -> None:
        """Set policy parameters from a flat decision variable vector.
        
        Called by Borg at each evaluation with new DV values.
        
        Args:
            flat_vector: 1D array of decision variable values
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> tuple:
        """Return decision variable bounds.
        
        Returns:
            (lower_bounds: np.ndarray, upper_bounds: np.ndarray)
        """
        pass
    
    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of decision variables for this policy."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Architecture identifier (e.g., 'RBF', 'Tree', 'ANN')."""
        pass
    
    @property
    def state_config(self) -> list:
        """State configuration for ExternalPolicyParameter.
        
        Default implementation: read aggregate NYC storage and inflows.
        Subclasses may override for different state features.
        
        Returns:
            List of dicts with keys: 'node', 'attribute', 'normalize_by'
        """
        return [
            {'node': 'cannonsville', 'attribute': 'volume',
             'normalize_by': 95706.0},  # Capacity in MG
            {'node': 'pepacton', 'attribute': 'volume',
             'normalize_by': 140190.0},
            {'node': 'neversink', 'attribute': 'volume',
             'normalize_by': 34941.0},
        ]
```

---

### 3.3 Model Replacement Utilities (NYCOptimization/src/model_builder_utils.py)

```python
"""Utilities for post-load model modification.

Used to replace FFMP release parameters with ExternalPolicyParameter instances,
or to set fixed MRF targets, without modifying existing pywrdrb code.
"""

import numpy as np
from typing import Callable, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def get_release_parameters_to_replace() -> Dict[str, str]:
    """Return mapping of reservoir names to release parameter keys.
    
    These parameters are hardcoded in pywrdrb's ModelBuilder and determine
    the max_flow constraint on each reservoir's release link.
    
    Returns:
        Dict mapping reservoir name -> parameter key name in model_dict
    """
    return {
        "cannonsville": "rel_max_release_cannonsville",
        "pepacton": "rel_max_release_pepacton",
        "neversink": "rel_max_release_neversink",
    }


def get_mrf_target_parameters() -> Dict[str, str]:
    """Return mapping of downstream location to MRF target parameter keys.
    
    In FFMP mode, these are computed by monthlyprofile parameters.
    For policy mode with fixed targets, these are replaced with constants.
    
    Returns:
        Dict mapping location (e.g., 'delMontague') -> parameter key
    """
    return {
        "delMontague": "mrf_target_delMontague",
        "delTrenton": "mrf_target_delTrenton",
    }


def replace_release_parameters(
    model,
    policy_fn: Callable,
    state_config: List[Dict],
    output_indices: Optional[Dict[str, int]] = None
) -> None:
    """Replace release parameters with ExternalPolicyParameter instances.
    
    This is the core post-load model replacement. Call after Model.load()
    and before model.run().
    
    Args:
        model: Pywr Model instance (already loaded from JSON)
        policy_fn: Callable(state) -> action vector
        state_config: State configuration to pass to ExternalPolicyParameter
        output_indices: Dict mapping reservoir name -> output_index in action vector.
            Default: {'cannonsville': 0, 'pepacton': 1, 'neversink': 2}
    
    Raises:
        KeyError: If a required parameter or link node is not in the model
    """
    if output_indices is None:
        output_indices = {
            "cannonsville": 0,
            "pepacton": 1,
            "neversink": 2,
        }
    
    from pywrdrb.parameters.external_policy import ExternalPolicyParameter, CacheClearParameter
    
    # Insert cache-clear parameter first
    cache_clear_param = CacheClearParameter(model, name="__cache_clear__")
    
    release_params = get_release_parameters_to_replace()
    
    for res_name, param_key in release_params.items():
        if param_key not in model.parameters:
            raise KeyError(
                f"Release parameter '{param_key}' not found for reservoir '{res_name}'. "
                f"Available parameters: {list(model.parameters.keys())[:5]}..."
            )
        
        output_idx = output_indices[res_name]
        
        # Create new ExternalPolicyParameter
        policy_param = ExternalPolicyParameter(
            model=model,
            policy_fn=policy_fn,
            state_config=state_config,
            output_index=output_idx,
            output_min=0.0,
            output_max=None,  # No upper bound; Pywr LP solver will enforce capacity
            name=f"policy_release_{res_name}",
        )
        
        # Replace in model's parameter dict
        model.parameters[param_key] = policy_param
        
        logger.info(
            f"Replaced release parameter '{param_key}' with ExternalPolicyParameter "
            f"(output_index={output_idx})"
        )


def set_fixed_mrf_targets(
    model,
    montague_mgd: float = 1131.05,
    trenton_mgd: float = 1938.95
) -> None:
    """Replace dynamic MRF target parameters with fixed constants.
    
    For policy-based evaluation with fair comparison, all architectures
    (FFMP and policies) are evaluated against the same fixed downstream
    flow targets. This function replaces the FFMP's dynamic monthlyprofile
    parameters with constant values.
    
    Args:
        model: Pywr Model instance
        montague_mgd: Fixed Montague target (MGD, default 1131.05)
        trenton_mgd: Fixed Trenton target (MGD, default 1938.95)
    
    Raises:
        KeyError: If MRF target parameters are not in the model
    """
    from pywrdrb.parameters.fixed_target import FixedMRFParameter
    
    mrf_params = get_mrf_target_parameters()
    targets = {
        "delMontague": montague_mgd,
        "delTrenton": trenton_mgd,
    }
    
    for loc, param_key in mrf_params.items():
        if param_key not in model.parameters:
            logger.warning(
                f"MRF target parameter '{param_key}' not found. Skipping fixed-target "
                f"replacement. Available parameters: {list(model.parameters.keys())[:5]}..."
            )
            continue
        
        target_val = targets[loc]
        
        # Create fixed parameter
        fixed_param = FixedMRFParameter(
            model=model,
            value=target_val,
            name=f"fixed_mrf_{loc}",
        )
        
        # Replace
        model.parameters[param_key] = fixed_param
        
        logger.info(f"Replaced MRF target '{param_key}' with fixed value {target_val} MGD")


def inspect_model_parameters(model, pattern: str = None) -> None:
    """Debug utility: print all model parameters, optionally filtered by pattern.
    
    Args:
        model: Pywr Model instance
        pattern: Optional substring to filter parameter names
    """
    print(f"\nModel has {len(model.parameters)} parameters:")
    for key in sorted(model.parameters.keys()):
        if pattern is None or pattern.lower() in key.lower():
            param = model.parameters[key]
            print(f"  {key}: {type(param).__name__}")
```

---

### 3.4 External Policy Evaluation (NYCOptimization/src/simulation_policy.py)

```python
"""Evaluation function for policy-based architectures (C, D, E).

Parallels the structure of dvs_to_config() and run_simulation_inmemory()
but uses ExternalPolicyParameter instead of FFMP parameters.
"""

import numpy as np
import json
import tempfile
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

# Cache the base model dict (same as for FFMP)
_CACHED_MODEL_DICT_POLICY = None


def _get_cached_model_dict_for_policy(use_trimmed: bool = None):
    """Get cached base model dict for policy evaluation (same as FFMP).
    
    This is the unmodified FFMP model structure. We will replace the release
    parameters after loading.
    """
    global _CACHED_MODEL_DICT_POLICY
    if _CACHED_MODEL_DICT_POLICY is None:
        from simulation import _get_cached_model_dict
        _CACHED_MODEL_DICT_POLICY = _get_cached_model_dict(use_trimmed=use_trimmed)
    return _CACHED_MODEL_DICT_POLICY


def evaluate_with_policy(
    policy,
    model_json_path: str = None,
    use_trimmed: bool = None,
    objective_set = None,
    fixed_mrf_targets: bool = True,
) -> list:
    """Evaluate a policy-based formulation (RBF, Tree, ANN).
    
    This replaces the FFMP evaluation path for Architectures C/D/E.
    
    Execution:
    1. Load base model dict from cache
    2. Write to JSON
    3. Load with Model.load()
    4. Replace release parameters with ExternalPolicyParameter
    5. Optionally replace MRF targets with fixed constants
    6. Run simulation
    7. Compute objectives
    
    Args:
        policy: PolicyBase instance (RBF, Tree, or ANN)
        model_json_path: Optional path to write model JSON. Default: temp file.
        use_trimmed: Whether to use trimmed model
        objective_set: ObjectiveSet for objective computation
        fixed_mrf_targets: If True, set fixed Montague/Trenton targets (fair comparison)
    
    Returns:
        List of objective values (Borg-compatible, all minimized)
    """
    import pywrdrb
    from config import USE_TRIMMED_MODEL, get_objective_set, RESULTS_SETS
    from simulation import InMemoryRecorder, _extract_results_from_recorder, _get_mpi_rank, _get_temp_dir
    from src.model_builder_utils import replace_release_parameters, set_fixed_mrf_targets
    
    if use_trimmed is None:
        use_trimmed = USE_TRIMMED_MODEL
    if objective_set is None:
        objective_set = get_objective_set()
    
    # Load base model dict
    base_dict = _get_cached_model_dict_for_policy(use_trimmed=use_trimmed)
    model_dict = copy.deepcopy(base_dict)  # Deep copy to avoid mutating cache
    
    # Write to JSON
    if model_json_path is None:
        rank = _get_mpi_rank()
        tmp_dir = _get_temp_dir()
        model_json_path = str(Path(tmp_dir) / f"policy_model_r{rank}.json")
    
    with open(model_json_path, "w") as f:
        json.dump(model_dict, f)
    
    # Load model
    try:
        model = pywrdrb.Model.load(model_json_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_json_path}: {e}")
        raise
    
    # Replace release parameters with external policy
    try:
        replace_release_parameters(
            model,
            policy_fn=policy,
            state_config=policy.state_config,
        )
    except Exception as e:
        logger.error(f"Failed to replace release parameters: {e}")
        raise
    
    # Optionally replace MRF targets with fixed values
    if fixed_mrf_targets:
        try:
            set_fixed_mrf_targets(model)
        except Exception as e:
            logger.warning(f"Failed to set fixed MRF targets: {e}")
            # Don't raise; proceed with whatever MRF targets are in the model
    
    # Run simulation
    mem_recorder = InMemoryRecorder(model)
    try:
        model.run()
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise
    
    # Extract results
    datetime_index = model.timestepper.datetime_index.to_timestamp()
    data = _extract_results_from_recorder(mem_recorder.recorder_dict, datetime_index)
    
    del model, mem_recorder
    
    # Compute objectives
    objectives = objective_set.compute_for_borg(data)
    
    return objectives
```

---

### 3.5 Variable-Resolution FFMP (NYCOptimization/optimization/formulations.py)

```python
"""Variable-resolution FFMP formulation generator.

Supports N-zone formulations where N can be 3, 4, 5, 6, 7, 8, 10, etc.
Interpolates zone thresholds and factor profiles to smooth transition
between different resolution levels.
"""

import numpy as np
from collections import OrderedDict
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Global registry of formulations
_FORMULATION_REGISTRY: Dict[str, dict] = {}


def register_formulation(name: str, formulation: dict) -> None:
    """Register a formulation in the global registry."""
    _FORMULATION_REGISTRY[name] = formulation
    logger.info(f"Registered formulation: {name}")


def get_formulation(name: str) -> dict:
    """Retrieve a registered formulation by name."""
    if name not in _FORMULATION_REGISTRY:
        raise KeyError(
            f"Formulation '{name}' not registered. Available: "
            f"{list(_FORMULATION_REGISTRY.keys())}"
        )
    return _FORMULATION_REGISTRY[name]


def generate_ffmp_formulation(n_zones: int, base_formulation: str = "ffmp") -> dict:
    """Generate an N-zone FFMP formulation.
    
    Generalizes the fixed 6-zone FFMP structure to support variable zone counts.
    For N=6, should match the existing FFMP exactly (within interpolation tolerance).
    
    Args:
        n_zones: Number of storage zones (drought levels). 
                 Typical: 3, 4, 5, 6 (standard), 8, 10, 15
        base_formulation: Name of base formulation to use for defaults.
                         Currently only "ffmp" is supported.
    
    Returns:
        Dict with keys:
        - 'description': Human-readable description
        - 'n_zones': Number of zones
        - 'decision_variables': OrderedDict of DVs with bounds
    """
    if base_formulation != "ffmp":
        raise NotImplementedError(f"Base formulation '{base_formulation}' not supported")
    
    dvs = OrderedDict()
    
    # ---- Zone-independent parameters (same for all N) ----
    # MRF baselines (5 DVs)
    dvs["mrf_cannonsville"] = {
        "baseline": 122.8,
        "bounds": [60.0, 250.0],
        "units": "MGD",
    }
    dvs["mrf_pepacton"] = {
        "baseline": 64.63,
        "bounds": [30.0, 130.0],
        "units": "MGD",
    }
    dvs["mrf_neversink"] = {
        "baseline": 48.47,
        "bounds": [20.0, 100.0],
        "units": "MGD",
    }
    dvs["mrf_montague"] = {
        "baseline": 1131.05,
        "bounds": [800.0, 1500.0],
        "units": "MGD",
    }
    dvs["mrf_trenton"] = {
        "baseline": 1938.95,
        "bounds": [1400.0, 2500.0],
        "units": "MGD",
    }
    
    # Max NYC delivery (1 DV)
    dvs["max_nyc_delivery"] = {
        "baseline": 800.0,
        "bounds": [500.0, 900.0],
        "units": "MGD",
    }
    
    # Flood release maximums (3 DVs)
    dvs["flood_max_cannonsville"] = {
        "baseline": 4200.0,
        "bounds": [2000.0, 8000.0],
        "units": "CFS",
    }
    dvs["flood_max_pepacton"] = {
        "baseline": 2400.0,
        "bounds": [1200.0, 5000.0],
        "units": "CFS",
    }
    dvs["flood_max_neversink"] = {
        "baseline": 3400.0,
        "bounds": [1500.0, 7000.0],
        "units": "CFS",
    }
    
    # Seasonal MRF scaling (4 DVs)
    for season in ["winter", "spring", "summer", "fall"]:
        dvs[f"mrf_profile_scale_{season}"] = {
            "baseline": 1.0,
            "bounds": [0.5, 1.5],
            "units": "multiplier",
        }
    
    # ---- Zone-specific parameters (N zones) ----
    # For each zone, we have:
    #   - zone threshold shift (1 DV per zone)
    #   - NYC delivery factor (1 DV per zone)
    #   - NJ delivery factor (1 DV per zone)
    #   - MRF scaling factor (1 DV per zone)
    # Total: 4*N DVs
    
    zone_names = [f"zone_{i}" for i in range(n_zones)]
    
    for i, zone_name in enumerate(zone_names):
        # Zone threshold: shift from the interpolated baseline
        baseline_shift = _interpolated_zone_shift(i, n_zones)
        dvs[f"zone_shift_{zone_name}"] = {
            "baseline": baseline_shift,
            "bounds": [-0.15, 0.15],
            "units": "fraction",
        }
        
        # NYC delivery factor
        baseline_nyc = _interpolated_delivery_factor(i, n_zones, demand_type="nyc")
        dvs[f"nyc_factor_{zone_name}"] = {
            "baseline": baseline_nyc,
            "bounds": [0.30, 1.0],
            "units": "fraction",
        }
        
        # NJ delivery factor
        baseline_nj = _interpolated_delivery_factor(i, n_zones, demand_type="nj")
        dvs[f"nj_factor_{zone_name}"] = {
            "baseline": baseline_nj,
            "bounds": [0.50, 1.0],
            "units": "fraction",
        }
        
        # MRF scaling factor
        baseline_mrf = _interpolated_mrf_factor(i, n_zones)
        dvs[f"mrf_factor_{zone_name}"] = {
            "baseline": baseline_mrf,
            "bounds": [0.50, 2.0],
            "units": "multiplier",
        }
    
    return {
        "description": f"Flexible FFMP with {n_zones} storage zones",
        "n_zones": n_zones,
        "decision_variables": dvs,
    }


def _interpolated_zone_shift(zone_idx: int, n_zones: int) -> float:
    """Interpolate zone threshold shift from FFMP 6-zone defaults.
    
    FFMP zones (approximate fractions of total capacity where level changes):
    - level1a: 0.85, level1b: 0.70, level1c: 0.55,
    - level2: 0.45, level3: 0.35, level4: 0.25, level5: 0.15
    
    For N zones, interpolate linearly between 0.85 (least severe) and 0.15 (most severe).
    The shift is relative to the interpolated position, so baseline=0.0 means no shift.
    """
    if n_zones == 1:
        return 0.0
    
    # Fraction spacing: least severe zone at 0.85, most severe at 0.15
    # Zone i (0-indexed) is at fraction: 0.85 - i * (0.70 / (n_zones - 1))
    target_frac = 0.85 - zone_idx * (0.70 / (n_zones - 1))
    
    # FFMP 6-zone target fracs
    ffmp_fracs = [0.85, 0.70, 0.55, 0.45, 0.35, 0.25, 0.15]
    
    # Find nearest FFMP fraction and return the difference as baseline shift
    # (This ensures N=6 defaults to matching FFMP exactly)
    if n_zones == 6:
        # Direct mapping
        ffmp_idx = min(zone_idx, len(ffmp_fracs) - 1)
        return 0.0  # No shift from default FFMP zone positions
    else:
        # Interpolation; for now, return 0.0 (could be enhanced)
        return 0.0


def _interpolated_delivery_factor(zone_idx: int, n_zones: int,
                                 demand_type: str) -> float:
    """Interpolate delivery (NYC/NJ) factor for a zone.
    
    FFMP values:
    - NYC: L1a-L2: 1,000,000 (unconstrained), L3: 0.85, L4: 0.70, L5: 0.65
    - NJ: L1a-L3: 1.0, L4: 0.90, L5: 0.80
    
    For N zones, interpolate smoothly between least-severe (1.0) and
    most-severe (0.65 for NYC, 0.80 for NJ).
    """
    if demand_type == "nyc":
        least_severe = 1.0
        most_severe = 0.65
    elif demand_type == "nj":
        least_severe = 1.0
        most_severe = 0.80
    else:
        raise ValueError(f"Unknown demand_type: {demand_type}")
    
    # Linear interpolation
    frac = zone_idx / max(n_zones - 1, 1)
    return least_severe - frac * (least_severe - most_severe)


def _interpolated_mrf_factor(zone_idx: int, n_zones: int) -> float:
    """Interpolate MRF factor (baseline MRF multiplier) for a zone.
    
    FFMP values (approximate):
    - L1a-L2: 1.0 (no reduction), L3: 0.95, L4: 0.85, L5: 0.75
    
    For N zones, interpolate smoothly.
    """
    least_severe = 1.0
    most_severe = 0.75
    
    frac = zone_idx / max(n_zones - 1, 1)
    return least_severe - frac * (least_severe - most_severe)
```

---

## Part IV: Execution Paths

### 4.1 Architecture A: Standard FFMP (Existing Code, No Changes)

```
User: run_simulation_inmemory(nyc_config) where nyc_config is from FFMP formulation

Execution:
  1. _get_cached_model_dict() → load base model dict from pywrdrb
  2. Deep-copy model dict
  3. _patch_model_dict() → modify parameter values for decision variables
  4. Write patched dict to JSON
  5. Model.load(json)
  6. Run simulation (standard FFMP logic)
  7. Extract results
  8. Compute objectives using FIXED flow targets (Tier 1)

Output: Objective vector for Borg
```

No changes to existing code. The objective computation is updated to use fixed targets instead of dynamic targets.

### 4.2 Architecture B: Variable-Resolution FFMP (New Registry, Same Execution Path)

```
User: run_simulation_inmemory(nyc_config) where nyc_config is from FFMP formulation with N zones

Execution:
  1. generate_ffmp_formulation(n_zones=N) → create formulation dict
  2. register_formulation(f"ffmp_vr_{N}", formulation)
  3. dvs_to_config(dv_vector, formulation_name=f"ffmp_vr_{N}") → NYCOperationsConfig
  4-8. Same as Architecture A

Output: Objective vector for Borg
```

All execution happens within existing `dvs_to_config()` and `run_simulation_inmemory()` paths. The formulation dict tells the system which DVs to expect, which parameter update methods to call.

**Implementation requirement:** Modify `_apply_ffmp_params()` to handle variable zone counts:

```python
def _apply_ffmp_params(config, params: dict, n_zones=6):
    """Apply formulation parameters, generalizing to N zones."""
    # ... existing code for zone-independent parameters ...
    
    zone_names = [f"zone_{i}" for i in range(n_zones)]
    
    # Dynamic zone shift application
    shifts = {zone_name: params[f"zone_shift_{zone_name}"]
              for zone_name in zone_names}
    _apply_zone_shifts_flexible(config, shifts, n_zones)
    
    # Dynamic delivery factor application
    nyc_factors = np.array([params[f"nyc_factor_zone_{i}"] for i in range(n_zones)])
    nj_factors = np.array([params[f"nj_factor_zone_{i}"] for i in range(n_zones)])
    config.update_delivery_constraints_flexible(nyc_factors, nj_factors, n_zones)
```

**CAVEAT:** This requires modifications to pywrdrb's `NYCOperationsConfig` class to handle variable zone counts. The alternative (less invasive) approach is to virtualize N zones onto the fixed 6-level structure using interpolation.

### 4.3 Architectures C/D/E: External Policy (New Evaluation Path)

```
User: optimize(policy_instance, formulation_name="rbf" or "tree" or "ann")

Execution:
  1. policy.set_params(dv_vector)
  2. Call evaluate_with_policy(policy) from simulation_policy.py
  3.   Load base model dict
  4.   Deep-copy model dict
  5.   Write to JSON
  6.   Model.load(json)
  7.   replace_release_parameters(model, policy_fn=policy, state_config=policy.state_config)
  8.   set_fixed_mrf_targets(model)
  9.   Run simulation
  10.  Extract results
  11.  Compute objectives using FIXED flow targets (Tier 1, same as all policies)
  
Output: Objective vector for Borg
```

All policies use the same execution logic. The difference is in the policy function (RBF vs Tree vs ANN) and the state features they use (defined by `policy.state_config`).

---

## Part V: Integration Checklist

### Minimal Changes Summary

**New Files (No Existing Code Modified):**
- `pywrdrb/parameters/external_policy.py` (350 lines)
- `pywrdrb/parameters/fixed_target.py` (50 lines)
- `NYCOptimization/optimization/policies/__init__.py` (minimal)
- `NYCOptimization/optimization/policies/base.py` (80 lines)
- `NYCOptimization/optimization/policies/rbf_policy.py` (~200 lines)
- `NYCOptimization/optimization/policies/tree_policy.py` (~250 lines)
- `NYCOptimization/optimization/policies/ann_policy.py` (~200 lines)
- `NYCOptimization/optimization/formulations.py` (~300 lines)
- `NYCOptimization/src/model_builder_utils.py` (~250 lines)
- `NYCOptimization/src/simulation_policy.py` (~200 lines)

**Modified Files (Minimal Changes):**
- `NYCOptimization/src/simulation.py`: Add dispatch in `evaluate()` function (5 lines)
- `NYCOptimization/config.py`: Replace FORMULATIONS dict initialization (10 lines)

**Total New Lines:** ~2,000  
**Total Modified Lines:** ~20

---

## Part VI: Key Design Questions Addressed

### Q1: How does the external policy interact with the LP solver?

**A:** The policy controls the `max_flow` of release links. The Pywr LP solver respects this as a hard constraint and allocates water accordingly. If the policy's releases are insufficient to meet the fixed MRF targets, the LP solver may fail to find a feasible solution. In practice, we allow some MRF violations and measure them in the objectives.

### Q2: Why is caching necessary in ExternalPolicyParameter?

**A:** If a single policy controls 3 reservoirs, we need 3 ExternalPolicyParameter instances (one per reservoir, one per output index). Without caching, the policy function would be called 3 times per timestep, tripling runtime. Caching ensures the policy is called ONCE, and all instances read from the cache.

### Q3: What if a policy's output is outside the plausible release range?

**A:** ExternalPolicyParameter clips to `output_min` and `output_max`. We set `output_min=0.0` (cannot release negative water) and `output_max=None` (LP solver will enforce capacity constraints). If the policy tries to release more than available storage + inflow, the LP solver will fail or produce infeasible results, which will be caught and recorded in objectives.

### Q4: How are state features (storage, inflow, time of year) normalized?

**A:** Each element of `state_config` has a `normalize_by` key. For storage, divide by reservoir capacity (0-1 fractions). For inflows, divide by mean or max inflow (may be domain-specific). For time of year, use sin/cos encoding (circular). The policy sees normalized 0-1 values, making it easier to learn across different reservoir sizes and inflow magnitudes.

### Q5: How does this handle multi-scenario runs?

**A:** Pywr supports multi-scenario evaluation using `scenario_index.global_id`. ExternalPolicyParameter reads this from node attributes (e.g., `volume[scenario_index.global_id]`) and uses it in the cache key `(timestep.index, scenario_index.global_id)`. Each scenario gets its own policy evaluation.

### Q6: Why fixed MRF targets instead of policy-determined targets?

**A:** For fair comparison. The FFMP's drought-level machinery relaxes flow targets during drought. If we measure all policies against the FFMP's dynamic targets, the FFMP has a measurement advantage. By measuring everyone against the same fixed targets, we isolate which policy is better at managing storage and timing releases, not which policy is best at declaring drought.

---

## Part VII: Performance Estimates

### Runtime Overhead

**ExternalPolicyParameter overhead per evaluation:**
- State extraction: ~0.1 ms (read 3 storage values, normalize)
- Policy call (RBF, Tree, ANN): ~0.5-5 ms (depends on policy complexity)
- Cache lookup/management: ~0.01 ms (negligible)
- Per timestep (78 years = ~28,500 days): ~15-140 seconds per simulation
- Total overhead vs. FFMP: ~10-30% slowdown expected

For 1M evaluations:
- FFMP runtime: ~150 seconds per sim × 1M = 150 million seconds (~1,736 days)
- Policy runtime: ~165-195 seconds per sim × 1M = 165-195 million seconds (~1,909-2,257 days)
- Feasible on HPC with parallelization (MPI ranks can evaluate independent samples)

**Memory overhead:**
- ExternalPolicyParameter instance per reservoir (3 total): ~1 KB each
- _POLICY_CACHE (28,500 timesteps × 1 scenario × 3 floats): ~340 KB per sim
- Policy instance (RBF ~100 KB, Tree ~500 KB, ANN ~1-50 MB depending on architecture)
- Total: ~2-60 MB per evaluation (negligible vs. Pywr's ~500 MB model)

---

## Part VIII: Testing Strategy

### Unit Tests

1. **ExternalPolicyParameter with constant policy:**
   - Policy returns [100, 100, 100] MGD always
   - Verify each reservoir releases exactly 100 MGD (if feasible)
   - Check cache is cleared at each timestep

2. **PolicyBase ABC compliance:**
   - All policy subclasses implement required methods
   - `set_params()` and `get_bounds()` have matching dimensions

3. **Model replacement utilities:**
   - `get_release_parameters_to_replace()` matches hardcoded pywrdrb keys
   - `replace_release_parameters()` creates ExternalPolicyParameter instances correctly
   - `set_fixed_mrf_targets()` replaces monthlyprofile params

### Integration Tests

1. **FFMP comparison (Architecture A):**
   - Run FFMP with standard evaluation (dynamic targets)
   - Run same FFMP with fixed-target evaluation
   - Compare objectives; expect slight worsening during drought (less target relaxation)

2. **Policy vs. FFMP (Architecture C, simple RBF):**
   - Train simple RBF policy on 5-year debug period
   - Evaluate against same fixed targets as FFMP
   - Expect performance within 10-20% of FFMP on test period (not optimization period)

3. **Variable-resolution FFMP (Architecture B):**
   - Run N=3, 6, 10 zone FFMP formulations
   - Verify N=6 matches standard FFMP exactly (within numerical precision ~1%)
   - Verify N=3 (fewer zones) and N=10 (more zones) produce valid objectives

---

## Part IX: Deployment Checklist

- [ ] Implement `ExternalPolicyParameter` and `CacheClearParameter` in pywrdrb/parameters/external_policy.py
- [ ] Implement `FixedMRFParameter` in pywrdrb/parameters/fixed_target.py
- [ ] Implement model replacement utilities in NYCOptimization/src/model_builder_utils.py
- [ ] Implement PolicyBase ABC in NYCOptimization/optimization/policies/base.py
- [ ] Implement RBFPolicy, ObliqueTreePolicy, ANNPolicy subclasses
- [ ] Implement variable-resolution FFMP generator in NYCOptimization/optimization/formulations.py
- [ ] Implement evaluate_with_policy() in NYCOptimization/src/simulation_policy.py
- [ ] Update simulation.py to dispatch based on formulation_name
- [ ] Update config.py to use formulation registry
- [ ] Update objectives.py to use fixed flow targets (Tier 1 evaluation)
- [ ] Write unit tests for ExternalPolicyParameter, policy classes, model replacement
- [ ] Integration test: FFMP A vs. fixed-target evaluation
- [ ] Integration test: simple policy C vs. FFMP on short period
- [ ] Integration test: variable-resolution FFMP N=3, 6, 10 reproduce expected differences
- [ ] Benchmark overhead: measure wall-clock time for FFMP vs. RBF policy on 1-year test period
- [ ] Prepare 5-year debug run with all architectures for team review

---

## Part X: Open Questions and Deferred Decisions

### Q1: NYCOperationsConfig Support for Variable Zones

Current design assumes N-zone FFMP is virtualized onto pywrdrb's fixed 6-level structure (interpolation). This works but may be less clean than modifying NYCOperationsConfig to store zone profiles as a list instead of a DataFrame with fixed row names.

**Decision:** Start with interpolation (less invasive). If performance or accuracy is insufficient, modify NYCOperationsConfig in a follow-up.

### Q2: State Features for Policies

Current design uses aggregate NYC storage (3 values) + time of year (2 features) = 5-D state space. Richer state (e.g., individual inflows, downstream demands, upstream releases) would give policies more information but increase state dimensionality.

**Decision:** Start with minimal state (3 storage + 2 temporal). Allow policies to override `state_config` property for custom state features.

### Q3: Multi-Reservoir Coordination

Current ExternalPolicyParameter assumes the policy controls all 3 reservoirs simultaneously (one call, 3-element output vector). What if we want independent policies per reservoir?

**Decision:** Keep unified interface (one policy, 3 outputs). If independent policies are needed, wrap each in its own PolicyBase that samples from a library of independent specialists.

### Q4: Pywr LP Solver Interaction Under Constraint Violation

If a policy's releases are insufficient to meet fixed MRF targets, what does the LP solver do? Does it override the policy's release, fail, or produce infeasible results?

**Decision:** Run empirical test. Expected: LP solver adjusts releases to meet MRF targets if possible, or produces infeasible results (recorded as high objective values). This is acceptable — the objectives measure actual flow compliance against fixed targets.

---

**Document Status:** Ready for implementation. See deployment checklist and testing strategy for next steps.
