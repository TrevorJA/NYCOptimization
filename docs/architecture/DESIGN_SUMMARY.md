# External Policy Architecture — Design Summary

**Quick Reference for Implementation**

---

## The Challenge

pywrdrb currently implements FFMP rule logic through a hardcoded control chain:
```
storage → controlcurveindex (0-6) → monthlyprofile MRF factors → LP solver constraints
```

We need to support THREE modes without modifying existing pywrdrb files:
- **Mode A:** Standard FFMP (baseline)
- **Mode B:** Variable-resolution FFMP (N zones instead of 6)
- **Mode C/D/E:** External policy (RBF, Tree, ANN) completely replaces release logic

---

## Solution: Post-Load Model Replacement (PLMR)

1. **Build standard FFMP model** (unchanged from existing code)
2. **Load the model into memory** as usual
3. **Replace release parameters** with `ExternalPolicyParameter` instances that call a user-supplied policy function
4. **Run simulation** with policy in control
5. **Measure against fixed flow targets** (fair comparison across all modes)

**Why this works:**
- No changes to existing pywrdrb code
- Policy function is passed at runtime (not serialized to JSON)
- Clean separation: model structure unchanged, only parameter implementations swapped

---

## Key Files to Create

### 1. Custom Pywr Parameter (pywrdrb/parameters/external_policy.py)

```python
class ExternalPolicyParameter(Parameter):
    """Reads state from model nodes, calls policy_fn, returns action."""
    
    def value(self, timestep, scenario_index):
        state = self._extract_state(timestep, scenario_index)  # [storage, storage, storage, sin(doy), cos(doy)]
        action = self.policy_fn(state)  # Returns [release_can, release_pep, release_nev]
        return np.clip(action[self.output_index], self.output_min, self.output_max)

class CacheClearParameter(Parameter):
    """Dummy parameter that clears policy cache at start of each timestep."""
```

**Why both classes?**
- One policy call produces 3 releases (3 reservoirs)
- Need 3 parameter instances, one per release link
- Without cache: policy called 3× per timestep (3× overhead)
- With cache: policy called 1× per timestep, all 3 instances read cached result

### 2. Model Replacement Utilities (src/model_builder_utils.py)

```python
def replace_release_parameters(model, policy_fn, state_config):
    """After Model.load(), swap release parameters with ExternalPolicyParameter."""
    # Find parameter keys: "rel_max_release_cannonsville", etc.
    # Create 3 ExternalPolicyParameter instances
    # Replace in model.parameters dict
    
def set_fixed_mrf_targets(model, montague_mgd=1131, trenton_mgd=1939):
    """Replace monthlyprofile dynamic targets with fixed constants."""
    # Finds MRF target parameters
    # Replaces with FixedMRFParameter (returns constant value)
```

### 3. Policy Base Class (optimization/policies/base.py)

```python
class PolicyBase(ABC):
    @abstractmethod
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """state → action (releases in MGD)"""
        
    @abstractmethod
    def set_params(self, flat_vector: np.ndarray) -> None:
        """Called by Borg with new DV values"""
        
    @property
    def state_config(self) -> list:
        """How to extract state from model (default: 3 storage values)"""
```

Subclasses: `RBFPolicy`, `ObliqueTreePolicy`, `ANNPolicy`

### 4. Evaluation Function (src/simulation_policy.py)

```python
def evaluate_with_policy(policy, use_trimmed=True, fixed_mrf_targets=True):
    """Borg-compatible evaluation for policy-based architectures."""
    # Load base model dict
    # model = Model.load(json)
    # replace_release_parameters(model, policy_fn=policy, state_config=policy.state_config)
    # set_fixed_mrf_targets(model)
    # model.run()
    # return compute_objectives(data)
```

### 5. Variable-Resolution FFMP (optimization/formulations.py)

```python
def generate_ffmp_formulation(n_zones: int) -> dict:
    """Create N-zone FFMP formulation dict."""
    # Returns OrderedDict of DVs: MRF baselines, delivery factors, zone shifts, etc.
    # For N zones: zone_shift_0, zone_shift_1, ..., zone_shift_N-1
    #            nyc_factor_0, ..., nj_factor_N-1
    # Works with existing dvs_to_config() and _apply_ffmp_params()
    # Requires generalization of _apply_ffmp_params() to handle variable zone counts
```

---

## Execution Paths

### Mode A: Standard FFMP
```
formulation_name = "ffmp"
    ↓
evaluate_ffmp()  [existing code, unchanged]
    ↓
dvs_to_config(dv_vector, "ffmp")
    ↓
run_simulation_inmemory(config)
    ↓
[UNCHANGED: patch model_dict, write JSON, load, run]
    ↓
compute_objectives(data, use_fixed_targets=True)  [CHANGED: evaluate against fixed Montague/Trenton targets]
```

### Mode B: Variable-Resolution FFMP
```
formulation_name = "ffmp_vr_10"
    ↓
evaluate_ffmp()
    ↓
dvs_to_config(dv_vector, "ffmp_vr_10")
    ↓
[NEW: _apply_ffmp_params() detects n_zones=10, handles variable zone DVs]
    ↓
run_simulation_inmemory(config)
    ↓
[UNCHANGED otherwise]
    ↓
compute_objectives(data, use_fixed_targets=True)
```

### Mode C/D/E: External Policy (RBF, Tree, ANN)
```
formulation_name = "rbf" | "tree" | "ann"
    ↓
[NEW: dispatch to evaluate_with_policy()]
    ↓
policy.set_params(dv_vector)
    ↓
evaluate_with_policy(policy)
    ↓
    Load base model dict
    ↓
    Model.load(json)
    ↓
    [NEW] replace_release_parameters(model, policy_fn=policy)
    ↓
    [NEW] set_fixed_mrf_targets(model)
    ↓
    model.run()
    ↓
compute_objectives(data, use_fixed_targets=True)
```

---

## State → Action Flow

### State Vector Extracted by ExternalPolicyParameter

```python
state_config = [
    {'node': 'cannonsville', 'attribute': 'volume', 'normalize_by': 95706},   # 0-1 fraction
    {'node': 'pepacton', 'attribute': 'volume', 'normalize_by': 140190},      # 0-1 fraction
    {'node': 'neversink', 'attribute': 'volume', 'normalize_by': 34941},      # 0-1 fraction
    # Temporal features added automatically:
    # sin(2π DOY / 366), cos(2π DOY / 366)
]

state = np.array([frac_can, frac_pep, frac_nev, sin_doy, cos_doy])  # Length 5
```

### Action Vector Returned by Policy

```python
action = policy(state)  # np.ndarray([release_cannonsville_mgd, release_pepacton_mgd, release_neversink_mgd])

# Each ExternalPolicyParameter extracts one element:
param_cannonsville.output_index = 0  → returns action[0]
param_pepacton.output_index = 1      → returns action[1]
param_neversink.output_index = 2     → returns action[2]

# Values are clipped to [0, reservoir_capacity]
```

---

## Fair Comparison: Fixed MRF Targets (Tier 1)

**Problem:** FFMP declares drought levels and relaxes flow targets during drought. If we measure policies against FFMP's dynamic targets, FFMP has an unfair advantage.

**Solution:** Evaluate ALL architectures (A, B, C/D/E) against the SAME fixed targets:
- Montague: 1,131 MGD (normal condition baseline)
- Trenton: 1,939 MGD (normal condition baseline)

This is implemented by:
1. `set_fixed_mrf_targets(model)` replaces MRF target parameters with fixed constants
2. LP solver enforces these fixed targets as constraints
3. Objectives measure flow compliance against the same fixed targets

**Result:** All policies face identical constraints. Comparison measures which policy is better at managing storage and release timing, NOT which policy is better at declaring drought and relaxing targets.

---

## Decision Variable Structure

### Mode A/B: FFMP (variable zones)

Example for N=6 (standard):
```
DVs = {
    'mrf_cannonsville': 122.8 (bounds: 60-250),
    'mrf_pepacton': 64.63,
    'mrf_neversink': 48.47,
    'mrf_montague': 1131.05,
    'mrf_trenton': 1938.95,
    'max_nyc_delivery': 800.0,
    'flood_max_*': [4200, 2400, 3400] (CFS),
    'mrf_profile_scale_{winter,spring,summer,fall}': 1.0 each (multipliers),
    'zone_shift_zone_0': 0.0 (bounds: -0.15 to 0.15),
    'zone_shift_zone_1': 0.0,
    ...
    'zone_shift_zone_5': 0.0,
    'nyc_factor_zone_0': 1.0,
    ...,
    'nj_factor_zone_0': 1.0,
    ...,
    'mrf_factor_zone_0': 1.0,
    ...
}
Total: 5 + 1 + 3 + 4 + 4*6 = 42 DVs for N=6
Total: 5 + 1 + 3 + 4 + 4*10 = 62 DVs for N=10
```

### Mode C/D/E: Policy

**RBFPolicy (C):**
```
DVs = RBF weights (e.g., 20 basis functions × 3 outputs = 60 DVs)
      RBF centers (learned offline, not optimized)
      Output scaling factors
```

**ObliqueTreePolicy (D):**
```
DVs = Tree weights (splits and leaf values, e.g., 50-100 DVs depending on tree depth)
```

**ANNPolicy (E):**
```
DVs = Neural network weights (e.g., 2-layer MLP: 5*32 + 32*3 = 196 DVs)
```

---

## Performance Characteristics

### Runtime Overhead per Evaluation

| Component | Time |
|-----------|------|
| FFMP baseline (78 years) | ~150 seconds |
| Policy RBF (78 years) | ~160-180 seconds (+10-20%) |
| Policy Tree (78 years) | ~160-180 seconds (+10-20%) |
| Policy ANN (78 years) | ~170-200 seconds (+15-30%) |

Cache overhead: ~1-5 ms per 28,500-day simulation (negligible)

### Memory per Evaluation

| Component | Size |
|-----------|------|
| Model instance | ~500 MB |
| ExternalPolicyParameter instances (3×) | ~5 KB |
| Policy cache (28,500 timesteps × 3 floats) | ~350 KB |
| RBF policy instance | ~100-500 KB |
| Tree policy instance | ~500 KB - 2 MB |
| ANN policy instance | ~1-50 MB |

Total: ~500-550 MB (negligible increase vs. Pywr model)

---

## Integration Summary

| File | Type | Change |
|------|------|--------|
| pywrdrb/parameters/external_policy.py | NEW | ~350 lines |
| pywrdrb/parameters/fixed_target.py | NEW | ~50 lines |
| src/model_builder_utils.py | NEW | ~250 lines |
| src/simulation_policy.py | NEW | ~200 lines |
| optimization/policies/base.py | NEW | ~80 lines |
| optimization/policies/{rbf,tree,ann}_policy.py | NEW | ~500 lines |
| optimization/formulations.py | NEW | ~300 lines |
| src/simulation.py | MODIFY | +5 lines (dispatch) |
| config.py | MODIFY | ~10 lines (registry) |

**Total: ~2,000 new lines, ~15 modified lines**

---

## Next Steps

1. **Implement ExternalPolicyParameter** (pywrdrb/parameters/external_policy.py)
   - Test with constant policy (always release 100 MGD)
   - Verify cache mechanism works

2. **Implement model replacement utilities** (src/model_builder_utils.py)
   - Test on full 78-year model

3. **Implement PolicyBase and RBFPolicy** (optimization/policies/)
   - Test evaluation on 5-year debug period

4. **Update src/simulation.py** to dispatch based on formulation_name

5. **Run integration tests** (FFMP fixed-target vs. RBF on short period)

6. **Benchmark** overhead on 1-year test run

7. **Full HPC optimization runs** once verified locally
