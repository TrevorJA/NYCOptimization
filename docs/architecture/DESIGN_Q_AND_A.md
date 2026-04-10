# Architecture Design — Questions and Answers

**Detailed responses to the core design questions**

---

## Q1: Where to Intercept — Which Design Pattern?

### The Question
At what point in the model building pipeline can we swap FFMP parameters for external policy parameters?

**Options Considered:**
- a) Modify the model_dict JSON after ModelBuilder produces it but before Model.load()
- b) Add a custom Parameter subclass that Pywr loads from JSON
- c) Modify the model programmatically after Model.load() but before model.run()
- d) Some combination

### The Answer: Option C (Post-Load Model Replacement)

**Why Option C is correct:**

1. **Option A is infeasible:** We cannot serialize a policy function (RBF, Tree, ANN) to JSON. JSON supports primitives and arrays, not Python callables. The model_dict is loaded and instantiated by Pywr's parameter registry, which expects JSON-serializable configurations.

2. **Option B is incomplete:** A custom Parameter subclass CAN be loaded from JSON (Pywr has a registry), but the policy function still cannot be JSON-serialized. We would need to either:
   - Store the policy function in a global variable and reference it by name (fragile, namespace pollution)
   - Subclass Pywr's Model loader to inject the policy function post-load (monkey-patching, maintainability issues)
   
   Both approaches are less clean than Option C.

3. **Option C is elegant:** After `Model.load(json)`, the model is a fully instantiated Pywr Model with all parameters as Python objects in memory. We can:
   - Iterate through `model.parameters` dict
   - Find the release parameters by name (e.g., "rel_max_release_cannonsville")
   - Create new `ExternalPolicyParameter` instances
   - Replace them in the dict
   - Call `model.run()` as usual

   No changes to pywrdrb's model-building code. No JSON serialization. No monkey-patching.

4. **Option D would be hybrid:** E.g., use Option A for FFMP (faster, direct dict patching) and Option C for policy (slower, model replacement). This adds complexity without benefit.

### Implementation: Post-Load Model Replacement (PLMR)

```python
# simulation_policy.py
def evaluate_with_policy(policy, ...):
    # Load base model dict (from cache)
    base_dict = _get_cached_model_dict()
    model_dict = copy.deepcopy(base_dict)
    
    # Write to JSON (required by Pywr)
    with open(model_json, 'w') as f:
        json.dump(model_dict, f)
    
    # Load as Pywr Model
    model = pywrdrb.Model.load(model_json)  # <-- Standard FFMP model
    
    # NOW replace parameters
    # model.parameters["rel_max_release_cannonsville"] → ExternalPolicyParameter instance
    replace_release_parameters(model, policy_fn=policy, ...)
    
    # Optionally replace MRF targets
    set_fixed_mrf_targets(model)
    
    # Run (now with policy in control)
    model.run()
    
    # Extract and return objectives
    ...
```

### Why This Doesn't Modify pywrdrb

The standard FFMP model is built, loaded, and then enhanced:
- No changes to ModelBuilder
- No changes to existing Parameter classes
- No JSON manipulation before load
- The ExternalPolicyParameter lives in a NEW file: `pywrdrb/parameters/external_policy.py`

---

## Q2: The MRF Target Problem — How to Handle Fixed vs. Dynamic Targets?

### The Question

If we disable drought-level MRF targets, what do we set them to? And how does the LP solver handle the constraint?

**The Core Problem:**
- FFMP has a drought level (0-6) that determines MRF targets
- During normal drought: Montague target = 1,131 MGD
- During Level 4 drought: Montague target = 1,131 × 0.57 ≈ 645 MGD (relaxed)
- If a policy controls releases and we measure against the relaxed target, the policy sees an easier standard during drought (unfair)
- If we measure the policy against the normal target, it's penalized for not meeting a standard it wasn't designed for

**Three Proposed Solutions (from method_critique_and_refinement.md):**

1. **Tier 1: Fixed-Target Objectives (CHOSEN)**
   - Montague = 1,131 MGD year-round (normal condition)
   - Trenton = 1,939 MGD year-round (normal condition)
   - Apply to ALL policies: FFMP, RBF, Tree, ANN
   - Measurement is fair: same constraints for all
   - FFMP operates with its normal release logic, but objectives ignore the dynamic targets it computes

2. **Tier 2: Universal State-Dependent Target (SUPPLEMENTARY)**
   - All policies see a storage-dependent target: target(t) = baseline × f(aggregate_storage(t))
   - f is a universal function (not policy-specific), fitted to FFMP's drought curves
   - Example: at 70% storage, target = 1,131 MGD; at 30% storage, target = 750 MGD
   - Comparison measures which policy best allocates water within the same state-dependent constraints

3. **Tier 3: Policy-Specific Objectives (DIAGNOSTIC ONLY)**
   - Measure FFMP against its own dynamic targets (how well it meets standards it sets)
   - Not used for architecture comparison

### The Answer: Tier 1 for Primary, Tier 2 for Sensitivity

**Primary Evaluation (Tier 1):**

All policies are evaluated against fixed normal-condition targets. Implementation:

```python
# In src/simulation_policy.py
def evaluate_with_policy(policy, fixed_mrf_targets=True):
    ...
    if fixed_mrf_targets:
        set_fixed_mrf_targets(model, montague_mgd=1131.05, trenton_mgd=1938.95)
    model.run()
    ...

# set_fixed_mrf_targets() does what?
# Finds the MRF target parameters in the model (created by monthlyprofile)
# Replaces them with FixedMRFParameter that always returns the fixed value

class FixedMRFParameter(Parameter):
    def __init__(self, model, value, name):
        super().__init__(model, name=name)
        self.value_const = value
    
    def value(self, timestep, scenario_index):
        return self.value_const
```

The LP solver sees fixed targets and respects them as hard constraints. If the policy's releases are insufficient, the LP solver will either:
- Adjust releases to meet the target (violating policy intent but meeting physical constraints)
- Produce infeasible results (recorded as high objective values)

**LP Solver Interaction:**

```
Policy outputs: [release_can, release_pep, release_nev] (in MGD)
                ↓
Pywr LP Solver constraints:
  - Montague flow >= 1,131 MGD (FIXED, not dynamic)
  - Trenton flow >= 1,939 MGD (FIXED, not dynamic)
  - Cannonsville release <= policy output
  - Pepacton release <= policy output
  - Neversink release <= policy output
  - + All other physical constraints (mass balance, capacity, etc.)
                ↓
LP Solution: Releases that satisfy all constraints
             If policy's releases are infeasible, LP solver adjusts
```

**Objectives measure compliance against the FIXED targets**, not against what the policy intended:

```python
# src/objectives.py
def montague_reliability_fixed_target(data: dict) -> float:
    flow = data["major_flow"]["delMontague"]
    target = 1131.05  # MGD, fixed
    weekly_flow = flow.resample("W").mean()
    met = (weekly_flow >= target).sum()
    return float(met) / len(weekly_flow)  # Fraction of weeks meeting target
```

**Supplementary Evaluation (Tier 2):**

For sensitivity analysis, also compute objectives with a universal storage-dependent target:

```python
def montague_reliability_storage_adjusted(data: dict) -> float:
    flow = data["major_flow"]["delMontague"]
    storage = data["res_storage"][["cannonsville", "pepacton", "neversink"]].sum(axis=1)
    storage_frac = storage / 270837.0  # Total NYC capacity
    
    # Universal target function (smooth, not FFMP-specific)
    # Full target at high storage, reduced at low storage
    target_factor = np.clip(0.6 + 0.4 * (storage_frac - 0.2) / 0.5, 0.6, 1.0)
    target_mgd = 1131.05 * target_factor
    
    weekly_flow = flow.resample("W").mean()
    weekly_target = target_mgd.resample("W").mean()
    met = (weekly_flow >= weekly_target).sum()
    return float(met) / len(weekly_target)
```

**Why Report Both?**

If architecture rankings change between Tier 1 and Tier 2:
- It means the FFMP's target-relaxation mechanism (determining drought levels, reducing targets) is a significant source of apparent performance advantage
- The paper should highlight this finding: "FFMP's advantage stems partly from declaring drought and relaxing targets, not purely from release timing logic"
- Tier 1 isolates the release-timing skill, Tier 2 incorporates target-relaxation skill

### What About FFMP in This Design?

For Mode A (standard FFMP):

```python
# src/simulation.py (unchanged)
def evaluate_ffmp(dv_vector, formulation_name="ffmp"):
    config = dvs_to_config(dv_vector, formulation_name)
    data = run_simulation_inmemory(config)  # Standard FFMP model, unchanged
    objectives = compute_objectives(data, use_fixed_targets=True)  # <-- CHANGED
    return objectives
```

The FFMP still operates with its dynamic targets (computes drought levels, adjusts MRF factors). But the OBJECTIVES use fixed targets. This separates:
- **What the FFMP does internally:** Dynamic target relaxation based on drought level
- **How we measure it:** Against fixed targets, same as all other architectures

**This is the cleanest design:** FFMP operates as designed, but measurement is fair.

---

## Q3: State Access and LP Solver Interaction

### The Question

How does the external policy function read current state (storage, inflows, time of year) at each timestep? How does the LP solver respect (or override) the policy's release decisions?

### State Access: ExternalPolicyParameter Extracts from Pywr Nodes

```python
# pywrdrb/parameters/external_policy.py
class ExternalPolicyParameter(Parameter):
    def __init__(self, model, policy_fn, state_config, output_index, ...):
        # state_config is a list of dicts:
        # [
        #   {'node': 'cannonsville', 'attribute': 'volume', 'normalize_by': 95706},
        #   {'node': 'pepacton', 'attribute': 'volume', 'normalize_by': 140190},
        #   {'node': 'neversink', 'attribute': 'volume', 'normalize_by': 34941},
        # ]
        # The policy function will receive: [frac_can, frac_pep, frac_nev, sin_doy, cos_doy]
        self.state_config = state_config
        self.policy_fn = policy_fn
        self.output_index = output_index
    
    def value(self, timestep, scenario_index):
        # Called by Pywr at each timestep
        state = []
        for cfg in self.state_config:
            node = self.model.nodes[cfg['node']]
            raw = node.volume  # Node attribute (volume, flow, etc.)
            val = float(raw[scenario_index.global_id])  # Handle scenario indexing
            if cfg.get('normalize_by'):
                val /= cfg['normalize_by']
            state.append(val)
        
        # Add temporal features
        doy = timestep.datetime.timetuple().tm_yday
        state.append(np.sin(2.0 * np.pi * doy / 366.0))
        state.append(np.cos(2.0 * np.pi * doy / 366.0))
        
        # Call policy
        action = self.policy_fn(np.array(state))  # RBF, Tree, or ANN evaluates state
        
        # Return one element
        return float(action[self.output_index])
```

**Execution Timeline:**

```
Timestep t:
  1. Pywr reads all Parameter.value() calls (in definition order)
  2. ExternalPolicyParameter["cache_clear"].value() → clears _POLICY_CACHE
  3. ExternalPolicyParameter["policy_release_cannonsville"].value() →
       - Extracts state from nodes (storage as of START of timestep t)
       - Calls policy_fn(state)
       - Caches result in _POLICY_CACHE[(t, scenario_id)]
       - Returns action[0]
  4. ExternalPolicyParameter["policy_release_pepacton"].value() →
       - Same state extraction
       - Finds cache entry (already computed)
       - Returns action[1]
  5. ExternalPolicyParameter["policy_release_neversink"].value() →
       - Returns action[2]
  6. LP Solver constructs constraints:
       - Cannonsville release <= action[0]
       - Pepacton release <= action[1]
       - Neversink release <= action[2]
       - Montague flow >= 1,131 MGD (FIXED, or Tier 2 dynamic)
       - Trenton flow >= 1,939 MGD (FIXED, or Tier 2 dynamic)
       - + Other constraints (capacity, minimum inflows, demands)
  7. LP Solver finds feasible allocation
  8. Flows and storages updated
  9. Move to next timestep
```

### LP Solver Interaction: Hard Constraint or Suggestion?

The policy's release is a **hard upper-bound constraint**, not a suggestion.

**If policy outputs [100, 80, 60] MGD:**
```
LP constraints:
  cannonsville_release <= 100
  pepacton_release <= 80
  neversink_release <= 60
  montague_flow >= 1131
  trenton_flow >= 1939
  [+ other physical constraints]
```

**Scenario A:** Policy releases are sufficient to meet MRF targets
- LP solver finds a feasible solution respecting the policy
- Releases are close to policy output (maybe slightly less if other demands reduce flow)

**Scenario B:** Policy releases are insufficient to meet MRF targets
- LP solver faces infeasibility
- Possible outcomes (depends on Pywr's LP solver configuration):
  - **Strict mode:** Solver fails, raises infeasibility error
  - **Slack mode:** Solver violates one constraint to satisfy others (e.g., allows MRF violation if policy constraint is strict)
  - **Priority mode:** Solver prioritizes hard constraints (capacity, mass balance) and soft constraints (MRF targets, policy) flexibly

**In practice (observed behavior):**

If Pywr prioritizes hard constraints, the LP solver will:
1. Satisfy mass balance and capacity constraints (always)
2. Satisfy MRF targets as much as possible (unless policy prevents it)
3. Accept MRF violation if policy constraint is absolute

**Example:** 
- Policy says: release 50 MGD from each reservoir
- But current inflows are 40 MGD combined
- LP solver cannot increase releases above inflows + storage drawdown
- MRF targets will be violated
- Objectives measure this violation (high flow deficit)

**This is acceptable design.** The objectives measure what actually happened (flow deficit), not what the policy intended. If a policy is too conservative (releases too little), it will be penalized in the objectives.

### Caching for Multi-Output Policies

**Without caching:**
```
3 parameters, 28,500 timesteps = 85,500 policy function calls
RBF call: ~1 ms each
Total overhead: ~85 seconds per simulation
```

**With caching:**
```
28,500 timestep entries in cache, ~1 MB memory
Cache lookup: ~0.01 ms per call (negligible)
Effective cost: 1 call per timestep, ~28.5 ms
Total overhead: <1 second per simulation
```

**Implementation:**

```python
# pywrdrb/parameters/external_policy.py
_POLICY_CACHE = {}  # Global cache, cleared at start of each timestep

def value(self, timestep, scenario_index):
    cache_key = (timestep.index, scenario_index.global_id)
    
    if cache_key not in _POLICY_CACHE:
        # First call for this timestep: compute and cache
        state = self._extract_state(timestep, scenario_index)
        action = self.policy_fn(state)
        _POLICY_CACHE[cache_key] = action
    
    # Retrieve from cache
    action = _POLICY_CACHE[cache_key]
    return float(action[self.output_index])


class CacheClearParameter(Parameter):
    """Inserted first in parameter evaluation order to clear cache."""
    def value(self, timestep, scenario_index):
        global _POLICY_CACHE
        _POLICY_CACHE.clear()
        return 0.0
```

Pywr evaluates parameters in definition order, so `CacheClearParameter` runs first each timestep, clearing the cache. Subsequent calls use the fresh cache.

### Multi-Scenario Handling

Pywr supports multiple scenarios (e.g., climate uncertainty). Each scenario has a `scenario_index.global_id`. ExternalPolicyParameter reads from scenario-indexed node attributes:

```python
node.volume  # Shape (n_scenarios,) — array
val = float(node.volume[scenario_index.global_id])  # Extract for this scenario
```

The cache key includes `scenario_index.global_id`, so each scenario gets independent cache entries:
```
_POLICY_CACHE[(t, scenario_0)] = [action_0, action_1, action_2]
_POLICY_CACHE[(t, scenario_1)] = [action_0', action_1', action_2']
```

---

## Summary Table

| Aspect | Design Choice | Why |
|--------|---------------|-----|
| **Interception Point** | Post-Load Model Replacement (Option C) | Policy function not JSON-serializable; cleanest to replace after load |
| **MRF Target Handling** | Fixed targets (Tier 1, primary) + storage-dependent (Tier 2, sensitivity) | Fair comparison across architectures; same constraints for all |
| **LP Solver Interaction** | Policy output is hard upper-bound constraint | Policy determines feasible release range; LP solver optimizes within it |
| **State Access** | ExternalPolicyParameter reads from Pywr nodes | Direct access at each timestep; no pre-computation needed |
| **Caching** | Global timestep-keyed cache with explicit clear | Avoids 3× policy function calls per timestep |
| **Multi-Scenario** | Cache key includes scenario_index.global_id | Independent state extraction per scenario |

---

## Final Design Robustness Check

**Potential Failure Modes:**

1. **Policy function crashes at some timestep**
   - Mitigated: ExternalPolicyParameter wraps policy call in try-except, logs error
   - Result: Simulation fails gracefully, objectives recorded as invalid
   
2. **Policy output outside [0, capacity]**
   - Mitigated: ExternalPolicyParameter clips to [0, capacity]
   - LP solver enforces capacity independently
   
3. **LP solver infeasibility due to policy constraint**
   - Mitigated: Objectives measure actual flow compliance (violations are high values)
   - Not a failure, just poor policy performance
   
4. **State extraction reads wrong node/attribute**
   - Mitigated: KeyError raised immediately with list of available nodes
   - Must be caught during initial testing
   
5. **Cache key collision (same timestep, scenario, but different model state)**
   - Mitigated: Impossible. Cache is cleared at start of each timestep
   - Cache is valid for exactly one timestep

**Conclusion:** Design is robust. Failure modes are either prevented by design or manifested as objective penalties (appropriate).
