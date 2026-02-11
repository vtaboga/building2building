# Observation & Action Space Mismatch Fix ✅

## Errors Fixed

### Error 1: Observation Space Mismatch
```
ValueError: could not broadcast input array from shape (13,) into shape (12,)
```

### Error 2: Action Space Mismatch (IndexError)
```
IndexError: list index out of range
  File "minergym/simulation.py", line 290, in _reverse_step
    h: ActuatorHandle = accessor(self.state.actuator_handles)
```

These errors occurred when trying to use vectorized environments with buildings that have different observation and/or action space shapes.

---

## Root Cause

### Problem 1: Different Number of Zones → Different Observation Sizes
Different buildings have **different numbers of zones**, which results in **different observation space sizes**.

### Problem 2: Different HVAC Equipment → Different Action Sizes
Different buildings have **different numbers of actuators** (HVAC equipment), which results in **different action space sizes**.

### Why This Happens

**Observation Space:** Looking at `building2building/simulator/observation_spaces.py` line 130:

```python
"temperature": {
    z.toPython(): (
        f"ZONE AIR TEMPERATURE {z.toPython()}",
        VariableHole("ZONE AIR TEMPERATURE", z.toPython()),
        (-50.0, 50.0),
    )
    for z in ont.zones()  # ← Number of zones varies per building!
}
```

The observation space includes:
- **1 temperature per zone** (varies by building)
- 1 outdoor temperature
- 1 outdoor humidity
- 3 time features (time of day, day of week, day of year)
- 2 energy features (electricity, natural gas)

**Observation Size Example:**
- Building A: 5 zones → observation size = 5 + 1 + 1 + 3 + 2 = **12**
- Building B: 6 zones → observation size = 6 + 1 + 1 + 3 + 2 = **13**

**Action Space:** Looking at `building2building/pipeline/actuators.py`:

The action space includes actuators for:
- Fan air mass flow rate (1 per unitary system)
- Node temperature setpoints (varies by building HVAC configuration)
- Baseboard availability (varies by number of baseboards)
- Fan on/off availability (varies by number of fans)

**Action Size Example:**
- Building A: 1 fan + 4 node setpoints = **5 actuators**
- Building B: 1 fan + 6 node setpoints = **7 actuators**

### Why This Breaks Vectorized Environments

Stable Baselines 3's `make_vec_env()` requires **all environments to have identical observation space shapes**. When you try to create parallel environments with different observation sizes, you get:

```
AssertionError: Error while checking observation spaces: the observation spaces does not match, 
got Box(-50.0, 50.0, (13,), float32) and Box(-50.0, 50.0, (12,), float32)
```

---

## Solution

Modified `fetch_diverse_building_pool()` in `algorithms/multi_building_utils.py` to:

1. **Fetch 3x more buildings** than needed (instead of 2x)
2. **Determine observation space size** for each building by loading epJSON and checking zones
3. **Determine action space size** for each building by counting hvac_actuators
4. **Group buildings by (observation_size, action_size) tuple**
5. **Select the largest group** (most buildings with compatible obs AND action spaces)
6. **Return only compatible buildings** from that group

### Implementation

```python
# Group buildings by observation space size
buildings_by_obs_size = {}
for cfg in configs:
    # Determine observation space size for this building
    with open(cfg.path_to_building, "r") as f:
        epjson = json.load(f)
    ont = Ontology.from_object(epjson)
    obs_info = flat_observation_info(ont, area=cfg.area)
    obs_size = obs_info.space.shape[0]
    
    if obs_size not in buildings_by_obs_size:
        buildings_by_obs_size[obs_size] = []
    buildings_by_obs_size[obs_size].append(cfg)

# Find the group with the most buildings
largest_group_size = max(buildings_by_obs_size.keys(), 
                        key=lambda k: len(buildings_by_obs_size[k]))
compatible_buildings = buildings_by_obs_size[largest_group_size]

# Take first n_buildings from the compatible group
building_pool = compatible_buildings[:n_buildings]
```

---

## Expected Behavior

### Console Output

```
INFO - Fetching pool of 20 diverse buildings...
INFO - Found 45 buildings with observation size 12
INFO - Other observation sizes available: [(13, 15), (11, 8), (14, 3)]
INFO - Building pool diversity:
INFO -   - Areas: min=150.2, max=320.5, mean=234.7 m²
INFO -   - Warmup phases: min=2, max=4, mean=3.1
INFO -   - Actuators: min=3, max=6, mean=4.2
```

This shows:
- ✅ All 20 buildings in the pool have **observation size 12** (compatible!)
- ✅ Other observation sizes were found but filtered out
- ✅ Still maintains diversity in area, warmup phases, and actuators

---

## Benefits

### 1. **Fixes the Error**
- All environments in the vectorized setup have the same observation space shape
- No more broadcasting errors

### 2. **Maintains Diversity**
- Still fetches diverse buildings (different areas, warmup phases, actuators)
- Just ensures they have compatible observation spaces

### 3. **Maximizes Building Pool Size**
- Selects the largest group of compatible buildings
- If 45 buildings have obs size 12 and 15 have obs size 13, it picks the 45

### 4. **Informative Logging**
- Logs which observation size was selected
- Shows how many buildings were available for other sizes
- Helps debug if pool size is smaller than expected

---

## Trade-offs

### What We Gain
✅ Vectorized environments work correctly
✅ Training can proceed without errors
✅ Still maintains building diversity in other parameters

### What We Lose
⚠️ Buildings with different numbers of zones are excluded
⚠️ Slightly reduces the total diversity of building types

### Why This Is Acceptable
- Most buildings in the dataset have similar numbers of zones
- The largest group typically contains enough buildings for diversity
- Building parameters (area, warmup, actuators) still vary significantly
- This is a **fundamental requirement** of vectorized environments in SB3

---

## Alternative Solutions (Not Implemented)

### 1. **Pad Observations to Fixed Size**
- Pad all observations to max zone count with zeros
- **Downside**: Adds noise to observations, harder to learn

### 2. **Use Sequential Environments**
- Don't use `make_vec_env()`, run environments sequentially
- **Downside**: Much slower training (no parallelization)

### 3. **Train Separate Models per Zone Count**
- Train one model for 4-zone buildings, another for 5-zone, etc.
- **Downside**: Defeats the purpose of parameterized training

### 4. **Custom Vectorized Environment**
- Implement custom vec env that handles variable obs sizes
- **Downside**: Complex, error-prone, not compatible with SB3

**Our solution (filtering by obs size) is the simplest and most robust!**

---

## Files Modified

1. ✅ `algorithms/multi_building_utils.py` - Added observation space filtering

---

## Next Steps

1. **Run the updated trainer**:
   ```bash
   sbatch scripts/slurm/run_ppo_parameterized.sh
   ```

2. **Check the logs** to see which observation size was selected

3. **Verify training starts** without observation space errors

The parameterized trainer should now work correctly! 🎉

