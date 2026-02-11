# Building Parameter Extraction - Debug Summary

## Issue

The `AugmentObservationWithBuildingParams` wrapper was not correctly extracting building parameters from the environment, causing it to always fall back to default values.

## Root Cause Analysis

### Problem 1: Missing Metadata Fields

The `create_simulator()` function in `building2building/simulator/__init__.py` was not storing `area` and `warmup_phases` in the environment's metadata dictionary.

**Before:**
```python
gymenv.metadata = {
    "controlled_zones": controlled_zones,
    "uncontrolled_zones": uncontrolled_zones,
    "observation_names": obs_info.slot_names,
    "action_names": action_names,
    "hvac_actuators": building_config.hvac_actuators,
    # Missing: area and warmup_phases
    "building_source_metadata": {...},
}
```

### Problem 2: Incorrect Attribute Access

The wrapper was trying to access a `_building_config` attribute that doesn't exist on the `EnergyPlusEnvironment` class.

**Before:**
```python
if hasattr(unwrapped, '_building_config'):  # This attribute doesn't exist!
    config = unwrapped._building_config
    params['area'] = float(config.area)
    # ...
```

## Solution

### Fix 1: Add Building Parameters to Metadata

**File:** `building2building/simulator/__init__.py` (lines 156-167)

```python
gymenv.metadata = {
    "controlled_zones": controlled_zones,
    "uncontrolled_zones": uncontrolled_zones,
    "observation_names": obs_info.slot_names,
    "action_names": action_names,
    "hvac_actuators": building_config.hvac_actuators,
    "area": building_config.area,                    # ✓ ADDED
    "warmup_phases": building_config.warmup_phases,  # ✓ ADDED
    "building_source_metadata": {...},
}
```

### Fix 2: Extract from Metadata

**File:** `building2building/simulator/wrappers.py` (lines 291-331)

```python
def _extract_building_params(self, env: gym.Env) -> dict[str, float]:
    """Extract building parameters from environment metadata."""
    params = {}
    
    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    # Extract from metadata instead of non-existent _building_config
    if hasattr(unwrapped, 'metadata') and isinstance(unwrapped.metadata, dict):
        metadata = unwrapped.metadata
        if 'area' in metadata:
            params['area'] = float(metadata['area'])
        if 'warmup_phases' in metadata:
            params['warmup_phases'] = float(metadata['warmup_phases'])
        if 'hvac_actuators' in metadata:
            params['num_actuators'] = float(len(metadata['hvac_actuators']))
    
    # Fallback to defaults with individual warnings
    if 'area' not in params:
        logger.warning("Could not extract 'area' from env metadata, using default")
        params['area'] = 100.0
    # ... (similar for other params)
    
    return params
```

## Verification

### How to Test

1. **Run the parameterized trainer:**
   ```bash
   sbatch scripts/slurm/run_ppo_parameterized.sh
   ```

2. **Check the logs for:**
   - ✓ "Augmented observation space from X to Y dimensions" - indicates wrapper is working
   - ✗ "Could not extract 'area' from env metadata" - indicates the fix didn't work

3. **Expected output:**
   ```
   INFO - Augmented observation space from 45 to 48 dimensions. Building params: ['area', 'warmup_phases', 'num_actuators']
   INFO - Observation space shape: (48,)
   ```

### What Changed

| Component | Before | After |
|-----------|--------|-------|
| `area` in metadata | ✗ Missing | ✓ Present |
| `warmup_phases` in metadata | ✗ Missing | ✓ Present |
| Extraction method | ✗ `_building_config` attribute | ✓ `metadata` dictionary |
| Error handling | ✗ Silent fallback | ✓ Individual warnings |

## Impact

- **Parameterized training**: Now correctly uses building-specific parameters
- **Observation space**: Properly augmented with 3 additional dimensions
- **Policy learning**: Can now condition on building characteristics
- **Backward compatibility**: Maintained (falls back to defaults if metadata missing)

## Files Modified

1. `building2building/simulator/__init__.py` - Added `area` and `warmup_phases` to metadata
2. `building2building/simulator/wrappers.py` - Fixed parameter extraction logic
3. `scripts/test_building_params.py` - Created test script (requires conda env)
4. `BUILDING_PARAMS_FIX.md` - Detailed documentation of the fix

