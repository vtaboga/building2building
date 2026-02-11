# Building Parameters Extraction - Debug and Fix

## Problem Identified

The `AugmentObservationWithBuildingParams` wrapper was trying to extract building parameters from a `_building_config` attribute that doesn't exist on the environment.

### Root Cause

1. **What we assumed**: The environment would have a `_building_config` attribute
2. **What actually happens**: The `create_simulator()` function creates an `EnergyPlusEnvironment` and stores building information in the `metadata` dictionary, NOT as a `_building_config` attribute

### Original Code (BROKEN)

```python
# In building2building/simulator/wrappers.py
def _extract_building_params(self, env: gym.Env) -> dict[str, float]:
    """Extract building parameters from environment metadata."""
    params = {}
    
    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    if hasattr(unwrapped, '_building_config'):  # ❌ This attribute doesn't exist!
        config = unwrapped._building_config
        params['area'] = float(config.area)
        params['warmup_phases'] = float(config.warmup_phases)
        params['num_actuators'] = float(len(config.hvac_actuators))
    else:
        # Would always fall back to defaults
        params['area'] = 100.0
        params['warmup_phases'] = 3.0
        params['num_actuators'] = 1.0
    
    return params
```

### Original Metadata (INCOMPLETE)

```python
# In building2building/simulator/__init__.py (line 156)
gymenv.metadata = {
    "controlled_zones": controlled_zones,
    "uncontrolled_zones": uncontrolled_zones,
    "observation_names": obs_info.slot_names,
    "action_names": action_names,
    "hvac_actuators": building_config.hvac_actuators,  # ✓ Has actuators
    # ❌ Missing: area
    # ❌ Missing: warmup_phases
    "building_source_metadata": dict(building_config.source_metadata)
    if isinstance(building_config.source_metadata, dict)
    else {},
}
```

## Solution Implemented

### Fix 1: Add Missing Fields to Metadata

Updated `building2building/simulator/__init__.py` to include `area` and `warmup_phases` in metadata:

```python
# In building2building/simulator/__init__.py (line 156)
gymenv.metadata = {
    "controlled_zones": controlled_zones,
    "uncontrolled_zones": uncontrolled_zones,
    "observation_names": obs_info.slot_names,
    "action_names": action_names,
    "hvac_actuators": building_config.hvac_actuators,
    "area": building_config.area,                      # ✓ ADDED
    "warmup_phases": building_config.warmup_phases,    # ✓ ADDED
    "building_source_metadata": dict(building_config.source_metadata)
    if isinstance(building_config.source_metadata, dict)
    else {},
}
```

### Fix 2: Extract from Metadata Instead of Non-existent Attribute

Updated `building2building/simulator/wrappers.py` to extract from metadata:

```python
# In building2building/simulator/wrappers.py
def _extract_building_params(self, env: gym.Env) -> dict[str, float]:
    """Extract building parameters from environment metadata."""
    params = {}
    
    # Try to get parameters from env metadata
    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    if hasattr(unwrapped, 'metadata') and isinstance(unwrapped.metadata, dict):
        metadata = unwrapped.metadata
        # Extract area
        if 'area' in metadata:
            params['area'] = float(metadata['area'])
        # Extract warmup_phases
        if 'warmup_phases' in metadata:
            params['warmup_phases'] = float(metadata['warmup_phases'])
        # Extract num_actuators from hvac_actuators list
        if 'hvac_actuators' in metadata:
            params['num_actuators'] = float(len(metadata['hvac_actuators']))
    
    # Use defaults for any missing parameters (with warnings)
    if 'area' not in params:
        logger.warning("Could not extract 'area' from env metadata, using default")
        params['area'] = 100.0
    if 'warmup_phases' not in params:
        logger.warning("Could not extract 'warmup_phases' from env metadata, using default")
        params['warmup_phases'] = 3.0
    if 'num_actuators' not in params:
        logger.warning("Could not extract 'num_actuators' from env metadata, using default")
        params['num_actuators'] = 1.0
    
    return params
```

## Expected Behavior After Fix

1. **Environment creation**: `create_simulator()` stores `area`, `warmup_phases`, and `hvac_actuators` in `metadata`
2. **Wrapper initialization**: `AugmentObservationWithBuildingParams` extracts these values from `metadata`
3. **Parameter normalization**: Building parameters are normalized to [-1, 1] range
4. **Observation augmentation**: Normalized parameters are concatenated to each observation

### Example

For a building with:
- Area: 150 m²
- Warmup phases: 3
- Num actuators: 2

The wrapper will:
1. Extract: `{'area': 150.0, 'warmup_phases': 3.0, 'num_actuators': 2.0}`
2. Normalize to [-1, 1]: `[0.22, -0.11, -0.78]` (approximately)
3. Append to each observation: `obs_augmented = np.concatenate([obs_original, [0.22, -0.11, -0.78]])`

## Files Modified

1. **`building2building/simulator/__init__.py`** (line 156-167)
   - Added `area` and `warmup_phases` to environment metadata

2. **`building2building/simulator/wrappers.py`** (line 291-331)
   - Changed extraction logic to read from `metadata` instead of `_building_config`
   - Added individual warnings for each missing parameter
   - More robust error handling

## Testing

To verify the fix works, you can:

1. Run the parameterized trainer and check logs for warnings
2. If you see warnings like "Could not extract 'area' from env metadata", the fix didn't work
3. If you see "Augmented observation space from X to Y dimensions", the fix is working

## Impact on Existing Code

- ✓ **Backward compatible**: Existing code that doesn't use the wrapper is unaffected
- ✓ **Metadata extension**: Adding fields to metadata doesn't break existing code
- ✓ **Graceful degradation**: Falls back to defaults if metadata is missing (with warnings)

