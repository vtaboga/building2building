# Test Policy Error Fix

## Error

```
TypeError: test_policy() got an unexpected keyword argument 'model'
```

## Root Causes

### Issue 1: Wrong Function Signature
The `test_policy()` function signature is:
```python
def test_policy(config, policy_model, output_dir: Path):
```

But it was being called with keyword arguments:
```python
test_policy(
    model=best_model,        # ❌ Wrong parameter name
    config=config,
    output_dir=test_dir,
    n_episodes=config.get('n_episodes', 1),  # ❌ Not a parameter
)
```

### Issue 2: Missing Building Parameter Augmentation in Test
The `test_policy()` function wasn't applying the `AugmentObservationWithBuildingParams` wrapper, so the observation space wouldn't match what the model was trained on.

### Issue 3: Missing `denormalize()` Method
The `AugmentObservationWithBuildingParams` wrapper didn't have a `denormalize()` method, which is called by `test_policy()` for logging.

## Fixes Applied

### Fix 1: Correct Function Call
**File:** `algorithms/parameterized_trainer.py` (line 224)

**Before:**
```python
test_policy(
    model=best_model,
    config=config,
    output_dir=test_dir,
    n_episodes=config.get('n_episodes', 1),
)
```

**After:**
```python
test_policy(config, best_model, output_dir)
```

### Fix 2: Apply Building Parameter Augmentation in Test
**File:** `algorithms/test.py` (lines 74-89)

**Before:**
```python
env = make_env(config=config, eplus_output_dir=str(test_dir / "eplus_outputs"))
norm_obs = config.env.normalize_obs
if norm_obs:
    env = NormalizeObservation(env)
```

**After:**
```python
env = make_env(config=config, eplus_output_dir=str(test_dir / "eplus_outputs"))

# Apply building parameter augmentation if configured
augment_params = config.env.get('augment_building_params', False)
if augment_params:
    env = AugmentObservationWithBuildingParams(env)

# Apply observation normalization if configured
norm_obs = config.env.normalize_obs
if norm_obs:
    env = NormalizeObservation(env)
```

### Fix 3: Add `denormalize()` Method to Wrapper
**File:** `building2building/simulator/wrappers.py` (lines 362-377)

**Added:**
```python
def denormalize(self, obs: np.ndarray) -> np.ndarray:
    """
    Remove building parameters and denormalize the original observation.
    
    This is used for logging/visualization purposes.
    """
    # Split off the building parameters (last N dimensions)
    n_params = len(self.normalized_params)
    obs_without_params = obs[:-n_params] if n_params > 0 else obs
    
    # If the wrapped env has a denormalize method, use it
    if hasattr(self.env, 'denormalize'):
        return self.env.denormalize(obs_without_params)
    
    # Otherwise, just return the observation without building params
    return obs_without_params
```

## Files Modified

1. **`algorithms/parameterized_trainer.py`** (line 224)
   - Fixed `test_policy()` call to use correct positional arguments

2. **`algorithms/test.py`** (lines 9-13, 74-89)
   - Added import for `AugmentObservationWithBuildingParams`
   - Added logic to apply building parameter augmentation when configured

3. **`building2building/simulator/wrappers.py`** (lines 362-377)
   - Added `denormalize()` method to `AugmentObservationWithBuildingParams`

## Expected Behavior

After these fixes, the parameterized trainer should:
1. ✅ Train successfully with building parameter augmentation
2. ✅ Test the policy with the same wrapper configuration
3. ✅ Generate test CSV files with denormalized observations
4. ✅ Complete without errors

## Testing

Re-run the parameterized training:
```bash
sbatch scripts/slurm/run_ppo_parameterized.sh
```

The job should now complete successfully and generate test outputs in the `test/` directory.

