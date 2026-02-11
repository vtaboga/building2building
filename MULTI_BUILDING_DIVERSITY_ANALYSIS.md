# Multi-Building Training Diversity Analysis

## Question
**Is this model successfully being trained on multiple, diverse buildings or is it always just the same building?**

## Answer: ❌ **SAME BUILDING - No Diversity**

### Current Behavior

The parameterized trainer is **NOT** training on diverse buildings. Here's why:

#### 1. How `make_env()` Works
```python
# In algorithms/utils.py
def make_env(config, eplus_output_dir: str):
    configs = hydroquebec.search_configs(config=config, n=1, ...)  # ← Requests only 1 building
    env_config = configs[0]  # ← Always the first building
    env = create_simulator(env_config)
    return env
```

#### 2. How `search_configs()` Works
```python
# In building2building/sources/hydroquebec.py
def search_configs(config, n=1, ...):
    rows = search_buildings(**config_nn)  # Filter by criteria
    
    # Sort deterministically - "single-family detached" first
    rows = rows.sort_values("_b2b_priority", kind="stable")
    
    # Return first n buildings that successfully build
    for _, row in rows.iterrows():
        if len(configs) >= n:
            break
        # ... build config ...
        configs.append(BuildingConfig(...))
    
    return configs  # ← ALWAYS returns same buildings in same order
```

#### 3. What Happens in Training
- **4 parallel environments** are created (from `configs/training/default.yaml`)
- Each environment calls `make_env()` independently
- Each call to `make_env()` calls `search_configs(n=1)`
- `search_configs()` returns buildings in **deterministic order**
- **Result**: All 4 environments get the **EXACT SAME BUILDING**

### Evidence

1. **No randomization** in `search_configs()` - it always returns buildings in the same order
2. **Deterministic sorting** - "single-family detached" buildings are prioritized first
3. **Each environment is stuck** with the same building for its entire lifetime
4. **Building parameters are constant** across all parallel environments

### Impact

| Aspect | Current State | Expected for Multi-Building |
|--------|--------------|----------------------------|
| Building diversity | ❌ Single building | ✅ Multiple diverse buildings |
| Parameter variation | ❌ Constant values | ✅ Varying area, warmup, actuators |
| Generalization | ❌ Overfits to one building | ✅ Learns general policy |
| Training data | ❌ Homogeneous | ✅ Heterogeneous |

## Solution: Implemented but NOT Integrated

### What Was Created
✅ **`algorithms/multi_building_utils.py`** with:
- `fetch_diverse_building_pool()` - Fetches N diverse buildings upfront
- `make_diverse_env()` - Samples randomly from the building pool

### What's Still Needed
❌ **Integration into `parameterized_trainer.py`**:
1. Fetch diverse building pool before creating environments
2. Pass building pool to environment creation
3. Use `make_diverse_env()` instead of `make_env()`
4. Add configuration for pool size

## Next Steps

### Step 1: Modify `parameterized_trainer.py`

```python
from algorithms.multi_building_utils import fetch_diverse_building_pool, make_diverse_env

def _make_parameterized_envs(config: OmegaConf, output_dir: Path):
    # Fetch diverse building pool upfront
    n_buildings = config.training.get('num_buildings_in_pool', 10)
    building_pool = fetch_diverse_building_pool(config, n_buildings)
    
    def wrapper_fn(env):
        if augment_params:
            env = AugmentObservationWithBuildingParams(env)
        if norm_obs:
            env = NormalizeObservation(env)
        return env
    
    num_envs = int(config.training.num_train_envs)
    train_env = make_vec_env(
        make_diverse_env,  # ← Changed from make_env
        n_envs=num_envs,
        env_kwargs={
            'config': config,
            'eplus_output_dir': str(output_dir / "train_eplus_outputs"),
            'building_pool': building_pool  # ← Pass the pool
        },
        wrapper_class=wrapper_fn,
    )
    # ... similar for eval_env
```

### Step 2: Add Configuration

```yaml
# In configs/training/default.yaml
num_buildings_in_pool: 20  # Fetch 20 diverse buildings for training
```

### Step 3: Verify Diversity

After integration, check logs for:
```
INFO - Building pool diversity:
INFO -   - Areas: min=100.0, max=500.0, mean=250.0 m²
INFO -   - Warmup phases: min=1, max=5, mean=3.0
INFO -   - Actuators: min=2, max=8, mean=4.5
```

## Current Status

- ✅ Problem identified
- ✅ Solution implemented (`multi_building_utils.py`)
- ❌ Solution NOT integrated into trainer
- ❌ Configuration NOT updated
- ❌ Diversity NOT verified

**The model is currently training on a SINGLE building repeatedly, defeating the purpose of parameterized training.**

