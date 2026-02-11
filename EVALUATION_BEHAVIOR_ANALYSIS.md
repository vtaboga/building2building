# Evaluation Behavior Analysis

## Question: What happens during eval?

There are **TWO types of evaluation** in the parameterized trainer:

---

## 1. **Periodic Evaluation During Training** (EvalCallback)

### When It Happens
- Every `eval_freq` timesteps (default: **35,040 timesteps**)
- With 4 parallel envs, this is approximately every **8,760 steps** per environment
- At 672 steps per episode (1 week), this is roughly every **13 episodes**

### What Happens
```python
# From _make_callbacks() in parameterized_trainer.py
eval_cb = EvalCallback(
    eval_env,                              # Single evaluation environment
    eval_freq=35040,                       # Evaluate every 35,040 timesteps
    n_eval_episodes=1,                     # Run 1 episode per evaluation
    deterministic=True,                    # Use deterministic policy (no exploration)
    best_model_save_path=str(model_dir),   # Save best model based on eval reward
)
```

### Current Behavior (✅ With Diverse Sampling)
```python
# From _make_parameterized_envs() in parameterized_trainer.py
eval_env = make_vec_env(
    make_diverse_env,                      # ← Uses diverse sampling!
    n_envs=1,
    env_kwargs={
        'building_pool': building_pool     # ← Same pool as training (20 buildings)
    },
)
```

**Result:**
- ✅ Eval environment **randomly samples from the 20-building pool**
- ✅ Each evaluation uses a **different building** (random from pool)
- ✅ Tests generalization across diverse buildings during training
- ✅ Best model is selected based on performance across diverse buildings

### Evaluation Flow
```
Training step 0 → 35,040 → 70,080 → 105,120 → ...
                     ↓         ↓          ↓
                   Eval 1    Eval 2    Eval 3
                     ↓         ↓          ↓
              Building M  Building C  Building Q  (random from pool)
                     ↓         ↓          ↓
              Reward: -150  Reward: -120  Reward: -180
                              ↑
                    Best model saved! (highest reward)
```

---

## 2. **Final Testing After Training** (test_policy)

### When It Happens
- **Once** at the end of training
- After loading the best model from periodic evaluations

### What Happens
```python
# From parameterized_trainer.py
test_policy(config, best_model, output_dir)
```

### Current Behavior (⚠️ Uses SINGLE Building)
```python
# From test.py
def test_policy(config, policy_model, output_dir: Path):
    env = make_env(config=config, ...)  # ← Uses make_env(), NOT make_diverse_env()
    # ... runs n_episodes (default: 1)
```

**Result:**
- ⚠️ Test uses `make_env()` which returns the **FIRST building** from search
- ⚠️ Always tests on the **SAME building** (deterministic)
- ⚠️ Does NOT test on diverse buildings
- ⚠️ Limited evaluation of generalization

---

## Summary Table

| Aspect | Periodic Eval (During Training) | Final Test (After Training) |
|--------|--------------------------------|----------------------------|
| **When** | Every 35,040 timesteps | Once at end |
| **Function** | `EvalCallback` | `test_policy()` |
| **Environment** | `make_diverse_env()` ✅ | `make_env()` ⚠️ |
| **Building diversity** | ✅ Random from 20-building pool | ⚠️ Always same building |
| **Episodes** | 1 per evaluation | 1 (configurable) |
| **Purpose** | Select best model | Generate CSV logs |
| **Deterministic** | ✅ Yes | ✅ Yes |

---

## Issues Identified

### ⚠️ **Issue 1: Final Test Uses Single Building**

The final test phase uses `make_env()` instead of `make_diverse_env()`, so it:
- Always tests on the **same building** (first match from search)
- Does NOT verify generalization across diverse buildings
- Generates CSV logs for only **one building**

### ⚠️ **Issue 2: Limited Test Coverage**

With `n_episodes: 1`, the final test only runs:
- **1 episode** on **1 building**
- No comprehensive evaluation of generalization

---

## Recommendations

### Option 1: Test on Diverse Buildings (Recommended)

Modify `test_policy()` to use diverse building sampling:

```python
# In algorithms/test.py
def test_policy(config, policy_model, output_dir: Path, building_pool=None):
    test_dir = output_dir / "test"
    
    if building_pool:
        # Use diverse sampling if pool provided
        env = make_diverse_env(
            config=config,
            eplus_output_dir=str(test_dir / "eplus_outputs"),
            building_pool=building_pool
        )
    else:
        # Fallback to single building
        env = make_env(config=config, eplus_output_dir=str(test_dir / "eplus_outputs"))
    
    # ... rest of test logic
```

Then pass the building pool from the trainer:
```python
# In parameterized_trainer.py
test_policy(config, best_model, output_dir, building_pool=building_pool)
```

### Option 2: Test on Multiple Buildings

Run multiple test episodes on different buildings:

```python
# In configs/parameterized.yaml
n_episodes: 5  # Test on 5 different buildings
```

Then modify `test_policy()` to sample a new building for each episode.

### Option 3: Test on Specific Unseen Buildings

Create a separate test set of buildings not in the training pool:

```python
# Fetch separate test buildings
test_building_pool = fetch_diverse_building_pool(config, n_buildings=10)
# Filter out buildings that were in training pool
# ... test on unseen buildings
```

---

## Current Evaluation Summary

### ✅ What Works Well
- Periodic evaluation uses diverse buildings from the pool
- Best model is selected based on diverse building performance
- Deterministic evaluation for reproducibility

### ⚠️ What Needs Improvement
- Final test only uses a single building
- No comprehensive generalization testing
- CSV logs only generated for one building

---

## Next Steps

1. **Decide on test strategy**: 
   - Test on diverse buildings from training pool?
   - Test on unseen buildings?
   - Test on multiple episodes?

2. **Modify `test_policy()`** to support diverse building testing

3. **Update configuration** to specify test behavior

4. **Generate comprehensive test reports** across multiple buildings

Would you like me to implement any of these recommendations?

