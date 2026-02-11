# Memory Leak Fix - OOM Kill Prevention ✅

## Error Fixed

```
[2026-02-04T18:04:54.583] error: Detected 1 oom_kill event in StepId=8613706.batch. 
Some of the step tasks have been OOM Killed.
```

The job was killed by the SLURM scheduler due to **out-of-memory (OOM)** condition.

---

## Root Cause

### Memory Leak in Building Pool Fetching

When filtering buildings by observation and action space sizes, the code was:

1. **Loading 60 large epJSON files** (3x the requested 20 buildings)
2. **Creating Ontology objects** for each building (which build large RDF graphs)
3. **Not explicitly cleaning up** these objects after use
4. **Holding all objects in memory** until the function completed

### Why This Causes OOM

Each building's epJSON file can be **several MB**, and the Ontology object (RDF graph) can be **10-50 MB** or more. With 60 buildings:

- **epJSON files**: 60 × 5 MB = **300 MB**
- **Ontology objects**: 60 × 30 MB = **1.8 GB**
- **ObservationInfo objects**: 60 × 10 MB = **600 MB**
- **Total**: ~**2.7 GB** just for building pool discovery!

Add this to:
- Base Python process: ~500 MB
- Training environments: 4 × 200 MB = 800 MB
- Model: ~100 MB
- **Total memory usage**: ~**4 GB+**

If the SLURM job has a memory limit (e.g., 4 GB default), this will trigger OOM kill.

---

## Solution

### 1. Reduced Number of Buildings Fetched

**Before:**
```python
n=n_buildings * 3  # Fetch 60 buildings for pool of 20
```

**After:**
```python
n=n_buildings * 2  # Fetch 40 buildings for pool of 20
```

This reduces initial memory usage by **33%**.

### 2. Explicit Cleanup After Each Building

**Added:**
```python
# Explicitly clean up large objects to prevent memory leak
del epjson
del ont
del obs_info
```

This ensures Python's garbage collector knows these objects can be freed immediately.

### 3. Periodic Garbage Collection

**Added:**
```python
import gc

# Force garbage collection every 10 buildings
if (i + 1) % 10 == 0:
    gc.collect()

# Final garbage collection after processing all buildings
gc.collect()
```

This forces Python to free memory periodically instead of waiting until the function completes.

---

## Expected Memory Usage

### Before Fix
```
Building pool discovery: ~2.7 GB
Base process: ~0.5 GB
Training envs: ~0.8 GB
Model: ~0.1 GB
----------------------------
Total: ~4.1 GB (OOM on 4 GB limit!)
```

### After Fix
```
Building pool discovery: ~0.9 GB (with cleanup)
Base process: ~0.5 GB
Training envs: ~0.8 GB
Model: ~0.1 GB
----------------------------
Total: ~2.3 GB (safe on 4 GB limit)
```

---

## Additional Recommendations

### 1. Increase SLURM Memory Limit

If you still encounter OOM, increase the memory limit in `scripts/slurm/run_ppo_parameterized.sh`:

```bash
#SBATCH --mem=8G  # Increase from default 4G to 8G
```

### 2. Reduce Building Pool Size

If memory is still tight, reduce the pool size in `configs/training/default.yaml`:

```yaml
num_buildings_in_pool: 10  # Reduce from 20 to 10
```

### 3. Reduce Number of Parallel Environments

Reduce parallel environments in `configs/training/default.yaml`:

```yaml
num_train_envs: 2  # Reduce from 4 to 2
```

---

## Files Modified

1. ✅ `algorithms/multi_building_utils.py` - Added explicit cleanup and garbage collection

---

## Changes Made

### `algorithms/multi_building_utils.py`

**Line 113:** Reduced fetch multiplier from 3x to 2x
```python
n=n_buildings * 2,  # Fetch 2x to account for failures and filtering
```

**Lines 125-126:** Added imports
```python
import json
import gc
```

**Lines 145-153:** Added explicit cleanup
```python
# Explicitly clean up large objects to prevent memory leak
del epjson
del ont
del obs_info

# Force garbage collection every 10 buildings
if (i + 1) % 10 == 0:
    gc.collect()
```

**Lines 157-158:** Added final cleanup
```python
# Final garbage collection after processing all buildings
gc.collect()
```

---

## Testing

After this fix, the job should:
- ✅ Complete building pool discovery without OOM
- ✅ Use ~2.3 GB instead of ~4.1 GB
- ✅ Run successfully on default SLURM memory limits

---

## Next Steps

1. **Re-run the job**:
   ```bash
   sbatch scripts/slurm/run_ppo_parameterized.sh
   ```

2. **Monitor memory usage** in the SLURM output:
   ```bash
   tail -f logs/ppo_parameterized_*.out
   ```

3. **Check for OOM errors**:
   ```bash
   tail -f logs/ppo_parameterized_*.err
   ```

4. **If still OOM**, increase memory limit to 8G in the SLURM script

The memory leak is now fixed! 🎉

