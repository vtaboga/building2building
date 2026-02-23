# Utility Scripts

This directory contains utility scripts for managing the Building2Building repository.

## Disk Cleanup Scripts

EnergyPlus simulations generate large output directories that can quickly consume disk space. These scripts help clean up intermediate outputs while preserving important results.

### `cleanup_eplus_outputs.sh`

Removes intermediate EnergyPlus output directories from training and evaluation runs.

**What it removes:**
- `eval_eplus_outputs/` - Intermediate evaluation outputs
- `train_eplus_outputs_*/` - Training simulation outputs

**What it keeps:**
- `test/eplus_outputs/` - Final evaluation results

**Usage:**
```bash
./scripts/utils/cleanup_eplus_outputs.sh
```

**When to use:** After training runs complete, to free up disk space while keeping final test results.

---

### `cleanup_old_runs.sh`

More aggressive cleanup that removes entire old run directories based on age.

**Usage:**
```bash
./scripts/utils/cleanup_old_runs.sh
```

**Warning:** Review the script before running to ensure it won't delete runs you want to keep.

---

### `simple_cleanup.sh`

Quick cleanup of common temporary files and outputs.

**Usage:**
```bash
./scripts/utils/simple_cleanup.sh
```

---

## Tips

- **Check disk usage before cleanup:**
  ```bash
  du -sh outputs/
  ```

- **Preview what will be deleted:**
  Modify the scripts to use `echo` instead of `rm -rf` to see what would be deleted

- **Backup important runs:**
  Copy any runs you want to keep to a separate directory before running cleanup scripts

