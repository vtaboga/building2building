#!/bin/bash
# Comprehensive cleanup script for Building2Building outputs

set -e

echo "========================================="
echo "Building2Building Disk Cleanup Script"
echo "========================================="
echo ""

# Function to get size before cleanup
get_size() {
    du -sh outputs 2>/dev/null | awk '{print $1}'
}

BEFORE=$(get_size)
echo "Current outputs directory size: $BEFORE"
echo ""

# 1. Remove incomplete/failed runs (no test results)
echo "1. Removing incomplete/failed runs..."
INCOMPLETE_COUNT=0
for dir in outputs/per_building_adaptive_dynamics/ppo/*/*/; do
    if [ -d "$dir" ] && [ ! -f "$dir/test/adaptive_dynamics_results.jsonl" ]; then
        rm -rf "$dir"
        ((INCOMPLETE_COUNT++))
    fi
done
for dir in outputs/parameterized_adaptive_dynamics/ppo/*/*/; do
    if [ -d "$dir" ] && [ ! -f "$dir/test/adaptive_dynamics_results.jsonl" ]; then
        rm -rf "$dir"
        ((INCOMPLETE_COUNT++))
    fi
done
for dir in outputs/baseline_adaptive_dynamics/ppo/*/*/; do
    if [ -d "$dir" ] && [ ! -f "$dir/test/adaptive_dynamics_results.jsonl" ]; then
        rm -rf "$dir"
        ((INCOMPLETE_COUNT++))
    fi
done
echo "   ✓ Removed $INCOMPLETE_COUNT incomplete runs"

# 2. Remove old runs (keep only most recent 3 per type)
echo ""
echo "2. Removing old completed runs (keeping 3 most recent per type)..."

# For parameterized runs
PARAM_RUNS=$(find outputs/parameterized_adaptive_dynamics/ppo -mindepth 2 -maxdepth 2 -type d -exec stat -c '%Y %n' {} \; 2>/dev/null | sort -rn | awk '{print $2}')
PARAM_COUNT=0
echo "$PARAM_RUNS" | tail -n +4 | while read dir; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
        ((PARAM_COUNT++))
    fi
done
echo "   ✓ Cleaned old parameterized runs"

# For baseline runs
BASELINE_RUNS=$(find outputs/baseline_adaptive_dynamics/ppo -mindepth 2 -maxdepth 2 -type d -exec stat -c '%Y %n' {} \; 2>/dev/null | sort -rn | awk '{print $2}')
echo "$BASELINE_RUNS" | tail -n +4 | while read dir; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
done
echo "   ✓ Cleaned old baseline runs"

# 3. Clean up old benchmark runs
echo ""
echo "3. Cleaning old benchmark runs..."
find outputs/benchmarks -mindepth 2 -maxdepth 2 -type d -exec stat -c '%Y %n' {} \; 2>/dev/null | sort -rn | tail -n +4 | awk '{print $2}' | while read dir; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
done
echo "   ✓ Cleaned old benchmark runs"

# 4. Clean up old dated runs
echo ""
echo "4. Cleaning old dated runs (2026-02-02, etc.)..."
find outputs -maxdepth 1 -type d -name "20*" -exec rm -rf {} + 2>/dev/null
echo "   ✓ Cleaned old dated runs"

# 5. Clean wandb cache
echo ""
echo "5. Cleaning wandb cache..."
if [ -d "wandb" ]; then
    find wandb -type d -name "run-*" -mtime +7 -exec rm -rf {} + 2>/dev/null
fi
echo "   ✓ Cleaned old wandb runs"

echo ""
echo "========================================="
echo "Cleanup Summary"
echo "========================================="
AFTER=$(get_size)
echo "Before: $BEFORE"
echo "After:  $AFTER"
echo ""
echo "Detailed breakdown:"
du -sh outputs/* 2>/dev/null | sort -hr
echo ""
echo "✓ Cleanup complete!"

