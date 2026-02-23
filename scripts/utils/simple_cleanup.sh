#!/bin/bash
echo "Cleaning incomplete runs..."

# Remove incomplete per-building runs
find outputs/per_building_adaptive_dynamics/ppo -mindepth 2 -maxdepth 2 -type d | while read dir; do
    if [ ! -f "$dir/test/adaptive_dynamics_results.jsonl" ]; then
        echo "Removing incomplete: $dir"
        rm -rf "$dir"
    fi
done

# Remove incomplete parameterized runs  
find outputs/parameterized_adaptive_dynamics/ppo -mindepth 2 -maxdepth 2 -type d | while read dir; do
    if [ ! -f "$dir/test/adaptive_dynamics_results.jsonl" ]; then
        echo "Removing incomplete: $dir"
        rm -rf "$dir"
    fi
done

# Remove incomplete baseline runs
find outputs/baseline_adaptive_dynamics/ppo -mindepth 2 -maxdepth 2 -type d 2>/dev/null | while read dir; do
    if [ ! -f "$dir/test/adaptive_dynamics_results.jsonl" ]; then
        echo "Removing incomplete: $dir"
        rm -rf "$dir"
    fi
done

# Remove old dated directories
rm -rf outputs/2026-02-02 outputs/parameterized outputs/unitary_pi 2>/dev/null

echo ""
echo "Cleanup complete!"
du -sh outputs
