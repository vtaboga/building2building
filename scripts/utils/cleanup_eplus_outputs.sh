#!/bin/bash
# Cleanup script to remove EnergyPlus output directories

echo "Finding and removing EnergyPlus output directories..."
echo "This will free up significant disk space."
echo ""

# Find and remove eval_eplus_outputs directories (intermediate eval outputs)
echo "Removing eval_eplus_outputs directories..."
find outputs -type d -name "eval_eplus_outputs" -exec du -sh {} \; 2>/dev/null | head -5
find outputs -type d -name "eval_eplus_outputs" -exec rm -rf {} + 2>/dev/null
echo "✓ Removed eval_eplus_outputs directories"

# Find and remove train_eplus_outputs_* directories (training outputs)
echo ""
echo "Removing train_eplus_outputs_* directories..."
find outputs -type d -name "train_eplus_outputs_*" -exec du -sh {} \; 2>/dev/null | head -5
find outputs -type d -name "train_eplus_outputs_*" -exec rm -rf {} + 2>/dev/null
echo "✓ Removed train_eplus_outputs_* directories"

# Keep test/eplus_outputs as they contain final evaluation results
echo ""
echo "Keeping test/eplus_outputs directories (final evaluation results)"

echo ""
echo "Cleanup complete!"
echo ""
echo "Disk usage after cleanup:"
du -sh outputs
