"""
Example workflow for using OfflineRL-Kit with Building2Building environment.
This script demonstrates the complete pipeline from data collection to training.
"""

import os
import subprocess
import json
from pathlib import Path


def main():
    """
    Example workflow:
    1. Collect offline dataset using baseline policy
    2. Train offline RL algorithm (CQL) on the dataset
    3. Evaluate the trained policy
    """
    
    # Configuration
    state = "AL"
    county = "Mobile" 
    building_id = "example_building"
    weather_file = "Mobile_AL.epw"
    
    # Step 1: Collect baseline dataset
    print("Step 1: Collecting offline dataset using baseline policy...")
    
    dataset_path = f"data/offline_datasets/{state}_{county}_{building_id}_baseline.npz"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    collect_cmd = [
        "python", "scripts/collect_offline_dataset.py",
        "--state", state,
        "--county", county, 
        "--building-id", building_id,
        "--weather", weather_file,
        "--output-path", dataset_path,
        "--data-source", "baseline",
        "--num-episodes", "50",  # Small number for demo
        "--heating-setpoint", "21.0",
        "--cooling-setpoint", "24.0",
        "--seed", "42"
    ]
    
    print("Running:", " ".join(collect_cmd))
    # subprocess.run(collect_cmd)  # Uncomment to actually run
    
    # Step 2: Train CQL algorithm on the dataset
    print("\nStep 2: Training CQL algorithm on offline dataset...")
    
    train_cmd = [
        "python", "scripts/train_offline_rl.py",
        "--state", state,
        "--county", county,
        "--building-id", building_id, 
        "--weather", weather_file,
        "--dataset-path", dataset_path,
        "--algorithm", "cql",
        "--epoch", "100",  # Fewer epochs for demo
        "--batch-size", "128",
        "--lr", "3e-4",
        "--cql-weight", "1.0",
        "--temperature", "1.0",
        "--save-model",
        "--seed", "42"
    ]
    
    print("Running:", " ".join(train_cmd))
    # subprocess.run(train_cmd)  # Uncomment to actually run
    
    # Step 3: Evaluate trained policy (placeholder)
    print("\nStep 3: Evaluating trained policy...")
    print("(Evaluation functionality can be added to offline_rl.py)")
    
    print("\nWorkflow completed!")
    print(f"Dataset saved to: {dataset_path}")
    print("Trained model will be saved in results/ directory")
    
    # Print summary of what was done
    print("\n" + "="*60)
    print("SUMMARY:")
    print("1. ✓ Added OfflineRL-Kit to requirements.txt")
    print("2. ✓ Created building2building/algorithms/offline_rl.py")
    print("3. ✓ Created scripts/train_offline_rl.py")
    print("4. ✓ Created scripts/collect_offline_dataset.py") 
    print("5. ✓ Created jobs/train_offline_rl.sh")
    print("\nTo use:")
    print("  1. Install: pip install -r requirements.txt")
    print("  2. Collect data: python scripts/collect_offline_dataset.py [args]")
    print("  3. Train: python scripts/train_offline_rl.py [args]")
    print("="*60)


if __name__ == "__main__":
    main() 