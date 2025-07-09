#!/usr/bin/env python3
"""
Script to run DQN hyperparameter optimization sweep using W&B and Hydra.
This script initializes the sweep and runs agents using the container system.
"""

import argparse
import subprocess
import sys
import os
import yaml
import tempfile
from pathlib import Path

def create_sweep_config(sweep_config_path: str, entity: str = None):
    """Create or modify the sweep configuration with the correct entity."""
    with open(sweep_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if entity:
        config['entity'] = entity
    
    # Create a temporary file with the updated config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        return f.name

def initialize_sweep(sweep_config_path: str, entity: str = None):
    """Initialize the W&B sweep and return the sweep ID."""
    print("Initializing W&B sweep...")
    
    # Create temporary config with entity if provided
    if entity:
        temp_config = create_sweep_config(sweep_config_path, entity)
        config_to_use = temp_config
    else:
        config_to_use = sweep_config_path
    
    try:
        # Initialize sweep
        result = subprocess.run(
            ['wandb', 'sweep', config_to_use],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract sweep ID from output
        for line in result.stdout.strip().split('\n'):
            if 'wandb agent' in line:
                sweep_id = line.split('wandb agent ')[-1].strip()
                print(f"Sweep initialized with ID: {sweep_id}")
                return sweep_id
        
        raise ValueError("Could not parse sweep ID from wandb output")
        
    finally:
        # Clean up temporary file
        if entity and os.path.exists(config_to_use):
            os.unlink(config_to_use)

def run_sweep_agent(sweep_id: str, scratch_dir: str, count: int = 1):
    """Run W&B sweep agent with the correct environment setup."""
    print(f"Running sweep agent for {count} runs...")
    
    # Set up environment variables
    env = os.environ.copy()
    env['WANDB_DIR'] = f"{scratch_dir}/wandb"
    env['SCRATCH_DIR'] = scratch_dir
    
    # Run the sweep agent
    cmd = ['wandb', 'agent', sweep_id, '--count', str(count)]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Environment: WANDB_DIR={env['WANDB_DIR']}, SCRATCH_DIR={env['SCRATCH_DIR']}")
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"Sweep agent failed with exit code: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run DQN hyperparameter optimization sweep')
    parser.add_argument('--entity', help='W&B entity (defaults to your default entity)')
    parser.add_argument('--count', type=int, default=1, help='Number of sweep runs per agent (default: 1)')
    parser.add_argument('--agents', type=int, default=1, help='Number of agents to run simultaneously (default: 1)')
    parser.add_argument('--sweep-config', default='configs/sweep_dqn_optimization.yaml', 
                       help='Path to sweep configuration file')
    parser.add_argument('--init-only', action='store_true', 
                       help='Only initialize sweep and print sweep ID')
    parser.add_argument('--sweep-id', help='Use existing sweep ID instead of creating new one')
    
    args = parser.parse_args()
    
    # Get scratch directory from environment variable
    scratch_dir = os.environ.get('SCRATCH_DIR')
    if not scratch_dir:
        print("Error: SCRATCH_DIR environment variable not set")
        print("This script should be run inside the container where SCRATCH_DIR is set")
        sys.exit(1)
    
    # Find sweep config file using robust path resolution
    config_possible_paths = [
        args.sweep_config,
        f"../{args.sweep_config}",
        f"/opt/repository/{args.sweep_config}",
        f"/opt/repository-host/{args.sweep_config}"
    ]
    
    sweep_config_path = None
    for path in config_possible_paths:
        if os.path.exists(path):
            sweep_config_path = path
            break
    
    if not sweep_config_path:
        print(f"Error: Sweep configuration file not found. Tried paths:")
        for path in config_possible_paths:
            print(f"   - {path}")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)
    
    print(f"Using sweep config: {sweep_config_path}")
    
    try:
        # Initialize sweep or use existing one
        if args.sweep_id:
            sweep_id = args.sweep_id
            print(f"Using existing sweep ID: {sweep_id}")
        else:
            sweep_id = initialize_sweep(sweep_config_path, args.entity)
        
        if args.init_only:
            print(f"Sweep initialized. To run agents, use:")
            print(f"  wandb agent {sweep_id}")
            print(f"Or run this script again with --sweep-id {sweep_id}")
            return
        
        # Run agents
        print(f"Starting {args.agents} agents...")
        for i in range(args.agents):
            print(f"Starting agent {i+1}/{args.agents}")
            success = run_sweep_agent(sweep_id, scratch_dir, args.count)
            if not success:
                print(f"Agent {i+1} failed!")
                sys.exit(1)
        
        print("All agents completed successfully!")
        print(f"View results at: https://wandb.ai/sweep/{sweep_id}")
        
    except KeyboardInterrupt:
        print("\nSweep interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 