#!/usr/bin/env python3
"""
W&B Sweep Runner for DQN Hyperparameter Optimization
"""

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import subprocess
import sys
from pathlib import Path
import os

# Import environment to register it
import building2building.simulator


@hydra.main(version_base=None, config_path="../configs", config_name="sweep_dqn_optimization")
def main_hydra(cfg: DictConfig) -> None:
    """Main function for running W&B sweep with Hydra configuration."""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting DQN hyperparameter sweep...")
    
    # Convert entire config to regular dict for W&B compatibility
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    # Initialize W&B sweep
    sweep_config = {
        "method": config_dict.get("method", "bayes"),
        "metric": config_dict.get("metric", {"name": "validation/episodic_return", "goal": "maximize"}),
        "early_terminate": config_dict.get("early_terminate", {"type": "hyperband", "min_iter": 100000}),
        "parameters": config_dict.get("parameters", {}),
        "project": config_dict.get("project", "building2building"),
        "entity": config_dict.get("entity"),
        "name": config_dict.get("name", "dqn_bayes_opt"),
        "program": config_dict.get("program", "scripts/train_dqn.py")
    }
    
    # Remove None values
    sweep_config = {k: v for k, v in sweep_config.items() if v is not None}
    
    logger.info(f"Sweep configuration: {sweep_config}")
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=config_dict.get("project", "building2building"))
    logger.info(f"Sweep initialized with ID: {sweep_id}")
    
    # Start the sweep agent
    wandb.agent(sweep_id, function=train_dqn_agent, count=config_dict.get("count", 50))
    logger.info("Sweep completed!")


def train_dqn_agent():
    """Agent function that will be called by W&B sweep."""
    # Initialize W&B run first
    run = wandb.init()
    config = wandb.config
    
    # Convert config to command line arguments for Hydra
    args = []
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)):
            args.append(f"{key}={value}")
        elif isinstance(value, list):
            # Use proper Hydra list syntax: key=[value1,value2]
            list_str = ','.join(map(str, value))
            args.append(f"{key}=[{list_str}]")
    
    # Build the command to run the training script
    cmd = [sys.executable, "scripts/train_dqn.py"] + args
    
    # Create unique output directory for this run
    run_name = run.name or f"run_{run.id}"
    output_dir = f"results/sweep_runs/{run_name}"
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Add Hydra overrides for proper output organization
    cmd.extend([
        f"hydra.run.dir={output_dir}",
        "hydra.mode=RUN",
        "track=true"
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Results will be stored in: {output_dir}")
    
    # Run the training script with better error handling
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully")
        print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        # Log the results directory to W&B
        if os.path.exists(output_dir):
            wandb.log({"results_dir": output_dir})
            print(f"Results saved to: {output_dir}")
        
        run.finish()
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        run.finish()
        raise


if __name__ == "__main__":
    main_hydra()
