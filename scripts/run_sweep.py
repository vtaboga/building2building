#!/usr/bin/env python3
"""
W&B Hyperparameter Sweep Runner for Building2Building

This script orchestrates hyperparameter sweeps for PPO and DQN training algorithms
using W&B sweeps with Hydra configuration management.

Key Features:
- Supports both PPO and DQN algorithms
- Uses W&B Bayesian optimization
- Integrates with Hydra for configuration management
- Proper W&B initialization to avoid double tracking
- Config restructuring for nested parameter access
"""

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
from pathlib import Path
import argparse

# Import environment to register it
import building2building.simulator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run W&B hyperparameter sweeps for Building2Building")
    parser.add_argument(
        "--algorithm", 
        choices=["dqn", "ppo"], 
        required=True,
        help="Algorithm to use for sweep (dqn or ppo)"
    )
    return parser.parse_args()


def main():
    """Main function for running W&B sweep with Hydra configuration."""
    
    args = parse_args()
    algorithm = args.algorithm
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {algorithm.upper()} hyperparameter sweep...")
    
    # Get absolute path to config directory
    current_dir = Path(__file__).parent
    config_dir = current_dir.parent / "configs" / "sweeps"
    
    # Load the appropriate configuration
    if algorithm == "dqn":
        config_name = "dqn_optimization"
    else:  # ppo
        config_name = "ppo_optimization"
    
    logger.info(f"Loading sweep configuration from: {config_dir}/{config_name}.yaml")
    
    # Load configuration using Hydra
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=config_name)
    
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
        "name": config_dict.get("name", f"{algorithm}_bayes_opt"),
        "program": f"scripts/train_{algorithm}.py"
    }
    
    # Remove None values
    sweep_config = {k: v for k, v in sweep_config.items() if v is not None}
    
    logger.info(f"Sweep configuration: {sweep_config}")
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=config_dict.get("project", "building2building"))
    logger.info(f"Sweep initialized with ID: {sweep_id}")
    
    # Start the sweep agent
    count = config_dict.get("count", 50)
    logger.info(f"Starting sweep agent with {count} runs...")
    wandb.agent(sweep_id, function=lambda: train_agent(algorithm), count=count)
    logger.info("Sweep completed!")


def train_agent(algorithm):
    """
    Agent function that will be called by W&B sweep.
    
    This function:
    1. Initializes a W&B run
    2. Converts W&B config to Hydra-compatible format
    3. Calls the appropriate training function
    4. Handles cleanup and error reporting
    """
    # Initialize W&B run first to ensure config is available
    run = wandb.init(settings=wandb.Settings(init_timeout=120))
    config = wandb.config
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training run {run.id} for {algorithm.upper()}")
    
    # Convert config to command line arguments for Hydra
    args = []
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)):
            args.append(f"{key}={value}")
        elif isinstance(value, list):
            # Use proper Hydra list syntax: key=[value1,value2]
            list_str = ','.join(map(str, value))
            args.append(f"{key}=[{list_str}]")
    
    # Set up output directory for this run
    run_name = run.name or f"run_{run.id}"
    sweep_run_dir = f"results/sweep_runs/{algorithm}_{run.id}_{run_name}"
    
    # Add Hydra overrides for proper output organization
    args.extend([
        f"hydra.run.dir={sweep_run_dir}",
        "hydra.mode=RUN",
        "track=false"  # Disable W&B tracking in training script since we already initialized
    ])
    
    logger.info(f"Running {algorithm.upper()} training with args: {' '.join(args)}")
    logger.info(f"Results will be stored in: {sweep_run_dir}")
    
    # Import and run the appropriate training function
    try:
        from building2building.training import get_training_function
        
        # Get the training function
        train_func = get_training_function(algorithm)
        
        # Temporarily modify sys.argv to pass Hydra arguments
        original_argv = sys.argv.copy()
        sys.argv = [f"train_{algorithm}.py"] + args
        
        # Convert wandb.config to a regular dict first, then restructure for nested config
        # wandb.config is not an OmegaConf object, so we convert it to dict first
        config_dict = dict(config)
        
        # Restructure flat keys to nested structure
        # This converts keys like 'ppo.learning_rate' to nested dicts
        restructured_config = {}
        for key, value in config_dict.items():
            if '.' in key:
                # Split key like 'ppo.learning_rate' into nested structure
                parts = key.split('.')
                current = restructured_config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                # Keep top-level keys as is
                restructured_config[key] = value
        
        from omegaconf import DictConfig
        mock_cfg = DictConfig(restructured_config)
        
        # Run the training
        train_func(mock_cfg, results_dir=sweep_run_dir)
        
        logger.info("Training completed successfully")
        run.log({"results_dir": sweep_run_dir})
        logger.info(f"Results saved to: {sweep_run_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        run.finish()
        raise
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    main() 