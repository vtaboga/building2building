import hydra
import wandb
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from building2building.generator.search_idf import search_idf
from building2building.generator.processing import add_hvac_meters_to_epjson, add_outdoor_air_meters_to_epjson, modify_timestep, add_setpoint_control_to_epjson
import numpy as np
import json


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for running building simulation with Hydra configuration."""
    
    # Get Hydra's output directory (automatically managed)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    
    # Create subdirectories
    (output_dir / 'data').mkdir(exist_ok=True)
    (output_dir / 'eplus_output').mkdir(exist_ok=True)
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    # Initialize W&B following best practices from https://docs.wandb.ai/guides/integrations/hydra/
    wandb_run = None
    if cfg.get('track', False):
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=cfg.get('name', 'experiment'),
            tags=cfg.wandb.get('tags', [])
        )
        logger.info(f"W&B initialized: {wandb_run.url}")
    
    # Extract building configuration
    building_cfg = cfg.building
    simulation_cfg = cfg.simulation
    
    logger.info("Searching for building files...")
    search_result = search_idf(
        state=building_cfg.state,
        county=building_cfg.county,
        building_type=building_cfg.building_type,
        area=building_cfg.area,
        num_floors=building_cfg.num_floors,
        height=building_cfg.height,
        n_buildings=building_cfg.n_buildings,
        n_weather_files=building_cfg.get('n_weather_files', 2)
    )
    
    if search_result is None:
        logger.error("No buildings found matching the criteria")
        return
    
    buildings, path_to_weather = search_result

    logger.info("Create simulator...")

    # Process first building
    building, characteristics = buildings[0]
    add_hvac_meters_to_epjson(building)
    add_outdoor_air_meters_to_epjson(building)
    modify_timestep(building)
    add_setpoint_control_to_epjson(building)

    # Import gymnasium and create registered environment
    import gymnasium as gym
    env_kwargs = {
        'path_to_building': building,
        'path_to_weather': path_to_weather,
        'building_characteristics': characteristics,
        'eplus_output_dir': str(output_dir / 'eplus_output')
    }
    env = gym.make('EnergyPlus-v0', **env_kwargs)

    logger.info("Starting simulation episodes...")
    
    # Create lists to store episode results
    episode_results = []
    episodes = simulation_cfg.episodes

    # Run episodes
    for episode in range(episodes):
        logger.info(f"Episode {episode + 1}/{episodes}")
        
        episode_data = {
            'steps': [],
            'rewards': [],
            'total_reward': 0,
            'episode_length': 0
        }
        
        # Reset environment
        obs, info = env.reset()
        total_reward = 0.0
        step = 0
        
        # Run episode
        for _ in range(10):
            action = np.array([20.0, 25.0]) 
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            
            # Store step data
            episode_data['steps'].append({
                'step': step,
                'reward': float(reward),
                'observation': obs.tolist() if isinstance(obs, np.ndarray) else obs,
                'info': info
            })
            
            if step % 24 == 0:  # Log every 24 steps
                logger.info(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
                
                # Log metrics to W&B
                if wandb_run:
                    wandb.log({
                        'reward': float(reward),
                        'total_reward': float(total_reward),
                        'episode': episode + 1
                    }, step=step)
                    
            step += 1
        
        # Store episode summary
        episode_data['total_reward'] = float(total_reward)
        episode_data['episode_length'] = step
        episode_results.append(episode_data)
        
        logger.info(f"Episode {episode + 1} finished. Total steps: {step}, Total reward: {total_reward:.2f}")
        
        # Save episode results using Hydra's output directory
        results_path = output_dir / 'data' / f'episode_{episode+1}_results.json'
        with open(results_path, 'w') as f:
            json.dump(episode_data, f, indent=2)

    # Save summary of all episodes
    summary = {
        'total_episodes': episodes,
        'average_reward': np.mean([ep['total_reward'] for ep in episode_results]),
        'average_length': np.mean([ep['episode_length'] for ep in episode_results]),
    }
    
    summary_path = output_dir / 'data' / 'simulation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Simulation complete. Results saved in: {output_dir}")
    env.close()
    
    # Hydra automatically saves config to .hydra/config.yaml
    logger.info(f"Config automatically saved to: {output_dir}/.hydra/config.yaml")


if __name__ == "__main__":
    main()
