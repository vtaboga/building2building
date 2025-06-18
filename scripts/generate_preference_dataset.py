import argparse
import json
import os
import numpy as np
import gymnasium as gym
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Import required modules
from src.core.run_manager import RunManager
from src.algorithms.dqn import QNetwork, make_env, dqn_evaluate
from src.algorithms.baselines import constant_policy
from src.simulator.utils import TrajectoryLogger
from src.simulator.wrappers import CustomRescaleAction, NormalizeObservation
import src.simulator


class RLAIFTrajectoryLogger(TrajectoryLogger):
    """Enhanced trajectory logger for RLAIF pipeline that ensures required data is captured."""
    
    def __init__(self, save_dir, observation_names, logger=None):
        super().__init__(save_dir, observation_names, logger)
        
    def log(self, state, action, reward, controlled_zones, uncontrolled_zones):
        """Log a single step with verification of required RLAIF data."""
        
        names = [
            f"{name} (uncontrolled)" if self._is_uncontrolled_zone(name, uncontrolled_zones) else name
            for name in self.observation_names
        ]

        # Convert NumPy types to native Python types for JSON serialization
        state_dict = {name: float(value) if hasattr(value, 'item') else value 
                     for name, value in zip(names, state)}

        # Verify that we have the required RLAIF data
        required_data = self._verify_rlaif_data(state_dict)
        
        self.total_reward += reward

        # Enhanced trajectory entry with explicit RLAIF fields
        trajectory_entry = {
            "state": state_dict,
            "action": action.tolist() if hasattr(action, 'tolist') else action,
            "reward": float(reward) if hasattr(reward, 'item') else reward,
            "rlaif_data": required_data  # Explicitly extract RLAIF required data
        }
        
        self.trajectories.append(trajectory_entry)
        
    def _verify_rlaif_data(self, state_dict: Dict[str, float]) -> Dict[str, Any]:
        """Extract and verify RLAIF required data from state."""
        rlaif_data = {}
        
        # Extract outdoor temperature
        outdoor_temp_key = None
        for key in state_dict.keys():
            if "Outdoor Air Temperature" in key:
                outdoor_temp_key = key
                break
        
        if outdoor_temp_key is None:
            self.logger.warning("Outdoor temperature not found in observations!")
            rlaif_data["outdoor_temperature"] = None
        else:
            rlaif_data["outdoor_temperature"] = state_dict[outdoor_temp_key]
        
        # Extract energy consumption (sum of electricity and gas)
        hvac_electricity = None
        hvac_gas = None
        
        for key, value in state_dict.items():
            if "HVAC Electricity Consumption" in key:
                hvac_electricity = value
            elif "HVAC Natural Gas Consumption" in key:
                hvac_gas = value
                
        if hvac_electricity is None or hvac_gas is None:
            self.logger.warning("HVAC energy consumption data incomplete!")
            rlaif_data["energy_consumption"] = None
        else:
            # Convert to kWh for better interpretability
            rlaif_data["energy_consumption"] = (hvac_electricity + hvac_gas) / 3600000.0
        
        # Extract indoor temperatures (all zones)
        indoor_temps = {}
        for key, value in state_dict.items():
            if "Zone Temperature" in key:
                zone_name = key.replace("Zone Temperature ", "").replace(" (uncontrolled)", "")
                indoor_temps[zone_name] = value
        
        if not indoor_temps:
            self.logger.warning("No indoor temperatures found in observations!")
            rlaif_data["indoor_temperatures"] = None
        else:
            rlaif_data["indoor_temperatures"] = indoor_temps
            # Also compute average indoor temperature
            rlaif_data["avg_indoor_temperature"] = sum(indoor_temps.values()) / len(indoor_temps)
        
        # Extract hour of day
        hour_key = None
        for key in state_dict.keys():
            if "Current Time of Day" in key:
                hour_key = key
                break
                
        if hour_key is None:
            self.logger.warning("Hour of day not found in observations!")
            rlaif_data["hour_of_day"] = None
        else:
            rlaif_data["hour_of_day"] = state_dict[hour_key]
            
        return rlaif_data

    def save_for_rlaif(self, filename="trajectories_rlaif.json"):
        """Save trajectories specifically formatted for RLAIF pipeline."""
        file_path = os.path.join(self.save_dir, filename)
        
        # Extract just the essential RLAIF data
        rlaif_trajectories = []
        for entry in self.trajectories:
            rlaif_entry = {
                "timestep": len(rlaif_trajectories),
                "action": entry["action"],
                "reward": entry["reward"],
                "outdoor_temperature": entry["rlaif_data"]["outdoor_temperature"],
                "energy_consumption": entry["rlaif_data"]["energy_consumption"],
                "indoor_temperatures": entry["rlaif_data"]["indoor_temperatures"],
                "avg_indoor_temperature": entry["rlaif_data"]["avg_indoor_temperature"],
                "hour_of_day": entry["rlaif_data"]["hour_of_day"]
            }
            rlaif_trajectories.append(rlaif_entry)
        
        with open(file_path, 'w') as f:
            json.dump(rlaif_trajectories, f, indent=4)
        
        self.logger.info(f"RLAIF trajectories saved to {file_path}")
        return file_path


def split_trajectory_into_chunks(trajectory: List[Dict[str, Any]], chunk_size: int = 24) -> List[List[Dict[str, Any]]]:
    """Split a trajectory into chunks of specified size (default: 24 hours)."""
    chunks = []
    for i in range(0, len(trajectory), chunk_size):
        chunk = trajectory[i:i + chunk_size]
        if len(chunk) == chunk_size:  # Only keep complete chunks
            chunks.append(chunk)
    return chunks


def create_preference_pairs(
    rule_based_chunks: List[List[Dict[str, Any]]], 
    dqn_chunks: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Create preference pairs from trajectory chunks for LLM evaluation."""
    
    # Ensure we have the same number of chunks
    min_chunks = min(len(rule_based_chunks), len(dqn_chunks))
    preference_pairs = []
    
    for i in range(min_chunks):
        rule_chunk = rule_based_chunks[i]
        dqn_chunk = dqn_chunks[i]
        
        # Calculate summary statistics for each chunk
        rule_summary = _calculate_chunk_summary(rule_chunk, "rule_based")
        dqn_summary = _calculate_chunk_summary(dqn_chunk, "dqn")
        
        preference_pair = {
            "pair_id": i,
            "chunk_size": len(rule_chunk),
            "trajectory_a": {
                "controller_type": "rule_based",
                "data": rule_chunk,
                "summary": rule_summary
            },
            "trajectory_b": {
                "controller_type": "dqn", 
                "data": dqn_chunk,
                "summary": dqn_summary
            },
            "comparison_metrics": {
                "energy_difference": dqn_summary["total_energy"] - rule_summary["total_energy"],
                "comfort_difference": dqn_summary["avg_comfort_deviation"] - rule_summary["avg_comfort_deviation"],
                "reward_difference": dqn_summary["total_reward"] - rule_summary["total_reward"]
            }
        }
        
        preference_pairs.append(preference_pair)
    
    return preference_pairs


def _calculate_chunk_summary(chunk: List[Dict[str, Any]], controller_type: str) -> Dict[str, Any]:
    """Calculate summary statistics for a trajectory chunk."""
    
    total_energy = sum(step["energy_consumption"] or 0 for step in chunk)
    total_reward = sum(step["reward"] for step in chunk)
    
    # Calculate comfort as deviation from target temperature (21°C)
    target_temp = 21.0
    comfort_deviations = []
    
    for step in chunk:
        if step["avg_indoor_temperature"] is not None:
            deviation = abs(step["avg_indoor_temperature"] - target_temp)
            comfort_deviations.append(deviation)
    
    avg_comfort_deviation = sum(comfort_deviations) / len(comfort_deviations) if comfort_deviations else 0
    
    # Time period info
    start_hour = chunk[0]["hour_of_day"] if chunk[0]["hour_of_day"] is not None else 0
    end_hour = chunk[-1]["hour_of_day"] if chunk[-1]["hour_of_day"] is not None else 0
    
    summary = {
        "controller_type": controller_type,
        "total_energy": total_energy,
        "total_reward": total_reward, 
        "avg_comfort_deviation": avg_comfort_deviation,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "timesteps": len(chunk),
        "avg_outdoor_temp": sum(step["outdoor_temperature"] or 0 for step in chunk) / len(chunk),
        "avg_indoor_temp": sum(step["avg_indoor_temperature"] or 0 for step in chunk) / len(chunk)
    }
    
    return summary


def parse_arguments():
    """Parse command line arguments for preference dataset generation."""
    parser = argparse.ArgumentParser(description='Generate preference dataset by comparing rule-based and DQN controllers')
    
    # Required arguments
    parser.add_argument('--state', '-s', type=str, required=True,
                       help='State code (e.g., AL)')
    parser.add_argument('--county', '-c', type=str, required=True,
                       help='County name')
    parser.add_argument('--building-id', '-b', type=str, required=True,
                       help='Building ID (epJSON file name without extension)')
    parser.add_argument('--weather', '-w', type=str, required=True,
                       help='Weather file name in data/weather/')
    parser.add_argument('--dqn-weights', type=str, required=True,
                       help='Path to DQN model weights file')
    
    # Optional arguments
    parser.add_argument('--heating-setpoint', type=float, default=21.0,
                       help='Heating setpoint for rule-based controller (°C)')
    parser.add_argument('--cooling-setpoint', type=float, default=24.0,
                       help='Cooling setpoint for rule-based controller (°C)')
    parser.add_argument('--reward-type', type=str, default="base", choices=["barrier", "base"],
                       help='Reward type for environment')
    parser.add_argument('--energy-weight', type=float, default=1.0,
                       help='Energy weight for reward function')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--track', action='store_true',
                       help='Track with wandb')
    parser.add_argument('--wandb-project', type=str, default="building2building",
                       help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='W&B entity')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--bins-per-dimension', type=int, default=20,
                       help='Number of discrete bins per action dimension for DQN')
    parser.add_argument('--chunk-size', type=int, default=24,
                       help='Size of trajectory chunks for comparison (default: 24 hours)')
    
    return parser.parse_args()


def load_building_data(state: str, county: str, building_id: str):
    """Load building epJSON file and characteristics."""
    building_path = f"data/processed_buildings/{state}/{county}/{building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{state}/{county}/{building_id}.json"
    
    # Check if files exist
    if not os.path.exists(building_path):
        raise FileNotFoundError(f"Building file not found: {building_path}")
    if not os.path.exists(characteristics_path):
        raise FileNotFoundError(f"Building characteristics file not found: {characteristics_path}")
    
    # Load building characteristics
    with open(characteristics_path, 'r') as f:
        building_characteristics = json.load(f)
    
    return building_path, building_characteristics


def run_rule_based_simulation(
    building_path: str,
    weather_path: str,
    building_characteristics: Dict[str, Any],
    heating_setpoint: float,
    cooling_setpoint: float,
    reward_type: str,
    energy_weight: float,
    gamma: float,
    seed: int,
    run_manager: RunManager
) -> Dict[str, Any]:
    """Run a simulation using rule-based constant setpoint controller."""
    
    logger = run_manager.logger
    logger.info(f"Starting rule-based simulation with heating={heating_setpoint}°C, cooling={cooling_setpoint}°C")
    
    # Create environment
    env = gym.make(
        "EnergyPlus-v0",
        path_to_building=building_path,
        path_to_weather=weather_path,
        building_characteristics=building_characteristics,
        reward_type=reward_type,
        energy_weight=energy_weight,
        run_manager=run_manager
    )
    
    # Store original action space before wrapping
    action_space = env.action_space
    
    # Apply wrappers
    env = CustomRescaleAction(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env)
    
    # Get environment properties
    uncontrolled_zones = env.unwrapped.uncontrolled_zones
    controlled_zones = env.unwrapped.controlled_zones
    observation_names = env.unwrapped.observation_names
    
    logger.info(f"Controlled zones: {controlled_zones}")
    logger.info(f"Uncontrolled zones: {uncontrolled_zones}")
    logger.info(f"Observation names: {observation_names}")
    
    # Initialize RLAIF trajectory logger
    trajectory_dir = os.path.join(run_manager.data_dir, "rule_based_trajectory")
    trajectory_logger = RLAIFTrajectoryLogger(
        trajectory_dir,
        observation_names,
        logger=logger
    )
    
    # Reset environment
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    timesteps = 0
    total_reward = 0
    
    logger.info("Running rule-based simulation...")
    
    # Run simulation
    while not (done or truncated):
        # Get action from constant policy
        action = constant_policy(
            obs, 
            heating_setpoint, 
            cooling_setpoint, 
            normalize=True, 
            action_space=action_space
        )
        
        # Take step in environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Get denormalized observation and scaled action for logging
        denorm_obs = env.denormalize(obs)
        scaled_action = env.env.env.scale_action(action)
        
        # Log the trajectory
        trajectory_logger.log(denorm_obs, scaled_action, reward, controlled_zones, uncontrolled_zones)
        
        # Update metrics
        total_reward += reward
        timesteps += 1
        obs = next_obs
        
        if timesteps % 24 == 0:  # Log every 24 timesteps (daily)
            logger.debug(f"Rule-based Day {timesteps//24}: Total reward = {total_reward:.2f}")
    
    # Save trajectory
    trajectory_logger.save()
    rlaif_path = trajectory_logger.save_for_rlaif()
    
    # Prepare results
    results = {
        "controller_type": "rule_based",
        "total_reward": float(total_reward),
        "mean_reward": float(total_reward / timesteps) if timesteps > 0 else 0.0,
        "total_timesteps": timesteps,
        "heating_setpoint": heating_setpoint,
        "cooling_setpoint": cooling_setpoint,
        "trajectory_path": trajectory_dir,
        "rlaif_trajectory_path": rlaif_path
    }
    
    logger.info(f"Rule-based simulation completed. Total reward: {total_reward:.2f}, Timesteps: {timesteps}")
    
    env.close()
    return results


def run_dqn_simulation(
    building_path: str,
    weather_path: str,
    building_characteristics: Dict[str, Any],
    dqn_weights_path: str,
    reward_type: str,
    energy_weight: float,
    gamma: float,
    bins_per_dimension: int,
    seed: int,
    run_manager: RunManager
) -> Dict[str, Any]:
    """Run a simulation using a trained DQN agent."""
    
    logger = run_manager.logger
    logger.info(f"Starting DQN simulation with weights from: {dqn_weights_path}")
    
    # Check if DQN weights file exists
    if not os.path.exists(dqn_weights_path):
        raise FileNotFoundError(f"DQN weights file not found: {dqn_weights_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create trajectory directory
    trajectory_dir = os.path.join(run_manager.data_dir, "dqn_trajectory")
    os.makedirs(trajectory_dir, exist_ok=True)
    
    # Run DQN evaluation using the existing function
    try:
        episodic_returns = dqn_evaluate(
            model_path=dqn_weights_path,
            make_env=make_env,
            reward_type=reward_type,
            energy_weight=energy_weight,
            env_id="EnergyPlus-v0",
            path_to_building=building_path,
            path_to_weather=weather_path,
            building_characteristics=building_characteristics,
            eval_episodes=1,  # Single episode
            run_name="dqn_preference_dataset",
            Model=QNetwork,
            device=device,
            run_manager=run_manager,
            gamma=gamma,
            save_trajectories=True,
            bins_per_dimension=bins_per_dimension
        )
        
        total_reward = episodic_returns[0] if episodic_returns else 0.0
        
        # Find the DQN trajectory file and convert it to RLAIF format
        dqn_trajectory_path = os.path.join(run_manager.data_dir, "episode_0", "trajectories.json")
        rlaif_path = None
        
        if os.path.exists(dqn_trajectory_path):
            # Convert DQN trajectory to RLAIF format
            rlaif_path = _convert_dqn_trajectory_to_rlaif(dqn_trajectory_path, trajectory_dir, logger)
        else:
            logger.warning(f"DQN trajectory file not found at {dqn_trajectory_path}")
        
        # Prepare results
        results = {
            "controller_type": "dqn",
            "total_reward": float(total_reward),
            "model_path": dqn_weights_path,
            "trajectory_path": os.path.join(run_manager.data_dir, "episode_0"),
            "rlaif_trajectory_path": rlaif_path
        }
        
        logger.info(f"DQN simulation completed. Total reward: {total_reward:.2f}")
        
    except Exception as e:
        logger.error(f"Error running DQN simulation: {e}")
        raise
    
    return results


def _convert_dqn_trajectory_to_rlaif(source_path: str, target_dir: str, logger) -> str:
    """Convert DQN trajectory format to RLAIF format."""
    
    with open(source_path, 'r') as f:
        dqn_trajectory = json.load(f)
    
    rlaif_trajectory = []
    
    for i, entry in enumerate(dqn_trajectory):
        state = entry["state"]
        
        # Extract RLAIF required data
        outdoor_temp = None
        hvac_electricity = None
        hvac_gas = None
        hour_of_day = None
        indoor_temps = {}
        
        for key, value in state.items():
            if "Outdoor Air Temperature" in key:
                outdoor_temp = value
            elif "HVAC Electricity Consumption" in key:
                hvac_electricity = value
            elif "HVAC Natural Gas Consumption" in key:
                hvac_gas = value
            elif "Current Time of Day" in key:
                hour_of_day = value
            elif "Zone Temperature" in key:
                zone_name = key.replace("Zone Temperature ", "").replace(" (uncontrolled)", "")
                indoor_temps[zone_name] = value
        
        # Calculate energy consumption in kWh
        energy_consumption = None
        if hvac_electricity is not None and hvac_gas is not None:
            energy_consumption = (hvac_electricity + hvac_gas) / 3600000.0
        
        # Calculate average indoor temperature
        avg_indoor_temp = None
        if indoor_temps:
            avg_indoor_temp = sum(indoor_temps.values()) / len(indoor_temps)
        
        rlaif_entry = {
            "timestep": i,
            "action": entry["action"],
            "reward": entry["reward"],
            "outdoor_temperature": outdoor_temp,
            "energy_consumption": energy_consumption,
            "indoor_temperatures": indoor_temps,
            "avg_indoor_temperature": avg_indoor_temp,
            "hour_of_day": hour_of_day
        }
        
        rlaif_trajectory.append(rlaif_entry)
    
    # Save RLAIF format trajectory
    rlaif_path = os.path.join(target_dir, "trajectories_rlaif.json")
    with open(rlaif_path, 'w') as f:
        json.dump(rlaif_trajectory, f, indent=4)
    
    logger.info(f"Converted DQN trajectory to RLAIF format: {rlaif_path}")
    return rlaif_path


def save_preference_dataset(
    rule_based_results: Dict[str, Any],
    dqn_results: Dict[str, Any],
    chunk_size: int,
    run_manager: RunManager
):
    """Save the preference dataset with trajectory chunks for RLAIF pipeline."""
    
    logger = run_manager.logger
    
    # Load RLAIF trajectories
    rule_based_rlaif_path = rule_based_results["rlaif_trajectory_path"]
    dqn_rlaif_path = dqn_results["rlaif_trajectory_path"]
    
    if not rule_based_rlaif_path or not os.path.exists(rule_based_rlaif_path):
        raise FileNotFoundError(f"Rule-based RLAIF trajectory not found: {rule_based_rlaif_path}")
    if not dqn_rlaif_path or not os.path.exists(dqn_rlaif_path):
        raise FileNotFoundError(f"DQN RLAIF trajectory not found: {dqn_rlaif_path}")
    
    with open(rule_based_rlaif_path, 'r') as f:
        rule_based_trajectory = json.load(f)
    
    with open(dqn_rlaif_path, 'r') as f:
        dqn_trajectory = json.load(f)
    
    logger.info(f"Rule-based trajectory length: {len(rule_based_trajectory)}")
    logger.info(f"DQN trajectory length: {len(dqn_trajectory)}")
    
    # Split trajectories into chunks
    rule_based_chunks = split_trajectory_into_chunks(rule_based_trajectory, chunk_size)
    dqn_chunks = split_trajectory_into_chunks(dqn_trajectory, chunk_size)
    
    logger.info(f"Rule-based chunks: {len(rule_based_chunks)}")
    logger.info(f"DQN chunks: {len(dqn_chunks)}")
    
    # Create preference pairs
    preference_pairs = create_preference_pairs(rule_based_chunks, dqn_chunks)
    
    # Create preference dataset summary
    preference_data = {
        "metadata": {
            "generation_timestamp": run_manager.timestamp,
            "run_id": run_manager.run_id,
            "seed": run_manager.seed,
            "chunk_size": chunk_size,
            "total_preference_pairs": len(preference_pairs),
            "rule_based_total_reward": rule_based_results["total_reward"],
            "dqn_total_reward": dqn_results["total_reward"]
        },
        "rule_based_controller": rule_based_results,
        "dqn_controller": dqn_results,
        "comparison": {
            "reward_difference": dqn_results["total_reward"] - rule_based_results["total_reward"],
            "better_controller": "dqn" if dqn_results["total_reward"] > rule_based_results["total_reward"] else "rule_based"
        },
        "preference_pairs": preference_pairs
    }
    
    # Save preference dataset
    preference_file = os.path.join(run_manager.run_dir, "preference_dataset_rlaif.json")
    with open(preference_file, 'w') as f:
        json.dump(preference_data, f, indent=4)
    
    # Also save just the preference pairs for easier LLM processing
    pairs_file = os.path.join(run_manager.run_dir, "preference_pairs_for_llm.json")
    with open(pairs_file, 'w') as f:
        json.dump(preference_pairs, f, indent=4)
    
    logger.info(f"RLAIF preference dataset saved to: {preference_file}")
    logger.info(f"Preference pairs for LLM saved to: {pairs_file}")
    logger.info(f"Total preference pairs generated: {len(preference_pairs)}")
    logger.info(f"Rule-based reward: {rule_based_results['total_reward']:.2f}")
    logger.info(f"DQN reward: {dqn_results['total_reward']:.2f}")
    logger.info(f"Reward difference (DQN - Rule-based): {preference_data['comparison']['reward_difference']:.2f}")
    logger.info(f"Better controller: {preference_data['comparison']['better_controller']}")
    
    return preference_data


def main():
    """Main function to generate preference dataset."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Create run manager
    run_manager = RunManager(
        experiment_name="rlaif_preference_dataset_generation",
        track_wandb=args.track,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
        tags={
            "state": args.state,
            "county": args.county,
            "building_id": args.building_id,
            "weather": args.weather,
            "task": "rlaif_preference_dataset",
            "chunk_size": args.chunk_size
        }
    )
    
    logger = run_manager.logger
    logger.info("Starting RLAIF preference dataset generation...")
    
    try:
        # Load building data
        logger.info("Loading building data...")
        building_path, building_characteristics = load_building_data(
            args.state, args.county, args.building_id
        )
        
        # Construct weather path
        weather_path = f"data/weather/{args.weather}"
        if not os.path.exists(weather_path):
            raise FileNotFoundError(f"Weather file not found: {weather_path}")
        
        logger.info(f"Building: {building_path}")
        logger.info(f"Weather: {weather_path}")
        logger.info(f"Building characteristics: {building_characteristics}")
        
        # Save configuration
        config = {
            "building_path": building_path,
            "weather_path": weather_path,
            "building_characteristics": building_characteristics,
            "dqn_weights_path": args.dqn_weights,
            "heating_setpoint": args.heating_setpoint,
            "cooling_setpoint": args.cooling_setpoint,
            "reward_type": args.reward_type,
            "energy_weight": args.energy_weight,
            "gamma": args.gamma,
            "bins_per_dimension": args.bins_per_dimension,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
            "required_rlaif_data": [
                "outdoor_temperature",
                "energy_consumption", 
                "indoor_temperatures",
                "hour_of_day"
            ]
        }
        run_manager.save_config(config)
        
        # Run rule-based simulation
        logger.info("=" * 60)
        logger.info("RUNNING RULE-BASED CONTROLLER SIMULATION")
        logger.info("=" * 60)
        
        rule_based_results = run_rule_based_simulation(
            building_path=building_path,
            weather_path=weather_path,
            building_characteristics=building_characteristics,
            heating_setpoint=args.heating_setpoint,
            cooling_setpoint=args.cooling_setpoint,
            reward_type=args.reward_type,
            energy_weight=args.energy_weight,
            gamma=args.gamma,
            seed=args.seed,
            run_manager=run_manager
        )
        
        # Run DQN simulation
        logger.info("=" * 60)
        logger.info("RUNNING DQN CONTROLLER SIMULATION")
        logger.info("=" * 60)
        
        dqn_results = run_dqn_simulation(
            building_path=building_path,
            weather_path=weather_path,
            building_characteristics=building_characteristics,
            dqn_weights_path=args.dqn_weights,
            reward_type=args.reward_type,
            energy_weight=args.energy_weight,
            gamma=args.gamma,
            bins_per_dimension=args.bins_per_dimension,
            seed=args.seed,
            run_manager=run_manager
        )
        
        # Save RLAIF preference dataset
        logger.info("=" * 60)
        logger.info("CREATING RLAIF PREFERENCE DATASET")
        logger.info("=" * 60)
        
        preference_data = save_preference_dataset(
            rule_based_results=rule_based_results,
            dqn_results=dqn_results,
            chunk_size=args.chunk_size,
            run_manager=run_manager
        )
        
        logger.info("RLAIF preference dataset generation completed successfully!")
        logger.info(f"Results saved in: {run_manager.run_dir}")
        logger.info("Ready for LLM labeling pipeline!")
        
    except Exception as e:
        logger.error(f"Error during RLAIF preference dataset generation: {e}")
        raise
    finally:
        # Finalize the run
        run_manager.finish()


if __name__ == "__main__":
    main()
