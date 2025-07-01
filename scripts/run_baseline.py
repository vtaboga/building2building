from building2building.algorithms.online.baselines import run_constant_baseline
import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run constant baseline policy on EnergyPlus environment")
    parser.add_argument('--state', '-s', type=str, required=True, help='State code (e.g., AL)')
    parser.add_argument('--county', '-c', type=str, required=True, help='County name')
    parser.add_argument('--building-id', '-b', type=str, required=True, help='Building ID')
    parser.add_argument('--weather', type=str, required=True, help='Weather file')
    parser.add_argument('--heating-setpoint', type=float, default=21.0, help='Constant heating setpoint (°C)')
    parser.add_argument('--cooling-setpoint', type=float, default=24.0, help='Constant cooling setpoint (°C)')
    parser.add_argument('--reward-type', type=str, default="base", choices=["barrier", "base"], help='Reward type')
    parser.add_argument('--energy-weight', type=float, default=0.0, help='Energy weight')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--results-dir', type=str, default="baseline_results", help='Directory to save results')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Setup paths
    building_path = f"data/processed_buildings/{args.state}/{args.county}/{args.building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{args.state}/{args.county}/{args.building_id}.json"
    weather_path = f"data/weather/{args.weather}"
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    try:
        with open(characteristics_path, 'r') as f:
            building_characteristics = json.load(f)
    except FileNotFoundError:
        print(f"Building characteristics file not found: {characteristics_path}")
        exit(1)
    
    # Run the constant baseline evaluation
    results = run_constant_baseline(
        env_id="EnergyPlus-v0",
        path_to_building=building_path,
        path_to_weather=weather_path,
        building_characteristics=building_characteristics,
        heating_setpoint=args.heating_setpoint,
        cooling_setpoint=args.cooling_setpoint,
        reward_type=args.reward_type,
        energy_weight=args.energy_weight,
        seed=args.seed,
        results_dir=args.results_dir
    )
    
    print("\nBaseline Evaluation Results:")
    print(f"Total reward: {results['total_reward']:.2f}")
    print(f"Mean reward per step: {results['mean_reward']:.2f}")
    print(f"Total timesteps: {results['total_timesteps']}")
    print(f"Results saved to: {Path(args.results_dir) / 'baseline_results.json'}")

