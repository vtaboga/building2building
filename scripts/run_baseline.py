from src.algorithms.baselines import run_constant_baseline
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Run constant baseline policy on EnergyPlus environment")
    parser.add_argument('--state', '-s', type=str, help='State code (e.g., AL)')
    parser.add_argument('--county', '-c', type=str, help='County name')
    parser.add_argument('--building-id', '-b', type=str, help='Building ID')
    parser.add_argument('--heating-setpoint', type=float, default=21.0, help='Constant heating setpoint (°C)')
    parser.add_argument('--cooling-setpoint', type=float, default=24.0, help='Constant cooling setpoint (°C)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load building characteristics
    building_path = f"data/processed_buildings/{args.state}/{args.county}/{args.building_id}.epJSON"
    characteristics_path = f"data/processed_buildings/{args.state}/{args.county}/{args.building_id}.json"
    weather_path = f"data/weather/USA_{args.state}_Albertville.Muni.AP.720376_TMYx.2004-2018.epw"
    
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
        seed=args.seed
    )
    
    print("\nBaseline Evaluation Results:")
    print(f"Total reward: {results['total_reward']:.2f}")
    print(f"Mean reward per step: {results['mean_reward']:.2f}")
    print(f"Total timesteps: {results['total_timesteps']}")

