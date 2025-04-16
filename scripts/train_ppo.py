import os
import gymnasium as gym
import torch
import torch.nn as nn
import json
from src.algorithms.ppo import main, Args

# Make sure to import your environment to register it
import src.simulator



if __name__ == "__main__":
    # Define the base directory for results
    base_dir = "results"
    os.makedirs(base_dir, exist_ok=True)

    building_id = "6014003401346"

    # Get building characteristics
    building_characteristics = json.load(open(f"data/processed_buildings/AL/Pike/{building_id}.json"))

    # Create an instance of Args with the desired environment ID and file paths
    args = Args(
        env_id="EnergyPlus-v0",
        path_to_building=f"data/processed_buildings/AL/Pike/{building_id}.epJSON",
        path_to_weather="data/weather/USA_AL_Albertville.Muni.AP.720376_TMYx.2004-2018.epw",
        building_characteristics=building_characteristics,
        track=False,
        wandb_project_name="building2building",
        wandb_entity="pierre-luc-bacon-mila-org",
        results_dir=base_dir  # Pass the base directory to Args
    )
    main(args) 