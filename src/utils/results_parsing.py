import json
import os
import matplotlib.pyplot as plt

def parse_trajectories(file_path, show_controlled_only=True):
    """
    Parse the trajectories and plot the data.

    Args:
        file_path (str): Path to the JSON file containing trajectories.
        show_controlled_only (bool): If True, only show temperatures for controlled zones.
    """
    # Load the trajectories from the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Group actions and states by trajectory
    actions = [entry['action'] for entry in data]
    states = [entry['state'] for entry in data]
    
    # Extract data for plotting
    heating_setpoints = [action[0] for action in actions]
    cooling_setpoints = [action[0] + action[1] for action in actions]
    
    # Extract indoor temperatures for each zone
    zone_temperatures = {}
    for state in states:
        for key, value in state.items():
            if "Zone Temperature" in key:
                if show_controlled_only and "uncontrolled" in key:
                    continue
                if key not in zone_temperatures:
                    zone_temperatures[key] = []
                zone_temperatures[key].append(value)
    
    energy_consumptions = [
        (state["HVAC Electricity Consumption"] + state["HVAC Natural Gas Consumption"]) / 3600000
        for state in states
    ]  # Convert Joules to kWh
    time_steps = list(range(len(actions)))
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(15, 10))

    # Plot heating and cooling setpoints
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature (°C)', color='tab:blue')
    ax1.plot(time_steps, heating_setpoints, label='Heating Setpoint', color='tab:red')
    ax1.plot(time_steps, cooling_setpoints, label='Cooling Setpoint', color='tab:green')
    
    # Plot each zone temperature
    for zone, temperatures in zone_temperatures.items():
        ax1.plot(time_steps, temperatures, label=zone, linestyle='--')
    
    ax1.set_ylim(15, 35)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # Create a second y-axis for energy consumption
    ax2 = ax1.twinx()
    ax2.set_ylabel('Energy Consumption (kWh)', color='tab:orange')
    ax2.plot(time_steps, energy_consumptions, label='Energy Consumption', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.legend(loc='upper right')

    # Save the plot
    output_dir = os.path.dirname(file_path)
    output_file = os.path.join(output_dir, 'graph.png')
    plt.grid(True)
    plt.title('Trajectory')
    plt.savefig(output_file)
    plt.close()
