# Building2Building

Benchmarking transfer learning in Reinforcement Learning on millions of buildings.

## Prerequisites

Before you begin, ensure you have one of the following container engines installed:
- Singularity (version 3.0 or later)
- Apptainer (the open-source continuation of Singularity)

## Installation Steps

1. Clone the repository
   ```bash
   git clone https://github.com/YOUR-USERNAME/Building2Building.git
   cd Building2Building
   ```

2. Verify container engine installation
   
   For Singularity:
   ```bash
   singularity --version
   ```

3. Run the setup script
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

### EnergyPlus path
To use the python EnergyPlus api, python needs to know the path to the EnergyPlus folder. By default, the installation script search EnergyPlus at the default location in Unbutu, Winbdows or Mac. To avoid any problem, you can also manually specify the path when executing the setup script. For instance:

   ```bash
./setup.sh --energyplus_path="/mnt/c/Program Files/EnergyPlus-24-1-0"
   ```
The path will be stored in `configs/energyplus_simulator.json` and loaded whenever needed. 

**Note:** If EnergyPlus is not installed on your machine, you can always use the singularity container. More details are given below.

## Workflow

### Searching the IDF files database

The `src/generator` directory contains scripts to search through the database for buildings matching a given high-level description.

Example usage:
```bash
python -m scripts.fetch_building --state "AL" --county "Pike" --building-type "SmallHotel" --area 1140 --num-floors 6 --n-buildings 4
```

This will search for the 4 buildings best matching the description in the Pike county in Alabama. The operations to get there are:

1. Download the IDF files of the county
2. Download the metadata of the state
3. Process the metadata to add the county of each building given its location (longitude/latitude)
4. Search the metadata for the n buildings matching the description best
5. Process the IDF files found to update the E+ version and convert to epJSON
6. Download a weather file from the state (and city, if specified)

Files are kept in memory in `data/` to only do the download and processing steps once.

**Note:** Step 3 takes a while (~30 to 60 min) because the metadata is organized per state and the data per county. We need to fetch the county for each building of the state. This step is only done the first time you specify a new state and is valid for all counties within the state.

**Note:** After an idf file has been processed and converted to an epJSON, the idf version in `data/idf` is deleted to free some memory. If a later query identify this file as a match in the metadata, it will be used without being re-processed.

### Runing baselines

To run a simulation with a constant policy, execute `scripts/run_baseline.py`. For instance:

```bash
python -m scripts.run_baseline -s AL -c Pike -b 6014003399143 --heating-setpoint 20.0 --cooling-setpoint 23.0
```

The results are stored in `results/constant_basline_{some_unique_name}`.

### Reinforcement Learning 

The implementation of RL algorithms is taken from (Clean RL)[https://github.com/vwxyzjn/cleanrl]. At the moment only PPO with continuous actions is supported. To launch a training, execute `scripts/train_ppo.py`. For instance:

```bash
python -m scripts.train_ppo -s "AL" -c "Pike" -b 6014003413384 --seed 1 --track
```
The ```--track``` arguments triggers the logging with wandb. 
The results are stored in `results/ppo_training_{some_unique_name}`. If wandb is used, the unique run id is added at the end of the results folder name for convenience. 

### Creating gym environments

From the processed building files, `src/simulator` contains scripts to wrap the EnergyPlus simulation in Gymnasium.
While `src/simulator/simulation.py` handle the EnergyPlus simulation itself, `src/simulator/environment.py` create a gym environment to interact with python. The environment is registered with the entry point `src/simulator/create_simulator.py`.
To create a gym environment, one needs to specify the epJSON and weather files. The epJSON file is parsed to fetch the names of the actuators (i.e. temperature setpoints of the controlled zones) and sensors. The action and observation spaces are described in `src/simulator/observations_spaces.py` and `src/simulator/action_spaces.py`. The reward functions to use in the environment are specified in `src/simulator/reward_functions.py`. 

**TODO:** There is an issue with the IDF files, only one floor (i.e., one zone) seems to be conditioned per building. In any case, only the controlled zone is consider to compute the reward.

## Singularity container

To run experiments on the cluster, we need a container to execute EnergyPlus.

Upon executing `setup.sh`, a Singularity container is built using `container/container.def`. This container contains the entire repository as well as EnergyPlus. 
Any python script can be executed in the container using `scripts/container_main.sh`. Simply execute the script and scpecify which python file to use. For intance:

```bash
./scripts/container_main.sh scripts/fetch_buildings.py -s AL -c Pike -b 6014003399143 --heating-setpoint 20.0 --cooling-setpoint 23.0
```

Don't forget to change the .sh permission the first time:

```bash
chmod +x /scripts/container_main.sh
```

## Slurm scripts

Slurm scripts are stored in `jobs/`. When a job is launched, a temporary directory is allocated and relevant data as well as the container are copied to this temporary directory.
The paths to the different folders are written in `.env` when `setup.sh` is executed. These paths are specific to the Mila cluster. 
The scripts execute `container_main.sh` with the specified python script and arguments. If multiple seeds are specified, runs are executed sequentially for each seed. 

An example of command is:

```bash
sbatch jobs/train_dqn.sh -s "AL" -c "Pike" -b 6014003413384 --seed 1,2,3,4,5 --weather "USA_AL_Albertville.Muni.AP.720376_TMYx.2004-2018.epw" --total-timesteps 250000 --eval-frequency 25000 --reward-type "barrier" --energy-weight 1.0 --track
```

## Minergym

The gym wrapper of energyplus is based on the minergym repository https://github.com/Terramorpha/minergym

