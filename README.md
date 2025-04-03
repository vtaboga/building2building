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

## Workflow

### Searching the IDF files database

The `src/generator` directory contains scripts to search through the database for buildings matching a given high-level description.

Example usage:
```bash
python -m scripts.main --state "AL" --county "Pike" --building-type "SmallHotel" --area 1140 --num-floors 6 --n-buildings 4
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

### Creating gym environments

From the processed building files, `src/simulator` contains scripts to wrap the EnergyPlus simulation in Gymnasium.

For this, one needs to specify the actuators and sensors in the building. This is done automatically by parsing and editing each epJSON file. The modified epJSON files are saved in place in `data/processed_idf`.

The action and observation spaces are described in `src/simulator/observations_spaces.py` and `src/simulator/action_spaces.py`. Reward functions to use in the environment are specified in `src/simulator/reward_functions.py`.

In the current state of the code, executing `scripts.main` ends with the n-buildings wrapped in gym wrappers.

**TODO:** There is an issue with the IDF files, only one floor (i.e., one zone) seems to be conditioned per building.

## Singularity container

To run experiments on the cluster, we need a container to execute EnergyPlus.

Upon executing `setup.sh`, a Singularity container is built using `container/container.def`. This container contains the entire repository as well as EnergyPlus. The script `scripts.container_main.sh` executes `scripts.main` through the repository with connections to `logs/` and `data/`.

## Minergym

The gym wrapper of energyplus is based on the minergym repository https://github.com/Terramorpha/minergym

