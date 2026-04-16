# Controller Tuning

Tune reactive controller parameters using Optuna.

## Overview

`tune_controller.py` uses Optuna's TPE sampler to optimize the hyperparameters
of the reactive controllers (`UnitaryHvacConfig` or `AirLoopConfig`). The
objective is the episode return on a single building.

## Usage

```bash
python -m baselines.tune_controller experiment=tune_controller \
    building_type=OfficeSmall climate_zone=1

# More trials
python -m baselines.tune_controller experiment=tune_controller \
    building_type=OfficeSmall climate_zone=1 n_trials=100

# Tune air-loop controller for OfficeMedium
python -m baselines.tune_controller experiment=tune_controller \
    building_type=OfficeMedium climate_zone=4
```

## How It Works

1. Optuna samples controller parameters within predefined ranges
2. A building environment is created for the specified building type and
   climate zone
3. The controller runs for one episode
4. The episode return is reported as the objective
5. After all trials, the best parameters are saved as a YAML config

## Output

The best config is saved to `baselines/configs/tuned_controllers/`:

```
baselines/configs/tuned_controllers/
├── unitary_hvac_officesmall_cz1.yaml
├── unitary_hvac_officesmall_cz2.yaml
├── air_loop_officemedium_cz1.yaml
└── ...
```

These tuned configs are automatically loaded by `run_reactive_control.py` when
evaluating on the matching building type and climate zone.

## Tuned Parameters

Currently 45 tuned controller configs are provided, covering all building types
and climate zones used in the paper.

## Configuration

The experiment config is at `baselines/configs/experiment/tune_controller.yaml`.

Key parameters:

| Parameter | Description |
|---|---|
| `building_type` | Building type to tune for |
| `climate_zone` | ASHRAE climate zone |
| `n_trials` | Number of Optuna trials |
| `task` | Task preset for evaluation |
