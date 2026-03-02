# SLURM Scripts for Building2Building

This directory contains SLURM batch scripts for running experiments on the cluster.

## Available Scripts

### 1. PPO Training
**Script:** `run_ppo.sh`

Trains a Proximal Policy Optimization (PPO) agent using the online trainer.

**Usage:**
```bash
sbatch scripts/slurm/run_ppo.sh
```

**What it does:**
- Uses `scripts/train_single_zone_houses.py` with `configs/base.yaml`
- Trains a PPO agent with default hyperparameters
- Runs for 1M timesteps by default
- Logs to WandB and TensorBoard
- Saves best model based on evaluation performance

**Configuration:** `configs/policy/ppo.yaml`

---

### 2. SAC Training
**Script:** `run_sac.sh`

Trains a Soft Actor-Critic (SAC) agent using the online trainer.

**Usage:**
```bash
sbatch scripts/slurm/run_sac.sh
```

**What it does:**
- Uses `scripts/train_single_zone_houses.py` with `configs/base.yaml` + SAC policy override
- Trains a SAC agent with default hyperparameters
- Runs for 1M timesteps by default
- Logs to WandB and TensorBoard
- Saves best model based on evaluation performance

**Configuration:** `configs/policy/sac.yaml`

---

## Customizing Runs

### Override Configuration Parameters

You can override any configuration parameter using Hydra's command-line syntax:

```bash
# Change training timesteps
sbatch --wrap="python scripts/train_single_zone_houses.py policy=ppo training.total_timesteps=2000000"

# Change number of parallel environments
sbatch --wrap="python scripts/train_single_zone_houses.py policy=sac training.num_train_envs=8"

# Change learning rate
sbatch --wrap="python scripts/train_single_zone_houses.py policy=ppo policy.learning_rate=0.0001"

# Run baseline with more episodes
sbatch --wrap="python scripts/baselines.py n_episodes=5"
```

### Modify SLURM Resources

Edit the `#SBATCH` directives at the top of each script:

- `--time`: Maximum runtime (format: HH:MM:SS or D-HH:MM:SS)
- `--cpus-per-task`: Number of CPU cores
- `--mem`: Memory allocation (e.g., 16G, 32G)
- `--partition`: Cluster partition to use

---

## Output Locations

### Baselines
Outputs are saved to: `outputs/<policy_type>/<date>/<time>/`

Example: `outputs/unitary_g36/02-02-2026/14-30-00/`

Contains:
- `rollout.csv`: Full rollout data
- `rollout.npz`: Numpy archive of rollout
- `config.json`: Configuration used

### RL Training (PPO/SAC)
Outputs are saved to Hydra's default output directory structure.

Contains:
- `models/`: Saved model checkpoints
- `logs/`: Evaluation logs
- `tb/`: TensorBoard logs
- `train_eplus_outputs/`: EnergyPlus simulation outputs (training)
- `eval_eplus_outputs/`: EnergyPlus simulation outputs (evaluation)
- `test/`: Test rollout results

---

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View job output in real-time
tail -f logs/<job_name>_<job_id>.out

# Cancel a job
scancel <job_id>
```

---

## Troubleshooting

### "algorithms module not found"
The scripts set `PYTHONPATH` to include the project root. If you still get this error:
1. Make sure you're submitting from the project root directory
2. Reinstall the package: `pip install -e .`

### Disk quota exceeded
Check your disk usage: `df -h ~`
Clean up old outputs or request more storage.

### Out of memory
Reduce `training.num_train_envs` or increase `--mem` in the SLURM script.

