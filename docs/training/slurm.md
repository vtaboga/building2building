# SLURM Cluster Execution

## Overview

B2B provides ready-made SLURM submission scripts in `scripts/slurm/` for
running training and evaluation on HPC clusters.

## Available Scripts

| Script | Purpose | Time | CPUs | Memory |
|---|---|---|---|---|
| `run_ppo.sh` | PPO training (single-zone) | 48 h | 8 | 32 GB |
| `run_sac.sh` | SAC training (single-zone) | 12 h | 8 | 32 GB |
| `run_baselines.sh` | Rule-based baseline rollouts | varies | varies | varies |
| `run_baseline_unitary_pi.sh` | PI controller rollout | varies | varies | varies |
| `run_rulebased_adaptive_dynamics.sh` | Rule-based adaptive dynamics benchmark | varies | varies | varies |
| `run_ppo_baseline_adaptive_dynamics.sh` | PPO adaptive dynamics evaluation | varies | varies | varies |
| `run_ppo_per_building_adaptive_dynamics.sh` | Per-building PPO training for adaptive dynamics | varies | varies | varies |
| `run_ppo_parameterized_adaptive_dynamics.sh` | Parameterised PPO adaptive dynamics | varies | varies | varies |

## Example: SAC Training

```bash title="scripts/slurm/run_sac.sh"
#!/bin/bash
#SBATCH --job-name=sac_training
#SBATCH --output=logs/sac_%j.out
#SBATCH --error=logs/sac_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=main

source $(conda info --base)/etc/profile.d/conda.sh
conda activate building2building

mkdir -p logs
cd $SLURM_SUBMIT_DIR
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

python scripts/train_single_zone_houses.py policy=sac
```

### Submitting

```bash
sbatch scripts/slurm/run_sac.sh
sbatch scripts/slurm/run_ppo.sh
```

## Example: PPO Training

```bash title="scripts/slurm/run_ppo.sh"
#!/bin/bash
#SBATCH --job-name=ppo_training
#SBATCH --output=logs/ppo_%j.out
#SBATCH --error=logs/ppo_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=main

source $(conda info --base)/etc/profile.d/conda.sh
conda activate building2building

mkdir -p logs
cd $SLURM_SUBMIT_DIR
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

python scripts/train_single_zone_houses.py
```

## Customising SLURM Parameters

### Resource Allocation

Adjust resources based on your workload:

```bash
#SBATCH --time=24:00:00       # Increase wall time
#SBATCH --cpus-per-task=16    # More CPUs for parallel envs
#SBATCH --mem=64G             # More memory for large replay buffers
#SBATCH --gres=gpu:1          # Request a GPU (for policy.device=cuda)
```

### Hydra Overrides

Pass Hydra overrides directly in the Python command:

```bash
python scripts/train_single_zone_houses.py \
    policy=sac \
    training.total_timesteps=5000000 \
    policy.buffer_size=500000
```

### Array Jobs

To train across multiple building instances in parallel:

```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --job-name=multizones_%a

python scripts/train_multizones.py \
    multizones.building_type=OfficeSmall \
    multizones.index=$SLURM_ARRAY_TASK_ID
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/sac_<job_id>.out

# Cancel a job
scancel <job_id>

# View job details after completion
sacct -j <job_id> --format=JobID,Elapsed,MaxRSS,State
```

## Directory Setup

SLURM scripts expect a `logs/` directory (created automatically) and assume the
conda environment `building2building` exists.  The project root is added to
`PYTHONPATH` so all modules are importable.
