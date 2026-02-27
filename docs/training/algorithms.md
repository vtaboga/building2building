# RL Algorithms

B2B integrates with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
(SB3) for RL training.  Algorithm selection is handled via Hydra policy config
groups under `configs/policy/`.

## PPO

Proximal Policy Optimisation — the default on-policy algorithm.

```yaml title="configs/policy/ppo.yaml"
algorithm: "ppo"
policy_type: "MlpPolicy"
device: "cpu"
n_steps: 2048
batch_size: 64
gamma: 0.99
learning_rate: 0.0003
clip_range: 0.2
ent_coef: 0.0
vf_coef: 0.5
max_grad_norm: 0.5
gae_lambda: 0.95
verbose: 1
```

| Parameter | Value | Description |
|---|---|---|
| `n_steps` | 2048 | Rollout buffer length per environment |
| `batch_size` | 64 | Minibatch size for gradient updates |
| `learning_rate` | 3e-4 | Adam learning rate |
| `clip_range` | 0.2 | PPO clipping parameter |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `ent_coef` | 0.0 | Entropy bonus coefficient |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |

### Running PPO

```bash
python scripts/train_single_zone_houses.py policy=ppo
python scripts/train_multizones.py policy=ppo
```

## SAC

Soft Actor-Critic — an off-policy algorithm with automatic entropy tuning.

```yaml title="configs/policy/sac.yaml"
algorithm: "sac"
buffer_size: 100000
batch_size: 128
learning_starts: 1000
tau: 0.005
gamma: 0.99
learning_rate: 0.0003
policy_type: "MlpPolicy"
```

| Parameter | Value | Description |
|---|---|---|
| `buffer_size` | 100,000 | Replay buffer capacity |
| `batch_size` | 128 | Minibatch size |
| `learning_rate` | 3e-4 | Adam learning rate |
| `learning_starts` | 1000 | Random exploration steps before training |
| `tau` | 0.005 | Soft target update coefficient |
| `gamma` | 0.99 | Discount factor |

### Running SAC

```bash
python scripts/train_single_zone_houses.py policy=sac
python scripts/train_multizones.py policy=sac
```

## Using Other SB3 Algorithms

Additional algorithm configs are available:

| Config | Algorithm |
|---|---|
| `configs/policy/trpo.yaml` | TRPO |
| `configs/policy/dqn.yaml` | DQN |

To use any SB3 algorithm, create a YAML file under `configs/policy/` with the
algorithm name and its hyperparameters, then reference it via `policy=<name>`.

## Hyperparameter Overrides

Override any algorithm parameter from the command line:

```bash
# Increase PPO learning rate
python scripts/train_single_zone_houses.py policy=ppo policy.learning_rate=0.001

# Larger SAC replay buffer
python scripts/train_multizones.py policy=sac policy.buffer_size=500000

# More training steps
python scripts/train_single_zone_houses.py training.total_timesteps=5000000
```

## Evaluating SB3 Checkpoints

After training, the best and final model checkpoints are saved as `.zip` files.
To evaluate a trained checkpoint on a baseline rollout:

```bash
python scripts/baselines.py policy=sb3 policy.checkpoint_path=outputs/.../best_model.zip
```

To evaluate on the adaptive dynamics benchmark:

```bash
python scripts/bm_adaptive_dynamics.py policy=sb3 policy.checkpoint_path=path/to/model.zip
```

## Training Configuration

The shared training configuration is in `configs/training/default.yaml`:

```yaml title="configs/training/default.yaml"
total_timesteps: 1000000
eval_freq: 262144
eval_episodes: 20
num_train_envs: 4
cb_gradient_save_freq: 500
```

Increase `total_timesteps` for longer training runs and decrease `eval_freq`
for more frequent evaluation checkpoints.
