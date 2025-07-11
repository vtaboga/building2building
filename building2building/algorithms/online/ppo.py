# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from typing import Callable, Optional
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import logging
from building2building.simulator.utils import TrajectoryLogger
from building2building.simulator.wrappers import NormalizeObservation, CustomRescaleAction

# Set default tensor type to float64 for better precision
torch.set_default_dtype(torch.float64)


def make_env(env_id, path_to_building, path_to_weather, building_characteristics, reward_type, energy_weight, gamma, eplus_output_dir=None):
    def thunk():
        env = gym.make(env_id, 
                      path_to_building=path_to_building, 
                      path_to_weather=path_to_weather, 
                      building_characteristics=building_characteristics,
                      reward_type=reward_type,
                      energy_weight=energy_weight,
                      eplus_output_dir=eplus_output_dir)
        env = CustomRescaleAction(env)
        env = gym.wrappers.ClipAction(env)
        norm_env = NormalizeObservation(env)
        return norm_env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
            nn.Sigmoid()
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def main(cfg, building_path, weather_path, weather_validation_path, building_characteristics, results_dir, wandb_run=None):
    """
    Main function to run PPO algorithm.
    
    Args:
        cfg: Hydra configuration object
        building_path: Path to building file
        weather_path: Path to weather file  
        weather_validation_path: Path to validation weather file
        building_characteristics: Building characteristics dict
        results_dir: Directory to save results (Hydra output directory)
    """
    # Use the results_dir directly (Hydra output directory) instead of creating nested folders
    run_name = f"EnergyPlus-v0__ppo__{cfg.seed}__{int(time.time())}"
    run_dir = results_dir  # Use Hydra's output directory directly
    
    # Create subdirectories for organized results
    eplus_outputs_dir = os.path.join(run_dir, "eplus_outputs")
    test_results_dir = os.path.join(run_dir, "test_results")
    os.makedirs(eplus_outputs_dir, exist_ok=True)
    os.makedirs(test_results_dir, exist_ok=True)

    # Setup logging to file
    log_file = os.path.join(run_dir, "train_ppo.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger("ppo")
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    if wandb_run:
        logger.info(f"TensorBoard logs will be synced from: {run_dir}")
    
    # Setup tensorboard - save in run-specific directory
    writer = SummaryWriter(os.path.join(run_dir, "logs"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in dict(cfg).items()])),
    )

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment setup with eplus_output_dir
    envs = gym.vector.SyncVectorEnv([
        make_env(
            env_id="EnergyPlus-v0",
            path_to_building=building_path,
            path_to_weather=weather_path,
            building_characteristics=building_characteristics,
            reward_type=cfg.reward_type,
            energy_weight=cfg.energy_weight,
            gamma=cfg.ppo.gamma,
            eplus_output_dir=eplus_outputs_dir
        ) for _ in range(cfg.ppo.num_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs_shape = envs.single_observation_space.shape or ()
    action_shape = envs.single_action_space.shape or ()
    
    obs = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs) + obs_shape).to(device)
    actions = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)
    rewards = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)
    dones = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)
    values = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.ppo.num_envs).to(device)

    batch_size = int(cfg.ppo.num_envs * cfg.ppo.num_steps)
    minibatch_size = int(batch_size // cfg.ppo.num_minibatches)
    num_iterations = cfg.training.total_timesteps // batch_size
    
    # Use the evaluation frequency from config
    eval_frequency = cfg.ppo.eval_frequency
    next_eval_step = eval_frequency

    for iteration in range(1, num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if cfg.ppo.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * cfg.ppo.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, cfg.ppo.num_steps):
            global_step += cfg.ppo.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            
            # Check if it's time to evaluate the current policy
            if global_step >= next_eval_step:
                logger.info(f"Evaluating policy at step {global_step}")
                
                # Save current model to a temporary file in run directory
                temp_model_path = os.path.join(run_dir, f"temp_model_{global_step}.pt")
                torch.save(agent.state_dict(), temp_model_path)
                
                # Use validation weather if provided, else training weather
                validation_weather = weather_validation_path if weather_validation_path else weather_path

                # Evaluate the current policy
                eval_returns = ppo_evaluate(
                    model_path=temp_model_path,
                    make_env=make_env,
                    env_id="EnergyPlus-v0",
                    reward_type=cfg.reward_type,
                    energy_weight=cfg.energy_weight,
                    path_to_building=building_path,
                    path_to_weather=validation_weather,
                    building_characteristics=building_characteristics,
                    eval_episodes=1,
                    run_name=f"{run_name}-validation-{global_step}",
                    device=device,
                    gamma=cfg.ppo.gamma,
                    save_trajectories=False,
                    trajectories_dir=test_results_dir,
                    eplus_output_dir=eplus_outputs_dir,
                )
                
                # Log the validation return
                mean_return = np.mean(eval_returns)
                writer.add_scalar("validation/episodic_return", mean_return, global_step)
                
                # Remove the temporary model file
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                
                # Set the next evaluation step
                next_eval_step = global_step + eval_frequency

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.ppo.num_steps)):
                if t == cfg.ppo.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.ppo.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(cfg.ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.ppo.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.ppo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.ppo.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.ppo.clip_coef,
                        cfg.ppo.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ppo.ent_coef * entropy_loss + v_loss * cfg.ppo.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.max_grad_norm)
                optimizer.step()

            if cfg.ppo.target_kl is not None and approx_kl > cfg.ppo.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if cfg.training.save_model:
        model_path = os.path.join(run_dir, "model.pt")
        torch.save(agent.state_dict(), model_path)       
        logger.info(f"Model saved to {model_path}")

        # Use validation weather if provided, else training weather
        validation_weather = weather_validation_path if weather_validation_path else weather_path

        episodic_returns = ppo_evaluate(
            model_path=model_path,
            make_env=make_env,
            reward_type=cfg.reward_type,
            energy_weight=cfg.energy_weight,
            env_id="EnergyPlus-v0",
            path_to_building=building_path,
            path_to_weather=validation_weather,
            building_characteristics=building_characteristics,
            eval_episodes=1,
            run_name=f"{run_name}-eval",
            device=device,
            gamma=cfg.ppo.gamma,
            save_trajectories=True,
            trajectories_dir=test_results_dir,
            eplus_output_dir=eplus_outputs_dir,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
    if wandb_run:
        wandb.finish()


def ppo_evaluate(
    model_path: str,
    make_env: Callable,
    reward_type: str,
    energy_weight: float,
    env_id: str,
    path_to_building: Optional[str],
    path_to_weather: Optional[str],
    building_characteristics: Optional[dict],
    eval_episodes: int,
    run_name: str,
    device: torch.device = torch.device("cpu"),
    gamma: float = 0.99,
    save_trajectories: bool = False,
    trajectories_dir: Optional[str] = None,
    eplus_output_dir: Optional[str] = None,
):
    """
    Evaluate a trained PPO model.
    
    Args:
        model_path: Path to the saved model
        make_env: Environment creation function
        env_id: Environment ID
        path_to_building: Path to building file
        path_to_weather: Path to weather file
        building_characteristics: Building characteristics dict
        eval_episodes: Number of episodes to evaluate
        run_name: Name for this evaluation run
        device: Device to run evaluation on
        gamma: Discount factor
        save_trajectories: Whether to save trajectories
        trajectories_dir: Directory to save trajectories (defaults to current directory)
        eplus_output_dir: Directory for EnergyPlus outputs
    """
    logger = logging.getLogger(__name__)
    
    # Create environment using the same setup as in training
    env = make_env(env_id, path_to_building, path_to_weather, 
                building_characteristics, reward_type, energy_weight, gamma, eplus_output_dir)()
    envs = gym.vector.SyncVectorEnv([
        make_env(env_id, path_to_building, path_to_weather, 
                building_characteristics, reward_type, energy_weight, gamma, eplus_output_dir)
    ])
    
    # Find both wrappers
    norm_wrapper = None
    rescale_wrapper = None
    temp_env = env

    while temp_env is not None:
        if isinstance(temp_env, NormalizeObservation):
            norm_wrapper = temp_env
        if isinstance(temp_env, CustomRescaleAction):
            rescale_wrapper = temp_env
        
        if hasattr(temp_env, 'env'):
            temp_env = getattr(temp_env, 'env')
        else:
            break
    
    # Create and load agent
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # Get the base environment to access its properties
    base_env = envs.envs[0].unwrapped
    observation_names = getattr(base_env, 'observation_names', None)
    controlled_zones = getattr(base_env, 'controlled_zones', None)
    uncontrolled_zones = getattr(base_env, 'uncontrolled_zones', None)

    episodic_returns = []
    
    # Set up trajectories directory - use test_results naming
    if trajectories_dir is None:
        trajectories_base = "test_results"
    else:
        trajectories_base = trajectories_dir
    
    # Run evaluation for the specified number of episodes
    for episode in range(eval_episodes):
        # Initialize trajectory logger for this episode
        trajectory_logger = TrajectoryLogger(
            os.path.join(trajectories_base, f"episode_{episode}"),
            observation_names,
            logger=logger
        )
        
        # Reset environment
        obs, _ = envs.reset()
        done = False
        
        logger.info(f"Starting evaluation episode {episode+1}/{eval_episodes}")
        
        # Run episode until done
        while not done:
            with torch.no_grad():
                actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminations, truncations, _ = envs.step(actions_np)
            
            # Denormalize observation if we have a normalization wrapper
            if norm_wrapper:
                denorm_obs = norm_wrapper.denormalize(obs[0])   
                
                # Use the rescale wrapper if available, otherwise just use the raw actions
                if rescale_wrapper:
                    scaled_actions = rescale_wrapper.scale_action(actions_np[0])
                else:
                    scaled_actions = actions_np[0]
                
                # Log the denormalized observation in the trajectory
                trajectory_logger.log(denorm_obs, scaled_actions, rewards[0], controlled_zones, uncontrolled_zones)
            else:
                # If no normalization wrapper, log the observation as is
                trajectory_logger.log(obs[0], actions_np[0], rewards[0], controlled_zones, uncontrolled_zones)
            
            # Check if episode is done
            done = terminations[0] or truncations[0]
            obs = next_obs
        
        # Episode is done, get the total reward from the trajectory logger
        episode_return = trajectory_logger.total_reward
        episodic_returns.append(episode_return)
        
        logger.info(f"Evaluation episode {episode+1} completed: return={episode_return:.2f}")
        
        # Save the trajectory if requested
        if save_trajectories:
            trajectory_logger.save()
            logger.info(f"Saved trajectory for episode {episode+1}")

    # Calculate and log average return
    mean_return = sum(episodic_returns) / len(episodic_returns)
    logger.info(f"Evaluation completed. Mean return: {mean_return:.2f}")
    
    envs.close()
    return episodic_returns
