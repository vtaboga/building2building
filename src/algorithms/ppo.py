# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Callable
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import logging
from src.simulator.utils import TrajectoryLogger, CustomNormalizeObservation, CustomRescaleAction

# Set default tensor type to float64 for better precision
torch.set_default_dtype(torch.float64)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    results_dir: str = "results"
    """the base directory for storing all results"""

    # Algorithm specific arguments
    env_id: str = "EnergyPlus-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 96
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    eval_frequency: int = 100000
    """how often (in steps) to evaluate the policy during training"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # EnergyPlus specific arguments
    path_to_building: str = None
    """the path to the EnergyPlus building file"""
    path_to_weather: str = None
    """the path to the EnergyPlus weather file"""
    building_characteristics: dict = None
    """the characteristics of the building"""


def make_env(env_id, path_to_building, path_to_weather, building_characteristics, gamma, run_manager=None):
    def thunk():
        env = gym.make(env_id, 
                      path_to_building=path_to_building, 
                      path_to_weather=path_to_weather, 
                      building_characteristics=building_characteristics,
                      run_manager=run_manager)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = CustomRescaleAction(env, min_action=-1.0, max_action=1.0)
        env = gym.wrappers.ClipAction(env)
        norm_env = CustomNormalizeObservation(env)
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


def main(args: Args, run_manager=None):
    """
    Main function to run PPO algorithm.
    
    Args:
        args: Arguments for the PPO algorithm
        run_manager: Optional RunManager instance for unified logging
    """
    # Create directories for logs and outputs
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = args.results_dir
    os.makedirs(run_dir, exist_ok=True)

    # Setup tracking based on run_manager
    # Only initialize wandb internally if RunManager isn't handling it
    use_internal_tracking = run_manager is None and args.track
    
    if use_internal_tracking:
        print("Initializing wandb in ppo!")
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # Use RunManager's logger and tensorboard if provided
    if run_manager:
        writer = run_manager.get_tensorboard_writer()
        logger = run_manager.logger
    else:
        writer = SummaryWriter(os.path.join(run_dir, "logs"))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = logging.getLogger("ppo")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.path_to_building, args.path_to_weather, 
                 args.building_characteristics, args.gamma, run_manager) for _ in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    # Use the evaluation frequency from args
    eval_frequency = args.eval_frequency
    next_eval_step = eval_frequency

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
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
                
                # Save current model to a temporary file
                temp_model_path = os.path.join(run_dir, f"temp_model_{global_step}.pt")
                torch.save(agent.state_dict(), temp_model_path)
                
                # Evaluate the current policy
                eval_returns = ppo_evaluate(
                    model_path=temp_model_path,
                    make_env=make_env,
                    env_id=args.env_id,
                    path_to_building=args.path_to_building,
                    path_to_weather=args.path_to_weather,
                    building_characteristics=args.building_characteristics,
                    eval_episodes=1,  # Just one episode for quick validation
                    run_name=f"{run_name}-validation-{global_step}",
                    Model=Agent,
                    device=device,
                    run_manager=run_manager,
                    gamma=args.gamma,
                    save_trajectories=False,  # Don't save trajectories during validation
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
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        # Determine the appropriate path for saving the model
        if run_manager:
            # Use RunManager's models directory if available
            models_dir = os.path.join(run_manager.run_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"{args.exp_name}.pt")
            norm_state_path = os.path.join(models_dir, f"{args.exp_name}_norm_state.pkl")
        else:
            # Default path if RunManager is not available
            model_path = os.path.join(run_dir, f"{args.exp_name}.cleanrl_model")
            norm_state_path = os.path.join(run_dir, f"{args.exp_name}_norm_state.pkl")
        
        # Save the model
        torch.save(agent.state_dict(), model_path)
        
        # Save normalization state from the first environment
        base_env = envs.envs[0]
        norm_wrapper = None
        
        # Find the normalization wrapper
        while base_env is not None:
            if isinstance(base_env, CustomNormalizeObservation):
                norm_wrapper = base_env
                break
            if hasattr(base_env, 'env'):
                base_env = base_env.env
            else:
                break
        
        if norm_wrapper:
            norm_wrapper.save_running_state(norm_state_path)
            logger.info(f"Normalization state saved to {norm_state_path}")
        
        logger.info(f"Model saved to {model_path}")

        episodic_returns = ppo_evaluate(
            model_path=model_path,
            make_env=make_env,
            env_id=args.env_id,
            path_to_building=args.path_to_building,
            path_to_weather=args.path_to_weather,
            building_characteristics=args.building_characteristics,
            eval_episodes=1,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            run_manager=run_manager,
            gamma=args.gamma,
            save_trajectories=True,  # Save trajectories for final evaluation
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", run_dir, f"videos/{run_name}-eval")

    envs.close()
    writer.close()


def ppo_evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    path_to_building: str,
    path_to_weather: str,
    building_characteristics: dict,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    run_manager=None,
    gamma: float = 0.99,
    save_trajectories: bool = False,
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
        Model: The model class to use
        device: Device to run evaluation on
        run_manager: Optional RunManager instance
        gamma: Discount factor
        save_trajectories: Whether to save trajectories
    """
    # Initialize logger first to avoid reference errors
    logger = run_manager.logger if run_manager else logging.getLogger(__name__)
    
    # Create environment using the same setup as in training
    env = make_env(env_id, path_to_building, path_to_weather, 
                building_characteristics, gamma, run_manager)()
    envs = gym.vector.SyncVectorEnv([
        make_env(env_id, path_to_building, path_to_weather, 
                building_characteristics, gamma, run_manager)
    ])
    
    # Find both wrappers
    norm_wrapper = None
    rescale_wrapper = None
    temp_env = env

    while temp_env is not None:
        if isinstance(temp_env, CustomNormalizeObservation):
            norm_wrapper = temp_env
        if isinstance(temp_env, CustomRescaleAction):
            rescale_wrapper = temp_env
        
        if hasattr(temp_env, 'env'):
            temp_env = temp_env.env
        else:
            break
    
    # Load normalization state if available
    norm_state_path = model_path.replace('.pt', '_norm_state.pkl')
    if norm_wrapper and os.path.exists(norm_state_path):
        try:
            norm_wrapper.load_running_state(norm_state_path)
            logger.info(f"Loaded normalization state from {norm_state_path}")
        except Exception as e:
            logger.warning(f"Failed to load normalization state: {e}")
    
    # Create and load agent
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # Get the base environment to access its properties
    base_env = envs.envs[0].unwrapped
    observation_names = base_env.observation_names if hasattr(base_env, 'observation_names') else None
    controlled_zones = base_env.controlled_zones if hasattr(base_env, 'controlled_zones') else None
    uncontrolled_zones = base_env.uncontrolled_zones if hasattr(base_env, 'uncontrolled_zones') else None

    episodic_returns = []
    
    # Run evaluation for the specified number of episodes
    for episode in range(eval_episodes):
        # Initialize trajectory logger for this episode
        trajectory_logger = TrajectoryLogger(
            os.path.join(run_manager.data_dir if run_manager else "trajectories", f"episode_{episode}"),
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
