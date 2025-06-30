# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
import logging
import building2building.simulator
from building2building.simulator.utils import TrajectoryLogger
from building2building.simulator.wrappers import NormalizeObservation, CustomRescaleAction

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
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    reward_type: str = "barrier"
    """the type of reward function to use"""
    energy_weight: float = 1.0
    """the weight of the energy consumption penalty"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the replay memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    eval_frequency: int = 100000
    """how often (in steps) to evaluate the policy during training"""
    
    # EnergyPlus specific arguments
    path_to_building: str = None
    """the path to the EnergyPlus building file"""
    path_to_weather: str = None
    """the path to the EnergyPlus weather file"""
    weather_validation: str = None
    """the path to the EnergyPlus weather file for validation (optional)"""
    building_characteristics: dict = None
    """the characteristics of the building"""


def make_env(env_id, path_to_building, path_to_weather, building_characteristics, reward_type, energy_weight, gamma, run_manager=None):
    def thunk():
        env = gym.make(env_id, 
                      path_to_building=path_to_building, 
                      path_to_weather=path_to_weather, 
                      building_characteristics=building_characteristics,
                      reward_type=reward_type,
                      energy_weight=energy_weight,
                      run_manager=run_manager)
        norm_env = NormalizeObservation(env)
        return norm_env

    return thunk


# Define discrete action binning for continuous spaces
def get_action_binning(action_space, bins_per_dimension=20):
    """Create discrete action bins for each continuous action dimension"""
    if isinstance(action_space, gym.spaces.Box):
        # Get dimensions and ranges of the action space
        num_actions = np.prod(action_space.shape)
        action_low = action_space.low
        action_high = action_space.high
        
        # Create a discretization for each dimension
        action_bins = []
        for dim in range(num_actions):
            bin_values = np.linspace(action_low[dim], action_high[dim], bins_per_dimension)
            action_bins.append(bin_values)
        
        return action_bins, bins_per_dimension**num_actions
    else:
        raise ValueError("Action space must be continuous (Box) for discretization")


class QNetwork(nn.Module):
    def __init__(self, env, bins_per_dimension=20):
        super().__init__()
        # Get observation and action dimensions
        self.observation_dim = np.array(env.single_observation_space.shape).prod()
        self.action_bins, self.action_dim = get_action_binning(env.single_action_space, bins_per_dimension)
        self.bins_per_dimension = bins_per_dimension
        self.num_action_dims = np.prod(env.single_action_space.shape)
        
        # Q-network layers
        self.network = nn.Sequential(
            nn.Linear(self.observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def get_action(self, obs, epsilon=0.0):
        """Get action using epsilon-greedy policy"""
        if random.random() < epsilon:
            # Random action: select random indices for each action dimension
            action_indices = [random.randint(0, self.bins_per_dimension-1) for _ in range(self.num_action_dims)]
            flat_index = self._action_indices_to_flat(action_indices)
            return self._decode_action(flat_index), flat_index
        else:
            q_values = self.forward(obs)
            flat_index = torch.argmax(q_values, dim=1).item()
            return self._decode_action(flat_index), flat_index
    
    def _action_indices_to_flat(self, action_indices):
        """Convert multi-dimensional action indices to flat index"""
        flat_idx = 0
        for i, idx in enumerate(action_indices):
            flat_idx += idx * (self.bins_per_dimension ** i)
        return flat_idx
    
    def _decode_action(self, flat_index):
        """Convert flat index to continuous action"""
        action = np.zeros(self.num_action_dims)
        
        # Convert flat index to multi-dimensional indices
        remaining = flat_index
        for dim in range(self.num_action_dims):
            idx = remaining % self.bins_per_dimension
            remaining //= self.bins_per_dimension
            action[dim] = self.action_bins[dim][idx]
        
        return action


def main(args: Args, run_manager=None):
    """
    Main function to run DQN algorithm.
    
    Args:
        args: Arguments for the DQN algorithm
        run_manager: Optional HydraManager instance for unified logging
    """
    # Create directories for logs and outputs
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = args.results_dir
    os.makedirs(run_dir, exist_ok=True)

    # Setup tracking based on run_manager
    # Only initialize wandb internally if HydraManager isn't handling it
    use_internal_tracking = run_manager is None and args.track
    
    if use_internal_tracking:
        print("Initializing wandb in dqn!")
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
    
    # Use HydraManager's logger and tensorboard if provided
    if run_manager:
        writer = run_manager.get_tensorboard_writer()
        logger = run_manager.logger
    else:
        writer = SummaryWriter(os.path.join(run_dir, "logs"))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = logging.getLogger("dqn")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id=args.env_id, path_to_building=args.path_to_building, path_to_weather=args.path_to_weather, 
                 building_characteristics=args.building_characteristics, reward_type=args.reward_type, energy_weight=args.energy_weight, gamma=args.gamma, run_manager=run_manager) for _ in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Set the number of discrete bins per dimension
    bins_per_dimension = 20
    
    # Initialize Q networks
    q_network = QNetwork(envs, bins_per_dimension).to(device)
    target_network = QNetwork(envs, bins_per_dimension).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    
    # Initialize replay buffer
    rb_obs = torch.zeros((args.buffer_size, np.array(envs.single_observation_space.shape).prod())).to(device)
    rb_actions = torch.zeros((args.buffer_size,), dtype=torch.int64).to(device)
    rb_rewards = torch.zeros((args.buffer_size,)).to(device)
    rb_next_obs = torch.zeros((args.buffer_size, np.array(envs.single_observation_space.shape).prod())).to(device)
    rb_dones = torch.zeros((args.buffer_size,)).to(device)
    
    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # Track evaluation frequency
    eval_frequency = args.eval_frequency
    next_eval_step = eval_frequency
    
    # Setup epsilon decay for exploration
    epsilon = args.start_e
    epsilon_decay = (args.start_e - args.end_e) / (args.exploration_fraction * args.total_timesteps)
    
    # Store experience in buffer
    rb_ptr = 0
    
    for step in range(1, args.total_timesteps + 1):
        global_step += 1
        
        # Decay epsilon
        epsilon = max(args.end_e, epsilon - epsilon_decay)
        
        # ALGO LOGIC: get action
        if global_step < args.learning_starts:
            # Random actions during initial exploration
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
            action_indices = np.zeros(args.num_envs, dtype=np.int64)
        else:
            # Get actions from Q-network
            with torch.no_grad():
                obs_tensor = next_obs.reshape(-1, np.array(envs.single_observation_space.shape).prod())
                actions_list = []
                action_indices_list = []
                for i in range(args.num_envs):
                    action, action_idx = q_network.get_action(obs_tensor[i:i+1], epsilon)
                    actions_list.append(action)
                    action_indices_list.append(action_idx)
                
                actions = np.array(actions_list)
                action_indices = np.array(action_indices_list)
        
        # Step the environment
        next_obs_step, rewards, terminations, truncations, infos = envs.step(actions)
        next_done_step = np.logical_or(terminations, truncations)
        
        # Store current transition in replay buffer
        if args.num_envs == 1:
            # Add experience to buffer
            rb_obs[rb_ptr] = next_obs.reshape(-1)
            rb_actions[rb_ptr] = torch.tensor(action_indices).long()
            rb_rewards[rb_ptr] = torch.tensor(rewards).reshape(-1)
            rb_next_obs[rb_ptr] = torch.tensor(next_obs_step).reshape(-1).to(device)
            rb_dones[rb_ptr] = torch.tensor(next_done_step).reshape(-1)
            rb_ptr = (rb_ptr + 1) % args.buffer_size
        
        # Update next observation and done flag
        next_obs = torch.tensor(next_obs_step).to(device)
        next_done = torch.tensor(next_done_step).to(device)
        
        # Log episode returns
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
        # Train the model if enough samples are collected
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            # Sample from replay buffer
            sample_indices = np.random.choice(args.buffer_size, size=args.batch_size, replace=(args.buffer_size < args.batch_size))
            s_obs = rb_obs[sample_indices]
            s_actions = rb_actions[sample_indices]
            s_rewards = rb_rewards[sample_indices]
            s_next_obs = rb_next_obs[sample_indices]
            s_dones = rb_dones[sample_indices]
            
            # Compute TD target
            with torch.no_grad():
                target_q_values = target_network(s_next_obs)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
                td_target = s_rewards.unsqueeze(1) + (1 - s_dones).unsqueeze(1) * args.gamma * max_target_q_values
            
            # Compute current Q values
            current_q_values = q_network(s_obs)
            selected_q_values = current_q_values.gather(1, s_actions.unsqueeze(1))
            
            # Compute loss
            loss = F.mse_loss(selected_q_values, td_target)
            
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update target network if it's time
            if global_step % args.target_network_frequency == 0:
                # If tau=1.0, it's a hard update, otherwise a soft update
                if args.tau == 1.0:
                    target_network.load_state_dict(q_network.state_dict())
                else:
                    for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
            
            # Log training metrics
            writer.add_scalar("losses/td_loss", loss.item(), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
        
        # Evaluate the current policy periodically
        if global_step >= next_eval_step:
            logger.info(f"Evaluating policy at step {global_step}")
            
            # Save current model to a temporary file
            temp_model_path = os.path.join(run_dir, f"temp_model_{global_step}.pt")
            torch.save(q_network.state_dict(), temp_model_path)
            
            # Use validation weather if provided, else training weather
            validation_weather = args.weather_validation if args.weather_validation else args.path_to_weather
            # Evaluate the current policy
            eval_returns = dqn_evaluate(
                model_path=temp_model_path,
                make_env=make_env,
                env_id=args.env_id,
                reward_type=args.reward_type,
                energy_weight=args.energy_weight,
                path_to_building=args.path_to_building,
                path_to_weather=validation_weather,
                building_characteristics=args.building_characteristics,
                eval_episodes=1,  # Just one episode for quick validation
                run_name=f"{run_name}-validation-{global_step}",
                Model=QNetwork,
                device=device,
                run_manager=run_manager,
                gamma=args.gamma,
                save_trajectories=False,  # Don't save trajectories during validation
                bins_per_dimension=bins_per_dimension
            )
            
            # Log the validation return
            mean_return = np.mean(eval_returns)
            writer.add_scalar("validation/episodic_return", mean_return, global_step)
            
            # Remove the temporary model file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            
            # Set the next evaluation step
            next_eval_step = global_step + eval_frequency
    
    # Save the final model if requested
    if args.save_model:
        # Determine the appropriate path for saving the model
        if run_manager:
            # Use HydraManager's models directory if available
            models_dir = os.path.join(run_manager.run_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"{args.exp_name}.pt")
        else:
            # Default path if HydraManager is not available
            model_path = os.path.join(run_dir, f"{args.exp_name}.cleanrl_model")
        
        # Save the model
        torch.save(q_network.state_dict(), model_path)       
        logger.info(f"Model saved to {model_path}")

        # Use validation weather if provided, else training weather
        validation_weather = args.weather_validation if args.weather_validation else args.path_to_weather

        # Run final evaluation with trajectory saving
        episodic_returns = dqn_evaluate(
            model_path=model_path,
            make_env=make_env,
            reward_type=args.reward_type,
            energy_weight=args.energy_weight,
            env_id=args.env_id,
            path_to_building=args.path_to_building,
            path_to_weather=validation_weather,
            building_characteristics=args.building_characteristics,
            eval_episodes=1,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            run_manager=run_manager,
            gamma=args.gamma,
            save_trajectories=True,  # Save trajectories for final evaluation
            bins_per_dimension=bins_per_dimension
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()


def dqn_evaluate(
    model_path: str,
    make_env: Callable,
    reward_type: str,
    energy_weight: float,
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
    bins_per_dimension: int = 5
):
    """
    Evaluate a trained DQN model.
    
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
        run_manager: Optional HydraManager instance
        gamma: Discount factor
        save_trajectories: Whether to save trajectories
        bins_per_dimension: Number of discrete bins per action dimension
    """
    # Initialize logger first to avoid reference errors
    logger = run_manager.logger if run_manager else logging.getLogger(__name__)
    
    # Create environment using the same setup as in training
    env = make_env(env_id, path_to_building, path_to_weather, 
                building_characteristics, reward_type, energy_weight, gamma, run_manager)()
    envs = gym.vector.SyncVectorEnv([
        make_env(env_id, path_to_building, path_to_weather, 
                building_characteristics, reward_type, energy_weight, gamma, run_manager)
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
            temp_env = temp_env.env
        else:
            break
    
    # Create and load agent
    q_network = Model(envs, bins_per_dimension).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()

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
                obs_tensor = torch.tensor(obs).reshape(1, -1).to(device)
                action, _ = q_network.get_action(obs_tensor, epsilon=0.0)  # No exploration during evaluation
            # Wrap the action in a list for SyncVectorEnv
            action_list = [action]  # Create a list with one action for the single environment
            next_obs, rewards, terminations, truncations, _ = envs.step(action_list)
            
            # Denormalize observation if we have a normalization wrapper
            if norm_wrapper:
                denorm_obs = norm_wrapper.denormalize(obs[0])   
                
                # Use the rescale wrapper if available, otherwise just use the raw actions
                if rescale_wrapper:
                    scaled_actions = rescale_wrapper.scale_action(action)
                else:
                    scaled_actions = action
                
                # Log the denormalized observation in the trajectory
                trajectory_logger.log(denorm_obs, scaled_actions, rewards[0], controlled_zones, uncontrolled_zones)
            else:
                # If no normalization wrapper, log the observation as is
                trajectory_logger.log(obs[0], action, rewards[0], controlled_zones, uncontrolled_zones)
            
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
