import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


class HydraManager:
    """
    Hydra-compatible manager for experiment runs that handles:
    - Directory creation with consistent naming (leveraging Hydra's output dir)
    - Logging configuration (file and console)
    - Result storage
    - Integration with wandb and TensorBoard
    - Works seamlessly with Hydra configurations
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize a new experiment run using Hydra configuration.
        
        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.experiment_name = cfg.get('name', 'experiment')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get Hydra's output directory
        try:
            hydra_cfg = HydraConfig.get()
            self.run_dir = hydra_cfg.runtime.output_dir
        except:
            # Fallback if not running under Hydra
            self.run_dir = f"results/{self.experiment_name}_{self.timestamp}"
            os.makedirs(self.run_dir, exist_ok=True)
        
        # Set up temporary logger for initialization
        self.logger = self._setup_temp_logger()
        
        # Initialize wandb if requested
        self.track_wandb = cfg.get('track', False)
        self.wandb_run_id = None
        
        if self.track_wandb:
            print("Initializing wandb")
            self._init_wandb()
            
        # Create run ID
        if self.track_wandb and hasattr(self, 'wandb') and self.wandb_run_id:
            self.run_id = f"{self.experiment_name}_{self.timestamp}_{self.wandb_run_id}"
        else:
            seed = cfg.get('seed', None)
            if seed is not None:
                self.run_id = f"{self.experiment_name}_{self.timestamp}_seed{seed}_{uuid.uuid4().hex[:6]}"
            else:
                self.run_id = f"{self.experiment_name}_{self.timestamp}_{uuid.uuid4().hex[:6]}"
        
        # Define directory paths
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.models_dir = os.path.join(self.run_dir, "models")
        self.eplus_dir = os.path.join(self.run_dir, "eplus_output")
        self.tensorboard_dir = os.path.join(self.run_dir, "tensorboard")
        self.data_dir = os.path.join(self.run_dir, "data")
        
        # Setup permanent logging
        self.logger = self._setup_logging()
        
        # Save initial config
        self.save_config()
        
        self.logger.info(f"Run initialized: {self.run_id}")
        self.logger.info(f"Results will be saved to: {self.run_dir}")
    
    def _setup_temp_logger(self):
        """Setup a temporary console-only logger for initialization phase."""
        logger = logging.getLogger("temp_" + str(uuid.uuid4()))
        logger.setLevel(logging.INFO)
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _ensure_dir(self, directory):
        """Create directory if it doesn't exist yet."""
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            if hasattr(self, 'logger'):
                self.logger.debug(f"Created directory: {directory}")
        return directory
    
    def _setup_logging(self):
        """Configure logging to both file and console."""
        # Ensure logs directory exists
        self._ensure_dir(self.logs_dir)
        
        logger = logging.getLogger(self.run_id)
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers if any
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)
        
        # File handler
        log_file = os.path.join(self.logs_dir, "run.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb
            
            # Get wandb config from Hydra config
            wandb_config = self.cfg.get('wandb', {})
            project_name = wandb_config.get('project', 'building2building')
            entity = wandb_config.get('entity', None)
            
            # Convert OmegaConf to regular dict for wandb
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)
            
            # Initialize wandb
            run = wandb.init(
                project=project_name,
                entity=entity,
                config=config_dict,
                sync_tensorboard=True,
                save_code=True,
                tags=[self.experiment_name]
            )
            
            # Store the wandb object and run ID
            self.wandb = wandb
            self.wandb_run_id = run.id
            
            self.logger.info(f"Weights & Biases initialized: {run.url}")
            self.logger.info(f"WandB Run ID: {run.id}")
            
        except ImportError:
            self.logger.warning("wandb package not found. Continuing without W&B tracking.")
            self.track_wandb = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.track_wandb = False
    
    def get_tensorboard_writer(self):
        """Get a TensorBoard SummaryWriter for this run."""
        try:
            # Ensure tensorboard directory exists
            self._ensure_dir(self.tensorboard_dir)
            
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(self.tensorboard_dir)
            
            # Convert config to string for tensorboard
            config_str = OmegaConf.to_yaml(self.cfg)
            writer.add_text("config", config_str.replace('\n', '  \n'))
            return writer
        except ImportError:
            self.logger.warning("torch.utils.tensorboard not found. Continuing without TensorBoard.")
            return None
    
    def save_config(self, additional_config=None):
        """Save the run configuration to files."""
        # Save Hydra config as YAML
        config_yaml_path = os.path.join(self.run_dir, "config.yaml")
        with open(config_yaml_path, "w") as f:
            OmegaConf.save(self.cfg, f)
        
        # Also save as JSON for backward compatibility
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        if additional_config:
            config_dict.update(additional_config)
        
        config_json_path = os.path.join(self.run_dir, "config.json")
        with open(config_json_path, "w") as f:
            json.dump(config_dict, f, indent=4)
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to both TensorBoard and W&B if enabled."""
        if self.track_wandb and hasattr(self, 'wandb'):
            self.wandb.log(metrics, step=step)
    
    def save_results(self, results, name="results.json"):
        """Save experiment results to a JSON file."""
        # Ensure data directory exists
        self._ensure_dir(self.data_dir)
        
        results_path = os.path.join(self.data_dir, name)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Results saved to {results_path}")
        return results_path
    
    def get_eplus_output_dir(self):
        """Get the directory for EnergyPlus output files."""
        # Ensure EnergyPlus output directory exists
        self._ensure_dir(self.eplus_dir)
        return self.eplus_dir
    
    def get_config(self, key: str = None, default=None):
        """Get configuration value using dot notation."""
        if key is None:
            return self.cfg
        return OmegaConf.select(self.cfg, key, default=default)
    
    def finish(self):
        """Finalize the run, close all loggers and trackers."""
        # Create a summary file with timing information
        duration = time.time() - time.mktime(datetime.strptime(
            self.timestamp, '%Y%m%d_%H%M%S').timetuple())
        
        summary = {
            "run_id": self.run_id,
            "duration_seconds": duration,
            "completed": True,
            "completed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add wandb URL to summary if available
        if self.track_wandb and hasattr(self, 'wandb'):
            summary["wandb_url"] = self.wandb.run.url
            summary["wandb_id"] = self.wandb_run_id
        
        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        self.logger.info(f"Run {self.run_id} completed after {duration:.2f} seconds")
        
        # Close wandb
        if self.track_wandb and hasattr(self, 'wandb'):
            try:
                self.logger.info("Finalizing WandB logging...")
                self.wandb.finish()
            except Exception as e:
                self.logger.warning(f"Error when finishing wandb: {e}")


# Backward compatibility function
def create_hydra_manager(cfg: DictConfig) -> HydraManager:
    """Create a HydraManager from Hydra configuration."""
    return HydraManager(cfg) 