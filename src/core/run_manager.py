import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

class RunManager:
    """
    Unified manager for experiment runs that handles:
    - Directory creation with consistent naming
    - Logging configuration (file and console)
    - Result storage
    - Integration with wandb and TensorBoard
    """
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "results",
        track_wandb: bool = False,
        wandb_project: str = None,
        wandb_entity: str = None,
        tags: Optional[Dict[str, Any]] = None,
        seed: int = None,
    ):
        """
        Initialize a new experiment run.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for all results
            track_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity (team or username)
            tags: Additional tags/metadata for the run
            seed: Random seed used for the experiment
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.config = {
            "experiment_name": experiment_name,
            "timestamp": self.timestamp,
            "seed": seed,
        }
        
        if tags:
            self.config.update(tags)
        
        # Set up temporary logger for initialization
        self.logger = self._setup_temp_logger()
        
        # Initialize wandb first if requested
        self.track_wandb = track_wandb
        self.wandb_run_id = None
        
        if track_wandb:
            print("Initializing wandb")
            self._init_wandb(wandb_project, wandb_entity)
            
        # Now create the run ID and directory structure, using wandb run ID if available
        if self.track_wandb and hasattr(self, 'wandb') and self.wandb_run_id:
            # Use wandb run ID in our directory name but keep the timestamp
            self.run_id = f"{experiment_name}_{self.timestamp}_{self.wandb_run_id}"
        else:
            # Create our own ID if not using wandb
            if seed is not None:
                self.run_id = f"{experiment_name}_{self.timestamp}_seed{seed}_{uuid.uuid4().hex[:6]}"
            else:
                self.run_id = f"{experiment_name}_{self.timestamp}_{uuid.uuid4().hex[:6]}"
        
        self.base_dir = base_dir
        self.run_dir = os.path.join(base_dir, self.run_id)
        
        # Create main run directory
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Define directory paths
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.models_dir = os.path.join(self.run_dir, "models")
        self.eplus_dir = os.path.join(self.run_dir, "eplus_output")
        self.tensorboard_dir = os.path.join(self.run_dir, "tensorboard")
        self.data_dir = os.path.join(self.run_dir, "data")
        
        # Update the config with run ID
        self.config["run_id"] = self.run_id
        
        # Setup permanent logging - this will create logs_dir
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
    
    def _init_wandb(self, project_name, entity):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb
            
            # Initialize wandb with default settings
            run = wandb.init(
                project=project_name,
                entity=entity,
                config=self.config,
                sync_tensorboard=True,
                save_code=True,
                tags=[self.experiment_name]
            )
            
            # Store the wandb object and run ID
            self.wandb = wandb
            self.wandb_run_id = run.id
            
            # Update config with wandb info
            self.config["wandb_url"] = run.url
            self.config["wandb_id"] = run.id
            
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
            writer.add_text(
                "config",
                "\n".join([f"**{k}:** {v}" for k, v in self.config.items()])
            )
            return writer
        except ImportError:
            self.logger.warning("torch.utils.tensorboard not found. Continuing without TensorBoard.")
            return None
    
    def save_config(self, additional_config=None):
        """Save the run configuration to a JSON file."""
        config = self.config.copy()
        if additional_config:
            config.update(additional_config)
            self.config.update(additional_config)  # Update internal config too
        
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
    
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