"""
Configuration management utilities with threshold method selection
================================================================
"""

import os
import yaml
import argparse
from dataclasses import dataclass
from typing import List, Optional
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Experiment configuration dataclass for type safety and validation"""

    # Data configuration
    normal_data_root: str = "./dataset/v10/0.05_8_32_CTCA/Normal"
    diseased_data_root: str = "./dataset/v10/0.05_8_32_CTCA/Diseased"

    # Experiment settings
    run_name: str = "simplescan"
    save_path: str = "./production_kfold_results"
    gpu_id: str = "0"
    num_workers: int = 1
    k_folds: int = 4
    confidence_level: float = 0.95
    random_seed: int = 42

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 2
    lr: float = 1e-4
    eta_min: float = 1e-6
    entropy_loss_weight: float = 0.0002
    early_stopping_patience: int = 10

    # Model architecture
    in_chans: int = 1
    dims: List[int] = None
    depths: List[int] = None
    ssm_d_state: int = 32
    mem_dim: int = 300

    # Evaluation settings
    anomaly_threshold: float = 0.01
    use_optimal_threshold: bool = True
    threshold_method: str = "accuracy"  # New: 'accuracy' or 'youden'
    threshold_percentile: float = 95

    def __post_init__(self):
        if self.dims is None:
            self.dims = [32, 64]
        if self.depths is None:
            self.depths = [2, 2]


class ConfigManager:
    """Handles configuration loading from YAML files and command line arguments"""

    @staticmethod
    def _flatten_yaml_config(config_dict):
        """
        Flatten nested YAML config to match ExperimentConfig structure
        """
        flattened = {}

        def flatten_dict(d, parent_key=''):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, parent_key)
                else:
                    flattened[key] = value

        flatten_dict(config_dict)
        return flattened

    @staticmethod
    def load_from_yaml(config_path: str) -> DictConfig:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found, using defaults")
            return OmegaConf.structured(ExperimentConfig())

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested structure to match ExperimentConfig
        flattened_config = ConfigManager._flatten_yaml_config(config_dict)

        # Create default config
        default_config = OmegaConf.structured(ExperimentConfig())

        # Create loaded config from flattened dictionary
        loaded_config = OmegaConf.create(flattened_config)

        # Merge configurations
        merged_config = OmegaConf.merge(default_config, loaded_config)

        logger.info(f"Configuration loaded from {config_path}")
        logger.info(f"Loaded parameters: {list(flattened_config.keys())}")

        return merged_config

    @staticmethod
    def get_args():
        """Parse command line arguments with YAML config support"""
        parser = argparse.ArgumentParser(
            description="Production-Ready K-Fold Cross Validation for MemAE-VSSM3D",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Primary config source
        parser.add_argument('--config', type=str, default=None,
                          help='Path to YAML config file (overrides individual args)')

        # Data configuration
        parser.add_argument('--normal_data_root', type=str,
                          default="./dataset/v10/0.05_8_32_CTCA/Normal")
        parser.add_argument('--diseased_data_root', type=str,
                          default="./dataset/v10/0.05_8_32_CTCA/Diseased")

        # Experiment settings
        parser.add_argument('--run_name', type=str, default="production_kfold_memae_vssm3d")
        parser.add_argument('--save_path', type=str, default='./production_kfold_results')
        parser.add_argument('--gpu_id', type=str, default='0')
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--k_folds', type=int, default=4)
        parser.add_argument('--confidence_level', type=float, default=0.95)
        parser.add_argument('--random_seed', type=int, default=42)

        # Training hyperparameters
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--eta_min', type=float, default=1e-6)
        parser.add_argument('--entropy_loss_weight', type=float, default=0.0002)
        parser.add_argument('--early_stopping_patience', type=int, default=10)

        # Model architecture
        parser.add_argument('--in_chans', type=int, default=1)
        parser.add_argument('--dims', type=int, nargs='+', default=[32, 64])
        parser.add_argument('--depths', type=int, nargs='+', default=[2, 2])
        parser.add_argument('--ssm_d_state', type=int, default=32)
        parser.add_argument('--mem_dim', type=int, default=300)

        # Evaluation settings
        parser.add_argument('--anomaly_threshold', type=float, default=0.01)
        parser.add_argument('--use_optimal_threshold', action='store_true')
        parser.add_argument('--threshold_method', type=str, default='accuracy',
                          choices=['accuracy', 'youden'],
                          help='Method for optimal threshold: accuracy (max acc) or youden (max sensitivity+specificity-1)')
        parser.add_argument('--threshold_percentile', type=float, default=95)

        args = parser.parse_args()

        # If config file specified, load from YAML and override with CLI args
        if args.config:
            config = ConfigManager.load_from_yaml(args.config)

            # Override YAML config with non-None command line arguments
            cli_dict = {k: v for k, v in vars(args).items()
                       if v is not None and k != 'config'}

            if cli_dict:
                cli_config = OmegaConf.create(cli_dict)
                final_config = OmegaConf.merge(config, cli_config)
                logger.info(f"CLI overrides applied: {list(cli_dict.keys())}")
            else:
                final_config = config
        else:
            # Use command line arguments only
            final_config = OmegaConf.create(vars(args))

        # Validate configuration
        ConfigManager._validate_config(final_config)

        return final_config

    @staticmethod
    def _validate_config(config: DictConfig):
        """Validate configuration parameters"""

        # Check required paths exist
        if hasattr(config, 'normal_data_root') and config.normal_data_root:
            if not os.path.exists(config.normal_data_root):
                logger.warning(f"Normal data path does not exist: {config.normal_data_root}")

        if hasattr(config, 'diseased_data_root') and config.diseased_data_root:
            if not os.path.exists(config.diseased_data_root):
                logger.warning(f"Diseased data path does not exist: {config.diseased_data_root}")

        # Validate numerical parameters
        if config.k_folds < 2:
            raise ValueError(f"k_folds must be >= 2, got {config.k_folds}")

        if config.confidence_level <= 0 or config.confidence_level >= 1:
            raise ValueError(f"confidence_level must be between 0 and 1, got {config.confidence_level}")

        if config.lr <= 0:
            raise ValueError(f"Learning rate must be > 0, got {config.lr}")

        # Validate threshold method
        if hasattr(config, 'threshold_method') and config.threshold_method not in ['accuracy', 'youden']:
            raise ValueError(f"threshold_method must be 'accuracy' or 'youden', got {config.threshold_method}")

        logger.info("Configuration validation passed")
        logger.info(f"Threshold method: {getattr(config, 'threshold_method', 'accuracy')}")


def create_compatible_sample_config():
    """Create a sample YAML configuration file that's compatible with the config manager"""

    os.makedirs('config', exist_ok=True)

    sample_config = {
        # Nested structure (will be flattened)
        'data': {
            'normal_data_root': "./dataset/v10/0.05_8_32_CTCA/Normal",
            'diseased_data_root': "./dataset/v10/0.05_8_32_CTCA/Diseased"
        },
        'experiment': {
            'run_name': "production_kfold_memae_vssm3d",
            'save_path': "./production_kfold_results",
            'gpu_id': "0",
            'num_workers': 1,
            'k_folds': 4,
            'confidence_level': 0.95,
            'random_seed': 42
        },
        'training': {
            'epochs': 100,
            'batch_size': 2,
            'lr': 0.0001,
            'eta_min': 0.000001,
            'entropy_loss_weight': 0.0002,
            'early_stopping_patience': 10
        },
        'model': {
            'in_chans': 1,
            'dims': [32, 64],
            'depths': [2, 2],
            'ssm_d_state': 32,
            'mem_dim': 300
        },
        'evaluation': {
            'anomaly_threshold': 0.01,
            'use_optimal_threshold': True,
            'threshold_method': 'accuracy',  # New parameter
            'threshold_percentile': 95
        }
    }

    with open('config/sample_config.yaml', 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)

    # Also create a flat version for reference
    flat_config = {
        'normal_data_root': "./dataset/v10/0.05_8_32_CTCA/Normal",
        'diseased_data_root': "./dataset/v10/0.05_8_32_CTCA/Diseased",
        'run_name': "production_kfold_memae_vssm3d",
        'k_folds': 4,
        'epochs': 100,
        'batch_size': 2,
        'lr': 0.0001,
        'dims': [32, 64],
        'mem_dim': 300,
        'use_optimal_threshold': True,
        'threshold_method': 'accuracy'  # New parameter
    }

    with open('config/flat_sample_config.yaml', 'w') as f:
        yaml.dump(flat_config, f, default_flow_style=False, indent=2)

    print("✅ Compatible configuration files created:")
    print("   📄 config/sample_config.yaml (nested structure)")
    print("   📄 config/flat_sample_config.yaml (flat structure)")
    print("\n🚀 Usage:")
    print("   python main.py --config config/sample_config.yaml")
    print("   python main.py --config config/flat_sample_config.yaml")

    return sample_config