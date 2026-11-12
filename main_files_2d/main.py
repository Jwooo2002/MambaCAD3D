"""
Main execution script for 2D MemAE-VSSM with K-Fold Cross Validation
================================================================

Usage:
    python main.py --config config.yaml
    python main.py --create-sample-config
    python main.py --run_name "my_experiment" --k_folds 5

Author: Research Team
Version: 2.0.0 (2D)
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Fixed import paths
from utils.config_manager import ConfigManager
from utils.reproducibility import ReproducibilityManager
from core.experiment_manager import ExperimentManager
from utils.sample_config import create_sample_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_2d.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""

    # Check if user wants to create sample config
    if len(sys.argv) > 1 and sys.argv[1] == '--create-sample-config':
        create_sample_config()
        return

    try:
        # Parse arguments and load configuration
        config = ConfigManager.get_args()

        # Set reproducibility
        ReproducibilityManager.set_seed(config.random_seed)

        # Create experiment manager and run experiment
        experiment_manager = ExperimentManager(config)
        experiment_manager.run_experiment()

        logger.info("Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
