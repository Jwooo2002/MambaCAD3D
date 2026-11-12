"""
Reproducibility management utilities
===================================
"""

import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """Ensures complete reproducibility across all random sources"""

    @staticmethod
    def set_seed(seed: int = 42):
        """Set seed for all random number generators"""
        logger.info(f"Setting random seed to {seed} for complete reproducibility")

        # Python random
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variables for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)

        logger.info("Random seed set successfully")