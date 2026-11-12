"""
Sample configuration generator
=============================
"""

from .config_manager import create_compatible_sample_config


def create_sample_config():
    """Create sample configuration files"""
    return create_compatible_sample_config()
    """
Sample configuration generator
=============================
"""


import yaml


def create_sample_config():
    """Create a sample YAML configuration file"""

    sample_config = {
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
            'threshold_percentile': 95
        }
    }

    with open('sample_config.yaml', 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)

    print("📄 Sample configuration file created: sample_config.yaml")
    print("\n🚀 To use this config file, run:")
    print("   python main.py --config sample_config.yaml")
    print("\n⚙️  You can also override specific parameters:")
    print("   python main.py --config sample_config.yaml --k_folds 5 --epochs 150")

    return sample_config