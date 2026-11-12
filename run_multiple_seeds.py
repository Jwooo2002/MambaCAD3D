#!/usr/bin/env python3
"""
Multiple Seeds Experiment Runner
Runs the same experiment configuration with different random seeds
"""

import os
import json
import yaml
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Base configuration
BASE_CONFIG = {
    "normal_data_root": "/home/jiwoong/PycharmProjects/VMamba/MemAE_kfold_simplescan/main_files/dataset/v11/0.05_8_8_CTCA_64/Normal",
    "diseased_data_root": "/home/jiwoong/PycharmProjects/VMamba/MemAE_kfold_simplescan/main_files/dataset/v11/0.05_8_8_CTCA_64/Diseased",
    "run_name": "test_8x8_patch_2",
    "save_path": "../multiple_seeds_res",
    "gpu_id": "1",
    "num_workers": 1,
    "k_folds": 4,
    "confidence_level": 0.95,
    "random_seed": 42,  # This will be overridden
    "epochs": 1000,
    "batch_size": 2,
    "lr": 0.0001,
    "eta_min": 1e-06,
    "entropy_loss_weight": 0.0002,
    "early_stopping_patience": 10,
    "in_chans": 1,
    "dims": [32, 64, 128, 128],
    "depths": [2, 2, 2, 2],
    "ssm_d_state": 8,
    "mem_dim": 300,
    "anomaly_threshold": 0.01,
    "use_optimal_threshold": True,
    "threshold_method": "accuracy",
    "threshold_percentile": 95
}

# Seeds to run experiments with
SEEDS = [42, 123, 456, 789, 2024]

# Directory setup
SCRIPT_DIR = Path(__file__).parent
MAIN_FILES_DIR = SCRIPT_DIR / "main_files"
CONFIG_DIR = SCRIPT_DIR / "temp_configs"
RESULTS_BASE_DIR = SCRIPT_DIR / "multiple_seeds_res"

def setup_directories():
    """Create necessary directories"""
    CONFIG_DIR.mkdir(exist_ok=True)
    RESULTS_BASE_DIR.mkdir(exist_ok=True)
    print(f"✓ Created directories: {CONFIG_DIR}, {RESULTS_BASE_DIR}")

def create_config_file(seed):
    """Create a temporary config file for a specific seed"""
    config = BASE_CONFIG.copy()
    config["random_seed"] = seed
    config["run_name"] = f"{BASE_CONFIG['run_name']}_seed{seed}"

    # Update save path to include seed (use absolute path)
    save_path = (RESULTS_BASE_DIR / f"seed_{seed}").resolve()
    config["save_path"] = str(save_path)

    config_path = CONFIG_DIR / f"config_seed_{seed}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✓ Created config file: {config_path}")
    return config_path

def run_experiment(seed, config_path):
    """Run experiment with a specific seed"""
    print("\n" + "="*70)
    print(f"Starting experiment with seed: {seed}")
    print("="*70)

    # Build command
    cmd = [
        sys.executable,  # Python interpreter
        str(MAIN_FILES_DIR / "main.py"),
        "--config", str(config_path)
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {MAIN_FILES_DIR}")

    try:
        # Run the experiment
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=str(MAIN_FILES_DIR)  # Run from main_files directory
        )
        print(f"✓ Experiment with seed {seed} completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment with seed {seed} failed!")
        print(f"Error: {e}")
        return False

def main():
    """Main execution function"""
    start_time = datetime.now()
    print("\n" + "="*70)
    print("Multiple Seeds Experiment Runner")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds to run: {SEEDS}")
    print(f"Total experiments: {len(SEEDS)}")
    print(f"Results directory: {RESULTS_BASE_DIR}")
    print("="*70 + "\n")

    # Setup
    setup_directories()

    # Track results
    results = {}

    # Run experiments for each seed
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n{'#'*70}")
        print(f"Experiment {i}/{len(SEEDS)}")
        print(f"{'#'*70}")

        config_path = create_config_file(seed)
        success = run_experiment(seed, config_path)
        results[seed] = "SUCCESS" if success else "FAILED"

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print("\nResults:")
    for seed, status in results.items():
        print(f"  Seed {seed}: {status}")

    success_count = sum(1 for s in results.values() if s == "SUCCESS")
    print(f"\nSuccess rate: {success_count}/{len(SEEDS)}")
    print(f"Results saved to: {RESULTS_BASE_DIR}")
    print("="*70 + "\n")

    # Save summary to file
    summary_path = RESULTS_BASE_DIR / "experiment_summary.json"
    summary = {
        "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "duration_seconds": duration.total_seconds(),
        "seeds": SEEDS,
        "results": results,
        "success_count": success_count,
        "total_count": len(SEEDS),
        "base_config": BASE_CONFIG
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
