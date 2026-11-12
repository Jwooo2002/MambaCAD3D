#!/usr/bin/env python3
"""
Flexible Multiple Seeds Experiment Runner
Usage:
    python run_multiple_seeds_flexible.py --seeds 42 123 456
    python run_multiple_seeds_flexible.py --config base_config.json --seeds 42 123 456 789 2024
    python run_multiple_seeds_flexible.py --help
"""

import os
import json
import yaml
import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Default configuration
DEFAULT_CONFIG = {
    "normal_data_root": "/home/jiwoong/PycharmProjects/VMamba/MemAE_kfold_simplescan/main_files/dataset/v11/0.05_8_8_CTCA_64/Normal",
    "diseased_data_root": "/home/jiwoong/PycharmProjects/VMamba/MemAE_kfold_simplescan/main_files/dataset/v11/0.05_8_8_CTCA_64/Diseased",
    "run_name": "test_8x8_patch_2",
    "save_path": "../multiple_seeds_res",
    "gpu_id": "1",
    "num_workers": 1,
    "k_folds": 4,
    "confidence_level": 0.95,
    "random_seed": 42,
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

SCRIPT_DIR = Path(__file__).parent
MAIN_FILES_DIR = SCRIPT_DIR / "main_files"
CONFIG_DIR = SCRIPT_DIR / "temp_configs"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run experiments with multiple random seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default seeds (42, 123, 456, 789, 2024)
  python run_multiple_seeds_flexible.py

  # Run with custom seeds
  python run_multiple_seeds_flexible.py --seeds 42 100 200

  # Run with custom config file
  python run_multiple_seeds_flexible.py --config my_config.json --seeds 42 123

  # Run with custom GPU and results directory
  python run_multiple_seeds_flexible.py --gpu 0 --results_dir my_results --seeds 42 123
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to base config JSON file (optional)'
    )

    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[1, 123, 456, 789, 2024],
        help='List of random seeds (default: 42 123 456 789 2024)'
    )

    parser.add_argument(
        '--gpu',
        type=str,
        default="1",
        help='GPU ID to use (default: 1)'
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default="multiple_seeds_res",
        help='Base directory for results (default: multiple_seeds_res)'
    )

    parser.add_argument(
        '--run_name',
        type=str,
        default="test_8x8_patch_2",
        help='Base run name (default: test_8x8_patch_2)'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print commands without executing'
    )

    return parser.parse_args()

def load_base_config(config_path):
    """Load base configuration from file"""
    if config_path is None:
        return DEFAULT_CONFIG.copy()

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found. Using default config.")
        return DEFAULT_CONFIG.copy()

    with open(config_path, 'r') as f:
        return json.load(f)

def setup_directories(results_base_dir):
    """Create necessary directories"""
    CONFIG_DIR.mkdir(exist_ok=True)
    results_base_dir = Path(results_base_dir)
    results_base_dir.mkdir(exist_ok=True)
    print(f"✓ Created directories: {CONFIG_DIR}, {results_base_dir}")
    return results_base_dir

def create_config_file(base_config, seed, results_base_dir, run_name):
    """Create a temporary config file for a specific seed"""
    config = base_config.copy()
    config["random_seed"] = seed
    config["run_name"] = f"{run_name}_seed{seed}"

    # Use absolute path for save_path
    save_path = (SCRIPT_DIR / results_base_dir / f"seed_{seed}").resolve()
    config["save_path"] = str(save_path)

    config_path = CONFIG_DIR / f"config_seed_{seed}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path

def run_experiment(seed, config_path, dry_run=False):
    """Run experiment with a specific seed"""
    print("\n" + "="*70)
    print(f"Experiment with seed: {seed}")
    print("="*70)

    cmd = [
        sys.executable,
        str(MAIN_FILES_DIR / "main.py"),
        "--config", str(config_path)
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {MAIN_FILES_DIR}")

    if dry_run:
        print("(DRY RUN - not executing)")
        return True

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
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
    args = parse_args()

    start_time = datetime.now()

    print("\n" + "="*70)
    print("Multiple Seeds Experiment Runner")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {len(args.seeds)}")
    print(f"GPU ID: {args.gpu}")
    print(f"Results directory: {args.results_dir}")
    print(f"Run name: {args.run_name}")
    if args.dry_run:
        print("MODE: DRY RUN (no actual execution)")
    print("="*70 + "\n")

    # Load base config
    base_config = load_base_config(args.config)
    base_config['gpu_id'] = args.gpu

    # Setup directories
    results_base_dir = setup_directories(args.results_dir)

    # Track results
    results = {}

    # Run experiments for each seed
    for i, seed in enumerate(args.seeds, 1):
        print(f"\n{'#'*70}")
        print(f"Experiment {i}/{len(args.seeds)}")
        print(f"{'#'*70}")

        config_path = create_config_file(base_config, seed, results_base_dir, args.run_name)
        print(f"✓ Created config: {config_path}")

        success = run_experiment(seed, config_path, args.dry_run)
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
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"  {status_symbol} Seed {seed}: {status}")

    success_count = sum(1 for s in results.values() if s == "SUCCESS")
    print(f"\nSuccess rate: {success_count}/{len(args.seeds)}")
    print(f"Results saved to: {results_base_dir}")
    print("="*70 + "\n")

    # Save summary
    if not args.dry_run:
        summary_path = results_base_dir / "experiment_summary.json"
        summary = {
            "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "duration_seconds": duration.total_seconds(),
            "seeds": args.seeds,
            "results": results,
            "success_count": success_count,
            "total_count": len(args.seeds),
            "base_config": base_config,
            "gpu_id": args.gpu,
            "run_name": args.run_name
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
