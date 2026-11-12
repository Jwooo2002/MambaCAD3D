#!/bin/bash

################################################################################
# PARALLEL Visualization Experiments Runner
################################################################################
# This script runs multiple visualization experiments with different parameters
# using PARALLEL processing for maximum speed.
#
# Features:
# - Multi-core parallel processing (up to 20x faster)
# - Multiple brightness/contrast configurations
# - Organized output directories per experiment
# - Comprehensive logging
#
# Usage:
#   bash run_visualize_experiments_parallel.sh
################################################################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="./dataset/v11/0.05_8_32_CTCA_64/Diseased"
BASE_OUTPUT_DIR="./visualizations_experiments_parallel"
NUM_WORKERS=20  # Number of parallel workers
LOG_FILE="experiments_run_parallel.log"

# Clear previous log
> "$LOG_FILE"

# Function to log messages
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local description=$2
    local alpha=$3
    local beta=$4
    local resolution=$5

    local output_dir="${BASE_OUTPUT_DIR}/${exp_name}"

    log "${BLUE}[Experiment: ${exp_name}]${NC}"
    log "  Description: ${description}"
    log "  Parameters: alpha=${alpha}, beta=${beta}, resolution=${resolution}, workers=${NUM_WORKERS}"
    log "  Output directory: ${output_dir}"

    local start_time=$(date +%s)
    log "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"

    # Run the parallel visualization
    python visualize_diseased_data_parallel.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$output_dir" \
        --alpha "$alpha" \
        --beta "$beta" \
        --resolution "$resolution" \
        --num_workers "$NUM_WORKERS" \
        >> "$LOG_FILE" 2>&1

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        log "  ${GREEN}✓ SUCCESS${NC} (Duration: ${duration}s, ${$((duration/60))}m)"
        log "  Results saved to: ${output_dir}"
    else
        log "  ${RED}✗ FAILED${NC} (Exit code: ${exit_code})"
        return 1
    fi

    log ""
}

################################################################################
# Main Execution
################################################################################

log "================================================================================"
log "Multiple Visualization Experiments (PARALLEL)"
log "================================================================================"
log "Configuration:"
log "  Data directory: ${DATA_DIR}"
log "  Base output directory: ${BASE_OUTPUT_DIR}"
log "  Number of parallel workers: ${NUM_WORKERS}"
log "  Number of experiments: 10"
log "  Execution mode: Sequential (but each uses parallel processing)"
log "  Max patches per file: All"
log "================================================================================"
log ""

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

log "================================================================================"
log "Running Experiments"
log "================================================================================"
log ""

# Define experiments
# Format: experiment_name "description" alpha beta resolution

# Experiment 1: Baseline
run_experiment \
    "exp1_baseline" \
    "Baseline (original brightness, standard contrast)" \
    1.0 0 448

# Experiment 2: Vessel enhanced
run_experiment \
    "exp2_vessel_enhanced" \
    "Vessel enhanced (brighter for contrast visualization)" \
    1.1 0 448

# Experiment 3: High contrast
run_experiment \
    "exp3_high_contrast" \
    "High contrast (enhanced vessel brightness)" \
    1.0 15 448

# Experiment 4: Dark background
run_experiment \
    "exp4_dark_bg" \
    "Dark background (vessels stand out more)" \
    1.0 -15 448

# Experiment 5: Bright overall
run_experiment \
    "exp5_bright_overall" \
    "Bright overall (good for detail inspection)" \
    1.2 10 448

# Experiment 6: High resolution baseline
run_experiment \
    "exp6_highres_baseline" \
    "High resolution baseline (512x512)" \
    1.0 0 512

# Experiment 7: Medium brightness
run_experiment \
    "exp7_medium_brightness" \
    "Medium brightness (balanced view)" \
    0.9 5 448

# Experiment 8: Sharp contrast
run_experiment \
    "exp8_sharp_contrast" \
    "Sharp contrast (alpha boosted)" \
    1.3 0 448

# Experiment 9: Soft appearance
run_experiment \
    "exp9_soft_appearance" \
    "Soft appearance (reduced contrast)" \
    0.8 -5 448

# Experiment 10: Ultra high detail
run_experiment \
    "exp10_ultra_detail" \
    "Ultra high detail (high resolution + enhanced)" \
    1.1 10 512

log "================================================================================"
log "${GREEN}All Experiments Completed!${NC}"
log "================================================================================"
log "Results directory: ${BASE_OUTPUT_DIR}"
log "Log file: ${LOG_FILE}"
log "================================================================================"
