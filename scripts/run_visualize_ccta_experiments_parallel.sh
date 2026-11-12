#!/bin/bash

################################################################################
# PARALLEL CCTA Window/Level Visualization Experiments Runner
################################################################################
# This script runs multiple CCTA visualization experiments with different
# Window/Level settings using PARALLEL processing for maximum speed.
#
# IMPORTANT: This data has been pre-clipped to [-300, 400] HU range
# - Min HU: -300 (air/background)
# - Max HU: 400 (contrast-enhanced vessels)
# - Total range: 700 HU
#
# Features:
# - Multi-core parallel processing (up to 20x faster)
# - Multiple Window/Level configurations
# - Medical imaging standard W/L parameters
# - Organized output directories per experiment
# - Comprehensive logging
#
# Usage:
#   bash run_visualize_ccta_experiments_parallel.sh
################################################################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="./dataset/v11/0.05_8_32_CTCA_64/Diseased"
BASE_OUTPUT_DIR="./ccta_window_level_experiments_parallel"
NUM_WORKERS=20  # Number of parallel workers
RESOLUTION=448
COLORMAP="gray"
LOG_FILE="ccta_experiments_parallel.log"

# Clear previous log
> "$LOG_FILE"

# Function to log messages
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Function to calculate display range from Window/Level
calculate_range() {
    local window=$1
    local level=$2
    local min_hu=$(echo "$level - $window / 2" | bc)
    local max_hu=$(echo "$level + $window / 2" | bc)
    echo "[$min_hu, $max_hu]"
}

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local description=$2
    local window=$3
    local level=$4

    local output_dir="${BASE_OUTPUT_DIR}/${exp_name}"
    local display_range=$(calculate_range $window $level)

    log "${BLUE}[Experiment: ${exp_name}]${NC}"
    log "  Description: ${description}"
    log "  Window/Level: W=${window}, L=${level}"
    log "  Display Range: ${display_range} HU"
    log "  Workers: ${NUM_WORKERS}"
    log "  Output directory: ${output_dir}"

    local start_time=$(date +%s)
    log "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"

    # Run the parallel visualization
    python visualize_ccta_window_level_parallel.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$output_dir" \
        --window "$window" \
        --level "$level" \
        --resolution "$RESOLUTION" \
        --colormap "$COLORMAP" \
        --num_workers "$NUM_WORKERS" \
        >> "$LOG_FILE" 2>&1

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        log "  ${GREEN}✓ SUCCESS${NC} (Duration: ${duration}s, $((duration/60))m)"
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
log "CCTA Window/Level Visualization Experiments (PARALLEL)"
log "================================================================================"
log "Data HU Range: [-300, 400] (pre-clipped)"
log "Configuration:"
log "  Data directory: ${DATA_DIR}"
log "  Base output directory: ${BASE_OUTPUT_DIR}"
log "  Number of parallel workers: ${NUM_WORKERS}"
log "  Number of experiments: 20"
log "  Execution mode: Sequential (but each uses parallel processing)"
log "  Resolution: ${RESOLUTION}x${RESOLUTION}"
log "  Max patches per file: All"
log "================================================================================"
log ""

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

log "================================================================================"
log "Running Experiments"
log "================================================================================"
log ""

# Define experiments with medical imaging standard Window/Level settings
# Format: experiment_name "description" window level

# Experiment 01: Full HU range
run_experiment \
    "exp01_full_range" \
    "Full HU range [-300, 400]" \
    700 50

# Experiment 02: Standard balanced view
run_experiment \
    "exp02_standard" \
    "Standard balanced view [-150, 350]" \
    500 100

# Experiment 03: Vessel-only focus
run_experiment \
    "exp03_vessel_only" \
    "Vessel-only focus [0, 400]" \
    400 200

# Experiment 04: Narrow vessel focus
run_experiment \
    "exp04_vessel_narrow" \
    "Narrow vessel focus [100, 400]" \
    300 250

# Experiment 05: Soft tissue window
run_experiment \
    "exp05_soft_tissue" \
    "Soft tissue window [-100, 200]" \
    300 50

# Experiment 06: Lung window
run_experiment \
    "exp06_lung_window" \
    "Lung window adapted [-300, 100]" \
    400 -100

# Experiment 07: Bone window
run_experiment \
    "exp07_bone_window" \
    "Bone/calcification window [100, 400]" \
    300 250

# Experiment 08: Wide soft tissue
run_experiment \
    "exp08_wide_soft" \
    "Wide soft tissue [-150, 250]" \
    400 50

# Experiment 09: Angiography optimized
run_experiment \
    "exp09_angio_optimal" \
    "Angiography optimized [50, 350]" \
    300 200

# Experiment 10: High contrast vessels
run_experiment \
    "exp10_high_contrast" \
    "High contrast vessels [150, 400]" \
    250 275

# Experiment 11: Medium vessels
run_experiment \
    "exp11_medium_vessels" \
    "Medium vessel detail [0, 300]" \
    300 150

# Experiment 12: Background reduced
run_experiment \
    "exp12_bg_reduced" \
    "Background reduced [-50, 350]" \
    400 150

# Experiment 13: Bright vessels
run_experiment \
    "exp13_bright_vessels" \
    "Bright vessels only [200, 400]" \
    200 300

# Experiment 14: All tissue types
run_experiment \
    "exp14_all_tissues" \
    "All tissue types visible [-200, 300]" \
    500 50

# Experiment 15: Narrow contrast range
run_experiment \
    "exp15_narrow_contrast" \
    "Narrow contrast enhanced [100, 300]" \
    200 200

# Experiment 16: Ultra-wide view
run_experiment \
    "exp16_ultra_wide" \
    "Ultra-wide HU range [-300, 400]" \
    700 50

# Experiment 17: Calcification focus
run_experiment \
    "exp17_calcification" \
    "Calcification focus [200, 400]" \
    200 300

# Experiment 18: Mixed tissue optimal
run_experiment \
    "exp18_mixed_optimal" \
    "Mixed tissue optimal [-100, 300]" \
    400 100

# Experiment 19: Standard CCTA protocol
run_experiment \
    "exp19_standard_ccta" \
    "Standard CCTA protocol [0, 350]" \
    350 175

# Experiment 20: Vessel wall detail
run_experiment \
    "exp20_vessel_wall" \
    "Vessel wall detail [50, 250]" \
    200 150

log "================================================================================"
log "${GREEN}All CCTA Experiments Completed!${NC}"
log "================================================================================"
log "Results directory: ${BASE_OUTPUT_DIR}"
log "Log file: ${LOG_FILE}"
log "================================================================================"
