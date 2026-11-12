#!/bin/bash

################################################################################
# CCTA Window/Level Visualization Experiments
################################################################################
# Multiple experiments with Window/Level settings optimized for CCTA stenosis detection
#
# DATA RANGE: [-300, 400] HU (pre-clipped)
# - Min: -300 HU (air/background)
# - Max: 400 HU (contrast-enhanced vessels)
# - Total: 700 HU range
#
# Usage:
#   bash run_visualize_ccta_experiments.sh
################################################################################

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

# Base data directory
DATA_DIR="./dataset/v11/0.05_8_32_CTCA_64/Diseased"

# Base output directory
BASE_OUTPUT_DIR="./ccta_window_level_experiments"

# Specific files to process (leave empty for all files)
SPECIFIC_FILES=""

# Max patches per file (leave empty for all patches)
MAX_PATCHES=""

# Resolution
RESOLUTION=448

# Colormap
COLORMAP="gray"

# ============================================================================
# DEFINE EXPERIMENTS - OPTIMIZED FOR [-300, 400] HU RANGE
# ============================================================================
# Format: "experiment_name|window_width|window_level|description"
#
# Window/Level Guide for [-300, 400] HU range:
# - Full range: W=700, L=50 (displays -300 to 400)
# - Vessel focus: W=400, L=200 (displays 0 to 400)
# - High contrast: W=300, L=150 (displays 0 to 300)
# - Background removal: W=500, L=250 (displays 0 to 500, clips at 400)

EXPERIMENTS=(
    # Baseline comparisons
    "exp01_full_range|700|50|Full HU range [-300, 400]"
    "exp02_standard|500|100|Standard balanced view [-150, 350]"

    # Vessel-focused (positive HU values, where contrast is)
    "exp03_vessel_only|400|200|Vessel-only focus [0, 400]"
    "exp04_vessel_narrow|300|250|Narrow vessel focus [100, 400]"
    "exp05_vessel_high_contrast|350|225|High contrast vessels [50, 400]"

    # Stenosis detection optimized
    "exp06_stenosis_wide|450|175|Wide stenosis detection [-50, 400]"
    "exp07_stenosis_focused|400|150|Focused stenosis detection [-50, 350]"
    "exp08_stenosis_tight|350|175|Tight stenosis contrast [0, 350]"

    # Background suppression (darker background, brighter vessels)
    "exp09_bg_suppress_mild|500|200|Mild background suppression [-50, 450]"
    "exp10_bg_suppress_strong|450|225|Strong background suppression [0, 450]"
    "exp11_bg_suppress_extreme|400|250|Extreme background suppression [50, 450]"

    # Fine-tuned for subtle differences
    "exp12_subtle_low|350|125|Subtle differences (low range) [-50, 300]"
    "exp13_subtle_mid|350|175|Subtle differences (mid range) [0, 350]"
    "exp14_subtle_high|350|225|Subtle differences (high range) [50, 400]"

    # Wide ranges for overview
    "exp15_overview_soft|600|100|Soft overview [-200, 400]"
    "exp16_overview_balanced|550|125|Balanced overview [-150, 400]"

    # Narrow ranges for specific features
    "exp17_narrow_low|250|125|Narrow low [0, 250]"
    "exp18_narrow_mid|250|175|Narrow mid [50, 300]"
    "exp19_narrow_high|250|275|Narrow high [150, 400]"

    # Recommended best settings (based on medical standards)
    "exp20_recommended|500|150|Recommended CCTA setting [-100, 400]"
)

# ============================================================================
# EXECUTION OPTIONS
# ============================================================================

# Run experiments in parallel (1) or sequentially (0)
PARALLEL=0

# Number of parallel jobs if PARALLEL=1
NUM_PARALLEL_JOBS=2

# ============================================================================
# END OF CONFIGURABLE PARAMETERS
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Track overall start time
OVERALL_START=$(date +%s)

echo "================================================================================"
echo "CCTA Window/Level Visualization Experiments"
echo "================================================================================"
echo "Data HU Range: [-300, 400] (pre-clipped)"
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Base output directory: $BASE_OUTPUT_DIR"
echo "  Number of experiments: ${#EXPERIMENTS[@]}"
echo "  Execution mode: $([ $PARALLEL -eq 1 ] && echo 'Parallel' || echo 'Sequential')"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  Max patches per file: ${MAX_PATCHES:-All}"
echo "================================================================================"
echo ""

# Create experiment log file
LOG_FILE="$BASE_OUTPUT_DIR/experiments_log.txt"
echo "CCTA Window/Level Experiment Log - $(date)" > "$LOG_FILE"
echo "Data HU Range: [-300, 400] (pre-clipped)" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Function to run a single experiment
run_experiment() {
    local exp_config=$1

    # Parse experiment configuration
    IFS='|' read -r exp_name window_width window_level description <<< "$exp_config"

    # Calculate display range
    min_display=$(awk "BEGIN {print $window_level - $window_width / 2}")
    max_display=$(awk "BEGIN {print $window_level + $window_width / 2}")

    # Create experiment output directory
    output_dir="$BASE_OUTPUT_DIR/$exp_name"
    mkdir -p "$output_dir"

    # Start timing
    start_time=$(date +%s)

    echo -e "${BLUE}[Experiment: $exp_name]${NC}"
    echo "  Description: $description"
    echo "  Window/Level: W=$window_width, L=$window_level"
    echo "  Display Range: [$min_display, $max_display] HU"
    echo "  Output directory: $output_dir"
    echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"

    # Build command
    cmd="python visualize_ccta_window_level.py \
        --data_dir \"$DATA_DIR\" \
        --output_dir \"$output_dir\" \
        --window $window_width \
        --level $window_level \
        --resolution $RESOLUTION \
        --colormap $COLORMAP"

    # Add max_patches if specified
    if [ -n "$MAX_PATCHES" ]; then
        cmd="$cmd --max_patches $MAX_PATCHES"
    fi

    # Add specific_files if specified
    if [ -n "$SPECIFIC_FILES" ]; then
        cmd="$cmd $SPECIFIC_FILES"
    fi

    # Execute command and capture output
    exp_log="$output_dir/experiment_log.txt"
    eval $cmd > "$exp_log" 2>&1
    exit_code=$?

    # End timing
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Report results
    if [ $exit_code -eq 0 ]; then
        echo -e "  ${GREEN}✓ SUCCESS${NC} (Duration: ${duration}s)"
        echo -e "  Results saved to: $output_dir"

        # Log to main log file
        echo "[$exp_name] SUCCESS - Duration: ${duration}s" >> "$LOG_FILE"
        echo "  W/L: $window_width/$window_level | Range: [$min_display, $max_display] HU" >> "$LOG_FILE"
        echo "  Output: $output_dir" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"

        return 0
    else
        echo -e "  ${RED}✗ FAILED${NC} (Duration: ${duration}s)"
        echo -e "  Check log: $exp_log"

        # Log to main log file
        echo "[$exp_name] FAILED - Duration: ${duration}s" >> "$LOG_FILE"
        echo "  W/L: $window_width/$window_level" >> "$LOG_FILE"
        echo "  Error log: $exp_log" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"

        return 1
    fi
}

# Export function for parallel execution
export -f run_experiment
export DATA_DIR BASE_OUTPUT_DIR MAX_PATCHES SPECIFIC_FILES RESOLUTION COLORMAP LOG_FILE
export RED GREEN YELLOW BLUE NC

# Run experiments
echo "================================================================================"
echo "Running Experiments"
echo "================================================================================"
echo ""

success_count=0
failed_count=0

if [ $PARALLEL -eq 1 ]; then
    # Parallel execution
    echo -e "${YELLOW}Running experiments in parallel (max $NUM_PARALLEL_JOBS jobs)...${NC}"
    echo ""

    if command -v parallel &> /dev/null; then
        printf "%s\n" "${EXPERIMENTS[@]}" | parallel -j $NUM_PARALLEL_JOBS run_experiment
    else
        printf "%s\n" "${EXPERIMENTS[@]}" | xargs -P $NUM_PARALLEL_JOBS -I {} bash -c 'run_experiment "$@"' _ {}
    fi

    success_count=$(grep "SUCCESS" "$LOG_FILE" | wc -l)
    failed_count=$(grep "FAILED" "$LOG_FILE" | wc -l)

else
    # Sequential execution
    echo -e "${YELLOW}Running experiments sequentially...${NC}"
    echo ""

    for exp_config in "${EXPERIMENTS[@]}"; do
        run_experiment "$exp_config"
        if [ $? -eq 0 ]; then
            ((success_count++))
        else
            ((failed_count++))
        fi
        echo ""
    done
fi

# Calculate overall duration
OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))

# Print summary
echo "================================================================================"
echo "Experiment Summary"
echo "================================================================================"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo -e "Successful: ${GREEN}$success_count${NC}"
echo -e "Failed: ${RED}$failed_count${NC}"
echo "Overall duration: ${OVERALL_DURATION}s ($(($OVERALL_DURATION / 60))m $(($OVERALL_DURATION % 60))s)"
echo "Results directory: $BASE_OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "================================================================================"

# Add summary to log file
echo "" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"
echo "SUMMARY" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"
echo "Total experiments: ${#EXPERIMENTS[@]}" >> "$LOG_FILE"
echo "Successful: $success_count" >> "$LOG_FILE"
echo "Failed: $failed_count" >> "$LOG_FILE"
echo "Overall duration: ${OVERALL_DURATION}s" >> "$LOG_FILE"
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"

echo ""
echo -e "${GREEN}All experiments completed!${NC}"
echo "Check $LOG_FILE for detailed results"
