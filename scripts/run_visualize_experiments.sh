#!/bin/bash

################################################################################
# Multiple Visualization Experiments Script
################################################################################
# This script runs multiple visualization experiments with different parameter
# combinations. Each experiment saves to a separate output directory.
#
# Usage:
#   bash run_visualize_experiments.sh
#
# Features:
#   - Run multiple experiments with different alpha/beta/resolution settings
#   - Each experiment saves to a separate directory
#   - Progress tracking and timing for each experiment
#   - Summary report at the end
################################################################################

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

# Base data directory
DATA_DIR="./dataset/v11/0.05_8_32_CTCA_64/Diseased"

# Base output directory (each experiment will create a subdirectory)
BASE_OUTPUT_DIR="./visualizations_experiments"

# Specific files to process (leave empty for all files)
# Examples:
#   SPECIFIC_FILES=""                                          # All files
#   SPECIFIC_FILES="--specific_files D_10_patch.pkl"         # Only D_10
#   SPECIFIC_FILES="--specific_files D_10_patch.pkl D_11_patch.pkl"  # Multiple files
SPECIFIC_FILES=""  # Process ALL files

# Max patches per file (leave empty for all patches)
MAX_PATCHES=""  # Process ALL patches

# ============================================================================
# DEFINE EXPERIMENTS - OPTIMIZED FOR CCTA STENOSIS DETECTION
# ============================================================================
# Format: "experiment_name|alpha|beta|resolution|description"
# These parameters are optimized for visualizing stenosis in CCTA data

EXPERIMENTS=(
    # Standard brightness for baseline comparison
    "exp1_baseline|1.0|0|448|Baseline (original brightness, standard contrast)"

    # Enhanced vessel visualization (slightly brighter to see contrast-enhanced vessels)
    "exp2_vessel_enhanced|1.1|0|448|Vessel enhanced (brighter for contrast visualization)"

    # Increased contrast (positive beta to enhance bright structures like vessels)
    "exp3_high_contrast|1.0|15|448|High contrast (enhanced vessel brightness)"

    # Reduced background (negative beta to darken background, highlight vessels)
    "exp4_dark_bg|1.0|-15|448|Dark background (vessels stand out more)"

    # Moderate enhancement (balanced approach)
    "exp5_moderate|1.05|5|448|Moderate enhancement (balanced vessel-background)"

    # High resolution for detailed stenosis detection
    "exp6_highres_baseline|1.0|0|512|High resolution baseline (detailed structures)"

    # High resolution + vessel enhancement (best for stenosis detection)
    "exp7_highres_enhanced|1.1|10|512|High-res + enhanced (optimal for stenosis)"

    # Alternative contrast settings
    "exp8_soft_contrast|0.95|10|448|Soft contrast (gentle enhancement)"

    # Strong enhancement for subtle stenosis
    "exp9_strong_enhancement|1.15|15|512|Strong enhancement (subtle stenosis detection)"

    # Balanced high-quality setting
    "exp10_balanced_hq|1.05|8|512|Balanced high-quality (recommended)"
)

# Uncomment experiments you want to run, or define your own above
# You can also comment out experiments you don't want to run

# ============================================================================
# EXECUTION OPTIONS
# ============================================================================

# Run experiments in parallel (1) or sequentially (0)
# WARNING: Running in parallel uses more memory and CPU
PARALLEL=0

# Number of parallel jobs if PARALLEL=1 (careful with memory usage!)
NUM_PARALLEL_JOBS=2

# ============================================================================
# END OF CONFIGURABLE PARAMETERS
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Track overall start time
OVERALL_START=$(date +%s)

echo "================================================================================"
echo "Multiple Visualization Experiments"
echo "================================================================================"
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Base output directory: $BASE_OUTPUT_DIR"
echo "  Number of experiments: ${#EXPERIMENTS[@]}"
echo "  Execution mode: $([ $PARALLEL -eq 1 ] && echo 'Parallel' || echo 'Sequential')"
echo "  Max patches per file: ${MAX_PATCHES:-All}"
echo "================================================================================"
echo ""

# Create experiment log file
LOG_FILE="$BASE_OUTPUT_DIR/experiments_log.txt"
echo "Experiment Log - $(date)" > "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Function to run a single experiment
run_experiment() {
    local exp_config=$1

    # Parse experiment configuration
    IFS='|' read -r exp_name alpha beta resolution description <<< "$exp_config"

    # Create experiment output directory
    output_dir="$BASE_OUTPUT_DIR/$exp_name"
    mkdir -p "$output_dir"

    # Start timing
    start_time=$(date +%s)

    echo -e "${BLUE}[Experiment: $exp_name]${NC}"
    echo "  Description: $description"
    echo "  Parameters: alpha=$alpha, beta=$beta, resolution=$resolution"
    echo "  Output directory: $output_dir"
    echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"

    # Build command
    cmd="python visualize_diseased_data.py \
        --data_dir \"$DATA_DIR\" \
        --output_dir \"$output_dir\" \
        --alpha $alpha \
        --beta $beta \
        --resolution $resolution"

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
        echo "  Parameters: alpha=$alpha, beta=$beta, resolution=$resolution" >> "$LOG_FILE"
        echo "  Output: $output_dir" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"

        return 0
    else
        echo -e "  ${RED}✗ FAILED${NC} (Duration: ${duration}s)"
        echo -e "  Check log: $exp_log"

        # Log to main log file
        echo "[$exp_name] FAILED - Duration: ${duration}s" >> "$LOG_FILE"
        echo "  Parameters: alpha=$alpha, beta=$beta, resolution=$resolution" >> "$LOG_FILE"
        echo "  Error log: $exp_log" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"

        return 1
    fi
}

# Export function for parallel execution
export -f run_experiment
export DATA_DIR BASE_OUTPUT_DIR MAX_PATCHES SPECIFIC_FILES LOG_FILE
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

    # Use GNU parallel if available, otherwise use xargs
    if command -v parallel &> /dev/null; then
        printf "%s\n" "${EXPERIMENTS[@]}" | parallel -j $NUM_PARALLEL_JOBS run_experiment
    else
        printf "%s\n" "${EXPERIMENTS[@]}" | xargs -P $NUM_PARALLEL_JOBS -I {} bash -c 'run_experiment "$@"' _ {}
    fi

    # Count results (approximate, since parallel execution makes tracking harder)
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

# Create a quick comparison HTML file (optional)
create_comparison_html() {
    html_file="$BASE_OUTPUT_DIR/experiments_comparison.html"

    cat > "$html_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Visualization Experiments Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .experiment {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .experiment h2 { color: #2196F3; margin-top: 0; }
        .params {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
        }
        img { max-width: 100%; height: auto; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Visualization Experiments Comparison</h1>
    <p>Generated: $(date '+%Y-%m-%d %H:%M:%S')</p>
EOF

    for exp_config in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_name alpha beta resolution description <<< "$exp_config"

        cat >> "$html_file" << EOF
    <div class="experiment">
        <h2>$exp_name</h2>
        <p><strong>Description:</strong> $description</p>
        <div class="params">
            Alpha: $alpha | Beta: $beta | Resolution: $resolution
        </div>
        <p><strong>Output Directory:</strong> $exp_name/</p>
    </div>
EOF
    done

    cat >> "$html_file" << 'EOF'
</body>
</html>
EOF

    echo "Comparison HTML created: $html_file"
}

create_comparison_html

echo ""
echo -e "${GREEN}All experiments completed!${NC}"
echo "Open $BASE_OUTPUT_DIR/experiments_comparison.html to view experiment details"
