#!/bin/bash

################################################################################
# Diseased Data Visualization Script
################################################################################
# This script runs the diseased data visualization with configurable parameters
#
# Usage:
#   bash run_visualize_diseased.sh
#
# Modify the parameters below to customize the visualization
################################################################################

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# Data directory (where the diseased .pkl files are located)
DATA_DIR="./dataset/v11/0.05_8_32_CTCA_64/Diseased"

# Output directory (where visualizations will be saved)
OUTPUT_DIR="./diseased_data_visualizations_F"

# Brightness adjustment (0.0 - 2.0)
# Lower values make images darker, higher values make them brighter
# Default: 0.8 (darker than original)
# Examples:
#   0.5 = very dark
#   0.8 = moderately dark (default)
#   1.0 = original brightness
#   1.5 = brighter
ALPHA=0.8

# Brightness offset (-100 to 100)
# Adds or subtracts a constant value to all pixels
# Default: 0 (no offset)
# Examples:
#   -50 = darker
#   0 = no change (default)
#   50 = brighter
BETA=0

# Resolution for each slice
# Higher values = better quality but larger file sizes
# Default: 448
# Examples:
#   256 = lower quality, smaller files
#   448 = good quality (default)
#   512 = high quality, larger files
RESOLUTION=448

# Maximum number of patches to visualize per file
# Set to empty string or remove to visualize all patches
# Examples:
#   MAX_PATCHES=5     # Only visualize 5 patches per patient
#   MAX_PATCHES=""    # Visualize all patches
MAX_PATCHES=""

# Specific files to visualize
# Leave empty to visualize all files, or specify specific files
# Examples:
#   SPECIFIC_FILES=""                                    # All files
#   SPECIFIC_FILES="--specific_files D_10_patch.pkl"    # Only D_10
#   SPECIFIC_FILES="--specific_files D_10_patch.pkl D_11_patch.pkl"  # D_10 and D_11
SPECIFIC_FILES=""

# ============================================================================
# END OF CONFIGURABLE PARAMETERS
# ============================================================================

echo "================================================================================"
echo "Diseased Data Visualization"
echo "================================================================================"
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Alpha (brightness): $ALPHA"
echo "  Beta (offset): $BETA"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  Max patches per file: ${MAX_PATCHES:-All}"
echo "================================================================================"
echo ""

# Build the command
CMD="python visualize_diseased_data.py \
    --data_dir \"$DATA_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --alpha $ALPHA \
    --beta $BETA \
    --resolution $RESOLUTION"

# Add max_patches if specified
if [ -n "$MAX_PATCHES" ]; then
    CMD="$CMD --max_patches $MAX_PATCHES"
fi

# Add specific_files if specified
if [ -n "$SPECIFIC_FILES" ]; then
    CMD="$CMD $SPECIFIC_FILES"
fi

# Display the command being run
echo "Running command:"
echo "$CMD"
echo ""
echo "================================================================================"
echo ""

# Execute the command
eval $CMD

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "Visualization completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "Error: Visualization failed!"
    echo "================================================================================"
    exit 1
fi
