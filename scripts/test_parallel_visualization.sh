#!/bin/bash

################################################################################
# Test Parallel Visualization (Quick Test)
################################################################################
# This script runs a quick test of the parallel visualization on a subset
# of data to verify everything works before running full experiments.
#
# Usage:
#   bash test_parallel_visualization.sh
################################################################################

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Testing Parallel Visualization${NC}"
echo -e "${BLUE}================================================================================${NC}"

DATA_DIR="./dataset/v11/0.05_8_32_CTCA_64/Diseased"
TEST_OUTPUT_DIR="./test_parallel_output"
NUM_WORKERS=4  # Use fewer workers for testing

# Clean up previous test output
if [ -d "$TEST_OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Cleaning up previous test output...${NC}"
    rm -rf "$TEST_OUTPUT_DIR"
fi

mkdir -p "$TEST_OUTPUT_DIR"

echo ""
echo -e "${BLUE}Test 1: visualize_diseased_data_parallel.py${NC}"
echo -e "  Using 2 specific files, $NUM_WORKERS workers"
echo ""

python visualize_diseased_data_parallel.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$TEST_OUTPUT_DIR/diseased_test" \
    --specific_files D_10_patch.pkl D_11_patch.pkl \
    --alpha 1.0 \
    --beta 0 \
    --resolution 448 \
    --num_workers $NUM_WORKERS

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test 1 PASSED${NC}"
else
    echo -e "${RED}✗ Test 1 FAILED${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Test 2: visualize_ccta_window_level_parallel.py${NC}"
echo -e "  Using 2 specific files, $NUM_WORKERS workers"
echo ""

python visualize_ccta_window_level_parallel.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$TEST_OUTPUT_DIR/ccta_test" \
    --specific_files D_10_patch.pkl D_11_patch.pkl \
    --window 500 \
    --level 100 \
    --resolution 448 \
    --num_workers $NUM_WORKERS

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test 2 PASSED${NC}"
else
    echo -e "${RED}✗ Test 2 FAILED${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}All Tests PASSED!${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo -e "Test output saved to: $TEST_OUTPUT_DIR"
echo ""
echo -e "${YELLOW}You can now run the full experiments:${NC}"
echo -e "  bash run_visualize_experiments_parallel.sh"
echo -e "  bash run_visualize_ccta_experiments_parallel.sh"
echo -e "${BLUE}================================================================================${NC}"
