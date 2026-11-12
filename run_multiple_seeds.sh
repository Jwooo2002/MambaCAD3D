#!/bin/bash
# Multiple Seeds Experiment Runner (Shell Script Version)

# Configuration
SEEDS=(42 123 456 789 2024)
GPU_ID="1"
BASE_RUN_NAME="test_8x8_patch_2"
BASE_SAVE_PATH="../multiple_seeds_res"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================="
echo "Multiple Seeds Experiment Runner"
echo "======================================================================="
echo "Seeds: ${SEEDS[@]}"
echo "GPU: ${GPU_ID}"
echo "Base save path: ${BASE_SAVE_PATH}"
echo "======================================================================="
echo ""

# Create base results directory
mkdir -p "${BASE_SAVE_PATH}"

# Track results
SUCCESS_COUNT=0
FAILED_COUNT=0
START_TIME=$(date +%s)

# Run experiments for each seed
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "======================================================================="
    echo -e "${BLUE}Starting experiment with seed: ${seed}${NC}"
    echo "======================================================================="

    SAVE_PATH="${BASE_SAVE_PATH}/seed_${seed}"
    RUN_NAME="${BASE_RUN_NAME}_seed${seed}"

    # Run the experiment
    cd main_files
    python main.py \
        --normal_data_root "../main_files/dataset/v11/0.05_8_8_CTCA_64/Normal" \
        --diseased_data_root "../main_files/dataset/v11/0.05_8_8_CTCA_64/Diseased" \
        --run_name "${RUN_NAME}" \
        --save_path "${SAVE_PATH}" \
        --gpu_id "${GPU_ID}" \
        --num_workers 1 \
        --k_folds 4 \
        --confidence_level 0.95 \
        --random_seed ${seed} \
        --epochs 1000 \
        --batch_size 2 \
        --lr 0.0001 \
        --eta_min 1e-06 \
        --entropy_loss_weight 0.0002 \
        --early_stopping_patience 10 \
        --in_chans 1 \
        --dims 32 64 128 \
        --depths 2 2 2 \
        --ssm_d_state 8 \
        --mem_dim 300 \
        --anomaly_threshold 0.01 \
        --use_optimal_threshold \
        --threshold_method "accuracy" \
        --threshold_percentile 95

    # Check if experiment succeeded
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Experiment with seed ${seed} completed successfully!${NC}"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}✗ Experiment with seed ${seed} failed!${NC}"
        ((FAILED_COUNT++))
    fi

    cd ..
done

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Print summary
echo ""
echo "======================================================================="
echo "EXPERIMENT SUMMARY"
echo "======================================================================="
echo "Total experiments: ${#SEEDS[@]}"
echo -e "${GREEN}Successful: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}Failed: ${FAILED_COUNT}${NC}"
echo "Duration: ${DURATION} seconds"
echo "Results directory: ${BASE_SAVE_PATH}"
echo "======================================================================="
