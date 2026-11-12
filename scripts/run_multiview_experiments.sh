#!/bin/bash

# Run multiview visualization with different gamma values
# Usage: bash run_multiview_experiments.sh

echo "=========================================="
echo "Running Multi-View Experiments with Different Gamma Values"
echo "=========================================="

# Backup original file
cp evaluate_and_visualize_multiview.py evaluate_and_visualize_multiview.py.backup

# Gamma values to test
GAMMA_VALUES=(1.0 1.5 1.8 2.0 2.5)

for gamma in "${GAMMA_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running with GAMMA = $gamma"
    echo "=========================================="

    # Set output directory name
    OUTPUT_DIR="./multiview_results_gamma${gamma}_exp01"

    echo "Output directory: $OUTPUT_DIR"

    # Restore from backup
    cp evaluate_and_visualize_multiview.py.backup evaluate_and_visualize_multiview.py

    # Modify GAMMA value (match any existing GAMMA value)
    sed -i "s/^GAMMA = [0-9.]\+/GAMMA = $gamma/g" evaluate_and_visualize_multiview.py

    # Modify OUTPUT_DIR (match any existing OUTPUT_DIR path)
    sed -i "s|^OUTPUT_DIR = \".*\"|OUTPUT_DIR = \"$OUTPUT_DIR\"|g" evaluate_and_visualize_multiview.py

    # Run the experiment
    python evaluate_and_visualize_multiview.py

    if [ $? -eq 0 ]; then
        echo "SUCCESS: Gamma = $gamma completed"
    else
        echo "FAILED: Gamma = $gamma"
    fi

    echo ""
done

# Restore original file
cp evaluate_and_visualize_multiview.py.backup evaluate_and_visualize_multiview.py

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved in:"
for gamma in "${GAMMA_VALUES[@]}"; do
    echo "  - multiview_results_gamma${gamma}_exp01/"
done

echo ""
echo "Original file restored."
