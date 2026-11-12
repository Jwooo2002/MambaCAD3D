#!/bin/bash
# run_experiments_depths.sh: depths 값을 변경하며 실험하는 스크립트

# ==========================================================
# 1. 고정 파라미터 설정
# ==========================================================
GPU_ID="0"
NORMAL_DATA_ROOT="./dataset/v11/0.05_8_32_CTCA_64/Normal"
DISEASED_DATA_ROOT="./dataset/v11/0.05_8_32_CTCA_64/Diseased"
K_FOLDS=4
EPOCHS=1000
BATCH_SIZE=2
LR=0.0001
MEM_DIM=300
IN_CHANS=1
SSM_D_STATE=8

# --- 이 실험에서는 DIMS를 '32 64 128'로 고정 ---
FIXED_DIMS="32 64 128"

# ==========================================================
# 2. 🎯 실험할 depths 조합 설정
# ==========================================================
depths_options=("1 1 1" "2 2 2" "3 3 3" "3 2 1")

# ==========================================================
# 3. 실험 루프 실행
# ==========================================================
echo "### Depths 변경 실험을 시작합니다 (dims=${FIXED_DIMS} 고정) ###"

for depths_val in "${depths_options[@]}"
do
    # 📁 run_name 과 save_path 를 depths 값으로 자동 생성
    depths_str=$(echo "${depths_val}" | tr ' ' '_')
    run_name="depths_${depths_str}"
    save_path="./results_depths_2/${run_name}"

    echo "=========================================================="
    echo ">> 실험 시작: ${run_name}"
    echo ">> Dims: ${FIXED_DIMS}, Depths: ${depths_val}"
    echo "=========================================================="

    mkdir -p ${save_path}

    PYTHONPATH=. python main.py \
        --normal_data_root "${NORMAL_DATA_ROOT}" \
        --diseased_data_root "${DISEASED_DATA_ROOT}" \
        --gpu_id "${GPU_ID}" \
        --run_name "${run_name}" \
        --save_path "${save_path}" \
        --k_folds ${K_FOLDS} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --mem_dim ${MEM_DIM} \
        --in_chans ${IN_CHANS} \
        --ssm_d_state ${SSM_D_STATE} \
        --dims ${FIXED_DIMS} \
        --depths ${depths_val} \
        --use_optimal_threshold \
        --threshold_method "accuracy"

    echo ">> 실험 ${run_name} 완료."
done

echo "🎉 모든 Depths 실험이 완료되었습니다."