#!/bin/bash
# run_experiments_linVol.sh: dims(linVol) 값을 변경하며 실험하는 스크립트

# ==========================================================
# 1. 고정 파라미터 설정
# ==========================================================
GPU_ID="1"
NORMAL_DATA_ROOT="./dataset/v11/0.05_8_32_CTCA_64/Normal"
DISEASED_DATA_ROOT="./dataset/v11/0.05_8_32_CTCA_64/Diseased"
K_FOLDS=4
EPOCHS=1000
BATCH_SIZE=2
LR=0.0001
MEM_DIM=300
IN_CHANS=1
SSM_D_STATE=8

# --- 이 실험에서는 DEPTHS를 '2 2 2'로 고정 ---
FIXED_DEPTHS="2 2 2"

# ==========================================================
# 2. 🎯 실험할 dims 조합 설정
# ==========================================================
dims_options=("16 32 64" "32 64 128" "64 128 256" "128 128 128")

# ==========================================================
# 3. 실험 루프 실행
# ==========================================================
echo "### Dims(linVol) 변경 실험을 시작합니다 (depths=${FIXED_DEPTHS} 고정) ###"

for dims_val in "${dims_options[@]}"
do
    # 📁 run_name 과 save_path 를 dims 값으로 자동 생성
    dims_str=$(echo "${dims_val}" | tr ' ' '_')
    run_name="dims_${dims_str}"
    save_path="./results_linVol_2/${run_name}"

    echo "=========================================================="
    echo ">> 실험 시작: ${run_name}"
    echo ">> Dims: ${dims_val}, Depths: ${FIXED_DEPTHS}"
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
        --dims ${dims_val} \
        --depths ${FIXED_DEPTHS} \
        --use_optimal_threshold \
        --threshold_method "accuracy"

    echo ">> 실험 ${run_name} 완료."
done

echo "🎉 모든 Dims(linVol) 실험이 완료되었습니다."