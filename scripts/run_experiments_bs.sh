#!/bin/bash
# run_experiments_batchsize_FIXED.sh: BATCH_SIZE 값을 변경하며 실험하는 스크립트 (dims/depths 인자 전달 오류 수정됨)

# ==========================================================
# 1. 고정 파라미터 설정
# ==========================================================
GPU_ID="1"
NORMAL_DATA_ROOT="./dataset/v11/0.05_16_32_CTCA_64/Normal"
DISEASED_DATA_ROOT="./dataset/v11/0.05_16_32_CTCA_64/Diseased"
K_FOLDS=4
EPOCHS=1000
LR=0.0001
MEM_DIM=300
IN_CHANS=1
SSM_D_STATE=8
FIXED_DEPTHS="2 2 2"

# --- 🎯 이 실험에서는 dims를 '32 64 128'로 고정 ---
FIXED_DIMS="32 64 128"

# ==========================================================
# 2. 🎯 실험할 BATCH_SIZE 설정
# ==========================================================
batch_size_options=(4 8) # 배치 사이즈 4와 8로 실험

# ==========================================================
# 3. 실험 루프 실행
# ==========================================================
echo "### BATCH_SIZE 변경 실험을 시작합니다 (dims=${FIXED_DIMS}, depths=${FIXED_DEPTHS} 고정) ###"

# save_path의 상위 디렉토리를 구분하기 위해 새로운 디렉토리 이름 설정
base_save_dir="./results_batchsize_32_64_128"
dims_str=$(echo "${FIXED_DIMS}" | tr ' ' '_')

for batch_size_val in "${batch_size_options[@]}"
do
    # 📁 run_name 과 save_path 를 batch_size 값으로 자동 생성
    run_name="dims_${dims_str}_bs_${batch_size_val}"
    save_path="${base_save_dir}/${run_name}"

    echo "=========================================================="
    echo ">> 실험 시작: ${run_name}"
    echo ">> Dims: ${FIXED_DIMS}, Batch Size: ${batch_size_val}, Depths: ${FIXED_DEPTHS}"
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
        --batch_size ${batch_size_val} \
        --lr ${LR} \
        --mem_dim ${MEM_DIM} \
        --in_chans ${IN_CHANS} \
        --ssm_d_state ${SSM_D_STATE} \
        --dims ${FIXED_DIMS} \
        --depths ${FIXED_DEPTHS} \
        --use_optimal_threshold \
        --threshold_method "accuracy"

    echo ">> 실험 ${run_name} 완료."
done

echo "🎉 모든 BATCH_SIZE 변경 실험이 완료되었습니다."