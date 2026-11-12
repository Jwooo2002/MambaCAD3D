#!/bin/bash
# run_experiments_dataRoot.sh: main_files의 3D Mamba 모델로 데이터셋 경로 변경 실험

# ==========================================================
# 1. 고정 파라미터 설정
# ==========================================================
GPU_ID="0"
BASE_DATA_DIR="../main_files/dataset/v11" # 데이터셋 상위 디렉토리 (scripts 폴더 기준)
K_FOLDS=4
EPOCHS=1000
BATCH_SIZE=2 # 🎯 BATCH_SIZE를 2로 고정
LR=0.0001
MEM_DIM=300
IN_CHANS=1
SSM_D_STATE=8
FIXED_DEPTHS="2 2 2" # Depths 값
FIXED_DIMS="32 64 128" # Dims 값

# ==========================================================
# 2. 🎯 실험할 데이터셋 폴더 이름 설정
# ==========================================================
data_root_options=(
    "0.05_8_32_CTCA_32"
    "0.05_8_32_CTCA_48"
    "0.05_8_32_CTCA_64"
    "0.05_8_32_CTCA_80"
    "0.05_8_32_CTCA_96"
    "0.05_8_32_CTCA_128"
)

# ==========================================================
# 3. 실험 루프 실행
# ==========================================================
echo "### main_files (3D Mamba) - 데이터셋 경로 변경 실험 시작 ###"
echo "### Dims: ${FIXED_DIMS}, Depths: ${FIXED_DEPTHS}, Batch Size: ${BATCH_SIZE} ###"

# save_path의 상위 디렉토리를 구분하기 위해 새로운 디렉토리 이름 설정
base_save_dir="../results_data_root"

for data_folder in "${data_root_options[@]}"
do
    # 📁 run_name 및 data_root 경로 설정
    NORMAL_DATA_ROOT="${BASE_DATA_DIR}/${data_folder}/Normal"
    DISEASED_DATA_ROOT="${BASE_DATA_DIR}/${data_folder}/Diseased"

    # run_name 과 save_path 를 데이터 폴더 이름으로 자동 생성 (특수문자 및 . 를 _ 로 치환)
    # Note: `tr -d '_'` was mistakenly included in previous versions; removed it to ensure paths are correctly named.
    run_name="data_$(echo "${data_folder}" | tr '.' '_')"
    save_path="${base_save_dir}/${run_name}"

    echo "=========================================================="
    echo ">> 실험 시작: ${run_name}"
    echo ">> Data Folder: ${data_folder}"
    echo ">> Batch Size: ${BATCH_SIZE}, Dims: ${FIXED_DIMS}, Depths: ${FIXED_DEPTHS}"
    echo "=========================================================="

    mkdir -p ${save_path}

    PYTHONPATH=.. python ../main_files/main.py \
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
        --depths ${FIXED_DEPTHS} \
        --use_optimal_threshold \
        --threshold_method "accuracy"

    echo ">> 실험 ${run_name} 완료."
done

echo "🎉 모든 데이터셋 경로 변경 실험이 완료되었습니다."