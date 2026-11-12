#!/bin/bash
# run_all_experiments.sh: 모든 실험을 통합 실행하는 스크립트
# Usage: ./run_all_experiments.sh [dataroot|states|depths|linvol|all]

# ==========================================================
# 공통 고정 파라미터 설정
# ==========================================================
GPU_ID="0"
BASE_DATA_DIR="../main_files/dataset/v11"
DEFAULT_NORMAL_DATA_ROOT="${BASE_DATA_DIR}/0.05_8_32_CTCA_64/Normal"
DEFAULT_DISEASED_DATA_ROOT="${BASE_DATA_DIR}/0.05_8_32_CTCA_64/Diseased"
K_FOLDS=4
EPOCHS=1000
BATCH_SIZE=2
LR=0.0001
MEM_DIM=300
IN_CHANS=1

# ==========================================================
# 실험 타입별 설정
# ==========================================================

# 1. DataRoot 실험 (데이터셋 경로 변경)
dataroot_options=(
    "0.05_8_32_CTCA_32"
    "0.05_8_32_CTCA_48"
    "0.05_8_32_CTCA_64"
    "0.05_8_32_CTCA_80"
    "0.05_8_32_CTCA_96"
    "0.05_8_32_CTCA_128"
)

# 2. States 실험 (ssm_d_state 변경)
states_options=(8 16 32 64)

# 3. Depths 실험 (depths 변경)
depths_options=("1 1 1" "2 2 2" "3 3 3")

# 4. LinVol 실험 (dims 변경)
linvol_options=("16 32 64" "32 64 128" "64 128 256" "128 128 128")

# 고정값
FIXED_DIMS="32 64 128"
FIXED_DEPTHS="2 2 2"
FIXED_SSM_STATE=8

# ==========================================================
# 실험 함수 정의
# ==========================================================

run_dataroot_experiments() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  📁 EXPERIMENT 1: DataRoot (데이터셋 경로 변경)           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"

    base_save_dir="../results_dataroot_ker3"

    for data_folder in "${dataroot_options[@]}"
    do
        NORMAL_DATA_ROOT="${BASE_DATA_DIR}/${data_folder}/Normal"
        DISEASED_DATA_ROOT="${BASE_DATA_DIR}/${data_folder}/Diseased"

        run_name="data_$(echo "${data_folder}" | tr '.' '_')"
        save_path="${base_save_dir}/${run_name}"

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 실험: ${run_name}"
        echo "   Data: ${data_folder}"
        echo "   Dims: ${FIXED_DIMS} | Depths: ${FIXED_DEPTHS} | State: ${FIXED_SSM_STATE}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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
            --ssm_d_state ${FIXED_SSM_STATE} \
            --dims ${FIXED_DIMS} \
            --depths ${FIXED_DEPTHS} \
            --use_optimal_threshold \
            --threshold_method "accuracy"

        echo "✅ 완료: ${run_name}"
    done

    echo ""
    echo "🎉 DataRoot 실험 완료!"
}

run_states_experiments() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  🔢 EXPERIMENT 2: States (SSM D_State 변경)              ║"
    echo "╚═══════════════════════════════════════════════════════════╝"

    base_save_dir="../results_all/results_states"

    for ssm_val in "${states_options[@]}"
    do
        run_name="state_${ssm_val}"
        save_path="${base_save_dir}/${run_name}"

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 실험: ${run_name}"
        echo "   SSM State: ${ssm_val}"
        echo "   Dims: ${FIXED_DIMS} | Depths: ${FIXED_DEPTHS}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        mkdir -p ${save_path}

        PYTHONPATH=.. python ../main_files/main.py \
            --normal_data_root "${DEFAULT_NORMAL_DATA_ROOT}" \
            --diseased_data_root "${DEFAULT_DISEASED_DATA_ROOT}" \
            --gpu_id "${GPU_ID}" \
            --run_name "${run_name}" \
            --save_path "${save_path}" \
            --k_folds ${K_FOLDS} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --mem_dim ${MEM_DIM} \
            --in_chans ${IN_CHANS} \
            --ssm_d_state ${ssm_val} \
            --dims ${FIXED_DIMS} \
            --depths ${FIXED_DEPTHS} \
            --use_optimal_threshold \
            --threshold_method "accuracy"

        echo "✅ 완료: ${run_name}"
    done

    echo ""
    echo "🎉 States 실험 완료!"
}

run_depths_experiments() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  📊 EXPERIMENT 3: Depths (네트워크 깊이 변경)             ║"
    echo "╚═══════════════════════════════════════════════════════════╝"

    base_save_dir="../results_all/results_depths"

    for depths_val in "${depths_options[@]}"
    do
        depths_str=$(echo "${depths_val}" | tr ' ' '_')
        run_name="depths_${depths_str}"
        save_path="${base_save_dir}/${run_name}"

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 실험: ${run_name}"
        echo "   Depths: ${depths_val}"
        echo "   Dims: ${FIXED_DIMS} | State: ${FIXED_SSM_STATE}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        mkdir -p ${save_path}

        PYTHONPATH=.. python ../main_files/main.py \
            --normal_data_root "${DEFAULT_NORMAL_DATA_ROOT}" \
            --diseased_data_root "${DEFAULT_DISEASED_DATA_ROOT}" \
            --gpu_id "${GPU_ID}" \
            --run_name "${run_name}" \
            --save_path "${save_path}" \
            --k_folds ${K_FOLDS} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --mem_dim ${MEM_DIM} \
            --in_chans ${IN_CHANS} \
            --ssm_d_state ${FIXED_SSM_STATE} \
            --dims ${FIXED_DIMS} \
            --depths ${depths_val} \
            --use_optimal_threshold \
            --threshold_method "accuracy"

        echo "✅ 완료: ${run_name}"
    done

    echo ""
    echo "🎉 Depths 실험 완료!"
}

run_linvol_experiments() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  📐 EXPERIMENT 4: LinVol (채널 차원 변경)                 ║"
    echo "╚═══════════════════════════════════════════════════════════╝"

    base_save_dir="../results_all/results_linvol"

    for dims_val in "${linvol_options[@]}"
    do
        dims_str=$(echo "${dims_val}" | tr ' ' '_')
        run_name="dims_${dims_str}"
        save_path="${base_save_dir}/${run_name}"

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 실험: ${run_name}"
        echo "   Dims: ${dims_val}"
        echo "   Depths: ${FIXED_DEPTHS} | State: ${FIXED_SSM_STATE}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        mkdir -p ${save_path}

        PYTHONPATH=.. python ../main_files/main.py \
            --normal_data_root "${DEFAULT_NORMAL_DATA_ROOT}" \
            --diseased_data_root "${DEFAULT_DISEASED_DATA_ROOT}" \
            --gpu_id "${GPU_ID}" \
            --run_name "${run_name}" \
            --save_path "${save_path}" \
            --k_folds ${K_FOLDS} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --mem_dim ${MEM_DIM} \
            --in_chans ${IN_CHANS} \
            --ssm_d_state ${FIXED_SSM_STATE} \
            --dims ${dims_val} \
            --depths ${FIXED_DEPTHS} \
            --use_optimal_threshold \
            --threshold_method "accuracy"

        echo "✅ 완료: ${run_name}"
    done

    echo ""
    echo "🎉 LinVol 실험 완료!"
}

# ==========================================================
# Main 실행 로직
# ==========================================================

print_usage() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔬 통합 실험 스크립트"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "사용법: ./run_all_experiments.sh [옵션]"
    echo ""
    echo "옵션:"
    echo "  dataroot   - 데이터셋 경로 변경 실험만 실행"
    echo "  states     - SSM D_State 변경 실험만 실행"
    echo "  depths     - Depths 변경 실험만 실행"
    echo "  linvol     - Dims(LinVol) 변경 실험만 실행"
    echo "  all        - 모든 실험 순차 실행 (기본값)"
    echo ""
    echo "예시:"
    echo "  ./run_all_experiments.sh dataroot"
    echo "  ./run_all_experiments.sh all"
    echo ""
}

# 시작 시간 기록
START_TIME=$(date +%s)

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║        🧪 3D Mamba MemAE 통합 실험 시작                  ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "⚙️  설정:"
echo "   GPU: ${GPU_ID}"
echo "   K-Folds: ${K_FOLDS}"
echo "   Epochs: ${EPOCHS}"
echo "   Batch Size: ${BATCH_SIZE}"
echo "   Learning Rate: ${LR}"
echo ""

# 인자 처리
EXPERIMENT_TYPE=${1:-all}

case $EXPERIMENT_TYPE in
    dataroot)
        run_dataroot_experiments
        ;;
    states)
        run_states_experiments
        ;;
    depths)
        run_depths_experiments
        ;;
    linvol)
        run_linvol_experiments
        ;;
    all)
        run_dataroot_experiments
        run_states_experiments
        run_depths_experiments
        run_linvol_experiments
        ;;
    help|--help|-h)
        print_usage
        exit 0
        ;;
    *)
        echo "❌ 알 수 없는 옵션: $EXPERIMENT_TYPE"
        print_usage
        exit 1
        ;;
esac

# 종료 시간 및 소요 시간 계산
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║        ✨ 모든 실험이 완료되었습니다! ✨                 ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  총 소요 시간: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "📁 결과 저장 위치:"
echo "   - ../results_dataroot/"
echo "   - ../results_states/"
echo "   - ../results_depths/"
echo "   - ../results_linvol/"
echo ""
