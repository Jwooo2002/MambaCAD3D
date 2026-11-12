#!/bin/bash
# run_parallel_experiments.sh: GPU별 병렬 실험 자동 실행 스크립트
# 여러 GPU에 실험을 자동으로 분배하여 병렬 실행합니다.

set -e  # 에러 발생 시 즉시 종료

# ==========================================================
# 색상 정의
# ==========================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ==========================================================
# 설정
# ==========================================================
LOG_DIR="./parallel_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${LOG_DIR}

# 실험 리스트
EXPERIMENTS=(states depths linvol)

# ==========================================================
# GPU 확인 함수
# ==========================================================
check_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}❌ nvidia-smi를 찾을 수 없습니다. CUDA가 설치되어 있는지 확인하세요.${NC}"
        exit 1
    fi

    NUM_GPUS=$(nvidia-smi -L | wc -l)

    if [ $NUM_GPUS -eq 0 ]; then
        echo -e "${RED}❌ 사용 가능한 GPU가 없습니다.${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ 사용 가능한 GPU: ${NUM_GPUS}개${NC}"
    nvidia-smi -L
    echo ""
}

# ==========================================================
# GPU 상태 모니터링 함수
# ==========================================================
monitor_gpus() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}🖥️  GPU 사용 현황${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    while IFS=, read -r idx name util mem_used mem_total; do
        echo -e "${YELLOW}GPU ${idx}${NC}: ${name} | ${GREEN}사용률: ${util}%${NC} | 메모리: ${mem_used}MB/${mem_total}MB"
    done
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# ==========================================================
# 실험 시작 함수
# ==========================================================
start_experiment() {
    local exp=$1
    local gpu_id=$2
    local log_file="${LOG_DIR}/${exp}_gpu${gpu_id}.log"
    local pid_file="${LOG_DIR}/${exp}_gpu${gpu_id}.pid"
    local done_file="${LOG_DIR}/${exp}_gpu${gpu_id}.done"

    echo -e "${BLUE}🚀 GPU ${gpu_id}에서 ${exp} 실험 시작...${NC}"
    echo -e "${BLUE}   로그 파일: ${log_file}${NC}"

    # 백그라운드에서 실행
    (
        cd "$(dirname "$0")"
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        ./run_all_experiments.sh ${exp} > "${log_file}" 2>&1
        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo -e "\n${GREEN}✅ ${exp} 실험 완료 (GPU ${gpu_id})${NC}" | tee -a "${log_file}"
            echo "SUCCESS" > "${done_file}"
        else
            echo -e "\n${RED}❌ ${exp} 실험 실패 (GPU ${gpu_id}) - Exit Code: ${exit_code}${NC}" | tee -a "${log_file}"
            echo "FAILED" > "${done_file}"
        fi

        rm -f "${pid_file}"
    ) &

    # PID 저장
    echo $! > "${pid_file}"

    echo -e "${GREEN}   PID: $(cat ${pid_file})${NC}"
    echo ""
}

# ==========================================================
# GPU가 사용 가능한지 확인 (실험이 실행 중인지)
# ==========================================================
is_gpu_free() {
    local gpu_id=$1
    local pid_files=(${LOG_DIR}/*_gpu${gpu_id}.pid)

    for pid_file in "${pid_files[@]}"; do
        if [ -f "${pid_file}" ]; then
            local pid=$(cat "${pid_file}")
            if ps -p $pid > /dev/null 2>&1; then
                return 1  # GPU 사용 중
            fi
        fi
    done

    return 0  # GPU 사용 가능
}

# ==========================================================
# 큐 시스템으로 실험 관리
# ==========================================================
run_with_queue() {
    local num_gpus=$1
    local experiments=("${@:2}")  # 나머지 인자들이 실험 리스트

    local exp_queue=("${experiments[@]}")
    local exp_index=0

    echo -e "${YELLOW}🎯 큐 시스템 시작: ${#exp_queue[@]}개 실험을 ${num_gpus}개 GPU에서 순차 실행${NC}"
    echo ""

    # 초기에 GPU 개수만큼 실험 시작
    for ((gpu_id=0; gpu_id<num_gpus && exp_index<${#exp_queue[@]}; gpu_id++)); do
        start_experiment "${exp_queue[$exp_index]}" ${gpu_id}
        exp_index=$((exp_index + 1))
        sleep 3
    done

    # 남은 실험들을 큐에서 처리
    while [ $exp_index -lt ${#exp_queue[@]} ]; do
        # GPU가 비는지 주기적으로 확인
        for ((gpu_id=0; gpu_id<num_gpus; gpu_id++)); do
            if is_gpu_free ${gpu_id}; then
                if [ $exp_index -lt ${#exp_queue[@]} ]; then
                    echo -e "${GREEN}💡 GPU ${gpu_id} 사용 가능! 다음 실험 시작...${NC}"
                    start_experiment "${exp_queue[$exp_index]}" ${gpu_id}
                    exp_index=$((exp_index + 1))
                    sleep 3
                fi
            fi
        done

        sleep 10  # 10초마다 체크
    done

    echo -e "${GREEN}✅ 모든 실험이 큐에 추가되었습니다. 완료 대기 중...${NC}"
}

# ==========================================================
# 진행 상황 모니터링 함수
# ==========================================================
monitor_progress() {
    local log_dir=$1

    while true; do
        clear
        echo ""
        echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${MAGENTA}║        🔬 병렬 실험 진행 상황 모니터링                   ║${NC}"
        echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════╝${NC}"
        echo ""

        # 실행 중인 프로세스 확인
        local running=0
        local completed=0

        for exp in "${EXPERIMENTS[@]}"; do
            local pid_files=(${log_dir}/${exp}_gpu*.pid)

            if [ -f "${pid_files[0]}" ]; then
                local pid=$(cat "${pid_files[0]}")
                if ps -p $pid > /dev/null 2>&1; then
                    echo -e "${YELLOW}⏳ ${exp}${NC}: 실행 중 (PID: ${pid})"
                    running=$((running + 1))

                    # 최근 로그 출력
                    local log_file=$(echo "${pid_files[0]}" | sed 's/.pid/.log/')
                    if [ -f "${log_file}" ]; then
                        echo -e "${CYAN}   마지막 로그:${NC}"
                        tail -3 "${log_file}" | sed 's/^/   │ /'
                    fi
                else
                    echo -e "${GREEN}✅ ${exp}${NC}: 완료"
                    completed=$((completed + 1))
                fi
            else
                echo -e "${GREEN}✅ ${exp}${NC}: 완료"
                completed=$((completed + 1))
            fi
            echo ""
        done

        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${CYAN}실행 중: ${running}개 | 완료: ${completed}개 | 전체: ${#EXPERIMENTS[@]}개${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

        # GPU 상태
        monitor_gpus

        # 모두 완료되었으면 종료
        if [ $running -eq 0 ]; then
            break
        fi

        echo -e "${YELLOW}⏰ 30초 후 업데이트... (Ctrl+C로 모니터링 종료, 실험은 계속 실행됨)${NC}"
        sleep 30
    done
}

# ==========================================================
# 종료 처리 함수
# ==========================================================
cleanup() {
    echo ""
    echo -e "${YELLOW}⚠️  스크립트가 중단되었습니다.${NC}"
    echo -e "${YELLOW}   백그라운드 실험은 계속 실행됩니다.${NC}"
    echo -e "${YELLOW}   진행 상황 확인: tail -f ${LOG_DIR}/*.log${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ==========================================================
# Main 실행
# ==========================================================

echo ""
echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                                                           ║${NC}"
echo -e "${MAGENTA}║        🚀 GPU 병렬 실험 자동 실행 시스템 🚀              ║${NC}"
echo -e "${MAGENTA}║                                                           ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# GPU 확인 (NUM_GPUS 변수는 함수 내에서 설정됨)
check_gpus

echo -e "${CYAN}📋 실험 계획:${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
total_exp=0
for exp in "${EXPERIMENTS[@]}"; do
    case $exp in
        dataroot)
            count=6
            ;;
        states)
            count=4
            ;;
        depths)
            count=3
            ;;
        linvol)
            count=4
            ;;
    esac
    echo -e "  ${YELLOW}${exp}${NC}: ${count}개 실험"
    total_exp=$((total_exp + count))
done
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}총 실험 수: ${total_exp}개${NC}"
echo -e "  ${GREEN}K-Fold: 4 folds/실험${NC}"
echo -e "  ${GREEN}총 학습 횟수: $((total_exp * 4)) folds${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# GPU 분배 전략 설명
echo -e "${YELLOW}🎯 GPU 분배 전략 (큐 시스템):${NC}"
if [ $NUM_GPUS -ge ${#EXPERIMENTS[@]} ]; then
    # GPU가 실험보다 많거나 같으면 각각 할당
    gpu_id=0
    for exp in "${EXPERIMENTS[@]}"; do
        echo -e "  ${exp} → GPU ${gpu_id}"
        gpu_id=$((gpu_id + 1))
    done
else
    # GPU가 부족하면 큐 시스템 사용
    echo -e "  ${GREEN}큐 시스템 활성화!${NC}"
    echo -e "  처음 ${NUM_GPUS}개 실험을 각 GPU에서 시작"
    echo -e "  한 실험이 끝나면 자동으로 다음 실험 시작"
    echo ""
    echo -e "  실행 순서:"
    for i in "${!EXPERIMENTS[@]}"; do
        gpu_id=$((i % NUM_GPUS))
        echo -e "  ${i}: ${EXPERIMENTS[$i]} → GPU ${gpu_id} (순서: $((i / NUM_GPUS + 1))번째 배치)"
    done
fi
echo ""

# 사용자 확인
echo -e "${YELLOW}⚠️  주의: 이 작업은 장시간 실행됩니다 (예상: 수일)${NC}"
echo -e "${YELLOW}   로그는 ${LOG_DIR}/ 에 저장됩니다.${NC}"
echo ""
read -p "계속하시겠습니까? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ 취소되었습니다.${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}✅ 실험을 시작합니다!${NC}"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)
echo "시작 시각: $(date)" > "${LOG_DIR}/execution_time.txt"

# 큐 시스템으로 실험 시작
run_with_queue $NUM_GPUS "${EXPERIMENTS[@]}"

echo ""
echo -e "${GREEN}✅ 모든 실험이 백그라운드에서 실행 중입니다!${NC}"
echo ""
echo -e "${CYAN}📊 실시간 모니터링을 시작합니다...${NC}"
echo -e "${CYAN}   (Ctrl+C를 눌러도 실험은 계속 실행됩니다)${NC}"
echo ""

sleep 5

# 진행 상황 모니터링
monitor_progress ${LOG_DIR}

# 종료 시간 기록
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "종료 시각: $(date)" >> "${LOG_DIR}/execution_time.txt"
echo "총 소요 시간: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "${LOG_DIR}/execution_time.txt"

echo ""
echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                                                           ║${NC}"
echo -e "${MAGENTA}║        ✨ 모든 병렬 실험이 완료되었습니다! ✨            ║${NC}"
echo -e "${MAGENTA}║                                                           ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}⏱️  총 소요 시간: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo ""
echo -e "${CYAN}📁 로그 파일:${NC}"
ls -lh ${LOG_DIR}/*.log
echo ""
echo -e "${CYAN}📊 결과 위치:${NC}"
echo "   ../results_all/results_states/"
echo "   ../results_all/results_depths/"
echo "   ../results_all/results_linvol/"
echo ""

# 요약 리포트 생성
echo -e "${CYAN}📝 실험 요약 생성 중...${NC}"
{
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║              병렬 실험 실행 요약                          ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "실행 시각: $(date)"
    echo "총 소요 시간: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "실험별 상태:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for exp in "${EXPERIMENTS[@]}"; do
        log_file="${LOG_DIR}/${exp}_gpu*.log"
        if grep -q "완료" ${log_file} 2>/dev/null; then
            echo "✅ ${exp}: 완료"
        else
            echo "❌ ${exp}: 실패 또는 미완료"
        fi
    done
    echo ""
} > "${LOG_DIR}/summary.txt"

cat "${LOG_DIR}/summary.txt"

echo -e "${GREEN}요약 파일 저장: ${LOG_DIR}/summary.txt${NC}"
echo ""
