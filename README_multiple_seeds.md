# Multiple Seeds Experiment Runner

여러 random seed로 동일한 실험을 반복 실행하는 스크립트 모음입니다.

## 파일 설명

### 1. `run_multiple_seeds.py` (기본 버전)
- 고정된 seeds로 실험 실행
- Seeds: 42, 123, 456, 789, 2024
- 가장 간단하게 사용 가능

### 2. `run_multiple_seeds.sh` (Bash 버전)
- Shell script 버전
- 고정된 seeds로 실행
- 리눅스/맥 환경에서 직접 실행 가능

### 3. `run_multiple_seeds_flexible.py` (유연한 버전)
- 커맨드라인에서 seeds, GPU, 저장 경로 등을 자유롭게 설정 가능
- 가장 권장되는 버전

---

## 사용법

### Option 1: 기본 버전 (고정된 seeds)

```bash
# Python 버전
python run_multiple_seeds.py

# Bash 버전
./run_multiple_seeds.sh
```

### Option 2: 유연한 버전 (권장)

#### 기본 실행 (기본 seeds 사용)
```bash
python run_multiple_seeds_flexible.py
```

#### 커스텀 seeds 지정
```bash
python run_multiple_seeds_flexible.py --seeds 42 100 200 300 500
```

#### GPU 변경
```bash
python run_multiple_seeds_flexible.py --seeds 42 123 456 --gpu 0
```

#### 결과 저장 디렉토리 변경
```bash
python run_multiple_seeds_flexible.py --seeds 42 123 --results_dir my_custom_results
```

#### Run name 변경
```bash
python run_multiple_seeds_flexible.py --seeds 42 123 --run_name my_experiment
```

#### 모든 옵션 조합
```bash
python run_multiple_seeds_flexible.py \
    --seeds 42 123 456 789 2024 \
    --gpu 0 \
    --results_dir multiple_seeds_res \
    --run_name test_8x8_patch_2
```

#### Dry run (실제 실행 없이 명령어만 확인)
```bash
python run_multiple_seeds_flexible.py --seeds 42 123 --dry_run
```

#### 커스텀 config 파일 사용
```bash
python run_multiple_seeds_flexible.py --config my_config.json --seeds 42 123 456
```

#### 도움말 보기
```bash
python run_multiple_seeds_flexible.py --help
```

---

## 설정 변경

### 기본 설정 (run_multiple_seeds.py)

`run_multiple_seeds.py` 파일 상단의 `BASE_CONFIG` 딕셔너리를 수정:

```python
BASE_CONFIG = {
    "normal_data_root": "../main_files/dataset/v11/0.05_8_8_CTCA_64/Normal",
    "diseased_data_root": "../main_files/dataset/v11/0.05_8_8_CTCA_64/Diseased",
    "run_name": "test_8x8_patch_2",
    "save_path": "../multiple_seeds_res",
    "gpu_id": "1",
    # ... 기타 설정들
}

# Seeds 변경
SEEDS = [42, 123, 456, 789, 2024]
```

### Shell 스크립트 설정 (run_multiple_seeds.sh)

`run_multiple_seeds.sh` 파일 상단의 변수들을 수정:

```bash
SEEDS=(42 123 456 789 2024)
GPU_ID="1"
BASE_RUN_NAME="test_8x8_patch_2"
BASE_SAVE_PATH="../multiple_seeds_res"
```

---

## 결과 구조

실험이 완료되면 다음과 같은 디렉토리 구조가 생성됩니다:

```
multiple_seeds_res/
├── seed_42/
│   ├── test_8x8_patch_2_seed42_timestamp/
│   │   ├── configs/
│   │   ├── models/
│   │   ├── results/
│   │   └── visualizations/
├── seed_123/
│   ├── test_8x8_patch_2_seed123_timestamp/
│   │   └── ...
├── seed_456/
│   └── ...
├── seed_789/
│   └── ...
├── seed_2024/
│   └── ...
└── experiment_summary.json
```

### experiment_summary.json

모든 실험의 요약 정보가 담긴 JSON 파일:

```json
{
  "start_time": "2025-11-20 10:00:00",
  "end_time": "2025-11-20 15:30:00",
  "duration_seconds": 19800,
  "seeds": [42, 123, 456, 789, 2024],
  "results": {
    "42": "SUCCESS",
    "123": "SUCCESS",
    "456": "SUCCESS",
    "789": "SUCCESS",
    "2024": "SUCCESS"
  },
  "success_count": 5,
  "total_count": 5,
  "base_config": { ... }
}
```

---

## 실험 진행 상황 모니터링

실험이 진행되는 동안 터미널에서 실시간으로 진행 상황을 확인할 수 있습니다:

```
======================================================================
Multiple Seeds Experiment Runner
======================================================================
Seeds: [42, 123, 456, 789, 2024]
Total experiments: 5
Results directory: multiple_seeds_res
======================================================================

######################################################################
Experiment 1/5
######################################################################
✓ Created config file: temp_configs/config_seed_42.json

======================================================================
Starting experiment with seed: 42
======================================================================
Command: python main_files/main.py --config temp_configs/config_seed_42.json
[실험 로그...]
✓ Experiment with seed 42 completed successfully!

[... 다음 실험들 ...]
```

---

## 주의사항

1. **GPU 메모리**: 각 실험은 순차적으로 실행됩니다. GPU 메모리가 충분한지 확인하세요.

2. **시간 소요**: epochs=1000으로 설정되어 있으므로, 각 실험은 상당한 시간이 소요됩니다.

3. **데이터 경로**: `normal_data_root`와 `diseased_data_root`가 올바른 경로인지 확인하세요.

4. **디스크 공간**: 각 seed마다 모델과 결과가 저장되므로 충분한 디스크 공간이 필요합니다.

---

## 트러블슈팅

### Config 파일을 찾을 수 없다는 에러
```bash
# main_files 디렉토리가 올바른 위치에 있는지 확인
ls main_files/main.py
```

### GPU 메모리 부족
```bash
# 더 작은 batch size로 실행
# BASE_CONFIG의 "batch_size"를 1로 변경
```

### 특정 seed에서 실험이 실패한 경우
```bash
# 실패한 seed만 다시 실행
python run_multiple_seeds_flexible.py --seeds 456
```

---

## 예시 워크플로우

### 1. 짧은 테스트 실행
```bash
# Dry run으로 명령어 확인
python run_multiple_seeds_flexible.py --seeds 42 --dry_run

# 1개의 seed로 짧은 테스트
python run_multiple_seeds_flexible.py --seeds 42 --epochs 10
```

### 2. 전체 실험 실행
```bash
# 모든 seeds로 전체 실험
python run_multiple_seeds_flexible.py --seeds 42 123 456 789 2024
```

### 3. 결과 확인
```bash
# 요약 파일 확인
cat multiple_seeds_res/experiment_summary.json

# 각 seed의 결과 확인
ls multiple_seeds_res/seed_*/
```

---

## 연락처

문제가 발생하거나 질문이 있으면 이슈를 남겨주세요.
