# MambaCAD-3D revision workspace

2026-11-07에 시작한 J-BHI major revision 작업 폴더입니다. 재제출 기한은 수신일 기준 2027-01-16이며, 제출 시스템에 표시되는 시각을 최종 확인해야 합니다.

주요 수정 내용과 실험 결과는 [Notion 리비전 변경 기록](https://app.notion.com/p/411d33500b3d81daa5a7de7fabd0e09b?pvs=204)에 계속 남깁니다. 날짜별 변경 이유·검증·남은 작업을 기록하며, 이어서 작업할 때는 같은 페이지를 갱신합니다.

현재 실행은 **ordered_mm_v2**입니다. [재시작 기록](docs/RESTART_ORDERED_V2.md)에 전처리·모델 수정과 검증을 정리했습니다. Astra의 독립 검토 근거는 [전처리 감사](docs/ASTRA_PREPROCESSING_AUDIT.md), 외부 원본 확보 내역은 [ImageCAS-X 준비 기록](docs/IMAGECAS_X_PREPARATION.md)에 있습니다. `python3 monitor.py`로 새 전처리와 학습 상태를 확인할 수 있습니다.

## 원본과 작업 범위

- 원본: `../MemAE_kfold_simplescan/main_files`.
- `legacy_source/main_files/`: 원본 Python 코드 21개의 보존본. 실행용이 아닙니다.
- `SOURCE_MANIFEST.json`: 원본 위치와 보존 당시 SHA-256.
- `main_files/`: 리비전 실행 코드.
- `configs/asoca_revision.json`: 수정된 입력·모델·외부 평가의 기본 설정.
- `configs/physical_sampling_v2.json`: 두 코호트의 공통 물리 표본화 규격.
- `configs/asoca_revision_legacy_inputs.json`: 초기 입력 설정의 보존본.
- `splits/asoca_ordered_mm_v2_4fold.json`: 새 입력의 고정 환자 목록. 환자별 역할은 초기 `asoca_4fold.json`과 동일합니다.
- 원본 데이터·체크포인트·실험 결과는 복사하거나 변경하지 않습니다. 설정의 데이터 경로는 이 작업 폴더 기준으로 해석합니다.

## 시작 설정과 남은 확인

ASOCA 원본 CT와 VTP 연결 정보로 정상 20명, 질환 20명을 다시 전처리했습니다. 공통 규격은 단면 8×8 mm / 32×32 pixels, 종축 간격 0.5 mm / 64단면, stride 32입니다. 실제 패치는 `[patch,64,32,32]`, 모델 입력은 `[batch,1,64,32,32]`입니다. 0.25 mm 단면 격자는 보간 간격이며 원본 CT의 실제 해상도를 높이지 않습니다.

원고의 3단계 설정에 맞춰 `dims=[32,64,128]`, `depths=[2,2,2]`, `d_state=8`, `mem_dim=300`을 사용합니다. 현재 소스로 계산한 파라미터 수는 **2,109,241**로, 원고의 **1.72M**과 다릅니다. 기존 다중 seed 설정은 4단계였으므로, 이 초기 실행은 수정 평가 절차를 확인하는 새 기준 실험입니다. 투고 당시 정확한 체크포인트/설정 재현은 아직 완료되지 않았습니다.

모델의 stage 수와 CoroSpiral 순서는 유지했습니다. 새 모델 버전은 방향별 grouped SSM의 dt/B/C 분리와 MemAE hard shrinkage를 수정했습니다. 이전 checkpoint를 이어 학습하지 않습니다. 원고의 짝/홀 scan 정의와 코드의 일치 여부는 별도 확인 항목입니다.

## 평가 절차

1. 정상 20명을 고정된 4개 outer fold로 나눕니다.
2. 매 fold에서 정상 **12명 학습 / 3명 검증 / 5명 테스트**, 질환 **20명 테스트**를 사용합니다.
3. 모든 학습 seed와 비교 모델은 같은 split manifest를 사용합니다. `split_seed`와 `training_seeds`는 별개입니다.
4. 조기 종료와 최적 모델 선택에는 정상 검증 환자의 평균 reconstruction MSE만 사용합니다.
5. 판정 기준은 정상 검증 환자의 patient score 95 percentile입니다. 검증 환자가 3명뿐이므로 이 기준의 불확실성이 큽니다. 테스트에서 sensitivity/specificity/F1을 최적화하지 않습니다.
6. 주 평가 aggregation은 원고와 같은 max입니다. mean, top-5 평균, 95 percentile은 사전 지정한 보조 비교로 함께 저장합니다. 테스트 성능으로 주 방법을 교체하지 않습니다.
7. 패치 점수는 원고 수식에 맞춰 squared-error sum(SSE)을 사용하며, MSE도 함께 보존합니다. 패치 크기가 일정하면 MSE와 SSE는 상수배 관계입니다. 손실 함수는 MSE입니다.
8. AUROC와 AP는 **같은 환자 라벨과 연속 점수**, 질환=1로 계산합니다. AP는 `average_precision_score`이며 patch-level AP나 trapezoidal PR-AUC와 혼용하지 않습니다. 질환 비율도 함께 기록합니다.
9. 주 지표의 95% CI는 각 fold/seed 내에서 정상·질환 환자를 각각 재표집하는 bootstrap입니다. 학습 모델과 클래스 수에 조건부인 CI입니다.
10. 질환 20명이 모든 fold에 반복됩니다. fold/seed의 결과를 독립 환자처럼 합치지 않습니다. 전체 요약의 mean/SD는 기술 통계이며 CI가 아닙니다. 최종 교차 fold/seed 추론은 환자 반복을 반영해 별도로 설계해야 합니다.

ASOCA는 이미 과거 개발에 사용되었습니다. 새 CV는 내부 검증이며 과거 개발 이력을 없애지는 않습니다. ImageCAS-X 800명은 전원 외부 시험 전용입니다. 각 모델의 내부 정상 검증 환자에서 정한 판정 기준을 그대로 적용하며, 외부 라벨로 학습·checkpoint 선택·판정 기준 조정을 하지 않습니다. 외부 자료의 질환 여부는 환자 단위이며 병변 위치 정답은 아닙니다.

## 실행

기존 Python 환경을 사용하며 추가 설치는 하지 않았습니다.

```bash
cd /mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision
/home/jiwoong/anaconda3/envs/VMamba/bin/python main.py prepare
/home/jiwoong/anaconda3/envs/VMamba/bin/python -m pytest -q
```

실제 자료로 1 epoch를 학습하고 평가까지 확인:

```bash
/home/jiwoong/anaconda3/envs/VMamba/bin/python -u main.py train --gpu 0 --folds 1 --seeds 1 --epochs 1 --run-name asoca_revision_smoke
```

4-fold × 5-seed 실행(최대 1,000 epochs, 검증 patience 10):

```bash
/home/jiwoong/anaconda3/envs/VMamba/bin/python -u main.py train --gpu 0
```

새 실행은 **4-fold × 5-seed**, 총 20개 학습입니다. GPU 0에서 fold 1–2, GPU 1에서 fold 3–4를 실행하며, 각 학습이 끝난 뒤 고정된 외부 800명을 평가합니다. 프로세스와 로그 위치는 `active_runs.json`에 기록합니다. 다음 명령으로 진행 상황을 확인합니다.

```bash
python3 monitor.py
```

서로 다른 fold를 두 GPU에 나누려면 같은 manifest와 설정으로 각각 `--gpu 0 --folds 1 2`, `--gpu 1 --folds 3 4`를 사용할 수 있습니다. 결과 폴더는 실행마다 분리됩니다. 이 경우 두 결과를 합칠 때 context 중복 여부를 확인해야 합니다.

GPU 드라이버가 보이는 실행 환경에서 실행해야 합니다. CUDA가 없으면 자동으로 대규모 CPU 학습으로 넘어가지 않습니다. 작은 CPU 검증만 `--device cpu`를 사용합니다.

## 출력과 검산

각 실행은 `outputs/<run_name>_<UTC timestamp>/`에 설정, 환경, split, 실행 소스 snapshot과 hash, 로그를 저장합니다. 각 `fold_N/seed_S/`에는 다음이 생성됩니다.

- `best_model.pt`: 검증으로 선택한 가중치와 실제 학습 seed, train/validation 환자 ID.
- `training_history.csv`: epoch별 손실, 메모리 entropy, 검증 손실과 시간.
- `patient_scores.csv`: 검증/테스트 역할, 환자 ID, 질환 라벨, 패치 수, 4가지 aggregation, 판정 기준과 예측.
- `patch_scores.csv`: 환자·패치별 MSE/SSE.
- `metrics.json`: AUROC/AP, prevalence, sensitivity/specificity/F1, 혼동행렬, 주 지표 CI.
- 실행 폴더의 `summary.json`: 진행 상태와 완료한 fold/seed의 기술 통계. 실패하면 실패 상태를 기록하고 정상 종료로 표시하지 않습니다.

`network_forward_seconds`는 환자별 모델 forward 시간이고, `scoring_seconds`는 준비된 패치에서 점수를 계산하는 시간입니다. 둘 다 원본 CT 읽기, 중심선 추출, 패치 전처리를 포함하지 않습니다. 리뷰어가 요구한 전체 처리 시간은 별도 측정해야 합니다.

저장된 단일 fold/seed의 환자별 점수로 AUROC/AP를 검산:

```bash
/home/jiwoong/anaconda3/envs/VMamba/bin/python main.py audit --scores /absolute/path/to/patient_scores.csv
```

동일 환자에 대한 baseline과 paired bootstrap 비교:

```bash
/home/jiwoong/anaconda3/envs/VMamba/bin/python main.py audit --scores /absolute/path/to/patient_scores.csv --baseline-scores /absolute/path/to/baseline_patient_scores.csv
```

Baseline CSV에도 `patient_id,label,score`가 필요합니다. `role`, `fold`, `seed`가 있으면 test 행만 사용하며 fold/seed가 일치해야 합니다. 반복 환자나 클래스 불일치는 오류로 처리합니다.
