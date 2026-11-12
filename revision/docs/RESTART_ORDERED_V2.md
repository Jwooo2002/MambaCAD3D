# 2026-11-07: ordered_mm_v2 재시작

사용자 요청에 따라 GPT-6 Astra가 중심선·좌표·모델 구현을 독립 검토했습니다. 기존 입력으로 돌던 두 학습은 중단하고 `outputs/superseded_runs/legacy_preprocessing_20261107.json`에 기록했습니다. 원본 CT, 코드, checkpoint와 이전 결과는 보존합니다. 새 실행은 초기 가중치에서 시작합니다.

## 확인하고 수정한 사항

- 기존 전처리가 set을 list로 변환하면서 중심선의 진행 순서를 잃었습니다. VTK/VTP의 실제 cell connectivity와 source root를 따라 경로를 구성합니다. ASOCA는 동일한 ordered prefix만 합치고, 분기 후 공간적으로 가까운 점을 임의 연결하지 않습니다.
- ASOCA raw VTP의 source-specific origin 보정 후 40명 모두 CT 내부에 위치했습니다. 전체 lumen 일치율은 98.9427–100%입니다. 이 변환은 일반 RAS/LPS 변환이 아니며 identity-direction ASOCA 배포에 한정합니다.
- 두 코호트 모두 LPS mm에서 경로를 표본화하고, parallel-transport frame에 수직인 단면을 full CT origin/direction/spacing을 반영해 보간합니다. 중심점을 정수 voxel로 먼저 반올림하지 않습니다.
- grouped Conv1d의 각 방향 안에서 dt/B/C를 분리하도록 수정했습니다. 기존 전체 채널 분리는 서로 다른 방향의 파라미터를 섞었습니다.
- softmax에 대한 hardtanh(0,1)를 실제 threshold hard shrinkage와 재정규화로 교체했습니다. 학습 및 환자 점수에 all-zero memory row 비율을 기록합니다. 현재 memory 300개, threshold 0.0025입니다.
- 전처리 NumPy 2 환경이 저장한 로컬 배열을 NumPy 1 학습 환경에서 읽을 때, `numpy._core`→`numpy.core` 이름을 명시적으로 대응시킵니다. 배열 수치나 형식을 바꾸지 않으며 검증된 로컬 pickle만 읽습니다.

## 고정 전처리 규격

| 항목 | 값 |
|---|---|
| 단면 | 8×8 mm, 32×32 pixels |
| 단면 pixel center 간격 | 0.25 mm; −3.875…+3.875 mm |
| 종축 | source arc length 0.5 mm 간격, 64단면 |
| 첫–끝 단면 중심 간 거리 | source arc 기준 31.5 mm |
| sliding stride | 32단면 / 16 mm |
| intensity | HU [−300,400] clip 후 trilinear 보간; 학습 시 [0,1] 정규화 |
| CT 밖 | −300 HU |
| 짧은 경로 | 마지막 단면 반복, valid_depth와 padded_patch_count 기록 |
| 마지막 잔여 구간 | full distal window 추가; 같은 start를 중복 추가하지 않음 |
| 동일 경로 중복 | 전체 window의 중심점·frame이 같으면 한 번만 유지 |

8 mm FOV는 Normal_10의 legacy 약 7 mm 범위와 가까워 불필요한 문맥 범위 변경을 줄입니다. 0.25 mm는 보간 간격이며 native CT 해상도 향상을 뜻하지 않습니다. 원래 legacy 단면/종축 간격은 환자마다 달랐으므로 새 규격을 투고 당시 규격과 동일하다고 쓰면 안 됩니다. 초기 16 mm FOV 샘플 네 건은 `outputs/preprocessing_initial_16mm_sample`에 별도 보관합니다.

Root-to-leaf 경로는 다른 분지의 window에 공통 근위부 조직이 일부 포함될 수 있습니다. 완전히 같은 window만 제거하며, 모든 혈관 조직이 균일하게 한 번씩 표본화된다고 주장하지 않습니다. 패치 수와 네 가지 aggregation을 함께 보존합니다.

## 원본 주석의 예외

- Diseased_6은 source start가 3개입니다. 별도 component를 보존하며 모두 대동맥 기시부라고 단정하지 않습니다.
- ImageCAS-X 455의 오른쪽 graph는 cycle 하나를 포함합니다. 이 source에만 사전 지정한 deterministic rooted shortest-path-tree 보정을 적용합니다. 제거 edge [362,411], 보존 노드와 인공 terminal을 기록하며 환자를 제외하지 않습니다. 다른 cycle은 자동으로 처리하지 않고 실패시킵니다.
- Lumen mask는 좌표 검증에만 사용하며 모델 입력 채널이 아닙니다. 제공된 reference centerline을 사용하므로 자동 중심선 추출 성능을 평가한 결과가 아닙니다.

## 새 실행과 검증

ASOCA 40명에서 1,718개 패치를 생성했습니다(환자당 23–72개). 초기 manifest와 새 manifest의 fold별 환자 역할은 완전히 같습니다. 정상 12명 학습 / 3명 검증 / 5명 내부 시험과 질환 20명 내부 시험을 사용하며, 4 folds × 5 seeds를 실행합니다. 외부 800명에서 32,098개 패치를 생성했습니다(환자당 21–67개). 전체 840명의 배열 content hash·HU·shape·직교 frame 검사, 원본 코드 21개 보존 검사를 모두 통과했습니다. 검사는 `processed_data/ordered_mm_v2/readiness_audit.json`에 보존합니다.

외부 코호트는 사전에 고정된 ImageCAS-X 800명(질환 없음 412 / 있음 388) 전체입니다. 각 ASOCA 모델의 checkpoint와 판정 기준 선택을 마친 뒤 평가하고, 내부 정상 validation의 판정 기준을 그대로 적용합니다. 모든 환자·패치 원점수와 max/mean/top-5/95-percentile 결과를 저장합니다.

물리 좌표의 analytic phantom, source connectivity/branch/cycle, 내부·외부 평가 흐름 등 총 39개 검증을 통과했습니다(학습 환경 pytest 16개, 전처리 환경 unittest 23개). NumPy 환경 호환을 포함한 실제 `2×1×64×32×32` 입력으로 GPU 0·1 모두 손실과 기울기가 유한했고 zero-memory-row 비율은 0이었습니다. 현재 모델 파라미터는 2,109,241개입니다.

2026-11-07 18:03 KST에 GPU 0에서 fold 1–2, GPU 1에서 fold 3–4, 각 seed [1,123,456,789,2024]를 시작했습니다. 실행별 PID·명령·로그는 `active_runs.json`에, 실제 source snapshot과 설정은 각 output에 저장합니다. 학습 및 외부 평가의 최종 완료 여부는 `python3 monitor.py`와 output summary로 확인해야 합니다.

이 새 실행은 투고 수치를 재현한 결과나 수정 원고의 최종 검증 완료를 뜻하지 않습니다. 원고의 1.72M 및 기존 AUROC/AP provenance, 같은 수정 입력으로 baseline/ablation 재실행, lesion-level 전문가 검증의 한계, 환자 반복을 반영한 최종 통계는 별도 작업입니다. 기존 입력/모델 결과와 새 결과를 동일 조건 비교로 제시하면 안 됩니다.
