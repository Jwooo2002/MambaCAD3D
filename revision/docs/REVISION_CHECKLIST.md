# J-BHI revision checklist

## 시작한 작업

- [x] `kfold_simplescan_revision` 생성, 원본 코드 보존 및 hash 기록.
- [x] ImageCAS-X v3 원본 1,000건 확보. 외부 코호트 800명(정상 412 / 질환 388)의 주석 연결·좌표 검사·중심선 그래프 및 수정된 공통 패치 32,098개 준비 완료. 각 모델 학습 후 외부 평가 실행을 연결함.
- [x] Astra 독립 검토 후 ASOCA 40명을 연결 순서·물리 좌표가 보존된 패치 1,718개로 재생성. 이전 환자별 fold 역할 보존.
- [x] 새 모델/전처리 버전으로 4-fold × 5-seed를 두 GPU에서 초기 가중치부터 재시작.
- [x] 원고 입력 크기에 해당하는 `64×32×32` 자료 확인.
- [x] 고정된 환자 단위 split과 별도 validation 도입.
- [x] 학습 seed와 split seed 분리.
- [x] 테스트 환자를 조기 종료와 판정 기준 선택에서 제외.
- [x] 최적 가중치를 CPU tensor clone으로 보관. 기존 shallow copy 문제 수정.
- [x] 메모리 entropy를 attention 마지막 축(memory slots)에 대해 계산. 기존 token 축 합산 수정.
- [x] 같은 환자 score에서 AUROC/AP 계산, 원점수 보존, 임상 지표 추가.
- [x] max/mean/top-k/percentile 사전 지정 비교와 검증 환자 기반 판정 기준.
- [x] 단일 fold/seed 환자 단위 bootstrap CI와 paired baseline 비교 도구.
- [x] 데이터 분할, 지표, checkpoint, 메모리 entropy, 실제 모델 forward/backward 및 전체 실행 흐름 검증.

## 제출 원본과 반드시 대조

- [x] 기존 중심선 직렬화의 set→list 순서 손실 확인 및 공통 물리 전처리 수정. `ASTRA_PREPROCESSING_AUDIT.md`, `RESTART_ORDERED_V2.md`에 원본 연결·좌표 근거와 예외를 기록함.

- [ ] 원고의 AUROC 0.860 / AP 0.980을 생성한 원본 점수·계산 스크립트 확인.
- [ ] 모든 baseline과 ablation의 평가 단위, 질환=1 정의, AP 계산 방식 통일.
- [ ] 투고 당시 모델 checkpoint와 실제 stage 수 확인. 현재 3-stage 모델은 2,109,241 parameters이며 원고의 1.72M과 다름.
- [ ] 4-stage 다중 seed 기록을 3-stage 원고 모델 결과로 사용하지 않기.
- [x] 새 HU window, 물리적 패치 길이/폭, 중심선 간격, 보간, 짧은 경로·distal window·중복 규칙 고정. 기존 연구/투고 설명과의 대조표는 원고에서 추가해야 함.
- [x] MemModule hardtanh 항등 문제를 threshold hard shrinkage로 수정하고 zero-memory-row 비율을 기록함. 수정 모델로 baseline/ablation 대조는 필요.
- [x] grouped x_proj 출력에서 group 내부 dt/B/C 분리를 수정하고 방향별 sentinel 회귀검증. 변경 전 모델과 동등 구현으로 혼용하지 않음.
- [ ] 각 downsampling 단계의 scan 순서, inverse merge(sum), 원고의 짝/홀 spiral 방향과 일치 여부 명시.
- [ ] 초기 lambda=0.0002는 기존 실험 설정에서 가져온 값. 투고 설정 및 선택 근거 확인.

## 에디터/리뷰어 대응

| 요구 | 다음 작업 | 상태 |
|---|---|---|
| AE, R1–R3: 작은 코호트·일반화 | 고정 CV/seed 결과 확보, 외부 ImageCAS-X 적합성·endpoint 검토 | 시작 |
| AE, R2, R3: CI·paired differences | 각 fold/seed CI, 동일 조건 baseline 점수 비교, 최종 환자 반복을 반영한 결합 추론 | 부분 구현 |
| AE, R2, R3: test-set overfitting | 과거 개발 이력 공개, 내부 validation과 독립 외부 평가 구분 | 코드 분리 완료, 원고 필요 |
| AE: expert-annotated abnormalities | 병변 위치 정답 확보 가능성 확인; 임상의 협조가 없으면 위치 검증 요구는 미해결로 명시 | 미해결 |
| R1–R3: aggregation | 저장된 점수로 사전 지정 4가지 방식 비교, 패치 수 영향도 검토 | 구현, 결과 필요 |
| R1, R2: centerline/branches/curvature | 위치·각도 perturbation, 곡률/직경과 정상 score 관계, branch 오탐 발생률 및 masking 손실 | 대기 |
| R1: processing time | CT부터 환자 점수까지 전체 처리 시간과 모델 forward를 분리 측정 | forward만 구현 |
| R2, R3: fair baselines | 같은 데이터/split/검증 예산의 3D CNN AE·Transformer AE·Mamba UAD | 대기 |
| R2, R3: scan contribution | 기존 scan/Hilbert/CoroSpiral을 동일 backbone에서 비교; 공간/종축 token continuity 정량화 | 대기 |
| R2: preliminary work | 기존 연구 대비 입력·구조·scan·평가 차이 비교표 | 대기 |
| R3: Serp-Mamba | 2D 망막 segmentation과 3D CTCA UAD의 공통점·차이 기술 | 대기 |
| R2, R3: presentation | VAE→AE, RD4AD reverse distillation, VMamba Cross-Scan, 그림/수식/용어 수정 | 대기 |
| R3: overstated results | 최고 baseline 대비 절대 AUROC 차이로 보고, 미검증 병변 위치와 전체 관상동맥 문맥 주장 제한 | 대기 |
| AE/R2: final submission | point-by-point response 별도 파일, 변경 표시 원고, 페이지/그림 규정 점검 | 대기 |

## 결과 해석 제한

현재 CI는 단일 학습 모델과 fold의 환자 구성에 조건부입니다. CV의 평균에 대한 CI나 seed 안정성 CI로 제시하면 안 됩니다. 20명 질환 환자가 반복되므로 단순히 모든 fold/seed 행을 합쳐 표본 수를 늘릴 수 없습니다.

정상 검증 3명의 95 percentile 판정 기준은 매우 불안정합니다. 임상 지표는 탐색적 운영점으로 해석하고 표본 수와 혼동행렬을 함께 제시해야 합니다.

이번 코드는 수정 실험 기반을 만드는 첫 버전입니다. 최초 원고 수치 재현, 외부 일반화, 병변 위치 검증, 최종 통계 결론 또는 논문 수정 완료를 의미하지 않습니다.
