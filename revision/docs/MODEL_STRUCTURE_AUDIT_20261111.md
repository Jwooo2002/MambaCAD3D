# 모델 본체 독립 감사 — 2026-11-11

사용자 요청에 따라 3개 서브에이전트가 memory, SSM·scan, 이상 탐지 설계를 독립적으로 검토했다. 초기 검토 뒤 중대한 dt 투영 소멸을 교차 검증했다. 부모는 정상 TRAIN 3명에서 각 1개 고정 middle patch만 사용해 기능 진단을 추가했다. 새 학습·질환/시험 평가·모델 교체는 없으며 원 모델 소스와 checkpoint를 수정하지 않았다.

## 핵심 판단

**메모리 수축 외에 Mamba의 입력 의존 Δ 투영도 학습 중 사실상 사라지고 있다. 이 현상은 legacy에만 국한되지 않는다.** 메모리에만 decay를 제외한 M3에서도 남아 있어, λ 후보 탐색보다 SSM 학습 경로의 수축 원인 분리를 우선할 근거다. 다만 이것이 탐지 성능 저하를 얼마나 설명하는지, decay·약한 복원 gradient·초기화 중 어느 원인이 지배적인지는 아직 확정하지 않았다.

| 우선순위 | 발견 | 확인 범위 / 해석 |
| --- | --- | --- |
| 1 | dt_proj.weight 소멸, Δ의 입력 의존성 소실 | Legacy 6개 모델의 36/36블록에서 모든 weight가 약 6×10⁻⁴⁰. 정상 TRAIN 108회 블록 관찰에서 출력이 bias와 정확히 동일. 수정 모델 20개의 120블록 중 102블록도 전원소 subnormal. |
| 2 | SSM 상태 경로와 시간척도 약화 | Legacy 초기 exp(A_log)=1…8이 최종 0.936…1.025로 좁아짐. 첫 stage의 recurrent/D·u RMS 비는 3개 정상 patch에서 최대 약 0.0004825. 다른 일부 블록은 재귀 성분이 살아 있음. |
| 3 | Memory는 연속 공간 코드이며 학습 신호도 그 기하에 좌우됨 | Tied key/value의 query Jacobian은 attention 가중 memory covariance. 중복·집중은 readout과 encoder gradient를 함께 제한할 수 있음. 반대로 M3는 full-rank가 가능하므로 강제 이산 정상 prototype 병목도 아님. |
| 4 | Encoder 특징도 이미 소수 방향에 집중 | M3 정상 TRAIN query의 99% 분산은 7방향, readout은 9/8방향. 낮은 차원이 데이터 특성·유용한 압축·퇴화 중 무엇인지는 미확인. |
| 5 | 전체 MSE→환자 max가 병변 선택성을 보장하지 않음 | 작은 국소 오차가 전체 voxel 평균에서 희석될 수 있고 가장 큰 오차 patch가 병변이라는 보장은 없음. 병변 주석/국소 민감도 및 matched AE 대조가 필요. |

## 1. 새 확정 관찰: Δ 투영 소멸

소스상 `delta=softplus(dt_proj(dt))`다. 각 checkpoint에 6개 dt weight tensor, 10,752개 원소가 있다. 초기 tensor L2 norm은 6.5573–13.1336이며 epoch20에 4.6841e−11–0.062484, epoch50·100에는 7.9649e−39–3.2766e−38로 줄었다. 최종 max|weight|는 5.9167e−40–6.3058e−40다. **정확한 0은 아니며 nonzero subnormal이다.** Float32 norm의 underflow를 피하기 위해 float64 norm·max와 count_nonzero를 함께 검사했다. 저장 Adam exp_avg는 dt weight에서0이었다. 과거의 모든 gradient가0이었다는 뜻은 아니다.

M0/M1/M3 × seed42/123 × 3개 정상 TRAIN patch × 6블록=108회 관찰에서 dt_proj의 출력은 해당 bias를 공간 전체로 펼친 것과 bitwise 동일했다. Softplus 이후도 동일하고 token간 최대 차이는0이었다. 이는 해당 입력에서 Δ가 입력에 따라 조절되지 않았다는 직접 증거다. corrected20개는 checkpoint 수치 확인이며 같은108회 기능 검증 대상으로 표현하지 않는다.

Corrected20개(source_snapshot과 현재 main_files/model.py hash 일치, best epoch48–135):

- 120개 블록 중102개는 모든 dt weight가 nonzero subnormal.
- 116/120개는 max|weight|<1e−8.
- 14/20모델은 6개 블록 모두 subnormal, 18/20은 모두 max|weight|<1e−8.
- 모든 tensor finite. 질환 점수·영상은 읽지 않았다.

이 현상은 과거 방향별 packing 버그만으로 설명할 수 없다. M3에서도 SSM을 포함한 memory 외 파라미터는 Adam weight_decay=1e−5가 유지된다. **Adam decay와 약한 task gradient의 기여는 유력하게 점검할 가설이지 확정 원인이 아니다.**

코드 위치: [수정본 dt projection](/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/main_files/model.py:39), [수정본 Δ 계산](/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/main_files/model.py:67), [전체 파라미터 Adam 설정](/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/main_files/core/model_trainer.py:67). 공식 Mamba의 입력 의존 Δ 및 A/D 설정은 [공식 구현](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)과 대조했다. 공식 초기화/decay 규칙과 다르다는 사실만으로 원인이나 성능 개선을 확정하지 않는다.

## 2. Mamba 전체가 사라졌다는 결론은 아님

Legacy 여섯 모델의 exp(A_log)는 초기1…8에서 최종0.935755…1.024741로 좁아졌고, softplus(dt_bias)는0.676474…0.716146이다. A_log→0은 A→−1이며 A→0이 아니다. D·u 직접 전달, gate, residual, 일부 B/C와 recurrent 성분은 남아 있다. 부모 hook의 첫 stage 두 블록에서는 recurrent/D·u RMS 비가0…0.0004825였지만 중간 블록에서는 약0.1…0.8 수준인 사례도 있다. LayerNorm/gate 이전의 비율이며 최종 성능 기여도나 전체 출력의 동일 비율로 해석하지 않는다.

직접 상태 전달은 exp(A·Δ)의 누적 곱이다. 실제 spiral에서 같은 xy의 인접 z단면 사이 scan 거리 중앙값은 단계별1024/256/64 token이다. 직접 상태 감쇠가 빠르면 이 경로는 약해질 수 있다. 그러나 3D convolution·여러 층·residual·gate 및 legacy 잘못된 방향 결합도 있으므로 전체 모델의 장거리 의존성이0이라고 말하지 않는다.

## 3. 메모리 자체의 구조와 encoder 쪽 단서

현재 memory는 위치별 q에 대해 a=softmax(Mq), r=Mᵀa를 계산하고16×8×8의 공간 배열을 유지한다. 300개 슬롯 중 하나로 반드시 이산 치환되는 구조가 아니다. Tied key/value에서 ∂r/∂q=Σaᵢ(mᵢ−r)(mᵢ−r)ᵀ이며 실제 source class의 CPU autograd와 최대차2.78e−17로 일치했다. 따라서 동일/근접 벡터와 한 슬롯 집중은 encoder에 전달되는 reconstruction gradient도 약하게 할 수 있다. 실제 환자 운용점에서의 크기 및 성능 기여는 별도 측정이 필요하다.

M0 seed42는 exact unique memory vector가3개라 각 위치의 readout은 최대2차원 affine span에 속한다. 반대로 M3의 실제 memory는 numerical rank128이고, 합성 q=0에서 Jacobian 최소고윳값이0.0009785/0.0009580으로 양수였다. 이는 구조적 full-rank 가능성을 보인 것이며 실제 환자 query의 Jacobian을 측정한 것은 아니다.

| 조건 | seed | query rank99 | readout rank99 | memory rank99 |
| --- | --- | --- | --- | --- |
| M0 | 42 | 4 | 1 | 2 |
| M0 | 123 | 6 | 2 | 3 |
| M1 | 42 | 6 | 4 | 8 |
| M1 | 123 | 7 | 5 | 8 |
| M3 | 42 | 7 | 9 | 118 |
| M3 | 123 | 7 | 8 | 119 |

정상 TRAIN 전체956patch의 중심화 covariance를 사용한99% 분산 차원이며 정보이론적 차원이 아니다. 1,024개 위치의 계수가 각각 다를 수 있고, 작은 분산 방향도 decoder가 사용할 수 있다. Memory만 rich128차원 입력을 갑자기8차원으로 떨어뜨린다는 해석은 맞지 않는다.

## 4. 이미 알려진 구현 오류와 배제한 의심

- Legacy는 방향별 grouped x_proj 출력을 전역 split하여 dt/B/C가 섞이고, hardtanh(softmax,0,1)는 실제 shrinkage가 아니다. 모두 기존에 확인했던 문제다. 현재 수정본에서는 고쳤으며, 최근 M0/M1/M3는 원인 분리 때문에 frozen legacy를 유지했다.
- Legacy entropy는 올바른 query entropy의1024/300배이며 고정 shape에서는 상수 배율 문제다. M1/M3 λ=0에서는 해당 항의 영향이0이다.
- Memory Parameter 등록·optimizer 포함·gradient 연결·tensor 축/transpose는 정상이며 memory 전체를 우회하는 encoder→decoder skip이 없다. VSSBlock 내부 residual과 D·u는 존재한다.
- 실제 세 단계의 scan에는 중복/누락이 없고 모든 연속 token은 L1거리1이다. 방향별 unscan은 x, 양방향 sum은2x로 정확하다. Sum은 의도된 merge다.
- Corrected hard shrink에서 생존 슬롯1개면 query gradient가0인 구간이 생긴다. 이는 조건부 설계 특성이며 현재 frozen 모델의 원인은 아니다. 300슬롯/threshold0.0025에서는 finite softmax의 all-zero support는 불가능하다.
- Corrected kernel3/stride2 transpose에는 합성 ones 검사상 위치별 겹침1/2/4/8 차이가 있다. 실제 학습 결과의 checkerboard나 성능 저하를 입증한 것은 아니다. Legacy kernel2의 원인으로 사용하지 않는다.

## 5. 정상 복원과 질환 점수의 연결

메모리 전후 shape는128×16×8×8=131,072 scalars로 입력65,536개보다 많다. 이것은 정보가 더 많다는 뜻이 아니라, shape만으로 강한 정보 병목을 주장할 수 없다는 뜻이다. 병변 위치를 모르는 전체 voxel MSE→환자 max는 정상 변이·배경·보간 오차와 국소 병변 잔차를 구별하지 않는다. 실제 어느 부위가 점수를 지배하는지는 병변 주석 또는 사전 고정된 국소/전역 perturbation으로 검증해야 한다. 이런 차이는 [MemAE 원 논문](https://arxiv.org/html/1904.02639v2)의 정상 복원 기반 가설과 구분하여 다룬다. 같은 patch shape에서 MSE→SSE만 바꾸면 양의 상수 배율이므로 AUROC/AP 개선이 생기지 않는다.

Memory 필요성은 동일 조건의 no-memory AE를 처음부터 학습한 대조가 필요하다. 추론 중 memory를 파괴해 성능이 낮아지는 것은 그 대조를 대신하지 못한다. SSM 효과도 matched local-conv 대조로 확인해야 한다. 이번 감사에서 이 재학습을 시작하지 않았다.

## 6. 실행·검증·다음 우선순위

- 독립 source/CPU 검토3개, legacy6개 최종 및 저장 시점·초기 checkpoint, corrected20개 checkpoint.
- 정상 TRAIN3개 고정patch×6개 legacy모델의 read-only GPU SSM 관찰. 관찰 hook 전후 출력·attention이 정확히 같고 모델 상태/checkpoint hash 불변. 질환/test자료 로드 없음.
- 별도108개 dt projection 관찰에서 bias-only와 정확히 일치.
- 기존 model correction 테스트3개 통과.
- 실제 installed CUDA oflex와 독립 CPU recurrence를 합성 입력에서 forward 및 u/Δ/A/B/C/D gradient로 비교하여 atol2e−6+rtol2e−4 기준 통과. 출력 최대차4.77e−7, Δ gradient 상대L2차3.34e−7. 전체실행/모든shape의무오류를보장하는검사는아니다.

다음 권고는 초기~수축 전 시점에서 SSM 파라미터별 reconstruction gradient와 decay 기여를 분리하고, Δ를 만드는 두 투영·A/D의 유지 여부를 조건으로 삼는 최소 optimizer/초기화 대조를 설계하는 것이다. 그 뒤 memory 기하와 no-memory AE 대조를 확인한다. ‘decay를 빼면 성능이 오른다’는 결론이나 새 학습은 아직 없다. 기존 λ 후보는 제안으로 보존한다.

## 근거 파일

- 실행/기능 진단: `outputs/model_structure_audit_20261110T152700Z/protocol.json`, `results.json`, `delta_input_dependence.json`, `cuda_gradient_parity.json`
- 독립 memory 보고서: [원문](/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/outputs/model_structure_audit_20261110T152700Z/independent_reviews/vmamba_audit_memory_20261111.md)
- 독립 SSM 보고서: [원문](/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/outputs/model_structure_audit_20261110T152700Z/independent_reviews/vmamba_audit_ssm_20261111.md)
- 독립 이상탐지 설계 보고서: [원문](/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/outputs/model_structure_audit_20261110T152700Z/independent_reviews/vmamba_audit_anomaly_20261111.md)
- `independent_reviews/`에 모든 CPU진단 script/JSON 보존. 원영상은 업로드하지 않았다.
