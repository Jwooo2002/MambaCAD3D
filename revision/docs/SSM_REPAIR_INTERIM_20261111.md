# SSM 복구 중간 분석 — 2026-11-11

16회 중 **14회 완료, S7의 두 seed 진행 중**인 시점의 분석이다. 완료된 S0–S6만 같은 epoch200에서 비교했다. 진행 중인 S7의 짧은 학습 결과를 최종 비교에 섞지 않았다. 실행과 설정을 변경하지 않았으며 질환·validation·test를 평가하지 않았다.

## 핵심 결론

**현재 실험 배경에서 Adam의 결합 weight decay를 SSM 전체에 일괄 적용하는 것이 입력 의존 Δ 경로를 수축시키는 주요 원인이라는 강한 대조 근거를 얻었다. Δ 최종 투영만 보호해서는 해결되지 않고, 그 입력과 B/C를 만드는 x_proj 및 상태 파라미터를 함께 다뤄야 한다.**

기존에는 가중치가 작다는 관찰만 있었다. 이제 같은 초기값·배치 순서·학습 예산의 개입에서 소실과 보존이 나뉜다. 다만 x_proj/A_log/D 중 최소한 어느 조합을 제외해야 하는지는 이번 S2 묶음 대조만으로 분리되지 않는다. 원래 논문 성능 저하의 모든 원인을 확정한 것도 아니다.

## 고정 epoch200 결과

MSE는 정상 **TRAIN 전체15명/956patch**의 값이다. 표의 수치는 seed42 / seed123 순서이고 MSE 열은 ×10⁻⁴ 단위다. Δ 기능은 고정 정상3patch×6블록×2seed=36회 관찰이다. 전체 환자·입력에 대한 증명이나 탐지 성능 지표가 아니다.

| 조건 | Δ 기능 관찰 | 정상 TRAIN MSE ×10⁻⁴ | 확인한 내용 |
| --- | --- | --- | --- |
| S0 기존 Adam 설정 | 36/36 bias-only | 5.173 / 6.721 | dt와 x_proj가 극소값으로 수축, 측정 raw recurrent 성분0 |
| S1 dt_proj만 decay 제외 | 36/36 bias-only | 5.208 / 4.820 | dt 가중치는 보존되지만 x_proj가 사라져 기능은 여전히 소실 |
| S2 x_proj·dt_proj·A_log·D 제외 | 0/36 bias-only | 13.177 / 4.294 | 두 seed 모두 입력 의존성과 재귀 성분 보존; 복원 개선은 seed 일관성이 없음 |
| S3 SS3D 내부 전체 제외 | 0/36 bias-only | 4.477 / 3.982 | 두 seed 모두 경로 보존, 정상 복원도 기준보다 낮은 오차 |
| S4 AdamW | 0/36 bias-only | 0.503 / 0.884 | 경로 보존, 정상 TRAIN 오차가 S0의0.097/0.131배 |
| S5 dt bias 초기화만 변경 | 36/36 bias-only | 15.986 / 4.983 | 초기화만으로 소실을 막지 못함 |
| S6 S2 + dt bias 초기화 | 0/36 bias-only | 5.543 / 6.035 | 두 seed 보존. S2 대비 복원 변화는 서로 반대 방향이며 S3/S4보다 좋은 정상 MSE도 아님 |

S0/S1의 raw scan recurrent 출력은 36회 모두 float32에서 D·u를 뺀 결과가0이다. 이를 전체 모델이0 또는 Mamba 블록 전체가 제거됐다는 뜻으로 해석하지 않는다. D 직접 전달, gate, normalization, residual 등은 남아 있다. S5 seed42는 일부 미약한 recurrence가 남으므로 S5까지 전부0이라고 표현하지 않는다.

## 인과 설명을 지지하는 근거

### 1. 초기 task gradient보다 decay 항이 훨씬 크다

S0 초기 고정 정상 patch에서 여섯 dt weight tensor의 reconstruction gradient L2는 seed42에서2.51×10⁻¹³–3.15×10⁻¹², seed123에서5.15×10⁻¹³–2.69×10⁻¹²다. 반면 decay×parameter L2는 약6.56×10⁻⁵–1.31×10⁻⁴다. Trainer와 같은 clipping을 반영한 비율은 **약2.74×10⁷–4.90×10⁸배**다. x_proj에서도 약2.50×10⁶–1.43×10⁷배다.

이는 초기 고정 한 patch의 gradient 진단이며 모든 minibatch의 gradient를 저장한 결과는 아니다. Norm 비율만으로 누적 optimizer 효과를 확정하지 않는다. 실제 S0/S1/S2의 개입 및 시간 경과 결과와 함께 해석한다. 이 코드에서는 task gradient를 clipping한 뒤 Adam이 decay 항을 더해 moment에 반영한다.

### 2. 최종 Δ 투영만 살려도 upstream이 사라지면 작동하지 않는다

S1의 epoch200 dt weight L2는6.56–13.11 정도로 보존된다. 하지만 x_proj weight L2는 약2.39×10⁻³⁸–5.53×10⁻³⁸로 수축한다. 따라서 **가중치 norm이 보존됐다는 지표만으로 성공을 판단하면 오판한다.** 실제로 두 seed의18회씩 관찰 모두 Δ가 bias-only이며 재귀 성분도 수치상0이다.

반면 S2는 두 seed 모두36회 중 bias-only가0이고, 재귀/D·u RMS 중앙값은0.360/0.0995다. S3도0.297/0.0949다. 이 비율은 LayerNorm·gate 이전 값이며 최종 출력에 대한 인과적 중요도 수치가 아니다.

### 3. 초기화가 단독 해결책이라는 가설은 지지되지 않는다

S5는 bias를 바꿨지만 두 seed 모두 dt weight가 소멸하고 입력 의존성도 사라졌다. S0는 저장된 epoch30부터 모든18회 관찰이 bias-only이고, S5도 epoch30까지 같은 상태에 도달한다. 먼저 decay 문제를 다루지 않은 초기화 변경만으로는 해결되지 않았다.

## 메모리와 AdamW의 해석

이번 모든 조건은 **memory decay=0, entropy=0**인 공통 기전 배경을 사용한다. S0 자체도 메모리 중복 이웃 최대크기가1이고 memory rank99가114/117이다. 따라서 이번 S0의 SSM 실패를 다시 메모리 벡터 중복 하나로 설명할 수 없다. 다만 이 지표가 의미 있는 정상 prototype 학습을 보장하지는 않는다.

S4에서 전체 정상 TRAIN query rank99는25/29로 S0의5/4보다 높다. 복원 오차는약10.3/7.6배 낮다. 하지만 실제 전체 슬롯 사용의 유효 개수는약26/26이고 memory rank99는120/120이다. 모든300슬롯을 균등 사용하는 것이 목표는 아니며, 이 상태가 탐지에 좋은지는 미확인이다.

**S4의 결과를 AdamW 이름 자체의 우월성으로 해석하면 안 된다.** 같은 weight_decay=1e-5라도 AdamW의 직접 감소율은 lr×weight_decay이며 초기 lr=1e-4에서1e-9다. 기존 결합 Adam과 규제 강도를 맞춘 대조가 아니다. 설치 PyTorch2.8의 코드와 작은 CPU float32 zero-gradient 예제에서 AdamW의 이 감소는 반올림으로 파라미터 변화가 없음을 확인했다. 이 예제로 GPU 전체 학습이 Adam(decay=0)과 bitwise 동일하다고 주장하지 않는다. S4의 큰 복원 개선에는 SSM 밖 encoder/decoder 규제의 약화도 포함되므로 SSM 하나의 효과로 돌릴 수 없다.

## 다음 판단

- SSM 기능 소실 문제를 고칠 우선순위는 **decay 대상과 적용 방식**이다. VRAM을 채우거나 memory 크기를 키우는 것이 이 결과에서 나온 해결책은 아니다.
- 구조를 유지하는 후보로 S3(SS3D 내부 decay 제외)와 S4(AdamW)를 남길 근거가 생겼다. S2는 경로를 살려도 seed42 정상 복원이 나빠서 경로 보존과 성능을 같은 것으로 취급하지 않는다.
- S7의 사전 고정 실행은 그대로 마친다. S4의 원인을 더 분리하려면 같은 조건의 Adam(decay=0) 대조가 유용하지만 이번 분석에서 추가 실행하지 않았다.
- **탐지 문제 해결은 아직 미확인이다.** 정상 TRAIN MSE 감소는 질환 AUROC/AP 상승이나 일반화를 입증하지 않는다. 더 잘 복원되는 모델이 병변까지 잘 복원할 가능성도 있어, 모델 채택에는 별도의 사전 고정 평가가 필요하다.
- 현재는 수정 모델 + frozen legacy 입력 + λ=0 배경이다. 원래 손실을 유지한 정식 k-fold 조건까지 일반화하지 않는다.

## 재현 근거

분석: `memory_pilot/analyze_ssm_partial.py`.
고정 분석 snapshot: `outputs/ssm_repair_analysis_20261111T051100Z/summary.json`, `table.md`.
원자료: `outputs/ssm_repair_20261110T155100Z`의 epoch별 probe, 전체 TRAIN 진단, 초기화·순서 hash 및 manifest.
분석 JSON에는 최종 입력 JSON의 SHA256, 조건별 수치, 초기 gradient 및 경로 소실 timeline을 보존했다. 원본 CT와 checkpoint를 변경하지 않았다.
