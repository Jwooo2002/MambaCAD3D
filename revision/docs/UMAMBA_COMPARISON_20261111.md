# U-Mamba의 CNN–SSM 연결과 우리 모델 비교 — 2026-11-11

## 결론

**CNN decoder가 SSM 뒤에 있다는 것 자체는 문제가 아니다. U-Mamba도 CNN decoder를 쓴다.** 기존 권고에 추가해야 할 것은 decoder·memory를 통과하는 학습 신호와 SSM의 실제 기여를 구분하는 진단이다. 우리 실험에서 decay 적용 변경으로 Δ 경로가 보존된 결과는 유효하지만, 초기 복원 gradient가 왜 작았는지는 별도 문제다. CNN decoder가 그 원인이라는 확정 근거는 아직 없다.

사용자 요청은 비교와 제안이며 모델·trainer·진행 중 실행·queue는 변경하지 않았다. 추가 GPU 평가/학습도 하지 않았다. 공식 U-Mamba의 3D Enc/Bot·trainer·초기화·loss 등10파일을 확인했고 `outputs/umamba_comparison_20261111/sources.json`에 고정 commit 28459e33ca03769800dd35e23c6e62491d1925b5로 보존했다.

## 비교한 실제 흐름

- 우리: CNN stem → [x + SS3D(LN(x)), CNN downsample] encoder → memory → Conv3d+GELU decoder → 복원 MSE.
- U-Mamba Enc: residual CNN+InstanceNorm → LN+Mamba를 각 encoder stage에서 순차 적용 → encoder의 여러 해상도 출력을 받는 residual CNN decoder → segmentation loss.
- U-Mamba Bot: residual CNN encoder → 가장 깊은 출력에만 Mamba → CNN decoder. 얕은 encoder 출력은 U-Net skip으로 decoder에 전달한다.

우리 모델에는 **encoder→decoder의 장거리 U-Net skip이 없고 모든 복원 정보가 memory 출력에서 시작한다.** VSSBlock 내부 residual과 SSM의 D·u 직접 경로는 따로 존재한다. CNN이 “memory를 통째로 건너뛴다”는 설명은 부정확하다. 메모리에는 공간 위치별 혼합 코드가 남아 있으므로 CNN이 그 표현을 활용하며 SSM 재귀 성분에는 덜 의존할 가능성은 있다.

## 코드로 확인한 차이

| 항목 | 우리 현재 코드 | U-Mamba 공식 3D 구현 |
| --- | --- | --- |
| Decoder | Conv3d(3³)+GELU 반복, 내부 norm/residual 없음 | BasicResBlock과 추가 residual block; InstanceNorm, LeakyReLU |
| CNN 초기화 | Conv3d/ConvTranspose3d/Linear에 공통 trunc_normal std=0.02 | 2D/3D convolution 계열에 Kaiming normal(He), bias=0 |
| Upsampling | kernel3, stride2 ConvTranspose3d | nearest interpolation + 1×1 convolution |
| Encoder→decoder 연결 | memory 출력 하나를 decode, U-Net skip 없음 | 여러 해상도 skip을 concatenate |
| Mamba wrapper | x + SS3D(LN(x)) | 해당 MambaLayer는 Mamba(LN(x))를 반환; wrapper 바깥 x+를 두지 않음 |
| 학습 신호 | 최종 복원 MSE + memory entropy; 현재 원인 분리 실험은 entropy=0 | label 기반 Dice+CE(일반 class 설정), 여러 decoder 출력의 deep supervision |
| Optimizer | 원 trainer 전체 Adam decay; 현재 S0–S7은 적용 범위/방식 대조 | 기본 trainer 전체 파라미터 SGD, lr=0.01, decay=3e-5, momentum=0.99, Nesterov |
| Tokenization | 3D spiral 정/역방향 | Bot은 공간 flatten; Enc는 stage 크기에 따라 spatial token 또는 channel token |
| Precision | 현재 진단에서 FP32 | MambaLayer에서 autocast를 끄고 저정밀 입력을 FP32로 변환; 별도 NoAMP trainer도 제공 |

현재 진단 config는 dims=[32,64,128], depths=[2,2,2]다. 따라서 memory 이후 기본 decoder 경로에는 **Conv+GELU 6개, ConvTranspose3d 2개, 마지막 Conv3d 1개**가 있다. 이 깊이/초기화/정규화 차이는 gradient 전달을 검사할 구체적 이유가 되지만, “9개 convolution이라 gradient가 사라진다”는 인과 결론은 아니다.

### 직접 확인한 경로와 한계

- [U-Mamba Enc: Mamba wrapper](https://github.com/bowang-lab/U-Mamba/blob/28459e33ca03769800dd35e23c6e62491d1925b5/umamba/nnunetv2/nets/UMambaEnc_3d.py#L42-L91), [residual CNN](https://github.com/bowang-lab/U-Mamba/blob/28459e33ca03769800dd35e23c6e62491d1925b5/umamba/nnunetv2/nets/UMambaEnc_3d.py#L94-L131), [CNN 다음 Mamba](https://github.com/bowang-lab/U-Mamba/blob/28459e33ca03769800dd35e23c6e62491d1925b5/umamba/nnunetv2/nets/UMambaEnc_3d.py#L281-L292).
- [Decoder 및 skip/중간 출력](https://github.com/bowang-lab/U-Mamba/blob/28459e33ca03769800dd35e23c6e62491d1925b5/umamba/nnunetv2/nets/UMambaEnc_3d.py#L307-L396), [InstanceNorm 설정·초기화 호출](https://github.com/bowang-lab/U-Mamba/blob/28459e33ca03769800dd35e23c6e62491d1925b5/umamba/nnunetv2/nets/UMambaEnc_3d.py#L510-L539), [He initializer](https://github.com/bowang-lab/U-Mamba/blob/28459e33ca03769800dd35e23c6e62491d1925b5/umamba/nnunetv2/utilities/network_initialization.py).
- [Bot의 bottleneck만 Mamba](https://github.com/bowang-lab/U-Mamba/blob/28459e33ca03769800dd35e23c6e62491d1925b5/umamba/nnunetv2/nets/UMambaBot_3d.py#L433-L440).
- [기본 optimizer](https://github.com/bowang-lab/U-Mamba/blob/28459e33ca03769800dd35e23c6e62491d1925b5/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L470-L474), [lr/decay/deep-supervision 설정](https://github.com/bowang-lab/U-Mamba/blob/28459e33ca03769800dd35e23c6e62491d1925b5/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L141-L149), [실제 multi-output loss](https://github.com/bowang-lab/U-Mamba/blob/28459e33ca03769800dd35e23c6e62491d1925b5/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L350-L374).
- [논문](https://arxiv.org/html/2401.04722v1), [공식 프로젝트](https://u-mamba.github.io/).
- 논문 본문은 decoder에 transposed convolution을 설명하지만 **확인한 GitHub revision의 UpsampleLayer는 nearest interpolation+1×1 conv**다. 논문 그림만 보고 현재 코드를 transposed convolution이라고 기록하지 않는다.
- Deep-supervision 코드의 마지막 저해상도 loss 가중치는0이다. 모든 해상도에 동일한 loss를 준다고 설명하지 않는다.
- U-Mamba는 mamba_ssm을 외부에서 import하고 설치 README가 버전을 고정하지 않는다. wrapper와 trainer 코드는 확인했지만 2024년 각 실험의 정확한 dependency binary까지 재현한 것은 아니다.
- U-Mamba 결과는 지도 segmentation이다. 정상만 학습하는 memory AE의 병변 탐지 성능과 직접 비교하지 않는다. 코드 차이만으로 residual/norm/deep supervision 각각의 효과를 분리할 수도 없다.

## 기존 결론의 보완

U-Mamba는 SSM에 별도 decay=0 group을 만들지 않고 전체 SGD를 사용한다. 따라서 “SSM 모델의 필수 규칙이 decay 제외”라는 일반화는 성립하지 않는다. 우리 후보 S3의 근거는 **우리 coupled Adam·복원 손실·memory 병목 조건에서 같은 초기값과 순서의 대조로 경로 보존이 나뉜 사실**이다. U-Mamba의 optimizer/손실/초기화/연결이 모두 다르므로 그 학습 성공이 우리 대조를 반박하는 것도 아니다. 반대로 U-Mamba의 성공을 근거로 우리 모델도 SGD로 바꾸면 해결된다고 단정하지 않는다.

기존 S0/S1 진단은 일부 고정 정상 patch에서 raw recurrent 성분이 수치상0이어도 정상 복원 loss가 감소하는 현상을 보여준다. 이는 **재귀 SSM이 약해도 다른 경로가 복원을 학습할 수 있다는 관찰**이다. CNN decoder가 단독으로 병목을 일으켰다거나 모든 input에서 SSM이 불필요하다는 증거는 아니다.

## 권고 변경

| 우선순위 | 제안 | 범위와 판정 |
| --- | --- | --- |
| 먼저 진단 | 같은 정상 입력·같은 MSE에서 출력→decoder 각 stage→memory 출력→memory 입력(encoder 출력)→SSM의 activation/gradient RMS를 기록 | 어느 경계에서 신호가 줄어드는지 분리. 단순 gradient norm만 비교해 채널 수/해상도 차이를 혼동하지 않음. weight decay/clipping 전 task gradient도 분리 |
| 먼저 기여 확인 | S0/S3/S4의 고정 checkpoint에서 raw scan의 재귀 성분만 제거하고 D·u, CNN/gate/residual, memory를 유지하여 복원·latent 변화 확인 | 추론 개입은 현재 모델의 의존성 진단이며, 처음부터 학습한 matched local-conv/no-recurrence 대조를 대신하지 않음 |
| 작은 새 후보 | 경로가 보존된 S3 설정에서 **CNN decoder 초기화만 fan-in을 반영하는 방식(He 계열 후보)**으로 변경 | 현재 std=0.02와 대조. GELU와 LeakyReLU가 다르므로 U-Mamba 값을 무비판적으로 최적값 취급하지 않음. SSM·memory·loss·배치 순서 유지 |
| 필요할 때 구조 후보 | decoder 내부 residual block 대조. Norm은 별도 변수로 분리 | 같은 memory 출력에서 시작하는 내부 residual은 memory를 우회하지 않음. block 연결 변경이므로 단순 optimizer 조정과 구분하여 방법/ablation에 기록 |
| 후순위 구조 후보 | 필요하면 nearest+1×1 upsampling 대조 | 실제 코드 차이를 재현하는 후보일 뿐 현재 ConvTranspose가 실패 원인/격자 artifact를 만든다는 관찰은 없음 |
| 현재 채택 안 함 | U-Net encoder→decoder skip, 원고 손실에 없는 deep supervision, encoder/bottleneck Mamba 재배치 | 구조 또는 목적식 변경. 특히 U-Net skip은 memory를 거치지 않는 정보 경로를 새로 만들어 병목 의미를 바꾸며, 이상 복원에 미칠 영향도 따로 검증해야 함 |

InstanceNorm을 바로 추가하는 것도 권하지 않는다. Feature의 강도/대비 정보를 바꾸는 설계이며 segmentation에서의 효용이 HU 복원·이상 점수에서 동일하다고 볼 수 없다. Decoder의 residual 추가와 norm 추가를 묶으면 개선 원인을 분리하기 어려워진다.

최종 채택은 정상 TRAIN MSE만으로 하지 않는다. 학습 경로 보존 확인 후 동일 fold/seed·사전 고정 모델 선택·평가 규칙에서 AUROC/AP를 비교한다. Mamba-3 교체보다 **현재 CNN–memory–SSM 경계의 신호 진단과 CNN 초기화 단독 대조**가 지금 질문에 직접 대응하는 후속 작업이다.

