# 공식 SSM 구현 비교와 수정 권고 — 2026-11-11

## 결론과 범위

**우선 수정 후보는 현재 SS3D의 optimizer grouping과 decay 적용 방식이다. Mamba-3 교체는 지금 채택하지 않는다.** 먼저 SSM이 실제로 입력에 반응하도록 학습을 복구한 뒤 탐지 성능을 판단한다. 사용자의 이번 요청에 따라 제안만 정리했으며 모델·trainer·실행 queue는 변경하지 않았다.

S4/S4D, Mamba-1/2/3, VMamba, Vim, MambaVision의 논문과 공식 GitHub를 읽었다. 공식 저장소 5개와 Vim 의존성 timm의 고정 버전에서 총22파일을 확인했다. 읽은 source와 commit은 `outputs/ssm_official_comparison_20261111/sources.json`에 보존했다. 아래는 해당 revision의 정적 코드 감사이며 공식 모델을 설치하거나 재학습한 결과는 아니다. 논문의 모든 실험 설정을 대표한다고 일반화하지 않는다.

## 공식 구현과 우리 코드

| 대상 | 직접 확인한 구현 | 우리 코드와의 차이 및 해석 |
| --- | --- | --- |
| S4/S4D | 커널의 log_dt, log_A_real, A_imag에 별도 optimizer 설정을 등록하고, example의 AdamW가 실제 별도 group으로 적용한다. | 민감한 SSM 파라미터를 일반 가중치와 구분한다. 우리 main trainer는 하나의 Adam group으로 전체에 decay를 적용한다. S4D의 C까지 모두 제외된다고 일반화하지 않는다. |
| Mamba-1 | dt bias를 inverse-softplus(log-uniform[0.001,0.1])로 초기화. A_log/D에 no-decay 표시. x_proj는 bias=False. | 우리 dt bias는 Conv1d 기본값이고 x_proj는 bias=True. A의 초기 형태(log(1..N))와 D=1은 이미 같다. dt weight의 기본 uniform 범위도 dt_rank^-0.5로 이미 대응한다. 표시만으로 optimizer 제외가 자동 적용되지는 않는다. |
| Mamba-2 | head 단위 상태 구조, dt_bias/A_log/D에 no-decay 표시, 명시적 Δ 초기화. | Mamba-1형 SS3D와 상태 구조·projection 형태가 달라 버전명만 바꾸는 교체가 아니다. dt/x_proj 전체 제외를 공식 규칙으로 주장할 수 없다. |
| VMamba | 2D 네 방향 scan, AdamW, 1D tensor·.bias·명시 skip 목록으로 decay group 분리. 기본 SSM_INIT=v0는 명시적 dt bias 초기화를 사용. | 우리는 3D spiral 두 방향이며 optimizer group이 없다. 방향 개수는 설계 차이이지 현재 수축의 원인으로 입증되지 않았다. VMamba에서도 모든 SSM 파라미터가 decay 제외되는 것은 아니다. |
| Vim | 양방향 Mamba, 전방 dt_proj의 특수 bias 초기화와 이를 보존하는 _no_reinit 처리. main의 기본 optimizer는 AdamW. | 두 방향이라는 이유만으로 우리 모델이 잘못됐다고 할 수 없다. 실제 timm0.4.12 grouping을 추적하면 A_log의 표시를 직접 읽지는 않는다. v2 역방향 dt_proj_b는 전방과 같은 특수 초기화가 아니므로 “모든 방향이 동일 초기화”라는 주장도 피한다. |
| MambaVision | Mamba와 attention의 혼합 구조. 공개 Tiny 설정은 LAMB이며 train CLI 기본값만 AdamW다. | attention 추가는 주요 모델 변경이다. 유명 모델들이 모두 AdamW라는 주장은 틀리다. 해당 revision에서는 mixer의 dt 특수 초기화 후 top-level Linear 초기화가 덮어쓰므로 constructor만 보고 최종 초기화를 판단하면 안 된다. |
| Mamba-3 | Δ 특수 초기화, B/C RMSNorm·bias, 입력 의존 A, 회전/복소 상태, 새 recurrence, 선택적 MIMO. | 선택성·상태 갱신·projection 자체가 바뀐다. 현재 학습 수축을 해결해야 한다는 근거에서 Mamba-3 교체 필요성이 바로 따라오지는 않는다. |

### optimizer 표시와 실제 적용을 구분한 근거

- S4D의 register는 실제 `_optim={weight_decay:0}`을 만들고 [example의 optimizer](https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/example.py#L247-L276)가 이 속성을 읽어 add_param_group을 한다. [S4D kernel](https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/models/s4/s4d.py#L14-L59).
- [Mamba-1](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py#L77-L115)은 A_log/D에 표시하지만 x_proj/dt_proj.weight 전체에 no-decay를 표시하지 않는다. [Mamba-2](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba2.py#L119-L140)는 dt_bias도 표시한다. 이 모듈 파일만으로 외부 trainer의 실제 제외 여부까지 확정하지 않는다.
- [VMamba optimizer](https://github.com/MzeroMiko/VMamba/blob/2ed52ead062a51a64521ed3871d52914bf532876/classification/utils/optimizer.py#L18-L68)는 `_no_weight_decay` 속성을 검사하지 않는다. 해당 model의 skip은 pos_embed, skip keyword는 빈 목록이다. 따라서 2D A_logs와 stacked 2D dt_projs_bias는 일반 decay group에 들어가며 1D Ds는 제외된다. [모델 skip](https://github.com/MzeroMiko/VMamba/blob/2ed52ead062a51a64521ed3871d52914bf532876/classification/models/vmamba.py#L1374-L1391). `_bias`는 `.bias`와 다르다. 정적 코드 경로에 따른 판정이다.
- Vim의 [skip 목록](https://github.com/hustvl/Vim/blob/dd0358ad1e42701f22afbefa0717cc8825cf9f45/vim/models_mamba.py#L377-L379)과 [timm0.4.12 grouping](https://github.com/huggingface/pytorch-image-models/blob/7096b52a613eefb4f6d8107366611c8983478b19/timm/optim/optim_factory.py#L29-L41)도 A_log의 속성을 읽지 않는다. 따라서 **공식 모델이면 A 전체가 항상 decay 제외된다는 주장은 하지 않는다.**
- MambaVision의 [Tiny 설정](https://github.com/NVlabs/MambaVision/blob/7860a506b2eb844eaaae676f08461ce8c3c26f43/mambavision/configs/mambavision_tiny_1k.yaml)은 opt=lamb, weight_decay=0.05다. [top-level initializer](https://github.com/NVlabs/MambaVision/blob/7860a506b2eb844eaaae676f08461ce8c3c26f43/mambavision/models/mamba_vision.py#L694-L700)는 _no_reinit을 검사하지 않는다. 배포 체크포인트의 학습 이력을 재현한 것은 아니므로 이 코드 관찰을 모든 공개 모델의 실패로 확대하지 않는다.

## 권고 우선순위

| 순서 | 구체적 제안 | 근거 / 판단 기준 |
| --- | --- | --- |
| 1A: 최소 개입 기준 | 현재 Adam을 유지하고 SS3D 내부를 명시적으로 decay=0 group에 둔다(S3 방식). | 우리 두 seed에서 Δ 경로 보존 및 정상 TRAIN MSE 개선. 공식 모델의 보편 규칙이 아니라 우리 실험에 근거한 후보다. |
| 1B: 후속 optimizer 후보 | AdamW + 명시적인 group을 검토한다. A_log, D, dt bias, norm/bias의 제외 여부를 이름과 실제 optimizer 값으로 기록한다. | S4 후보는 이미 경로 보존과 큰 정상 TRAIN 개선을 보였다. 다만 그룹을 정리한 AdamW 조합 자체는 아직 실행되지 않았다. 전체 SS3D 제외까지 한 번에 추가하지 말고 범위 효과를 분리한다. |
| 2: 시간척도 초기화 | optimizer 문제를 다룬 뒤 dt bias를 log-uniform[0.001,0.1]의 inverse-softplus로 초기화하는 후보를 대조한다. 모델 전체 초기화가 끝난 값도 점검한다. | Mamba-1/2/3의 코드 근거. 우리 S5는 초기화만으로 실패했고 S6도 일관된 개선이 아니므로 필수 성능 개선으로 단정하지 않는다. dt weight는 이미 같은 uniform 범위라 불필요하게 다시 “교정”하지 않는다. |
| 3: 관찰 가능한 학습 | 실제 optimizer group, Δ 입력 의존성, recurrent 성분, x_proj/dt_proj norm·gradient를 기록한다. | S1은 dt weight가 살아 있어도 x_proj 소실로 기능을 잃었다. loss와 weight norm만으로 성공을 판단하면 안 된다. |

우선 코드 한 군데를 고른다면 `main_files/core/model_trainer.py:67`의 전체 파라미터 단일 Adam group을 바꾸겠다. 초기화 후보의 대상은 `main_files/model.py:39`의 dt_proj.bias이며, 모든 모듈을 통째로 교체하는 것보다 뒤에 둔다.

AdamW의 같은 decay 숫자는 Adam과 규제 강도가 같지 않다. 현재 S4는 초기 lr×decay=1e-9라 수축이 극미약하고 SSM 밖 규제도 바뀐다. 따라서 S4의 향상을 optimizer 이름이나 SSM 하나의 효과로 돌리지 않는다. 추가 원인 대조로 Adam(decay=0)이 유용하지만 이번 요청에서는 실행하지 않았다. 공식 ImageNet의 weight_decay=0.05와 learning rate를 소수 정상 환자 학습에 그대로 복사하지 않는다.

Memory decay=0, entropy=0은 현재 공통 진단 배경이다. 이것을 원래 논문 목적식과 동일한 최종 설정으로 취급하지 않는다. 정상 TRAIN MSE가 낮아졌다는 것과 질환 AUROC/AP 개선은 별개이며, 채택 전 같은 fold/seed·사전 고정 checkpoint 선정·평가 규칙으로 검증해야 한다.

## 구조 변경 후보는 구분

- **x_proj bias=False:** Mamba-1/VMamba와 다른 점은 확인했지만 우리 수축의 원인이라고 입증하지 않았다. optimizer 수정과 동시에 섞지 말고 필요할 때 별도 ablation으로 둔다.
- **3D scan 방향 확대:** VMamba의 2D 네 방향을 그대로 3D에 복사할 근거가 없다. Vim도 양방향을 쓴다. 현재는 방향별 packing을 교정한 동일 구조에서 optimizer 개입으로 기능 보존이 갈리므로 우선순위가 낮다.
- **B/C normalization, attention 추가, Mamba-2/3 교체:** 새로운 블록 설계/표현력 실험이다. minor 학습 설정 변경으로 묶지 않는다.
- **Memory 2048 복귀:** 현재 SSM 문제 해결책으로 바로 이어지지 않는다. Memory300에서도 S0와 S3/S4의 결과가 갈린다. 슬롯 수 문제는 별도다.

## Mamba-3 판단

Mamba-3는 실제 공개된 2026년 연구이며 [논문](https://arxiv.org/html/2603.15569v1), [ICLR2026 기록](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html), [공식 구현](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py)을 확인했다.

새 discretization/recurrence, 복소 상태를 구현하는 입력 의존 회전, MIMO 등이 주요 변경이다. 공식 코드에는 B/C RMSNorm과 bias도 있다. 현재 모델의 입력·상태·scan 경로에 맞춰 다시 설계해야 하므로 라이브러리 버전만 올리는 작업이 아니다. 논문의 주 평가 맥락은 언어 모델링·state tracking·autoregressive decoding이다. 우리 3D 정상 학습/이상 탐지에서 향상된다는 직접 증거는 확인하지 못했다. MIMO는 논문에서도 학습 속도와 표현력 사이의 비용 교환을 인정하므로 GPU/VRAM 사용률을 높이는 목적으로 즉시 교체할 근거가 없다.

코드의 H100 테스트 한정 주석은 decode의 step 경로에 대한 것이다. 이 주석을 전체 학습이 RTX4090에서 불가능하다는 주장으로 확대하지 않는다. 현재 GPU에서 Mamba-3 설치·속도·수치 적합성은 아직 시험하지 않았다.

**권고:** 이번 revision에서는 현재 SS3D의 학습 방식 복구를 먼저 평가한다. Mamba-3는 그 후에도 SSM 표현력 문제가 남을 때 별도 backbone 실험으로 고려한다. Optimizer 종류·decay 제외 범위·Δ 초기화도 채택한다면 재현 가능한 학습 설정에 명시한다. Mamba-3는 방법 변경으로 설명해야 한다.

## 논문과 로컬 근거

- [S4](https://arxiv.org/abs/2111.00396), [Mamba-1](https://arxiv.org/abs/2312.00752), [Mamba-2](https://arxiv.org/abs/2405.21060), [VMamba](https://arxiv.org/abs/2401.10166), [Vim](https://arxiv.org/abs/2401.09417), [MambaVision](https://arxiv.org/abs/2407.08083), Mamba-3 위 링크.
- 우리 결과: `docs/SSM_REPAIR_INTERIM_20261111.md`, `outputs/ssm_repair_analysis_20261111T051100Z/summary.json`. 이번 제안은 S0–S6의 고정 epoch200 결과까지만 사용했으며 진행 중인 S7의 부분 결과를 비교에 섞지 않았다.
- 우리 구현: `main_files/model.py`, `main_files/core/model_trainer.py`. 공식 code snapshot 22개와 고정 commit은 위 sources.json에 보존.


## U-Mamba 후속 비교

사용자의 CNN decoder 질문에 따라 공식 U-Mamba의 3D Enc/Bot·trainer를 추가 확인했다. U-Mamba도 CNN decoder를 사용하고 전체 SGD decay를 적용한다. CNN 존재 자체보다 residual/norm/초기화·memory 병목·학습 신호의 차이를 구분해야 한다. 상세 비교와 추가 제안은 `docs/UMAMBA_COMPARISON_20261111.md`에 기록했다. 기존 optimizer 권고를 보완하며 모델이나 학습을 변경하지 않았다.
