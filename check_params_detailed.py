import torch
import sys
sys.path.insert(0, './main_files')
from model import MemAE_VSSM3D

def analyze_parameters(model):
    """Analyze parameters in detail"""
    print("=" * 80)
    print("상세 파라미터 분석")
    print("=" * 80)

    total_params = 0

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:60s} | Shape: {str(list(param.shape)):30s} | Params: {param_count:>10,}")

    print("=" * 80)
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print("=" * 80)

    return total_params

def count_by_module(model):
    """Count parameters by module"""
    print("\n" + "=" * 80)
    print("모듈별 파라미터 집계")
    print("=" * 80)

    module_counts = {}

    for name, param in model.named_parameters():
        module_name = name.split('.')[0]  # Get top-level module name
        if module_name not in module_counts:
            module_counts[module_name] = 0
        module_counts[module_name] += param.numel()

    for module_name, count in sorted(module_counts.items()):
        print(f"{module_name:30s}: {count:>12,} params ({count / 1e6:.3f}M)")

    print("=" * 80)

if __name__ == '__main__':
    # Model configuration from sample_config.yaml
    model = MemAE_VSSM3D(
        in_chans=1,
        depths=[2, 2, 2],
        dims=[32, 64, 128],
        ssm_d_state=8,
        mem_dim=300,
        shrink_thres=0.0025
    )

    print("\n현재 설정:")
    print(f"  - in_chans: 1")
    print(f"  - depths: [2, 2, 2]")
    print(f"  - dims: [32, 64, 128]")
    print(f"  - ssm_d_state: 8")
    print(f"  - mem_dim: 300")
    print()

    total = analyze_parameters(model)
    count_by_module(model)

    # Try with torchinfo if available
    try:
        from torchinfo import summary
        print("\n" + "=" * 80)
        print("torchinfo Summary")
        print("=" * 80)

        summary(model, input_size=(2, 1, 64, 32, 32),
                device='cpu',
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                depth=3)
    except ImportError:
        print("\n⚠️  torchinfo 라이브러리가 설치되어 있지 않습니다.")
        print("더 자세한 정보를 보려면: pip install torchinfo")
