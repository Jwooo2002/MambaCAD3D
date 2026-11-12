import torch
import sys
sys.path.insert(0, './main_files')
from model import MemAE_VSSM3D

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_flops_manual(model, input_shape):
    """Manual FLOPs estimation for the model"""
    # Input shape: (B, C, Z, H, W) = (1, 1, 64, 8, 8) 기본값
    B, C, Z, H, W = input_shape
    total_flops = 0

    # Stem: Conv3d(1, 32, 3, 1, 1)
    stem_flops = B * C * 32 * (3*3*3) * Z * H * W
    total_flops += stem_flops

    print(f"Stem FLOPs: {stem_flops / 1e9:.4f} GFLOPs")

    # For each stage, we need to calculate VSSBlock3D FLOPs
    # This is a rough estimation
    dims = [32, 64, 128]
    depths = [2, 2, 2]

    current_z, current_h, current_w = Z, H, W

    for stage_idx, (dim, depth) in enumerate(zip(dims, depths)):
        stage_flops = 0

        # VSSBlock3D operations
        for _ in range(depth):
            L = current_z * current_h * current_w
            d_inner = int(2.0 * dim)  # ssm_ratio = 2.0

            # in_proj: Linear(dim, d_inner * 2)
            stage_flops += B * dim * (d_inner * 2) * L

            # conv3d: Conv3d(d_inner, d_inner, 3, groups=d_inner)
            stage_flops += B * d_inner * (3*3*3) * L

            # x_proj, dt_proj, and SSM operations (simplified)
            stage_flops += B * d_inner * L * 100  # Simplified SSM computation

            # out_proj: Linear(d_inner, dim)
            stage_flops += B * d_inner * dim * L

        # Downsampling: Conv3d(dim, next_dim, 2, 2)
        if stage_idx < len(dims) - 1:
            next_dim = dims[stage_idx + 1]
            stage_flops += B * dim * next_dim * (2*2*2) * (current_z//2) * (current_h//2) * (current_w//2)
            current_z, current_h, current_w = current_z // 2, current_h // 2, current_w // 2

        total_flops += stage_flops
        print(f"Stage {stage_idx} FLOPs: {stage_flops / 1e9:.4f} GFLOPs")

    # Memory module (simplified)
    mem_dim = 300
    fea_dim = dims[-1]
    L = current_z * current_h * current_w
    mem_flops = B * L * fea_dim * mem_dim * 2  # Forward and backward memory access
    total_flops += mem_flops
    print(f"Memory Module FLOPs: {mem_flops / 1e9:.4f} GFLOPs")

    # Decoder (simplified, usually similar to encoder)
    decoder_flops = stage_flops * 0.5  # Rough estimate (already includes B)
    total_flops += decoder_flops
    print(f"Decoder FLOPs (estimated): {decoder_flops / 1e9:.4f} GFLOPs")

    return total_flops

if __name__ == '__main__':
    print("=" * 60)
    print("MemAE_VSSM3D 모델 분석")
    print("=" * 60)

    # Model configuration from sample_config.yaml
    model = MemAE_VSSM3D(
        in_chans=1,
        depths=[2, 2, 2],  # 3 stages
        dims=[32, 64, 128],
        ssm_d_state=8,
        mem_dim=300,
        shrink_thres=0.0025
    )

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 Parameters:")
    print(f"  - Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  - Trainable Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    # Calculate FLOPs (manual estimation)
    # Typical input size from config: (B, 1, 64, 8, 8) or similar
    input_shapes = [
        (1, 1, 64, 32, 32),
        (2, 1, 64, 32, 32),
    ]

    for input_shape in input_shapes:
        print(f"\n📈 FLOPs for input shape {input_shape}:")
        flops = calculate_flops_manual(model, input_shape)
        print(f"  Total FLOPs (estimated): {flops / 1e9:.4f} GFLOPs ({flops / 1e6:.2f} MFLOPs)")
        print("-" * 60)

    # Try using thop library if available
    try:
        from thop import profile, clever_format
        print("\n🔬 Using THOP library for accurate FLOPs calculation:")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        for input_shape in input_shapes:
            dummy_input = torch.randn(input_shape).to(device)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_str, params_str = clever_format([flops, params], "%.3f")
            print(f"\nInput shape {input_shape}:")
            print(f"  - FLOPs: {flops_str}")
            print(f"  - Params: {params_str}")

    except ImportError:
        print("\n⚠️  thop 라이브러리가 설치되어 있지 않습니다.")
        print("정확한 FLOPs 계산을 위해 설치하려면: pip install thop")

    print("\n" + "=" * 60)
