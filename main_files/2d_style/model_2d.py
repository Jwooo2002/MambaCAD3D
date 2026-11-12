import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from timm.models.layers import DropPath

# 2D scan functions
sys.path.insert(0, os.path.dirname(__file__))
from utils_scan_2d import cross_scan_2d, cross_unscan_2d
# SSM core function (from parent directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from csms6s import selective_scan_fn


# --- VSSM 2D 관련 클래스 ---
class SS2D(nn.Module):
    def __init__(
            self, d_model, d_state=16, d_conv=3, ssm_ratio=2.0,
            dt_rank="auto", act_layer=nn.SiLU, **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.ssm_ratio = ssm_ratio
        self.d_inner = int(self.ssm_ratio * self.d_model)
        self.dt_rank = (math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.k_group = 4  # 4-directional scan (row-fwd, row-bwd, col-fwd, col-bwd)

        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=(d_conv - 1) // 2, groups=self.d_inner, bias=True,
        )
        self.act = act_layer()
        self.x_proj = nn.Conv1d(
            self.d_inner * self.k_group,
            (self.dt_rank + self.d_state * 2) * self.k_group,
            kernel_size=1, groups=self.k_group,
        )
        self.dt_proj = nn.Conv1d(
            self.dt_rank * self.k_group,
            self.d_inner * self.k_group,
            kernel_size=1, groups=self.k_group,
        )

        # State-space parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1).repeat(self.d_inner * self.k_group, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner * self.k_group))

        # Output projections
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        x_reshaped = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_proj, gate = self.in_proj(x_reshaped).chunk(2, dim=-1)
        x_conv = x_proj.permute(0, 3, 1, 2)  # (B, d_inner, H, W)
        x_conv = self.act(self.conv2d(x_conv))

        # 4-directional cross scan
        scanned_x = cross_scan_2d(x_conv)  # (B, 4, d_inner, H*W)
        scanned_x = scanned_x.reshape(B, self.k_group * self.d_inner, L)

        x_ssm_params = self.x_proj(scanned_x)
        dt, B_ssm, C_ssm = torch.split(
            x_ssm_params,
            [self.dt_rank * self.k_group, self.d_state * self.k_group, self.d_state * self.k_group],
            dim=1,
        )
        delta = F.softplus(self.dt_proj(dt))

        # Reshape B_ssm and C_ssm for selective scan
        B_ssm = B_ssm.reshape(B, self.k_group, self.d_state, L)
        C_ssm = C_ssm.reshape(B, self.k_group, self.d_state, L)

        y = selective_scan_fn(
            u=scanned_x,
            delta=delta.to(dtype=scanned_x.dtype),
            A=-torch.exp(self.A_log.float()),
            B=B_ssm,
            C=C_ssm,
            D=self.D.float(),
            delta_bias=None,
            delta_softplus=False,
        )

        y = y.reshape(B, self.k_group, self.d_inner, L)
        # Unscan from 4 directions
        y = cross_unscan_2d(y, (B, self.d_inner, H, W))
        y = y.permute(0, 2, 3, 1)  # (B, H, W, d_inner)
        y = self.out_norm(y)
        y = y * self.act(gate)
        y = self.out_proj(y)
        return y.permute(0, 3, 1, 2)  # (B, C, H, W)


class VSSBlock2D(nn.Module):
    def __init__(self, hidden_dim, drop_path=0.0, ssm_d_state=16, **kwargs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.module = SS2D(d_model=hidden_dim, d_state=ssm_d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x_norm = self.ln_1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.drop_path(self.module(x_norm))


# --- MemAE의 핵심: 메모리 모듈 (2D) ---
class MemModule2D(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.01):
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = nn.Parameter(torch.randn(self.mem_dim, self.fea_dim))
        self.memory.data = F.normalize(self.memory.data, dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # (B, C, H*W)
        att_weight = F.linear(x_flat.permute(0, 2, 1), self.memory)  # (B, H*W, mem_dim)
        att_weight = F.softmax(att_weight, dim=2)

        if self.shrink_thres > 0:
            att_weight = F.hardtanh(att_weight, 0., 1.)
            att_weight = F.normalize(att_weight, p=1, dim=2)

        mem_trans = self.memory.permute(1, 0)
        output_flat = F.linear(att_weight, mem_trans)  # (B, H*W, C)
        output = output_flat.permute(0, 2, 1).view(B, C, H, W)
        return output, att_weight


# --- 최종 MemAE + VSSM 2D 모델 ---
class MemAE_VSSM2D(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2], dims=[64, 128],
                 ssm_d_state=16, drop_path_rate=0.,
                 mem_dim=2000, shrink_thres=0.0025):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], 3, 1, 1),
            nn.GELU()
        )

        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        for i in range(len(depths)):
            stage_blocks = [VSSBlock2D(dims[i], drop_path_rate, ssm_d_state) for _ in range(depths[i])]
            if i < len(depths) - 1:
                # Downsample: Conv2d with stride 2
                stage_blocks.append(nn.Conv2d(dims[i], dims[i + 1], 2, 2))
            self.encoder_stages.append(nn.Sequential(*stage_blocks))

        # Memory module
        self.mem_rep = MemModule2D(
            mem_dim=mem_dim,
            fea_dim=dims[-1],
            shrink_thres=shrink_thres
        )

        # Decoder stages
        decoder_dims = list(reversed(dims))
        decoder_depths = list(reversed(depths))
        self.decoder_stages = nn.ModuleList()
        for i in range(len(decoder_depths)):
            stage_blocks = []
            current_dim = decoder_dims[i]
            for _ in range(decoder_depths[i]):
                stage_blocks.append(nn.Sequential(
                    nn.Conv2d(current_dim, current_dim, 3, 1, 1),
                    nn.GELU()
                ))
            if i < len(decoder_depths) - 1:
                next_dim = decoder_dims[i + 1]
                # Upsample: ConvTranspose2d with stride 2
                stage_blocks.append(nn.ConvTranspose2d(current_dim, next_dim, 2, 2))
            self.decoder_stages.append(nn.Sequential(*stage_blocks))

        self.final_recon_conv = nn.Conv2d(decoder_dims[-1], in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode(self, x):
        x = self.stem(x)
        for stage in self.encoder_stages:
            x = stage(x)
        return x

    def decode(self, z):
        for stage in self.decoder_stages:
            z = stage(z)
        return self.final_recon_conv(z)

    def forward(self, x):
        z = self.encode(x)
        res_mem, att = self.mem_rep(z)
        x_recon = self.decode(res_mem)
        return x_recon, att


if __name__ == '__main__':
    print("🚀 MemAE_VSSM2D (2D 버전) 테스트 시작")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MemAE_VSSM2D(
        in_chans=1,
        depths=[2, 2],
        dims=[64, 128],
        ssm_d_state=16,
        mem_dim=2000
    ).to(device)

    dummy_input = torch.randn(2, 1, 256, 256).to(device)
    print(f"\n- 입력 데이터 Shape: {dummy_input.shape}")

    recon_x, attention = model(dummy_input)

    print(f"- 복원된 데이터 Shape: {recon_x.shape}")
    print(f"- 어텐션 가중치 Shape: {attention.shape}")

    recon_loss = F.mse_loss(recon_x, dummy_input)
    entropy_loss = -torch.mean(torch.sum(attention * torch.log(attention + 1e-12), dim=2))
    total_loss = recon_loss + 0.0002 * entropy_loss

    print("-" * 20)
    print(f"- 재구성 손실 (MSE): {recon_loss.item():.4f}")
    print(f"- 엔트로피 손실: {entropy_loss.item():.4f}")
    print(f"- 전체 손실 (예시): {total_loss.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n- Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    if recon_x.shape == dummy_input.shape:
        print("\n✅ 테스트 통과! 입출력 Shape가 동일합니다.")
    else:
        print("\n❌ 테스트 실패. 입출력 Shape가 다릅니다.")
