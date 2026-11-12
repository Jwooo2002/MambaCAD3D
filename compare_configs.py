import torch
import sys
sys.path.insert(0, './main_files')
from model import MemAE_VSSM3D

def count_params(model):
    return sum(p.numel() for p in model.parameters())

configs = [
    {
        'name': '현재 설정 (작음)',
        'in_chans': 1,
        'depths': [2, 2, 2],
        'dims': [32, 64, 128],
        'ssm_d_state': 8,
        'mem_dim': 300,
    },
    {
        'name': '중간 크기',
        'in_chans': 1,
        'depths': [2, 2, 2],
        'dims': [64, 128, 256],
        'ssm_d_state': 16,
        'mem_dim': 1000,
    },
    {
        'name': '큰 크기',
        'in_chans': 1,
        'depths': [3, 3, 3],
        'dims': [64, 128, 256],
        'ssm_d_state': 16,
        'mem_dim': 2000,
    },
    {
        'name': '매우 큰 크기',
        'in_chans': 1,
        'depths': [4, 4, 4],
        'dims': [96, 192, 384],
        'ssm_d_state': 32,
        'mem_dim': 2000,
    },
]

print("=" * 100)
print(f"{'Config Name':<25} | {'dims':<20} | {'depths':<15} | {'mem_dim':<10} | {'Parameters':<20}")
print("=" * 100)

for config in configs:
    model = MemAE_VSSM3D(
        in_chans=config['in_chans'],
        depths=config['depths'],
        dims=config['dims'],
        ssm_d_state=config['ssm_d_state'],
        mem_dim=config['mem_dim'],
    )

    params = count_params(model)
    dims_str = str(config['dims'])
    depths_str = str(config['depths'])

    print(f"{config['name']:<25} | {dims_str:<20} | {depths_str:<15} | {config['mem_dim']:<10} | {params:>12,} ({params/1e6:>5.2f}M)")

print("=" * 100)
