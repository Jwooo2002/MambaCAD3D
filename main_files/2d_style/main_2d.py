"""
Main execution script for 2D MemAE-VSSM
================================================================

Usage:
    python main_2d.py --help

Author: Research Team
Version: 1.0.0 (2D)
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_2d import MemAE_VSSM2D


class ImageDataset2D(Dataset):
    """2D image dataset - extracts 2D slices from 3D volumes"""
    def __init__(self, data_root, is_normal=True):
        import pickle

        self.data_root = Path(data_root)
        self.is_normal = is_normal

        # Load all pkl files
        pkl_files = sorted(list(self.data_root.glob("*.pkl")))

        all_slices = []
        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                clips = pickle.load(f)  # List of 3D volumes
                for clip in clips:
                    # clip shape: (Z, H, W) or similar
                    clip = np.array(clip, dtype=np.float32)

                    # Extract all Z slices as 2D images
                    if len(clip.shape) == 3:
                        for z_idx in range(clip.shape[0]):
                            slice_2d = clip[z_idx]  # (H, W)
                            all_slices.append(slice_2d)
                    else:
                        # If already 2D
                        all_slices.append(clip)

        self.data = np.array(all_slices, dtype=np.float32)
        print(f"Loaded {len(self.data)} 2D slices from {len(pkl_files)} files")

        # Normalize to [0, 1]
        if len(self.data) > 0:
            self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min() + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]  # (H, W)
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        return img


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='2D MemAE-VSSM Training')

    # Data
    parser.add_argument('--normal_data_root', type=str,
                        default="/home/jiwoong/PycharmProjects/VMamba/MemAE_kfold_simplescan/main_files/dataset/v11/0.05_8_32_CTCA_1/Normal",
                        help='Path to normal data')
    parser.add_argument('--diseased_data_root', type=str, default="/home/jiwoong/PycharmProjects/VMamba/MemAE_kfold_simplescan/main_files/dataset/v11/0.05_8_32_CTCA_1/Diseased",
                        help='Path to diseased data (for validation)')

    # Model
    parser.add_argument('--in_chans', type=int, default=1)
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 2])
    parser.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--ssm_d_state', type=int, default=8)
    parser.add_argument('--mem_dim', type=int, default=300)
    parser.add_argument('--shrink_thres', type=float, default=0.0025)

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.0002)

    # System
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='./results_2d')
    parser.add_argument('--run_name', type=str, default='test_2d')
    parser.add_argument('--random_seed', type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, optimizer, device, entropy_weight):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_entropy_loss = 0

    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)

        optimizer.zero_grad()

        # Forward
        recon, att = model(data)

        # Loss
        recon_loss = nn.MSELoss()(recon, data)
        entropy_loss = -torch.mean(torch.sum(att * torch.log(att + 1e-12), dim=2))
        loss = recon_loss + entropy_weight * entropy_loss

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_entropy_loss += entropy_loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Recon: {recon_loss.item():.4f} "
                  f"Entropy: {entropy_loss.item():.4f}")

    return {
        'loss': total_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader),
        'entropy_loss': total_entropy_loss / len(dataloader)
    }


def validate(model, dataloader, device, entropy_weight):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            recon, att = model(data)

            recon_loss = nn.MSELoss()(recon, data)
            entropy_loss = -torch.mean(torch.sum(att * torch.log(att + 1e-12), dim=2))
            loss = recon_loss + entropy_weight * entropy_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()

    return {
        'loss': total_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader)
    }


def main():
    """Main execution function"""

    # Parse arguments
    args = get_args()

    # Set seed
    set_seed(args.random_seed)

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Using device: {device}")
    print(f"{'='*80}\n")

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_path) / f"{args.run_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {save_dir}\n")

    # Create model
    print("Creating model...")
    model = MemAE_VSSM2D(
        in_chans=args.in_chans,
        depths=args.depths,
        dims=args.dims,
        ssm_d_state=args.ssm_d_state,
        mem_dim=args.mem_dim,
        shrink_thres=args.shrink_thres
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters ({total_params/1e6:.2f}M)\n")

    # Create datasets
    print("Loading data...")
    train_dataset = ImageDataset2D(args.normal_data_root, is_normal=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Training samples: {len(train_dataset)}\n")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    print(f"{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}]")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.entropy_loss_weight
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics
        print(f"  Train Loss: {train_metrics['loss']:.4f} "
              f"(Recon: {train_metrics['recon_loss']:.4f}, "
              f"Entropy: {train_metrics['entropy_loss']:.4f})")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_dir / 'best_model.pth')
            print(f"  ✓ Best model saved!")

        print()

    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
