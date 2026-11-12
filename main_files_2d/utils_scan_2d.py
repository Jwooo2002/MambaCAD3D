"""
2D Cross-Scan for VMamba (일반적인 VMamba 방식)
4-directional scanning: row-forward, row-backward, col-forward, col-backward
"""

import torch


def cross_scan_2d(x):
    """
    2D Cross-scan (일반적인 VMamba 방식)

    Input: x with shape (B, C, H, W)
    Output: scanned sequences with shape (B, 4, C, H*W)

    4 directions:
    - Direction 0: row-wise, left to right, top to bottom
    - Direction 1: row-wise, right to left, bottom to top (reverse of 0)
    - Direction 2: column-wise, top to bottom, left to right
    - Direction 3: column-wise, bottom to top, right to left (reverse of 2)
    """
    B, C, H, W = x.shape
    L = H * W

    # Direction 0: row-wise scan (normal order)
    # Shape: (B, C, H, W) -> (B, C, H*W)
    x_row = x.flatten(2, 3)  # (B, C, H*W)

    # Direction 1: reverse of direction 0
    x_row_flip = torch.flip(x_row, dims=[2])  # (B, C, H*W)

    # Direction 2: column-wise scan
    # Transpose H and W, then flatten
    x_col = x.transpose(2, 3).flatten(2, 3)  # (B, C, W, H) -> (B, C, W*H)

    # Direction 3: reverse of direction 2
    x_col_flip = torch.flip(x_col, dims=[2])  # (B, C, W*H)

    # Stack all 4 directions
    # Shape: (B, 4, C, H*W)
    xs = torch.stack([x_row, x_row_flip, x_col, x_col_flip], dim=1)

    return xs


def cross_unscan_2d(xs, original_shape):
    """
    2D Cross-unscan (일반적인 VMamba 방식)

    Input: xs with shape (B, 4, C, H*W)
    Output: reconstructed image with shape (B, C, H, W)
    """
    B, C, H, W = original_shape

    # Extract each direction
    x_row = xs[:, 0]  # (B, C, H*W)
    x_row_flip = xs[:, 1]  # (B, C, H*W)
    x_col = xs[:, 2]  # (B, C, H*W)
    x_col_flip = xs[:, 3]  # (B, C, H*W)

    # Unflip direction 1 and 3
    x_row_flip = torch.flip(x_row_flip, dims=[2])
    x_col_flip = torch.flip(x_col_flip, dims=[2])

    # Reshape row scans
    x_row = x_row.view(B, C, H, W)
    x_row_flip = x_row_flip.view(B, C, H, W)

    # Reshape column scans (need to transpose back)
    x_col = x_col.view(B, C, W, H).transpose(2, 3)
    x_col_flip = x_col_flip.view(B, C, W, H).transpose(2, 3)

    # Average all 4 directions
    x_out = (x_row + x_row_flip + x_col + x_col_flip) / 4.0

    return x_out


# Test code
if __name__ == "__main__":
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)

    print(f"Input shape: {x.shape}")

    # Scan
    xs = cross_scan_2d(x)
    print(f"Scanned shape: {xs.shape}")  # (B, 4, C, H*W)

    # Unscan
    x_recon = cross_unscan_2d(xs, x.shape)
    print(f"Reconstructed shape: {x_recon.shape}")  # (B, C, H, W)

    # Check reconstruction error
    error = torch.mean(torch.abs(x - x_recon))
    print(f"Reconstruction error: {error.item():.6f}")

    if error < 1e-5:
        print("✅ Perfect reconstruction!")
    else:
        print("⚠️  Reconstruction has errors")
