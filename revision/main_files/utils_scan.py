# utils_scan.py - Modified for single axis scanning

import torch
import numpy as np

spiral_cache = {}


def generate_spiral_indices(D, H, reverse=False, device='cpu'):
    key = (D, H, reverse, str(device))
    if key in spiral_cache:
        return spiral_cache[key]
    indices = []
    top, bottom, left, right = 0, D - 1, 0, H - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1): indices.append((top, j))
        top += 1
        for i in range(top, bottom + 1): indices.append((i, right))
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1): indices.append((bottom, j))
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1): indices.append((i, left))
            left += 1
    if reverse:
        indices = list(reversed(indices))
    linear = torch.tensor([d * H + h for d, h in indices], dtype=torch.long, device=device)
    spiral_cache[key] = linear
    return linear


def scan_on_plane_alternating_vectorized(plane_data):
    B, C, L_dim, D_dim, H_dim = plane_data.shape
    device = plane_data.device
    even_slices = plane_data[:, :, 0::2]
    odd_slices = plane_data[:, :, 1::2]
    in_to_out_idx = generate_spiral_indices(D_dim, H_dim, reverse=True, device=device)
    B_e, C_e, L_e, D_e, H_e = even_slices.shape
    idx_expanded_even = in_to_out_idx.view(1, 1, 1, -1).expand(B_e, C_e, L_e, -1)
    scanned_evens = torch.gather(even_slices.reshape(B_e, C_e, L_e, -1), 3, idx_expanded_even)
    out_to_in_idx = generate_spiral_indices(D_dim, H_dim, reverse=False, device=device)
    B_o, C_o, L_o, D_o, H_o = odd_slices.shape
    idx_expanded_odd = out_to_in_idx.view(1, 1, 1, -1).expand(B_o, C_o, L_o, -1)
    scanned_odds = torch.gather(odd_slices.reshape(B_o, C_o, L_o, -1), 3, idx_expanded_odd)
    fwd_zigzag_seq = torch.empty(B, C, L_dim, D_dim * H_dim, device=device, dtype=plane_data.dtype)
    fwd_zigzag_seq[:, :, 0::2] = scanned_evens
    fwd_zigzag_seq[:, :, 1::2] = scanned_odds
    fwd_zigzag_seq = fwd_zigzag_seq.flatten(2, 3)
    bwd_zigzag_seq = torch.flip(fwd_zigzag_seq, dims=[-1])
    return torch.stack([fwd_zigzag_seq, bwd_zigzag_seq], dim=1)


def single_axis_spiral_scan(x):
    """
    Single axis spiral scan for Z direction only
    Input: x with shape (B, C, Z=64, H=8, W=8)
    Output: scanned sequences with shape (B, 2, C, Z*H*W) - only forward and backward
    """
    # Only scan along Z axis (8x8 planes)
    scans_z = scan_on_plane_alternating_vectorized(x)
    return scans_z  # Shape: (B, 2, C, Z*H*W)


def unscan_from_plane_final_vectorized(sequences, target_shape):
    B, _, C, L_total = sequences.shape;
    L_dim, D_dim, H_dim = target_shape;
    device = sequences.device
    fwd_seq = sequences[:, 0].view(B, C, L_dim, D_dim * H_dim);
    bwd_seq = sequences[:, 1].view(B, C, L_dim, D_dim * H_dim)
    bwd_seq_flipped = torch.flip(bwd_seq, dims=[2, 3]);
    combined_seq = fwd_seq + bwd_seq_flipped
    even_seq = combined_seq[:, :, 0::2];
    odd_seq = combined_seq[:, :, 1::2]
    out_to_in_idx = generate_spiral_indices(D_dim, H_dim, reverse=False, device=device);
    inv_out_to_in_idx = torch.argsort(out_to_in_idx)
    in_to_out_idx = generate_spiral_indices(D_dim, H_dim, reverse=True, device=device);
    inv_in_to_out_idx = torch.argsort(in_to_out_idx)
    B_e, C_e, L_e, _ = even_seq.shape;
    B_o, C_o, L_o, _ = odd_seq.shape
    inv_idx_expanded_even = inv_in_to_out_idx.view(1, 1, 1, -1).expand(B_e, C_e, L_e, -1)
    unscanned_evens = torch.gather(even_seq, 3, inv_idx_expanded_even)
    inv_idx_expanded_odd = inv_out_to_in_idx.view(1, 1, 1, -1).expand(B_o, C_o, L_o, -1)
    unscanned_odds = torch.gather(odd_seq, 3, inv_idx_expanded_odd)
    unscanned_vol = torch.empty_like(combined_seq)
    unscanned_vol[:, :, 0::2] = unscanned_evens
    unscanned_vol[:, :, 1::2] = unscanned_odds
    return unscanned_vol.reshape(B, C, L_dim, D_dim, H_dim)


def single_axis_spiral_unscan_final(y, original_shape):
    """
    Single axis spiral unscan for Z direction only
    Input: y with shape (B, 2, C, Z*H*W)
    Output: reconstructed volume with original_shape (B, C, Z=64, H=8, W=8)
    """
    B, C, Z, H, W = original_shape
    # Only unscan from Z axis
    vol_z = unscan_from_plane_final_vectorized(y, (Z, H, W))
    return vol_z  # Shape: (B, C, Z, H, W)


# Example usage:
if __name__ == "__main__":
    # Test with 8x8x64 input
    B, C, Z, H, W = 1, 256, 64, 8, 8
    x = torch.randn(B, C, Z, H, W)

    # Single axis scan
    scanned = single_axis_spiral_scan(x)
    print(f"Input shape: {x.shape}")
    print(f"Scanned shape: {scanned.shape}")  # Should be (1, 2, 256, 4096)

    # Single axis unscan
    reconstructed = single_axis_spiral_unscan_final(scanned, x.shape)
    print(f"Reconstructed shape: {reconstructed.shape}")  # Should be (1, 256, 64, 8, 8)

    # Check if reconstruction is close to original
    print(f"Reconstruction error: {torch.mean(torch.abs(x - reconstructed))}")