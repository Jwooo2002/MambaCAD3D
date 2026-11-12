#!/usr/bin/env python3
"""Show cross sections and ordered longitudinal views for geometry inspection."""

import json
import os
from pathlib import Path
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vmamba_revision_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    data_root = ROOT / "processed_data/ordered_mm_v2"
    cases = [data_root / "ASOCA/Normal/N_10_patch.pkl", data_root / "ASOCA/Diseased/D_6_patch.pkl",
             data_root / "ImageCAS-X/patients/ImageCAS-X_1.pkl", data_root / "ImageCAS-X/patients/ImageCAS-X_455.pkl"]
    cases = [path for path in cases if path.exists()]
    fig, axes = plt.subplots(len(cases), 3, figsize=(10, 3.8 * len(cases)), squeeze=False, constrained_layout=True)
    for row, path in enumerate(cases):
        metadata = json.loads(path.with_suffix(".json").read_text())
        with path.open("rb") as handle:
            patches = np.asarray(pickle.load(handle))
        # First full window rather than a potentially padded minor branch.
        index = next((r["patch_index"] for r in metadata["patches"] if r["valid_depth"] == 64), 0)
        patch = patches[index]
        fov = metadata["protocol"]["cross_section_fov_mm"]
        length = (patch.shape[0] - 1) * metadata["protocol"]["longitudinal_spacing_mm"]
        for column, (image, extent, xlabel, ylabel) in enumerate([
            (patch[32], [-fov/2, fov/2, fov/2, -fov/2], "u (mm)", "v (mm)"),
            (patch[:, 16, :], [-fov/2, fov/2, length, 0], "u (mm)", "source arc (mm)"),
            (patch[:, :, 16], [-fov/2, fov/2, length, 0], "v (mm)", "source arc (mm)")]):
            ax = axes[row, column]
            ax.imshow(image, cmap="gray", vmin=-300, vmax=400, extent=extent, aspect="auto", interpolation="nearest")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{metadata['patient_id']} | patch {index}" if column == 0 else "Ordered longitudinal view")
        axes[row, 0].axhline(0, color="#36cfb4", lw=.6)
        axes[row, 0].axvline(0, color="#36cfb4", lw=.6)
    destination = ROOT / "outputs/ordered_mm_v2_geometry_preview.png"
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
