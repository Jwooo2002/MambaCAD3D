#!/usr/bin/env python3
"""Render actual CT/mask/centerline overlays for physical-coordinate QC."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import vtk_to_numpy


ROOT = Path(__file__).resolve().parents[1] / "external_data/imagecas_x"


def main():
    frame = pd.read_csv(ROOT / "prepared/external_cohort.csv")
    rows = [frame.loc[frame.scan_id == scan_id].iloc[0] for scan_id in (1, 11)]
    figure, axes = plt.subplots(len(rows), 3, figsize=(11, 7), facecolor="white")
    for row_index, row in enumerate(rows):
        image = sitk.ReadImage(row.volume_path)
        ct = sitk.GetArrayFromImage(image)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(row.mask_path))
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(row.left_centerline_path)
        reader.Update()
        points = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
        point = tuple(map(float, points[len(points) // 2]))
        x, y, z = image.TransformPhysicalPointToIndex(point)
        views = [(ct[z], mask[z], x, y, "Axial"),
                 (ct[:, y, :], mask[:, y, :], x, z, "Coronal"),
                 (ct[:, :, x], mask[:, :, x], y, z, "Sagittal")]
        for column, (values, labels, cx, cy, title) in enumerate(views):
            axis = axes[row_index, column]
            axis.imshow(values, cmap="gray", vmin=-300, vmax=400, origin="lower")
            axis.contour(labels > 0, levels=[0.5], colors=["#2bd67b"], linewidths=0.9)
            axis.plot(cx, cy, "+", color="#ff654f", markersize=12, markeredgewidth=1.5)
            axis.set_xlim(max(0, cx - 30), min(values.shape[1], cx + 30))
            axis.set_ylim(max(0, cy - 30), min(values.shape[0], cy + 30))
            axis.set_title(f"ID {int(row.scan_id)} | {title}", fontsize=11)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("ImageCAS-X v3: physical-coordinate checks", fontsize=16, y=0.98)
    figure.text(0.5, 0.03, "Green: reference lumen mask   |   Red: reference centerline point\nGeometry QC only; these overlays do not identify plaque or stenosis.",
                ha="center", fontsize=10)
    figure.tight_layout(rect=[0.02, 0.09, 0.98, 0.94])
    target = ROOT / "reports/imagecas_x_geometry_preview.png"
    figure.savefig(target, dpi=150)
    plt.close(figure)
    print(target)


if __name__ == "__main__":
    main()
