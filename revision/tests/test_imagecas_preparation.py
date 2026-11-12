"""Run with journal_preprocess Python's unittest; no pytest dependency required."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_imagecas_x as fetch
import prepare_imagecas_x as prepare


class ExternalCohortTests(unittest.TestCase):
    def make_metadata(self, root):
        source = root / "raw/annotations/ImageCAS-X_dataset_v3"
        (source / "filelist").mkdir(parents=True)
        frame = pd.DataFrame({"Scan ID": range(1, 1001),
                              "Image Quality": [3] * 800 + [0] * 200,
                              "Dominance": ["R"] * 800 + [None] * 200,
                              "Disease": ["yes"] * 388 + ["no"] * 412 + [None] * 200})
        frame.to_excel(source / "Descriptors.xlsx", index=False)
        for name, ids in {"train": range(1, 561), "val": range(561, 641),
                          "test": range(641, 801), "exclude": range(801, 1001)}.items():
            (source / "filelist" / f"{name}.txt").write_text("\n".join(map(str, ids)))
        return source

    def test_source_exclusions_and_disease_mapping_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_metadata(root)
            frame, cohort, _ = prepare.read_metadata(root)
            self.assertEqual(len(frame), 1000)
            self.assertEqual(len(cohort), 800)
            self.assertEqual(cohort["label"].value_counts().to_dict(), {0.0: 412, 1.0: 388})
            rows, manifest = prepare.build_index(root)
            self.assertEqual(manifest["status"], "waiting_for_files")
            self.assertEqual(set(rows["revision_role"]), {"external_test"})
            self.assertEqual(len(set(rows["patient_id"])), 800)

    def test_overlapping_official_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_metadata(root)
            with (source / "filelist/test.txt").open("a") as handle:
                handle.write("\n1\n")
            with self.assertRaisesRegex(ValueError, "overlap"):
                prepare.read_metadata(root)

    def test_unknown_disease_label_is_not_inferred_from_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_metadata(root)
            frame = pd.read_excel(source / "Descriptors.xlsx")
            frame.loc[0, "Disease"] = "unknown"
            frame.to_excel(source / "Descriptors.xlsx", index=False)
            with self.assertRaisesRegex(ValueError, "disease labels"):
                prepare.read_metadata(root)

    def test_archive_parent_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "bad")
            with patch.object(fetch, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "escapes"):
                    fetch.extract_labels(archive)

    def test_kaggle_wrapper_crc_size_and_raw_hash_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapper = root / "wrapper.zip"
            payload = b"an original archive part"
            with zipfile.ZipFile(wrapper, "w") as handle:
                handle.writestr("1-200.z01", payload)
            (root / "downloads/imagecas_v3").mkdir(parents=True)
            with patch.object(fetch, "ROOT", root), patch.object(fetch, "download", return_value=wrapper):
                fetch.fetch_kaggle_file({"name": "1-200.z01", "totalBytes": len(payload)})
            raw = root / "downloads/imagecas_v3/1-200.z01"
            self.assertEqual(raw.read_bytes(), payload)
            verification = json.loads(raw.with_name(raw.name + ".verified.json").read_text())
            self.assertEqual(verification["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertTrue(verification["kaggle_wrapper_crc_checked"])

    def test_geometry_identifies_ras_to_lps_without_using_disease_labels(self):
        try:
            import SimpleITK as sitk
            import vtk
        except ImportError:
            self.skipTest("Run geometry checks with the existing journal_preprocess environment")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            array = np.zeros((10, 10, 10), dtype=np.uint8)
            array[2:8, 2:8, 2:8] = 1
            mask = sitk.GetImageFromArray(array)
            mask.SetSpacing((0.5, 0.5, 0.5))
            mask.SetOrigin((100.0, 110.0, -20.0))
            image = sitk.Cast(mask, sitk.sitkInt16)
            sitk.WriteImage(mask, str(root / "mask.nii.gz"))
            sitk.WriteImage(image, str(root / "ct.nii.gz"))
            points = vtk.vtkPoints()
            for point in [(-101.5, -111.5, -18.5), (-102.0, -112.0, -18.0), (-102.5, -112.5, -17.5)]:
                points.InsertNextPoint(*point)
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            lines = vtk.vtkCellArray()
            for _ in range(2):
                lines.InsertNextCell(3)
                for point_id in range(3):
                    lines.InsertCellPoint(point_id)
            polydata.SetLines(lines)
            for side in ("left", "right"):
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileName(str(root / f"{side}.vtk"))
                writer.SetInputData(polydata)
                self.assertEqual(writer.Write(), 1)
            result = prepare.geometry_one({"patient_id": "synthetic", "scan_id": 1, "label": 0,
                                          "volume_path": str(root / "ct.nii.gz"), "mask_path": str(root / "mask.nii.gz"),
                                          "graph_output_path": str(root / "graph.npz"),
                                          "left_centerline_path": str(root / "left.vtk"), "right_centerline_path": str(root / "right.vtk")})
            self.assertTrue(result["qc_pass"], result)
            self.assertEqual(result["centerline_coordinate_convention"], "ras_to_lps")
            with np.load(root / "graph.npz", allow_pickle=False) as graph:
                self.assertEqual(graph["left_edges"].shape, (2, 2))
                self.assertEqual(graph["left_directed_polyline_edges"].shape, (4, 2))
                self.assertEqual(graph["left_node_degree"].tolist(), [1, 2, 1])


if __name__ == "__main__":
    unittest.main()
