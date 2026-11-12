#!/usr/bin/env python3
"""Pair the external cohort and audit CT/mask/centerline physical geometry."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "external_data/imagecas_x"
LOG = logging.getLogger(__name__)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_metadata(root=ROOT):
    candidates = [root / "raw/annotations/ImageCAS-X_dataset_v3",
                  root / "raw/annotation_metadata_bootstrap/ImageCAS-X_dataset_v3"]
    source = next((path for path in candidates if (path / "Descriptors.xlsx").exists()), None)
    if source is None:
        raise FileNotFoundError("ImageCAS-X Descriptors.xlsx has not been downloaded")
    frame = pd.read_excel(source / "Descriptors.xlsx")
    expected_columns = {"Scan ID", "Image Quality", "Dominance", "Disease"}
    if not expected_columns <= set(frame):
        raise ValueError("Unexpected descriptor columns")
    frame["Scan ID"] = pd.to_numeric(frame["Scan ID"], errors="raise").astype(int)
    if frame["Scan ID"].duplicated().any() or set(frame["Scan ID"]) != set(range(1, 1001)):
        raise ValueError("Expected unique original scan IDs 1..1000")
    groups = {name: set(map(int, (source / "filelist" / f"{name}.txt").read_text().split()))
              for name in ("train", "val", "test", "exclude")}
    group_values = list(groups.values())
    if any(a & b for i, a in enumerate(group_values) for b in group_values[i + 1:]):
        raise ValueError("Official source file lists overlap")
    if set.union(*group_values) != set(frame["Scan ID"]) or len(groups["exclude"]) != 200:
        raise ValueError("Source inclusion/exclusion lists are incomplete")
    frame["source_split"] = frame["Scan ID"].map({i: name for name, ids in groups.items() for i in ids})
    frame["included"] = ~frame["Scan ID"].isin(groups["exclude"])
    frame["label"] = frame["Disease"].map({"no": 0, "yes": 1})
    included = frame[frame["included"]].copy()
    if len(included) != 800 or included["label"].isna().any() or (included["Image Quality"] == 0).any():
        raise ValueError("Included cohort must have 800 diagnostic scans with disease labels")
    if included["label"].value_counts().to_dict() != {0.0: 412, 1.0: 388}:
        raise ValueError("Disease counts differ from pinned version: inspect before changing the cohort")
    return frame, included, source


def build_index(root=ROOT):
    frame, included, metadata_source = read_metadata(root)
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    volume_dir = prepared / "volumes"
    volume_dir.mkdir(exist_ok=True)
    original_volumes = {}
    for path in (root / "raw/imagecas_volumes").rglob("*.img.nii.gz"):
        scan_id = int(path.name.split(".")[0])
        if scan_id in original_volumes:
            raise ValueError(f"Duplicated original volume ID: {scan_id}")
        original_volumes[scan_id] = path.resolve()
    rows = []
    annotation_root = root / "raw/annotations/ImageCAS-X_dataset_v3"
    for _, record in included.sort_values("Scan ID").iterrows():
        scan_id = int(record["Scan ID"])
        volume = volume_dir / f"{scan_id}.img.nii.gz"
        if scan_id in original_volumes and not volume.exists():
            volume.symlink_to(original_volumes[scan_id])
        paths = {"volume_path": volume,
                 "mask_path": annotation_root / "segmentations" / f"{scan_id}.coronary.nii.gz",
                 "left_centerline_path": annotation_root / "centerlines" / f"{scan_id}.coronary_left_centerline.vtk",
                 "right_centerline_path": annotation_root / "centerlines" / f"{scan_id}.coronary_right_centerline.vtk"}
        present = all(path.is_file() for path in paths.values())
        rows.append({"patient_id": f"ImageCAS-X:{scan_id}", "scan_id": scan_id, "label": int(record["label"]),
                     "image_quality": int(record["Image Quality"]), "dominance": record["Dominance"],
                     "source_split": record["source_split"], "revision_role": "external_test",
                     "graph_output_path": str(prepared / "graphs" / f"{scan_id}.npz"),
                     **{key: str(value.resolve()) for key, value in paths.items()}, "all_files_present": present})
    cohort = pd.DataFrame(rows)
    cohort.to_csv(prepared / "external_cohort.csv", index=False)
    frame.to_csv(prepared / "source_descriptors.csv", index=False)
    excluded = frame[~frame["included"]][["Scan ID", "Image Quality", "source_split"]].copy()
    excluded["reason"] = "Excluded by source authors as non-diagnostic/invalid; no CAD label assigned"
    excluded.to_csv(prepared / "excluded_scans.csv", index=False)
    membership = cohort[["patient_id", "scan_id", "label", "source_split", "revision_role"]].to_csv(index=False)
    cohort_hash = hashlib.sha256(membership.encode()).hexdigest()
    membership_path = prepared / "cohort_membership.csv"
    if membership_path.exists() and membership_path.read_text() != membership:
        raise ValueError("Frozen external cohort membership changed")
    membership_path.write_text(membership)
    manifest = {"created_or_checked_utc": datetime.now(timezone.utc).isoformat(), "annotation_version": 3,
                "zenodo_record": 23894668, "imagecas_dataset_version": 3,
                "source_urls": ["https://zenodo.org/records/23894668", "https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas"],
                "annotation_archive_md5": "4e7ec922065905fe29cc820276e6f283",
                "cohort_membership_sha256": cohort_hash,
                "original_subject_count": 1000, "included_subject_count": 800, "excluded_subject_count": 200,
                "normal_subject_count": 412, "diseased_subject_count": 388,
                "complete_file_sets": int(cohort["all_files_present"].sum()),
                "all_external_test": True, "training_or_selection_on_external_labels": False,
                "source_split_note": "560/80/160 are the source segmentation benchmark splits, retained as provenance; this revision assigns all 800 included scans to external evaluation.",
                "annotation_scope": "Disease is a scan-level descriptor. Segmentation labels represent lumen/arterial branches, not plaque/stenosis localization ground truth.",
                "centerline_scope": "Provided reference centerlines support preprocessing; this does not validate automatic end-to-end centerline extraction.",
                "status": "paired_files_ready" if cohort["all_files_present"].all() else "waiting_for_files",
                "patch_preprocessing_status": "Not yet converted: reconcile physical patch sampling with the ASOCA preprocessing before producing model inputs."}
    write_json(prepared / "external_cohort_manifest.json", manifest)
    LOG.info("External cohort: 800 subjects (412 normal / 388 diseased), %d complete file sets", manifest["complete_file_sets"])
    return cohort, manifest


def geometry_one(row):
    import SimpleITK as sitk
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    result = {"patient_id": row["patient_id"], "scan_id": row["scan_id"], "label": row["label"]}
    try:
        image_reader = sitk.ImageFileReader()
        image_reader.SetFileName(row["volume_path"])
        image_reader.ReadImageInformation()
        mask = sitk.ReadImage(row["mask_path"])
        array = sitk.GetArrayFromImage(mask)
        result.update({"ct_size_xyz": list(image_reader.GetSize()), "ct_spacing_mm_xyz": list(image_reader.GetSpacing()),
                       "mask_size_xyz": list(mask.GetSize()), "mask_label_min": int(array.min()),
                       "mask_label_max": int(array.max()), "ct_pixel_type": sitk.GetPixelIDValueAsString(image_reader.GetPixelID())})
        result["ct_mask_geometry_matches"] = bool(tuple(image_reader.GetSize()) == tuple(mask.GetSize()) and
            np.allclose(image_reader.GetSpacing(), mask.GetSpacing(), atol=1e-4) and
            np.allclose(image_reader.GetOrigin(), mask.GetOrigin(), atol=1e-3) and
            np.allclose(image_reader.GetDirection(), mask.GetDirection(), atol=1e-5))
        points, edge_lengths, graphs = [], [], {}
        for key in ("left_centerline_path", "right_centerline_path"):
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(row[key])
            reader.Update()
            polydata = reader.GetOutput()
            if polydata.GetNumberOfPoints() == 0:
                raise ValueError(f"Empty centerline: {key}")
            coordinates = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)
            points.append(coordinates)
            side = "left" if key.startswith("left") else "right"
            side_edges = []
            lines = polydata.GetLines()
            lines.InitTraversal()
            ids = vtk.vtkIdList()
            while lines.GetNextCell(ids):
                if ids.GetNumberOfIds() > 1:
                    indices = np.array([ids.GetId(i) for i in range(ids.GetNumberOfIds())], dtype=np.int64)
                    side_edges.extend(np.stack([indices[:-1], indices[1:]], axis=1).tolist())
            directed_edges = np.asarray(side_edges, dtype=np.int64).reshape(-1, 2)
            edges = np.unique(np.sort(directed_edges, axis=1), axis=0)
            edge_lengths.extend(np.linalg.norm(coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1).tolist())
            graphs[f"{side}_points_mm"] = coordinates
            graphs[f"{side}_edges"] = edges
            graphs[f"{side}_directed_polyline_edges"] = directed_edges
            graphs[f"{side}_node_degree"] = np.bincount(edges.ravel(), minlength=len(coordinates))
            point_data = polydata.GetPointData()
            names = []
            for index in range(point_data.GetNumberOfArrays()):
                values = point_data.GetArray(index)
                if values is not None:
                    graphs[f"{side}_point_attribute_{index}"] = vtk_to_numpy(values)
                    names.append({"index": index, "name": point_data.GetArrayName(index)})
            graphs[f"{side}_point_attributes_json"] = np.asarray(json.dumps(names))
        points = np.concatenate(points)
        result["centerline_point_count"] = len(points)
        result["topology_edge_count"] = len(edge_lengths)
        result["topology_edge_median_mm"] = float(np.median(edge_lengths)) if edge_lengths else None
        result["topology_edge_max_mm"] = float(np.max(edge_lengths)) if edge_lengths else None
        result["sequence_extraction_note"] = "Traverse VTK line connectivity; stored point-array order is not a vessel trajectory."
        points = points[::max(1, len(points) // 4000)]
        direction_inverse = np.linalg.inv(np.asarray(mask.GetDirection()).reshape(3, 3))
        origin, spacing, size = np.asarray(mask.GetOrigin()), np.asarray(mask.GetSpacing()), np.asarray(mask.GetSize())
        for name, signs in (("lps", [1, 1, 1]), ("ras_to_lps", [-1, -1, 1])):
            indices = np.rint(((points * signs - origin) @ direction_inverse.T) / spacing).astype(int)
            inside = ((indices >= 0) & (indices < size)).all(1)
            lumen = np.zeros(len(indices), dtype=bool)
            valid = indices[inside]
            lumen[inside] = array[valid[:, 2], valid[:, 1], valid[:, 0]] > 0
            result[f"{name}_in_bounds_fraction"] = float(inside.mean())
            result[f"{name}_in_lumen_fraction"] = float(lumen.mean())
        convention = max(("lps", "ras_to_lps"), key=lambda name: result[f"{name}_in_lumen_fraction"])
        result["centerline_coordinate_convention"] = convention
        result["qc_pass"] = bool(result["ct_mask_geometry_matches"] and result["mask_label_min"] == 0 and
                                  result["mask_label_max"] <= 14 and result[f"{convention}_in_lumen_fraction"] >= 0.85)
        if "graph_output_path" in row and result["qc_pass"]:
            for side in ("left", "right"):
                values = graphs.pop(f"{side}_points_mm")
                graphs[f"{side}_points_lps_mm"] = values if convention == "lps" else values * [-1, -1, 1]
            graphs["ct_origin_lps_mm"] = np.asarray(image_reader.GetOrigin())
            graphs["ct_spacing_mm_xyz"] = np.asarray(image_reader.GetSpacing())
            graphs["ct_direction_lps"] = np.asarray(image_reader.GetDirection()).reshape(3, 3)
            graphs["scan_id"] = np.asarray(row["scan_id"], dtype=np.int64)
            graphs["format_note"] = np.asarray("Coordinates are LPS millimetres. edges are unique undirected connectivity; directed_polyline_edges retain source traversal. Point storage order is not a path.")
            graph_path = Path(row["graph_output_path"])
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = graph_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **graphs)
            temporary.replace(graph_path)
            result["prepared_graph_path"] = str(graph_path)
    except Exception as exc:
        result.update({"qc_pass": False, "error": f"{type(exc).__name__}: {exc}"})
    return result


def audit_image_inventory(root=ROOT):
    """Check extracted byte sizes against ZIP catalogs and screen exact duplicates."""
    binary = root / "tools/7zip/7zz"
    rows = []
    for group in ("1-200", "201-400", "401-600", "601-800", "801-1000"):
        archive = root / "downloads/imagecas_v3" / f"{group}.zip"
        result = subprocess.run([str(binary), "l", "-slt", str(archive)], capture_output=True, text=True, check=True)
        (root / "provenance" / f"archive_catalog_{group}.txt").write_text(result.stdout)
        for block in result.stdout.split("\n\n"):
            values = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
            name = values.get("Path", "")
            if name.endswith(".img.nii.gz"):
                rows.append({"scan_id": int(Path(name).name.split(".")[0]), "archive_group": group,
                             "relative_path": name, "size_bytes": int(values["Size"]), "zip_crc32": values["CRC"]})
    if len(rows) != 1000 or {row["scan_id"] for row in rows} != set(range(1, 1001)):
        raise ValueError("Original archive catalogs do not describe 1000 unique CT scans")
    volumes = {int(p.name.split(".")[0]): p for p in (root / "raw/imagecas_volumes").rglob("*.img.nii.gz")}
    for row in rows:
        path = volumes[row["scan_id"]]
        if path.stat().st_size != row["size_bytes"]:
            raise ValueError(f"Extracted size differs from source ZIP catalog: {row['scan_id']}")
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "reports/image_integrity_inventory.csv", index=False)
    candidates = frame[frame.duplicated(["size_bytes", "zip_crc32"], keep=False)]
    duplicates = []
    for _, group in candidates.groupby(["size_bytes", "zip_crc32"]):
        hashes = {}
        for scan_id in group["scan_id"]:
            digest = hashlib.sha256()
            with volumes[int(scan_id)].open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                    digest.update(chunk)
            hashes.setdefault(digest.hexdigest(), []).append(int(scan_id))
        duplicates.extend(ids for ids in hashes.values() if len(ids) > 1)
    summary = {"original_ct_count": len(rows), "source_catalog_sizes_match": True,
               "crc_size_duplicate_candidates": int(len(candidates)), "sha256_confirmed_byte_duplicates": duplicates,
               "scope": "Exact compressed-file duplicates; does not establish absence of clinically repeated acquisitions."}
    write_json(root / "reports/image_integrity_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-geometry", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--available-only", action="store_true", help="Audit completed file sets while other groups download")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cohort, manifest = build_index()
    if not args.validate_geometry:
        print(json.dumps(manifest, indent=2))
        return
    if not args.available_only and not cohort["all_files_present"].all():
        raise ValueError("Download/extraction is incomplete; physical geometry validation cannot yet run")
    selected = cohort[cohort["all_files_present"]] if args.available_only else cohort
    selected = selected.iloc[:args.limit] if args.limit else selected
    if selected.empty:
        raise ValueError("No complete cases are available for geometry validation")
    if not args.available_only and not args.limit:
        manifest["image_integrity"] = audit_image_inventory()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(geometry_one, selected.to_dict("records")), 1):
            results.append(result)
            if index % 25 == 0:
                LOG.info("Geometry checked %d / %d", index, len(selected))
    suffix = "_sample" if args.available_only or args.limit else ""
    pd.DataFrame(results).to_csv(ROOT / "reports" / f"geometry_qc{suffix}.csv", index=False)
    counts = {name: sum(row.get("centerline_coordinate_convention") == name for row in results)
              for name in ("lps", "ras_to_lps")}
    summary = {"checked_subjects": len(results), "passed": sum(row["qc_pass"] for row in results),
               "failed_subject_ids": [row["scan_id"] for row in results if not row["qc_pass"]],
               "coordinate_convention_counts": counts, "is_full_cohort": len(results) == 800,
               "mask_overlap_is_preprocessing_qc_not_lesion_detection_validation": True}
    write_json(ROOT / "reports" / f"geometry_qc{suffix}_summary.json", summary)
    manifest[f"geometry_qc{suffix}"] = summary
    if len(results) == 800 and summary["passed"] == 800:
        manifest["status"] = "raw_cohort_geometry_verified"
    elif summary["failed_subject_ids"]:
        manifest["status"] = "geometry_review_required"
    write_json(ROOT / "prepared/external_cohort_manifest.json", manifest)
    print(json.dumps(summary, indent=2))
    if summary["failed_subject_ids"]:
        raise SystemExit("Some cases need geometry review; no extra subjects were silently excluded")


if __name__ == "__main__":
    main()
