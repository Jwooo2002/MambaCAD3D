#!/usr/bin/env python3
"""Rebuild ASOCA and ImageCAS-X from connected centerlines in physical mm."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.centerline_paths import extract_paths, read_asoca_graph, read_imagecas_graph
from scripts.physical_patches import make_patches, physical_to_indices

LOG = logging.getLogger(__name__)
ASOCA_SOURCE = Path("/home/jiwoong/PycharmProjects/journal_preprocess/ASOCADataAccess")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sources(dataset):
    if dataset == "asoca":
        return [{"dataset": dataset, "patient_id": f"{group}_{i}", "label": label,
                 "volume_path": str(ASOCA_SOURCE / group / "CTCA" / f"{group}_{i}.nrrd"),
                 "mask_path": str(ASOCA_SOURCE / group / "Annotations" / f"{group}_{i}.nrrd"),
                 "centerline_path": str(ASOCA_SOURCE / group / "Centerlines" / f"{group}_{i}.vtp")}
                for label, group in ((0, "Normal"), (1, "Diseased")) for i in range(1, 21)]
    with (ROOT / "external_data/imagecas_x/prepared/external_cohort.csv").open(newline="") as handle:
        return [{**row, "dataset": dataset, "label": int(row["label"])} for row in csv.DictReader(handle)]


def prepare_subject(task):
    row, protocol, output_root, implementation_sha256 = task
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    started = time.perf_counter()
    output_root = Path(output_root)
    if row["dataset"] == "asoca":
        # Retain the legacy patient filenames so frozen fold membership remains
        # identical after correcting the input. Names never choose model inputs.
        name = ("N" if row["label"] == 0 else "D") + "_" + row["patient_id"].rsplit("_", 1)[1] + "_patch"
    else:
        name = row["patient_id"].replace(":", "_")
    group = ("Normal" if row["label"] == 0 else "Diseased") if row["dataset"] == "asoca" else "patients"
    directory = output_root / ("ASOCA" if row["dataset"] == "asoca" else "ImageCAS-X") / group
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{name}.pkl"
    info_path = output.with_suffix(".json")
    geometry_path = output.with_suffix(".geometry.npz")
    source_keys = [key for key in ("volume_path", "mask_path", "centerline_path", "graph_output_path") if key in row]
    signatures = {key: file_signature(row[key]) for key in source_keys}
    fingerprint = hashlib.sha256(json.dumps({"protocol": protocol, "sources": signatures,
                                            "implementation_sha256": implementation_sha256}, sort_keys=True).encode()).hexdigest()
    if info_path.exists() and output.exists() and geometry_path.exists():
        existing = json.loads(info_path.read_text())
        if existing["input_fingerprint"] != fingerprint:
            raise ValueError(f"Frozen preprocessing differs: {info_path}. Use a new version/output directory.")
        if output.stat().st_size != existing["output_bytes"]:
            raise ValueError(f"Incomplete previous output: {output}")
        return {**existing, "resumed_existing": True}
    image = sitk.ReadImage(row["volume_path"])
    origin, spacing, direction = image.GetOrigin(), image.GetSpacing(), image.GetDirection()
    if row["dataset"] == "asoca":
        graphs = [read_asoca_graph(row["centerline_path"], origin, direction)]
    else:
        graphs = [read_imagecas_graph(row["graph_output_path"], side) for side in ("left", "right")]
    paths, audits = [], []
    for graph in graphs:
        audit = {}
        override_key = f"{row['patient_id']}:{graph.metadata.get('side', 'asoca')}"
        cycle_policy = protocol.get("topology_repair_overrides", {}).get(override_key, protocol["cycle_policy"])
        paths.extend(extract_paths(graph, mode=protocol["path_policy"], cycle_policy=cycle_policy, audit=audit))
        audit["source_side"] = graph.metadata.get("side", "asoca_all")
        audit["cycle_policy"] = cycle_policy
        audit["explicit_override"] = override_key if cycle_policy != protocol["cycle_policy"] else None
        edges = np.unique(np.sort(graph.edges, axis=1), axis=0)
        removed = {tuple(sorted(pair)) for pair in audit["cycle_edges_removed"]}
        kept = np.asarray([pair for pair in edges if tuple(pair) not in removed], dtype=np.int64)
        degree_before = np.bincount(edges.ravel(), minlength=len(graph.points))
        degree_after = np.bincount(kept.ravel(), minlength=len(graph.points))
        audit["artificial_terminal_nodes"] = np.flatnonzero((degree_before > 1) & (degree_after == 1)).tolist()
        audit["retained_active_node_count"] = int((degree_after > 0).sum())
        audit["source_terminal_node_count"] = int((degree_before == 1).sum())
        audit["root_count"] = len(graph.roots)
        audits.append(audit)
    mask = sitk.ReadImage(row["mask_path"])
    if (image.GetSize() != mask.GetSize() or
        not all(np.allclose(a, b, atol=1e-5) for a, b in ((origin, mask.GetOrigin()), (spacing, mask.GetSpacing()),
                                                        (direction, mask.GetDirection())))):
        raise ValueError(f"CT/mask geometry differs: {row['patient_id']}")
    mask_array = sitk.GetArrayViewFromImage(mask)
    # Lumen masks serve only to verify coordinates; they are never input channels.
    centers = np.concatenate([graph.points for graph in graphs])
    centers = centers[::max(1, len(centers) // 20000)]
    indices = np.rint(physical_to_indices(centers, origin, spacing, direction)).astype(np.int64)
    inside = ((indices >= 0) & (indices < np.asarray(image.GetSize()))).all(axis=1)
    lumen = np.zeros(len(indices), dtype=bool)
    valid = indices[inside]
    lumen[inside] = mask_array[valid[:, 2], valid[:, 1], valid[:, 0]] > 0
    lumen_fraction = float(lumen.mean())
    if lumen_fraction < protocol["minimum_centerline_in_lumen_fraction"]:
        raise ValueError(f"Centerline misses lumen for {row['patient_id']}: {lumen_fraction:.4f}")
    del mask_array, mask
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    np.clip(volume, *protocol["hu_window"], out=volume)
    patches, patch_rows, path_rows, geometry, qc = make_patches(volume, paths, origin, spacing, direction, protocol)
    if qc["resampled_adjacent_chord_max_mm"] > protocol["longitudinal_spacing_mm"] + 1e-5:
        raise ValueError(f"Discontinuous resampled centerline: {row['patient_id']}")
    temp = output.with_suffix(".pkl.tmp")
    with temp.open("wb") as handle:
        pickle.dump(patches, handle, protocol=pickle.HIGHEST_PROTOCOL)
    output_hash = hashlib.sha256(temp.read_bytes()).hexdigest()
    temp.replace(output)
    with geometry_path.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, **geometry)
    geometry_path.with_suffix(".npz.tmp").replace(geometry_path)
    info = {"status": "prepared", "patient_id": row["patient_id"], "dataset": row["dataset"],
            "label": row["label"], "revision_role": "external_test" if row["dataset"] != "asoca" else "internal_cv",
            "output_path": str(output.resolve()), "geometry_path": str(geometry_path.resolve()),
            "output_bytes": output.stat().st_size, "output_sha256": output_hash,
            "input_fingerprint": fingerprint, "implementation_sha256": implementation_sha256,
            "source_signatures": signatures, "protocol": protocol,
            "source_geometry": {"origin_lps_mm": origin, "spacing_mm_xyz": spacing, "direction_lps": direction},
            "coordinate_correction": graphs[0].metadata.get("affine_lps"),
            "topology_audits": audits, "centerline_in_lumen_fraction": lumen_fraction,
            "centerline_in_bounds_fraction": float(inside.mean()), "patches": patch_rows, "paths": path_rows,
            **qc, "preprocessing_seconds": time.perf_counter() - started,
            "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(info_path, info)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=("asoca", "imagecas-x"), default=["asoca", "imagecas-x"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/physical_sampling_v2.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "processed_data/ordered_mm_v2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="+", help="Patient IDs for a first geometry check; reuse outputs in full run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.workers < 1:
        parser.error("--workers must be positive")
    protocol = json.loads(args.config.read_text())
    code_files = [Path(__file__), ROOT / "scripts/physical_patches.py", ROOT / "scripts/centerline_paths.py"]
    implementation = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files}
    implementation_sha256 = hashlib.sha256(json.dumps(implementation, sort_keys=True).encode()).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = [row for dataset in args.dataset for row in sources(dataset)
                if args.only is None or row["patient_id"] in args.only]
    if args.only and set(args.only) != {row["patient_id"] for row in selected}:
        raise ValueError("Some --only IDs were not found")
    if not selected:
        raise ValueError("No selected patients")
    progress_path = args.output_root / ("preprocessing_" + "_".join(args.dataset) + ("_sample" if args.only else "") + ".json")
    progress = {"status": "running", "protocol": protocol, "implementation": implementation,
                "expected_patients": len(selected), "completed_patients": 0,
                "started_utc": datetime.now(timezone.utc).isoformat(), "patients": [], "errors": []}
    write_json(progress_path, progress)
    tasks = [(row, protocol, str(args.output_root), implementation_sha256) for row in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(prepare_subject, task): task[0]["patient_id"] for task in tasks}
        for future in as_completed(futures):
            patient = futures[future]
            try:
                result = future.result()
                progress["patients"].append({key: result[key] for key in ("patient_id", "label", "output_path", "output_sha256",
                    "patch_count", "path_count", "padded_patch_count", "duplicate_windows_removed", "centerline_in_lumen_fraction",
                    "resampled_adjacent_chord_max_mm", "preprocessing_seconds")})
                progress["completed_patients"] += 1
                LOG.info("%d/%d %s: %d patches, lumen %.4f, %.1f s", progress["completed_patients"], len(tasks),
                         patient, result["patch_count"], result["centerline_in_lumen_fraction"], result["preprocessing_seconds"])
            except Exception as exc:
                progress["errors"].append({"patient_id": patient, "error": f"{type(exc).__name__}: {exc}"})
                LOG.exception("Failed %s", patient)
            write_json(progress_path, progress)
    progress["status"] = "complete" if not progress["errors"] else "failed"
    progress["finished_utc"] = datetime.now(timezone.utc).isoformat()
    # Freeze model-ready external paths only for a complete source cohort.
    external = [row for row in selected if row["dataset"] == "imagecas-x"]
    if not args.only and len(external) == 800 and not progress["errors"]:
        lookup = {row["patient_id"]: row for row in progress["patients"]}
        target = args.output_root / "external_cohort_patches.csv"
        with target.with_suffix(".csv.tmp").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["patient_id", "label", "revision_role", "path", "patch_sha256", "patch_count"])
            writer.writeheader()
            for row in external:
                result = lookup[row["patient_id"]]
                writer.writerow({"patient_id": row["patient_id"], "label": row["label"], "revision_role": "external_test",
                                 "path": result["output_path"], "patch_sha256": result["output_sha256"], "patch_count": result["patch_count"]})
        target.with_suffix(".csv.tmp").replace(target)
        progress["external_patch_index_path"] = str(target.resolve())
        progress["external_patch_index_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(progress_path, progress)
    if progress["errors"]:
        raise RuntimeError(f"{len(progress['errors'])} patients failed; see {progress_path}")
    print(json.dumps({key: value for key, value in progress.items() if key != "patients"}, indent=2))


if __name__ == "__main__":
    main()
