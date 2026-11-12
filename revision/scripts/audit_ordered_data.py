#!/usr/bin/env python3
"""Verify every model-ready array, content hash, physical frame and cohort role."""

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "processed_data/ordered_mm_v2"


def audit_one(path):
    metadata = json.loads(path.with_suffix(".json").read_text())
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == metadata["output_sha256"], str(path)
    assert len(payload) == metadata["output_bytes"], str(path)
    array = np.asarray(pickle.loads(payload))
    assert array.dtype == np.float32 and array.shape == (metadata["patch_count"], 64, 32, 32), str(path)
    assert np.isfinite(array).all() and array.min() >= -300.001 and array.max() <= 400.001, str(path)
    assert len(metadata["patches"]) == len(array), str(path)
    assert metadata["protocol"] == json.loads((ROOT / "configs/physical_sampling_v2.json").read_text()), str(path)
    assert metadata["centerline_in_lumen_fraction"] >= .85, str(path)
    assert metadata["resampled_adjacent_chord_max_mm"] <= .50001, str(path)
    with np.load(path.with_suffix(".geometry.npz"), allow_pickle=False) as data:
        tangent, u, v = [data[key] for key in ("tangent_lps", "u_lps", "v_lps")]
        for axis in (tangent, u, v):
            assert np.isfinite(axis).all() and np.allclose(np.linalg.norm(axis, axis=1), 1, atol=1e-6), str(path)
        assert np.allclose(np.cross(tangent, u), v, atol=1e-6), str(path)
        assert np.allclose((tangent * u).sum(1), 0, atol=1e-6), str(path)
        assert len(data["points_lps_mm"]) == data["path_offsets"][-1], str(path)
    return {"patient_id": metadata["patient_id"], "dataset": metadata["dataset"], "label": metadata["label"],
            "patch_count": len(array), "padded_patch_count": metadata["padded_patch_count"],
            "duplicate_windows_removed": metadata["duplicate_windows_removed"], "output_bytes": len(payload),
            "lumen_fraction": metadata["centerline_in_lumen_fraction"],
            "input_implementation": metadata["implementation_sha256"]}


def main():
    asoca = list((DATA / "ASOCA").rglob("*.pkl"))
    external = list((DATA / "ImageCAS-X/patients").glob("*.pkl"))
    assert len(asoca) == 40 and len(external) == 800, (len(asoca), len(external))
    index = DATA / "external_cohort_patches.csv"
    with index.open(newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    assert len(index_rows) == 800 and len({row["patient_id"] for row in index_rows}) == 800
    assert all(row["revision_role"] == "external_test" for row in index_rows)
    assert sum(int(row["label"]) == 0 for row in index_rows) == 412
    assert sum(int(row["label"]) == 1 for row in index_rows) == 388
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(audit_one, asoca + external))
    assert len({row["patient_id"] for row in records}) == 840
    assert len({row["input_implementation"] for row in records}) == 1
    old = json.loads((ROOT / "splits/asoca_4fold.json").read_text())
    new = json.loads((ROOT / "splits/asoca_ordered_mm_v2_4fold.json").read_text())
    assert old["folds"] == new["folds"], "Patient roles changed"
    original = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for row in original["source_files"]:
        assert hashlib.sha256(Path(row["source"]).read_bytes()).hexdigest() == row["sha256"], row["source"]
        assert hashlib.sha256((ROOT / row["snapshot"]).read_bytes()).hexdigest() == row["sha256"], row["snapshot"]
    counts = {}
    for dataset in ("asoca", "imagecas-x"):
        rows = [r for r in records if r["dataset"] == dataset]
        counts[dataset] = {"patients": len(rows), "normal": sum(r["label"] == 0 for r in rows),
                           "diseased": sum(r["label"] == 1 for r in rows), "patches": sum(r["patch_count"] for r in rows),
                           "patch_count_range": [min(r["patch_count"] for r in rows), max(r["patch_count"] for r in rows)],
                           "padded_patches": sum(r["padded_patch_count"] for r in rows),
                           "duplicate_windows_removed": sum(r["duplicate_windows_removed"] for r in rows),
                           "minimum_centerline_lumen_fraction": min(r["lumen_fraction"] for r in rows),
                           "total_bytes": sum(r["output_bytes"] for r in rows)}
    report = {"status": "passed", "cohorts": counts, "identical_fold_patient_roles": True,
              "all_array_hashes_verified": True, "all_frame_geometries_verified": True,
              "original_code_files_preserved": len(original["source_files"]),
              "external_patch_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
              "preprocessing_implementation_sha256": records[0]["input_implementation"],
              "native_gpu_check": json.loads((ROOT / "outputs/ordered_v2_native_gpu_check.json").read_text())}
    destination = DATA / "readiness_audit.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
