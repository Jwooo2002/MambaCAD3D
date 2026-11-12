"""Score the frozen external cohort using only internal validation calibration."""

import csv
import hashlib
import logging
from pathlib import Path

from main_files.data.patients import PatientPatchDataset
from .evaluation import evaluate_held_out, score_patients

LOG = logging.getLogger(__name__)


def load_external_records(path, expected_sha256=None, expected_count=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Frozen external patch index changed")
    with path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records or (expected_count is not None and len(records) != expected_count):
        raise ValueError("External cohort is incomplete")
    ids, paths = set(), set()
    for row in records:
        row["label"] = int(row["label"])
        resolved = str(Path(row["path"]).resolve())
        if (row["revision_role"] != "external_test" or row["label"] not in (0, 1) or
            row["patient_id"] in ids or resolved in paths):
            raise ValueError("Invalid external role, label, or duplicated patient/volume")
        if not Path(resolved).is_file():
            raise ValueError(f"Missing external patch file: {resolved}")
        row["path"] = resolved
        ids.add(row["patient_id"])
        paths.add(resolved)
    if {row["label"] for row in records} != {0, 1}:
        raise ValueError("External evaluation needs both classes")
    return records


def score_external(model, records, validation_records, config, device, fold_dir, context):
    """Load one CT scan's prepared patches at a time; never fit on this cohort."""
    patient_rows = []
    raw_path = Path(fold_dir) / "external_patient_scores_raw.csv"
    patch_path = Path(fold_dir) / "external_patch_scores.csv"
    with raw_path.open("w", newline="") as patient_handle, patch_path.open("w", newline="") as patch_handle:
        patient_writer, patch_writer = None, None
        for i, patient in enumerate(records):
            dataset = PatientPatchDataset([patient], config["hu_window"], config["expected_patch_shape"])
            scored, patches = score_patients(model, dataset, config, device, warmup=(i == 0))
            row = {**context, "role": "external_test", **scored[0], "score": scored[0][config["primary_aggregation"]]}
            patch_batch = [{**context, "role": "external_test", **p} for p in patches]
            if patient_writer is None:
                patient_writer = csv.DictWriter(patient_handle, fieldnames=list(row))
                patch_writer = csv.DictWriter(patch_handle, fieldnames=list(patch_batch[0]))
                patient_writer.writeheader()
                patch_writer.writeheader()
            patient_writer.writerow(row)
            patch_writer.writerows(patch_batch)
            patient_handle.flush()
            patch_handle.flush()
            patient_rows.extend(scored)
            if (i + 1) % 50 == 0 or i + 1 == len(records):
                LOG.info("External fold=%d seed=%d: scored %d/%d frozen patients", context["fold"], context["seed"], i + 1, len(records))
    metrics = evaluate_held_out(validation_records, patient_rows, config)
    for result in metrics.values():
        result["evaluation_cohort"] = "external_test"
        result["threshold_source"] = "ASOCA normal validation patients only; no external recalibration"
    return {"patient_count": len(records), "normal_count": sum(p["label"] == 0 for p in records),
            "diseased_count": sum(p["label"] == 1 for p in records), "aggregation_results": metrics,
            "patient_scores_path": str(raw_path), "patch_scores_path": str(patch_path),
            "scope": "Frozen reference-centerline-based external evaluation; no lesion localization ground truth."}
