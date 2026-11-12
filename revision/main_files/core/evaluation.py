"""Score patients once; fix operating thresholds using normal validation only."""

import time

import numpy as np
import torch

from .metrics import aggregate_patch_scores, bootstrap_metrics, normal_validation_threshold, patient_metrics


AGGREGATIONS = ("max", "mean", "top_k", "percentile")


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def score_patients(model, dataset, config, device, warmup=False):
    model.eval()
    records, patch_records = [], []
    warmed_up = False
    for patient, volumes in zip(dataset.records, dataset.volumes):
        mse_scores, sse_scores = [], []
        forward_seconds = 0.0
        zero_memory_rows, memory_rows = 0, 0
        patient_start = time.perf_counter()
        for start in range(0, len(volumes), config["batch_size"]):
            batch = volumes[start:start + config["batch_size"]].to(device)
            if warmup and not warmed_up:
                model(batch)
                _synchronize(device)
                warmed_up = True
                patient_start = time.perf_counter()
            _synchronize(device)
            before = time.perf_counter()
            reconstruction, attention = model(batch)
            _synchronize(device)
            forward_seconds += time.perf_counter() - before
            zero_memory_rows += int((attention.sum(-1) == 0).sum())
            memory_rows += attention.numel() // attention.shape[-1]
            if reconstruction.shape != batch.shape:
                raise ValueError(f"Reconstruction shape mismatch: {reconstruction.shape} vs {batch.shape}")
            errors = (batch - reconstruction).square().flatten(1)
            mse_scores.extend(errors.mean(1).cpu().tolist())
            sse_scores.extend(errors.sum(1).cpu().tolist())
        scores = sse_scores if config["patch_score_reduction"] == "sum" else mse_scores
        aggregates = {name: aggregate_patch_scores(scores, name, config["top_k"], config["aggregation_percentile"])
                      for name in AGGREGATIONS}
        if not np.isfinite(mse_scores).all() or not np.isfinite(sse_scores).all():
            raise ValueError(f"Non-finite reconstruction errors for {patient['patient_id']}")
        records.append({
            "patient_id": patient["patient_id"], "label": patient["label"], "patch_count": len(volumes),
            "mean_patch_mse": float(np.mean(mse_scores)), **aggregates,
            "zero_memory_row_fraction": zero_memory_rows / memory_rows,
            "network_forward_seconds": forward_seconds,
            "scoring_seconds": time.perf_counter() - patient_start,
        })
        patch_records.extend({"patient_id": patient["patient_id"], "label": patient["label"],
                              "patch_index": i, "mse": mse, "sse": sse}
                             for i, (mse, sse) in enumerate(zip(mse_scores, sse_scores)))
    return records, patch_records


def evaluate_held_out(validation_records, test_records, config):
    if not validation_records or any(row["label"] != 0 for row in validation_records):
        raise ValueError("Threshold calibration requires normal validation patients only")
    validation_ids = {row["patient_id"] for row in validation_records}
    ids = [row["patient_id"] for row in test_records]
    if validation_ids & set(ids):
        raise ValueError("Validation and test patient IDs overlap")
    labels = [row["label"] for row in test_records]
    results = {}
    for method in AGGREGATIONS:
        threshold = normal_validation_threshold([row[method] for row in validation_records],
                                                config["threshold_normal_percentile"])
        scores = [row[method] for row in test_records]
        metrics = patient_metrics(labels, scores, threshold, ids)
        metrics["threshold_source"] = "normal validation patient scores only"
        metrics["threshold_percentile"] = config["threshold_normal_percentile"]
        metrics["validation_patient_ids"] = sorted(validation_ids)
        metrics["role"] = "primary" if method == config["primary_aggregation"] else "prespecified secondary comparison"
        if method == config["primary_aggregation"]:
            metrics["bootstrap_ci"] = bootstrap_metrics(labels, scores, ids, config["bootstrap_samples"],
                                                        config["bootstrap_seed"], config["confidence_level"])
        results[method] = metrics
    return results
