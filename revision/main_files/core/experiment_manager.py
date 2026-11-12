"""Run frozen folds/seeds and retain scores needed to audit every metric."""

import csv
import hashlib
import json
import logging
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch

from main_files import csms6s
from main_files.data.patients import PatientPatchDataset
from .evaluation import evaluate_held_out, score_patients
from .external import load_external_records, score_external
from .model_trainer import train_one_fold
from .protocol import PROJECT_ROOT


logger = logging.getLogger(__name__)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config, manifest, device_name="cuda", selected_folds=None, selected_seeds=None):
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Use a GPU-enabled execution environment; CPU is only suitable for small tests.")
        if not any((csms6s.WITH_SELECTIVESCAN_OFLEX, csms6s.WITH_SELECTIVESCAN_CORE, csms6s.WITH_SELECTIVESCAN_MAMBA)):
            raise RuntimeError("No compiled selective-scan kernel is available; refusing a full-volume Python fallback on CUDA.")
    selected_folds = selected_folds or [fold["fold"] for fold in manifest["folds"]]
    selected_seeds = selected_seeds or config["training_seeds"]
    if not set(selected_folds) <= {fold["fold"] for fold in manifest["folds"]}:
        raise ValueError("Unknown fold selection")
    if not set(selected_seeds) <= set(config["training_seeds"]):
        raise ValueError("Selected seeds must be declared in the configuration")
    external_records = None
    if config.get("external_patch_index"):
        external_records = load_external_records(config["external_patch_index"], config.get("external_index_sha256"),
                                                 config.get("expected_external_patients"))
        if {p["patient_id"] for p in external_records} & {p["patient_id"] for p in manifest["patients"]}:
            raise ValueError("External and internal patients overlap")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(config["save_path"]) / f"{config['run_name']}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    file_handler = logging.FileHandler(run_dir / "experiment.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "selected_folds": selected_folds, "selected_seeds": selected_seeds,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "cuda": torch.version.cuda, "device": str(device),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "scan_kernels": {"oflex": csms6s.WITH_SELECTIVESCAN_OFLEX,
                                         "core": csms6s.WITH_SELECTIVESCAN_CORE,
                                         "mamba": csms6s.WITH_SELECTIVESCAN_MAMBA},
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "timing_scope": "Already prepared patches; forward and scoring times exclude raw CT loading, centerline extraction and patch preprocessing.",
        "selection": "Normal validation mean reconstruction MSE; no held-out test labels or scores are used for checkpoint/threshold selection.",
        "limitations": ["ASOCA was already used during historical development; renewed CV is internal validation.",
                        "Three normal validation patients give an unstable percentile threshold.",
                        "Fold/seed observations are correlated; summary SD is descriptive, not a confidence interval.",
                        "Architecture and preprocessing provenance must still be reconciled with the submitted checkpoint."],
    }
    write_json(run_dir / "run_config.json", metadata)
    write_json(run_dir / "split_manifest.json", manifest)
    sources = {}
    paths_to_snapshot = list((PROJECT_ROOT / "main_files").rglob("*.py")) + list((PROJECT_ROOT / "scripts").glob("*.py"))
    for path in sorted(paths_to_snapshot):
        relative = path.relative_to(PROJECT_ROOT)
        target = run_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(PROJECT_ROOT / "main.py", run_dir / "source_snapshot" / "main.py")
    sources["main.py"] = hashlib.sha256((PROJECT_ROOT / "main.py").read_bytes()).hexdigest()
    write_json(run_dir / "source_hashes.json", sources)
    if "preprocessing_config" in config:
        shutil.copy2(config["preprocessing_config"], run_dir / "preprocessing_config.json")
        shutil.copy2(config["preprocessing_manifest"], run_dir / "preprocessing_manifest.json")
    if external_records is not None:
        shutil.copy2(config["external_patch_index"], run_dir / "external_patch_index.csv")
    logger.info("Run directory: %s", run_dir)
    patients = {row["patient_id"]: row for row in manifest["patients"]}
    finished = []
    try:
        for fold in manifest["folds"]:
            if fold["fold"] not in selected_folds:
                continue
            for seed in selected_seeds:
                actual_seed = seed + 1000 * (fold["fold"] - 1)
                context = {"fold": fold["fold"], "seed": seed}
                fold_dir = run_dir / f"fold_{fold['fold']}" / f"seed_{seed}"
                logger.info("Starting fold=%d seed=%d (actual training seed=%d)", fold["fold"], seed, actual_seed)
                model, validation_records, validation_patches, training = train_one_fold(
                    [patients[pid] for pid in fold["train"]], [patients[pid] for pid in fold["validation"]],
                    config, device, actual_seed, fold_dir)
                # Load and score test subjects only after checkpoint selection has finished.
                test_dataset = PatientPatchDataset([patients[pid] for pid in fold["test_normal"] + fold["test_diseased"]],
                                                    config["hu_window"], config["expected_patch_shape"])
                test_records, test_patches = score_patients(model, test_dataset, config, device, warmup=True)
                # Preserve raw predictions before statistical reporting can fail/interruption occurs.
                raw_patient_rows, patch_rows = [], []
                for role, records, patches in (("validation", validation_records, validation_patches),
                                                ("test", test_records, test_patches)):
                    raw_patient_rows.extend({**context, "role": role, **row,
                                             "score": row[config["primary_aggregation"]]} for row in records)
                    patch_rows.extend({**context, "role": role, **row} for row in patches)
                write_csv(fold_dir / "patient_scores_raw.csv", raw_patient_rows)
                write_csv(fold_dir / "patch_scores.csv", patch_rows)
                evaluations = evaluate_held_out(validation_records, test_records, config)
                primary = config["primary_aggregation"]
                threshold = evaluations[primary]["threshold"]
                patient_rows = []
                for role, records, patches in (("validation", validation_records, validation_patches),
                                                ("test", test_records, test_patches)):
                    patient_rows.extend({**context, "role": role, **row, "score": row[primary],
                                         "aggregation": primary, "threshold": threshold,
                                         "prediction": int(row[primary] > threshold)} for row in records)
                write_csv(fold_dir / "patient_scores.csv", patient_rows)
                result = {**context, "training": training, "aggregation_results": evaluations}
                write_json(fold_dir / "metrics.json", result)
                if external_records is not None:
                    result["external_evaluation"] = score_external(model, external_records, validation_records, config, device, fold_dir, context)
                    write_json(fold_dir / "external_metrics.json", result["external_evaluation"])
                write_json(fold_dir / "metrics.json", result)
                finished.append(result)
                update_summary(run_dir, finished, config, "running")
                logger.info("Finished fold=%d seed=%d AUROC=%.4f AP=%.4f threshold=%.6f",
                            fold["fold"], seed, evaluations[primary]["auroc"], evaluations[primary]["ap"], threshold)
                del model, test_dataset
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        update_summary(run_dir, finished, config, "complete")
    except BaseException as exc:
        update_summary(run_dir, finished, config, "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                       f"{type(exc).__name__}: {exc}")
        raise
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
    return run_dir


def update_summary(run_dir, results, config, status, error=None):
    primary = config["primary_aggregation"]
    descriptive = {}
    for key in ("auroc", "ap", "sensitivity", "specificity", "f1"):
        values = [row["aggregation_results"][primary][key] for row in results]
        if values:
            descriptive[key] = {"mean": float(np.mean(values)),
                                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                "values": values}
    write_json(Path(run_dir) / "summary.json", {
        "status": status, "error": error, "completed_fold_seed_runs": len(results),
        "primary_aggregation": primary, "descriptive_fold_seed_summary": descriptive,
        "warning": "Repeated patients and overlapping training sets make folds/seeds correlated. Do not treat these as independent patients or use SD as a confidence interval.",
        "results": results,
    })
