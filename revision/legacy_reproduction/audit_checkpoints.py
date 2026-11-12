#!/usr/bin/env python3
"""Compare recovered original inference against archived per-patch scores."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(HERE / "source"), str(PROJECT / "legacy_source/main_files")]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torchvision import transforms
    import model as implementation
    from core.fp_based_evaluator import FPBasedEvaluator

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for original compiled scan inference")
    recovered = json.loads((HERE / "recovery_manifest.json").read_text())
    historical = Path(recovered["historical_run"])
    original_config = json.loads((historical / "configs/experiment_config.json").read_text())
    config = OmegaConf.create(original_config)
    roots = {label: PROJECT.parent / "MemAE_kfold_simplescan/main_files/dataset/v11/0.05_8_32_CTCA_64" / group
             for label, group in ((0, "Normal"), (1, "Diseased"))}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x + 300) / 700)])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = PROJECT / "outputs" / f"legacy_checkpoint_audit_{stamp}"
    output.mkdir(exist_ok=False)
    report = {"status": "running", "historical_run": str(historical), "output_path": str(output),
              "expected_folds": args.folds, "original_config": original_config,
              "source_model": str(Path(implementation.__file__).resolve()),
              "recovery_manifest": recovered, "environment": {"torch": torch.__version__, "numpy": np.__version__,
              "gpu": torch.cuda.get_device_name(0)}, "folds": [],
              "scope": "Historical reconstruction using original unordered v11 inputs, legacy score function and recovered kernel-2 architecture. This is not an unbiased final revision estimate."}
    write_json(output / "audit.json", report)
    device = torch.device("cuda")
    try:
        for fold in args.folds:
            path = historical / "models" / f"fold_{fold}_model.pt"
            # User-owned local training artifacts; results include NumPy arrays.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            archived = checkpoint["results"]
            model = implementation.MemAE_VSSM3D(in_chans=config.in_chans, dims=config.dims,
                depths=config.depths, ssm_d_state=config.ssm_d_state, mem_dim=config.mem_dim).to(device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            predicted, reference, labels, ids, patch_rows = [], [], [], [], []
            max_absolute_error = 0.0
            comparisons = []
            for label, key in ((0, "normal_patient_scores"), (1, "diseased_patient_scores")):
                archived_scores = archived[key]
                filenames = [patient + ".pkl" for patient in archived_scores]
                logging.info("fold=%d class=%d scoring %d original patients", fold, label, len(filenames))
                current = FPBasedEvaluator._calculate_patient_scores(model, filenames, str(roots[label]), transform, device, config)
                for patient in archived_scores:
                    expected = np.asarray(archived_scores[patient], dtype=float)
                    actual = np.asarray(current[patient], dtype=float)
                    if actual.shape != expected.shape:
                        raise ValueError(f"Original patch count differs for {patient}: {actual.shape} vs {expected.shape}")
                    error = float(np.max(np.abs(actual - expected)))
                    max_absolute_error = max(max_absolute_error, error)
                    close = bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-7))
                    comparisons.append(close)
                    ids.append(patient)
                    labels.append(label)
                    predicted.append(float(actual.max()))
                    reference.append(float(expected.max()))
                    patch_rows.extend({"patient_id": patient, "label": label, "patch_index": i,
                        "recomputed_mse": float(score), "archived_mse": float(old)}
                        for i, (score, old) in enumerate(zip(actual, expected)))
            with (output / f"fold_{fold}_patch_scores.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(patch_rows[0])); writer.writeheader(); writer.writerows(patch_rows)
            with (output / f"fold_{fold}_patient_scores.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["patient_id", "label", "recomputed_max_mse", "archived_max_mse"])
                writer.writeheader()
                writer.writerows({"patient_id": patient, "label": label, "recomputed_max_mse": value, "archived_max_mse": old}
                                 for patient, label, value, old in zip(ids, labels, predicted, reference))
            result = {"fold": fold, "checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "parameter_count": sum(p.numel() for p in model.parameters()), "normal_count": labels.count(0),
                      "diseased_count": labels.count(1), "patch_count": len(patch_rows),
                      "all_patch_scores_match": all(comparisons), "comparison_tolerance": {"rtol": 1e-4, "atol": 1e-7},
                      "max_absolute_patch_mse_difference": max_absolute_error,
                      "stored_auroc": float(archived["patient_auc"]),
                      "archived_auroc_recomputed": float(roc_auc_score(labels, reference)),
                      "archived_ap_recomputed": float(average_precision_score(labels, reference)),
                      "recovered_auroc": float(roc_auc_score(labels, predicted)),
                      "recovered_ap": float(average_precision_score(labels, predicted))}
            report["folds"].append(result)
            write_json(output / "audit.json", report)
            logging.info("FOLD RESULT %s", json.dumps(result))
            if not result["all_patch_scores_match"]:
                raise ValueError("Recovered inference differs from archived per-patch scores; investigate before claiming reproduction")
            del model
            torch.cuda.empty_cache()
        report["status"] = "verified"
        report["mean_auroc"] = float(np.mean([f["recovered_auroc"] for f in report["folds"]]))
        report["mean_ap"] = float(np.mean([f["recovered_ap"] for f in report["folds"]]))
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
