#!/usr/bin/env python3
"""Rerun archived settings using recovered model and preserved legacy trainer.

The legacy test-as-validation and checkpoint-copy behavior are intentionally
retained to trace the historical result. Outputs are separate from revision CV.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
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
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit_report.read_text())
    if (audit["status"] != "verified" or {f["fold"] for f in audit["folds"]} != {1, 2, 3, 4} or
        not all(f["all_patch_scores_match"] and f["parameter_count"] == 1720121 for f in audit["folds"])):
        raise ValueError("All four original checkpoints must match archived inference before retraining")
    output = args.output.resolve()
    output.mkdir(exist_ok=False, parents=True)
    source = output / "source_snapshot/main_files"
    shutil.copytree(PROJECT / "legacy_source/main_files", source, ignore=shutil.ignore_patterns("__pycache__"))
    recovery = json.loads((HERE / "recovery_manifest.json").read_text())
    for name in ("model.py", "csms6s.py", "utils_scan.py"):
        path = HERE / "source" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != recovery["source_files"][name]["reproduction_sha256"]:
            raise ValueError(f"Recovered source changed after audit: {name}")
        shutil.copy2(path, source / name)
    shutil.copy2(args.audit_report, output / "checkpoint_audit.json")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["WANDB_MODE"] = "offline"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_cache")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import timm
    from omegaconf import OmegaConf
    from core.experiment_manager import ExperimentManager
    from utils.reproducibility import ReproducibilityManager

    if not torch.cuda.is_available():
        raise RuntimeError("Original training requires a native GPU environment")
    historical = Path(recovery["historical_run"])
    original = json.loads((historical / "configs/experiment_config.json").read_text())
    config = dict(original)
    data = PROJECT.parent / "MemAE_kfold_simplescan/main_files/dataset/v11/0.05_8_32_CTCA_64"
    config.update({"normal_data_root": str(data / "Normal"), "diseased_data_root": str(data / "Diseased"),
                   "save_path": str(output / "results"), "run_name": "legacy_recovered_original", "gpu_id": args.gpu})
    write_json(output / "original_config.json", original)
    write_json(output / "effective_config.json", config)
    write_json(output / "source_hashes.json", {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in source.rglob("*.py")})
    reference = []
    for fold in range(1, 5):
        checkpoint = torch.load(historical / "models" / f"fold_{fold}_model.pt", map_location="cpu", weights_only=False)
        history = checkpoint["results"]["train_history"]
        reference.append({"fold": fold, "epoch_1": {key: values[0] for key, values in history.items()},
                          "stored_auroc": checkpoint["results"]["patient_auc"]})
    write_json(output / "archived_training_reference.json", reference)
    working = output / "work"
    working.mkdir()
    os.chdir(working)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "training.log")])
    state = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "expected_folds": 4,
             "output_path": str(output), "historical_run": str(historical), "root_seed": config["random_seed"],
             "fold_execution": "All four folds sequentially in the original manager; global RNG is not reset per fold.",
             "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                             "timm": timm.__version__, "gpu": torch.cuda.get_device_name(0)},
             "retained_historical_behavior": ["Original v11 unordered centerlines and voxel patches", "Kernel-2 recovered 1,720,121-parameter architecture", "Global direction-parameter split and identity memory clamp", "Legacy entropy axis", "Normal test subjects reused for early stopping", "Shallow-copy checkpoint selection", "Original score and reporting utilities"],
             "operational_adaptations": ["Absolute data/output paths and isolated working directory", "W&B local/offline logging and headless figures", "Direct OmegaConf loading bypasses later four-stage-only CLI validation; saved three-stage config unchanged"],
             "provenance_limit": "Original Oct-2025 model source was not archived. Recovered inference is validated against all archived patch scores. Available Nov-2025 trainer is reused; exact historical training trajectory is not yet verified.",
             "interpretation": "Historical reproduction only; not an unbiased final revision experiment."}
    write_json(output / "run_state.json", state)
    try:
        ReproducibilityManager.set_seed(config["random_seed"])
        manager = ExperimentManager(OmegaConf.create(config))
        state["legacy_results_directory"] = manager.save_paths["main"]
        write_json(output / "run_state.json", state)
        manager.run_experiment()
        state["status"] = "complete"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "run_state.json", state)


if __name__ == "__main__":
    main()
