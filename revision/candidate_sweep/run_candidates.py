#!/usr/bin/env python3
"""Prepare immutable legacy-matched candidates, then run a finite GPU queue."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
VARIANTS = {
    "R3_cosine": {"cosine": True},
    "R1_dt": {"explicit_dt": True},
    "R2_lambda": {"entropy_loss_weight": 0.0002 * 1024 / 300},
    "R4_sampling": {"patient_uniform": True},
    "R5_dt_lambda": {"explicit_dt": True, "entropy_loss_weight": 0.0002 * 1024 / 300},
    "R6_cosine_lambda": {"cosine": True, "entropy_loss_weight": 0.0002 * 1024 / 300},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one patch target in {path}: {old}")
    path.write_text(text.replace(old, new, 1))


def patch_source(source):
    shutil.copy2(HERE / "candidate_hooks.py", source / "candidate_hooks.py")
    model = source / "model.py"
    replace_once(model, "import math\n", "import math\nfrom candidate_hooks import memory_logits, initialize_dt\n")
    replace_once(model, "att_weight = F.linear(x_flat.permute(0, 2, 1), self.memory)",
                 "att_weight = memory_logits(x_flat.permute(0, 2, 1), self.memory)")
    replace_once(model, "        self.apply(self._init_weights)",
                 "        self.apply(self._init_weights)\n        initialize_dt(self)")
    trainer = source / "core/model_trainer.py"
    replace_once(trainer, "from model import MemAE_VSSM3D", "from model import MemAE_VSSM3D\nfrom candidate_hooks import training_sampling")
    replace_once(trainer, "            shuffle=True,", "            **training_sampling(train_dataset, train_files, data_roots['normal']),")
    manager = source / "core/experiment_manager.py"
    replace_once(manager, "from core.model_trainer import ModelTrainer", "from core.model_trainer import ModelTrainer\nfrom candidate_hooks import record_fold")
    replace_once(manager, "            fold_results.append(results)",
                 "            record_fold(fold_idx, results)\n            fold_results.append(results)")


def prepare(baseline, root, gpu):
    baseline, root = baseline.resolve(), root.resolve()
    config = json.loads((baseline / "effective_config.json").read_text())
    hashes = json.loads((baseline / "source_hashes.json").read_text())
    original_source = baseline / "source_snapshot/main_files"
    for name, expected in hashes.items():
        if digest(original_source / name) != expected:
            raise ValueError(f"Baseline source changed: {name}")
    if (config['k_folds'], config['random_seed'], config['entropy_loss_weight']) != (4, 42, 0.0002):
        raise ValueError("Unexpected baseline settings; inspect before preparing candidates")
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), root / "run_candidates.py")
    shutil.copy2(HERE / "candidate_hooks.py", root / "candidate_hooks.py")
    from sklearn.model_selection import KFold
    normal = sorted(Path(config["normal_data_root"]).glob("*.pkl"))
    diseased = sorted(Path(config["diseased_data_root"]).glob("*.pkl"))
    if len(normal) != 20 or len(diseased) != 20:
        raise ValueError("Expected 20 normal and 20 diseased patients")
    splits = []
    for index, (train, test) in enumerate(KFold(4, shuffle=True, random_state=42).split(normal), 1):
        splits.append({"fold": index, "train": [normal[i].name for i in train],
                       "normal_test_and_validation": [normal[i].name for i in test],
                       "diseased_test": [p.name for p in diseased]})
    write_json(root / "split_manifest.json", splits)
    inputs = [{"path": str(p), "size_bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
              for p in normal + diseased]
    write_json(root / "input_manifest.json", inputs)
    jobs = []
    for name, options in VARIANTS.items():
        output = root / name
        source = output / "source_snapshot/main_files"
        shutil.copytree(original_source, source, ignore=shutil.ignore_patterns("__pycache__"))
        patch_source(source)
        effective = dict(config, save_path=str(output / "results"), run_name=name, gpu_id=gpu)
        if "entropy_loss_weight" in options:
            effective["entropy_loss_weight"] = options["entropy_loss_weight"]
        write_json(output / "effective_config.json", effective)
        write_json(output / "variant.json", options)
        write_json(output / "source_hashes.json", {str(p.relative_to(source)): digest(p) for p in source.rglob("*.py")})
        jobs.append({"variant": name, "output": str(output), "status": "queued"})
    manifest = {"status": "prepared", "created_utc": now(), "gpu": gpu,
                "baseline": str(baseline), "baseline_source_hashes": hashes,
                "root_seed": 42, "folds_per_variant": 4, "candidate_training_runs": 24,
                "scope": "Exploratory interventions on the ongoing historical reproduction; not corrected T0 or final revision CV.",
                "retained": ["Original input and normalization", "Kernel-2 model and legacy SSM/memory/entropy",
                             "Normal test-as-validation", "Shallow-copy checkpoint behavior", "All test patches",
                             "Sequential global RNG, without reseeding at each fold"],
                "rng_limit": "Same root seed and fold order, not matched per-fold initialization after trajectories diverge.",
                "sampling_budget": "Same draws and optimizer steps per epoch; legacy early stopping can change total epochs/updates.",
                "lambda_meaning": "3.4133x stronger coefficient on unchanged legacy entropy, not corrected-entropy compensation.",
                "omitted": {"T7": "Legacy memory clamp already does not hard-shrink; disabling it is a redundant control."},
                "external_evaluation": False, "jobs": jobs}
    write_json(root / "sweep_manifest.json", manifest)
    print(json.dumps({"root": str(root), "candidate_training_runs": 24, "gpu": gpu}))


def run_one(output):
    output = output.resolve()
    config = json.loads((output / "effective_config.json").read_text())
    source = output / "source_snapshot/main_files"
    for name, expected in json.loads((output / "source_hashes.json").read_text()).items():
        if digest(source / name) != expected:
            raise ValueError(f"Candidate source changed: {name}")
    for row in json.loads((output.parent / "input_manifest.json").read_text()):
        stat = Path(row["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["size_bytes"], row["mtime_ns"]):
            raise ValueError(f"Input changed: {row['path']}")
    os.environ.update(CUDA_VISIBLE_DEVICES=config["gpu_id"], WANDB_MODE="offline", MPLBACKEND="Agg",
                      MPLCONFIGDIR=str(output / "matplotlib_cache"))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from candidate_hooks import configure
    configure(json.loads((output / "variant.json").read_text()), output)
    from omegaconf import OmegaConf
    import torch
    from core.experiment_manager import ExperimentManager
    from utils.reproducibility import ReproducibilityManager
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    working = output / "work"
    working.mkdir(exist_ok=False)
    os.chdir(working)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "training.log")])
    state = {"status": "running", "started_utc": now(), "pid": os.getpid(), "variant": output.name,
             "gpu": config["gpu_id"], "root_seed": config["random_seed"], "expected_folds": 4}
    write_json(output / "run_state.json", state)
    try:
        ReproducibilityManager.set_seed(config["random_seed"])
        manager = ExperimentManager(OmegaConf.create(config))
        state["legacy_results_directory"] = manager.save_paths["main"]
        write_json(output / "run_state.json", state)
        manager.run_experiment()
        state.update(status="complete", finished_utc=now())
    except BaseException as error:
        state.update(status="failed", finished_utc=now(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "run_state.json", state)


def queue(root):
    root = root.resolve()
    path = root / "sweep_manifest.json"
    state = json.loads(path.read_text())
    if state["status"] != "prepared":
        raise ValueError("Queue is not newly prepared; inspect state before relaunch")
    state.update(status="running", queue_pid=os.getpid(), started_utc=now())
    write_json(path, state)
    for job in state["jobs"]:
        output = Path(job["output"])
        with (output / "launcher.log").open("xb") as log:
            process = subprocess.Popen([sys.executable, "-B", "-u", str(root / "run_candidates.py"),
                                        "run", "--output", str(output)], stdout=log, stderr=subprocess.STDOUT,
                                       cwd=root)
            job.update(status="running", pid=process.pid, started_utc=now())
            write_json(path, state)
            code = process.wait()
        job.update(status="complete" if code == 0 else "failed", returncode=code, finished_utc=now())
        write_json(path, state)
        if code != 0:
            state.update(status="failed", finished_utc=now())
            write_json(path, state)
            raise RuntimeError(f"{job['variant']} failed; later jobs remain queued")
    state.update(status="complete", finished_utc=now())
    write_json(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", default="1")
    for command in ("run", "queue"):
        p = commands.add_parser(command)
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.baseline, args.output, args.gpu)
    elif args.command == "run":
        run_one(args.output)
    else:
        queue(args.output)


if __name__ == "__main__":
    main()
