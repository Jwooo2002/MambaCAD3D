#!/usr/bin/env python3
"""Prepare frozen splits, train revision experiments, or audit patient score CSVs."""

import argparse
import csv
import json
import logging
import os
from pathlib import Path

from main_files.core.protocol import PROJECT_ROOT, load_config, prepare_manifest


def audit_csv(path, baseline=None):
    from main_files.core.metrics import bootstrap_metrics, patient_metrics
    with Path(path).open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("role", "test") == "test"]
    if not rows:
        raise ValueError("No test patient rows")
    contexts = {(row.get("fold"), row.get("seed")) for row in rows}
    if len(contexts) != 1:
        raise ValueError("Audit one fold and training seed at a time; repeated patients cannot be pooled")
    ids = [row["patient_id"] for row in rows]
    labels, scores = [int(row["label"]) for row in rows], [float(row["score"]) for row in rows]
    result = patient_metrics(labels, scores, patient_ids=ids)
    if baseline:
        with Path(baseline).open(newline="") as handle:
            other_rows = [row for row in csv.DictReader(handle) if row.get("role", "test") == "test"]
        if {(row.get("fold"), row.get("seed")) for row in other_rows} != contexts:
            raise ValueError("Baseline fold/seed does not match")
        other = {row["patient_id"]: row for row in other_rows}
        if len(other) != len(other_rows) or set(other) != set(ids):
            raise ValueError("Baseline must contain the same unique held-out patients")
        if any(int(other[pid]["label"]) != label for pid, label in zip(ids, labels)):
            raise ValueError("Baseline patient labels differ")
        baseline_scores = [float(other[pid]["score"]) for pid in ids]
        result["baseline"] = patient_metrics(labels, baseline_scores, patient_ids=ids)
        result["paired_difference_ci"] = bootstrap_metrics(labels, scores, ids, baseline=baseline_scores)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "audit"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/asoca_revision.json")
    parser.add_argument("--gpu", default="0", help="Physical GPU index; applied before importing torch")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--epochs", type=int, help="Override for a smoke run; use a distinct run name")
    parser.add_argument("--run-name")
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--baseline-scores", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.action == "audit":
        if not args.scores:
            parser.error("audit requires --scores")
        print(json.dumps(audit_csv(args.scores, args.baseline_scores), indent=2))
        return
    config = load_config(args.config)
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        config["epochs"] = args.epochs
        config["run_name"] += f"_smoke_{args.epochs}epoch"
    if args.run_name:
        config["run_name"] = args.run_name
    manifest = prepare_manifest(config)
    if args.action == "prepare":
        print(json.dumps({"split_manifest": config["split_manifest"],
                          "normal_subjects": sum(p["label"] == 0 for p in manifest["patients"]),
                          "diseased_subjects": sum(p["label"] == 1 for p in manifest["patients"]),
                          "folds": [{"fold": f["fold"], **{key: len(f[key]) for key in
                                    ("train", "validation", "test_normal", "test_diseased")}} for f in manifest["folds"]]}, indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from main_files.core.experiment_manager import run_experiment
    run_dir = run_experiment(config, manifest, args.device, args.folds, args.seeds)
    print(f"Completed: {run_dir}")


if __name__ == "__main__":
    main()
