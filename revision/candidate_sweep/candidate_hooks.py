"""Opt-in interventions for a frozen copy of the ongoing legacy run."""

import json
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


OPTIONS = {}
OUTPUT = None


def configure(options, output):
    global OPTIONS, OUTPUT
    OPTIONS = dict(options)
    OUTPUT = Path(output)


def memory_logits(query, memory):
    if OPTIONS.get("cosine", False):
        return F.linear(F.normalize(query, dim=-1), F.normalize(memory, dim=-1))
    return F.linear(query, memory)


def initialize_dt(model):
    if not OPTIONS.get("explicit_dt", False):
        return
    # The model is still on CPU. Do not shift the RNG stream used for sampling
    # or alter initialization of any other parameter.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for module in model.modules():
            if not hasattr(module, "dt_proj"):
                continue
            bound = module.dt_rank ** -0.5
            module.dt_proj.weight.uniform_(-bound, bound)
            dt = torch.exp(torch.rand_like(module.dt_proj.bias) *
                           (np.log(0.1) - np.log(0.001)) + np.log(0.001))
            module.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))


def sampling_from_counts(counts):
    if not counts or any(n <= 0 for n in counts):
        raise ValueError("Every training patient must have at least one patch")
    weights = torch.cat([torch.full((n,), 1.0 / n, dtype=torch.double) for n in counts])
    return WeightedRandomSampler(weights, num_samples=sum(counts), replacement=True)


def training_sampling(dataset, files, data_root):
    if not OPTIONS.get("patient_uniform", False):
        return {"shuffle": True}
    counts = []
    for name in files:
        with (Path(data_root) / name).open("rb") as handle:
            counts.append(len(pickle.load(handle)))
    if sum(counts) != len(dataset):
        raise ValueError("Sampler patch order/count differs from the training dataset")
    return {"shuffle": False, "sampler": sampling_from_counts(counts)}


def record_fold(fold_idx, results):
    from sklearn.metrics import average_precision_score, roc_auc_score

    patients = []
    for label, key in ((0, "normal_patient_scores"), (1, "diseased_patient_scores")):
        for name, scores in results[key].items():
            values = np.asarray(scores, dtype=float)
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError("Invalid patient patch scores")
            patients.append({"patient_id": name, "label": label,
                             "patch_scores": values.tolist(), "max_score": float(values.max())})
    labels = [row["label"] for row in patients]
    scores = [row["max_score"] for row in patients]
    auc = float(roc_auc_score(labels, scores))
    if not np.isclose(auc, results["patient_auc"]):
        raise ValueError("Independent AUROC calculation differs from the legacy evaluator")
    summary = {"fold": fold_idx + 1, "auroc": auc,
               "ap": float(average_precision_score(labels, scores)),
               "total_epochs": results["total_epochs"], "best_epoch_label": results["best_epoch"],
               "checkpoint_note": "Legacy shallow-copy behavior retained; best_epoch is not a frozen best checkpoint.",
               "patients": patients, "train_history": results["train_history"]}
    destination = OUTPUT / f"fold_{fold_idx + 1}_observation.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
