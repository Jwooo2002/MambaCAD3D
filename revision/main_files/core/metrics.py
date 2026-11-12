"""Patient-level metrics and paired resampling for a single held-out cohort."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def aggregate_patch_scores(scores, method="max", top_k=5, percentile=95):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Patch scores must be a nonempty finite vector")
    if method == "max":
        return float(values.max())
    if method == "mean":
        return float(values.mean())
    if method == "top_k":
        if top_k < 1:
            raise ValueError("top_k must be positive")
        return float(np.sort(values)[-min(top_k, len(values)):].mean())
    if method == "percentile":
        return float(np.percentile(values, percentile))
    raise ValueError(f"Unknown aggregation: {method}")


def validate_cohort(labels, scores, patient_ids=None):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("labels and scores must be matching nonempty vectors")
    if set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Both binary classes and finite scores are required; disease must be 1")
    if patient_ids is not None and (len(patient_ids) != len(labels) or len(set(patient_ids)) != len(labels)):
        raise ValueError("One unique patient ID per observation is required; do not pool repeated folds")
    return labels.astype(int), scores


def patient_metrics(labels, scores, threshold=None, patient_ids=None):
    labels, scores = validate_cohort(labels, scores, patient_ids)
    result = {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "prevalence_reference": float(labels.mean()),
        "normal_count": int((labels == 0).sum()),
        "diseased_count": int((labels == 1).sum()),
        "positive_class": "diseased",
        "ap_definition": "sklearn.metrics.average_precision_score; patient-level continuous scores",
    }
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError("Threshold must be finite and fixed before testing")
        predicted = scores > threshold
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        result.update({
            "threshold": float(threshold), "decision_rule": "score > threshold",
            "sensitivity": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp)),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "accuracy": float((tp + tn) / len(labels)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })
    return result


def normal_validation_threshold(scores, percentile=95):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Normal validation patient scores must be finite and nonempty")
    return float(np.percentile(scores, percentile))


def bootstrap_metrics(labels, scores, patient_ids, n_bootstrap=2000, seed=2026, confidence=0.95, baseline=None):
    labels, scores = validate_cohort(labels, scores, patient_ids)
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    if baseline is not None:
        _, baseline = validate_cohort(labels, baseline, patient_ids)
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(labels == c) for c in (0, 1)]
    samples = {"auroc": [], "ap": []}
    for _ in range(n_bootstrap):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
        for name, metric in (("auroc", roc_auc_score), ("ap", average_precision_score)):
            value = metric(labels[indices], scores[indices])
            if baseline is not None:
                value -= metric(labels[indices], baseline[indices])
            samples[name].append(float(value))
    tail = (1 - confidence) / 2
    return {
        "method": "stratified patient bootstrap" if baseline is None else "paired stratified patient bootstrap of differences",
        "confidence_level": confidence, "n_resamples": n_bootstrap, "seed": seed,
        "scope": "single fold and training seed, conditional on fitted model and fixed class counts",
        **{name: {"lower": float(np.quantile(values, tail)), "upper": float(np.quantile(values, 1 - tail))}
           for name, values in samples.items()},
    }
