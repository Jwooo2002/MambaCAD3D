"""Frozen patient splits. Test subjects never enter training or validation."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    with Path(path).open() as handle:
        config = json.load(handle)
    for key in ("normal_data_root", "diseased_data_root", "save_path", "split_manifest", "preprocessing_config", "preprocessing_manifest", "external_patch_index"):
        if key not in config:
            continue
        path = Path(config[key]).expanduser()
        config[key] = str((PROJECT_ROOT / path).resolve() if not path.is_absolute() else path)
    if not config["dims"] or len(config["dims"]) != len(config["depths"]):
        raise ValueError("dims and depths must have equal, nonzero lengths")
    if any(int(x) <= 0 for key in ("dims", "depths", "expected_patch_shape") for x in config[key]):
        raise ValueError("Architecture and patch dimensions must be positive")
    factor = 2 ** (len(config["dims"]) - 1)
    if len(config["expected_patch_shape"]) != 3 or any(x % factor for x in config["expected_patch_shape"]):
        raise ValueError("All three patch dimensions must be divisible by encoder downsampling")
    for key in ("epochs", "batch_size", "early_stopping_patience", "top_k", "bootstrap_samples"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["primary_aggregation"] not in ("max", "mean", "top_k", "percentile"):
        raise ValueError("Unknown primary_aggregation")
    if config["patch_score_reduction"] not in ("sum", "mean"):
        raise ValueError("patch_score_reduction must be sum or mean")
    if not 0 < config["confidence_level"] < 1:
        raise ValueError("confidence_level must be between zero and one")
    for key in ("aggregation_percentile", "threshold_normal_percentile"):
        if not 0 <= config[key] <= 100:
            raise ValueError(f"{key} must be in [0, 100]")
    if len(config["hu_window"]) != 2 or config["hu_window"][0] >= config["hu_window"][1]:
        raise ValueError("hu_window must contain increasing lower/upper values")
    if not config["training_seeds"] or len(set(config["training_seeds"])) != len(config["training_seeds"]):
        raise ValueError("training_seeds must be nonempty and unique")
    return config


def discover_patients(config):
    patients = []
    for label, key in ((0, "normal_data_root"), (1, "diseased_data_root")):
        root = Path(config[key])
        files = sorted(root.glob("*.pkl"))
        if not files:
            raise ValueError(f"No patient pickle files in {root}")
        for path in files:
            stat = path.stat()
            record = {
                "patient_id": f"{'normal' if label == 0 else 'diseased'}:{path.stem}",
                "label": label,
                "path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            if path.with_suffix(".json").is_file():
                metadata = json.loads(path.with_suffix(".json").read_text())
                if metadata["label"] != label or metadata["output_bytes"] != stat.st_size:
                    raise ValueError(f"Patch metadata differs: {path}")
                record.update({"source_patient_id": metadata["patient_id"],
                               "patch_sha256": metadata["output_sha256"],
                               "preprocessing_fingerprint": metadata["input_fingerprint"],
                               "geometry_path": metadata["geometry_path"]})
            patients.append(record)
    if len({p["path"] for p in patients}) != len(patients):
        raise ValueError("Normal and diseased sources overlap")
    return patients


def make_splits(patients, k_folds, validation_count, split_seed):
    normal = sorted(p["patient_id"] for p in patients if p["label"] == 0)
    diseased = sorted(p["patient_id"] for p in patients if p["label"] == 1)
    if len(set(normal + diseased)) != len(patients) or not diseased:
        raise ValueError("Patient IDs must be unique and both classes must be present")
    if not 2 <= k_folds <= len(normal) or validation_count < 1:
        raise ValueError("Invalid fold count or validation size")
    splitter = KFold(n_splits=k_folds, shuffle=True, random_state=split_seed)
    folds = []
    for fold, (development, test) in enumerate(splitter.split(normal), 1):
        if len(development) <= validation_count:
            raise ValueError("Not enough normal subjects for separate training and validation")
        development = np.random.default_rng(split_seed + fold).permutation(development)
        folds.append({
            "fold": fold,
            "train": sorted(normal[i] for i in development[validation_count:]),
            "validation": sorted(normal[i] for i in development[:validation_count]),
            "test_normal": sorted(normal[i] for i in test),
            "test_diseased": diseased,
        })
    for fold in folds:
        groups = [set(fold[key]) for key in ("train", "validation", "test_normal", "test_diseased")]
        if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
            raise ValueError("A patient appears in multiple roles in one fold")
    return folds


def prepare_manifest(config):
    patients = discover_patients(config)
    fingerprint = hashlib.sha256(json.dumps(patients, sort_keys=True).encode()).hexdigest()
    protocol = {key: config[key] for key in ("k_folds", "validation_normal_count", "split_seed")}
    folds = make_splits(patients, config["k_folds"], config["validation_normal_count"], config["split_seed"])
    manifest = {
        "schema_version": 1,
        "fingerprint_kind": "patient paths, sizes and modification times; not a content hash",
        "data_fingerprint": fingerprint,
        "protocol": protocol,
        "patients": patients,
        "folds": folds,
        "diseased_reuse": "The same diseased patients are evaluated in every fold; observations are correlated.",
    }
    if "preprocessing_config" in config:
        manifest["preprocessing_protocol"] = json.loads(Path(config["preprocessing_config"]).read_text())
    path = Path(config["split_manifest"])
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            raise ValueError(f"Frozen split or source data changed: {path}. Use a new manifest path.")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest
