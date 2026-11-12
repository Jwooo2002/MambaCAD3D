import json
import math
import pickle

import numpy as np
import pytest
import torch

from main_files.core.evaluation import evaluate_held_out
from main_files.core.experiment_manager import run_experiment
from main_files.core.metrics import aggregate_patch_scores, bootstrap_metrics, patient_metrics
from main_files.core.model_trainer import memory_entropy, snapshot_state
from main_files.core.protocol import load_config, make_splits, prepare_manifest, PROJECT_ROOT
from main_files.data.patients import PatientPatchDataset
from main_files.model import MemAE_VSSM3D
from main_files.utils_scan import single_axis_spiral_scan, single_axis_spiral_unscan_final


def synthetic_patients():
    return [{"patient_id": f"normal:{i}", "label": 0} for i in range(20)] + [
        {"patient_id": f"diseased:{i}", "label": 1} for i in range(20)]


def test_frozen_folds_have_no_patient_leakage_and_seed_independent_roles():
    first = make_splits(synthetic_patients(), 4, 3, 42)
    assert first == make_splits(synthetic_patients(), 4, 3, 42)
    all_test_normals = []
    for fold in first:
        train, validation, normal, disease = [set(fold[k]) for k in ("train", "validation", "test_normal", "test_diseased")]
        assert [len(train), len(validation), len(normal), len(disease)] == [12, 3, 5, 20]
        assert len(train | validation | normal | disease) == 40
        all_test_normals.extend(normal)
    assert len(all_test_normals) == len(set(all_test_normals)) == 20


def test_patient_ap_auroc_and_strict_threshold_agree_with_known_ranking():
    metrics = patient_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], threshold=0.8,
                              patient_ids=["a", "b", "c", "d"])
    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["ap"] == pytest.approx(5 / 6)
    assert metrics["prevalence_reference"] == 0.5
    assert (metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]) == (1, 2, 0, 1)


def test_repeated_patient_observations_cannot_be_pooled():
    with pytest.raises(ValueError, match="unique patient"):
        patient_metrics([0, 1], [0.1, 0.9], patient_ids=["same", "same"])


def test_paired_bootstrap_identical_models_have_zero_difference():
    labels, scores, ids = [0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9], ["a", "b", "c", "d"]
    result = bootstrap_metrics(labels, scores, ids, n_bootstrap=30, baseline=scores)
    assert result["auroc"] == {"lower": 0.0, "upper": 0.0}
    assert result["ap"] == {"lower": 0.0, "upper": 0.0}


def test_threshold_is_independent_of_test_scores():
    config = load_config(PROJECT_ROOT / "configs/asoca_revision.json")
    config["bootstrap_samples"] = 10
    def record(pid, label, score):
        return {"patient_id": pid, "label": label, **{k: score for k in ("max", "mean", "top_k", "percentile")}}
    validation = [record("v1", 0, 1), record("v2", 0, 2), record("v3", 0, 3)]
    a = evaluate_held_out(validation, [record("n", 0, 0.5), record("d", 1, 5)], config)
    b = evaluate_held_out(validation, [record("n", 0, 500), record("d", 1, -5)], config)
    assert a["max"]["threshold"] == b["max"]["threshold"] == pytest.approx(2.9)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_held_out(validation, [record("v1", 0, 1), record("d", 1, 5)], config)


def test_aggregation_handles_fewer_patches_than_top_k():
    assert aggregate_patch_scores([1, 3], "top_k", top_k=5) == 2
    assert aggregate_patch_scores([1, 3, 2], "top_k", top_k=2) == 2.5
    with pytest.raises(ValueError):
        aggregate_patch_scores([float("nan")])


def test_memory_entropy_sums_memory_slots_instead_of_tokens():
    attention = torch.full((2, 7, 4), 0.25)
    assert memory_entropy(attention).item() == pytest.approx(math.log(4))


def test_best_checkpoint_does_not_change_with_live_model_weights():
    model = torch.nn.Linear(2, 1)
    saved = snapshot_state(model)
    original = saved["weight"].clone()
    with torch.no_grad():
        model.weight.add_(100)
    assert torch.equal(saved["weight"], original)
    assert not torch.equal(saved["weight"], model.weight)


def test_scan_is_permutation_and_directional_merge_preserves_spatial_positions():
    # The legacy reverse merge sums both directions, so unscan(scan(x)) == 2*x.
    value = torch.arange(2 * 3 * 5).reshape(1, 1, 2, 3, 5).float()
    scanned = single_axis_spiral_scan(value)
    assert torch.equal(scanned[:, 1], scanned[:, 0].flip(-1))
    assert torch.equal(scanned[:, 0].sort().values, value.flatten(2).sort().values)
    assert torch.equal(single_axis_spiral_unscan_final(scanned, value.shape), 2 * value)


def test_actual_mamba_model_forward_backward_on_small_cpu_volume():
    model = MemAE_VSSM3D(in_chans=1, dims=[2, 4], depths=[1, 1], ssm_d_state=2, mem_dim=8)
    value = torch.rand(1, 1, 4, 4, 4)
    output, attention = model(value)
    assert output.shape == value.shape
    loss = (output - value).square().mean() + 0.0002 * memory_entropy(attention)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


class TinyAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 1, 1)
        self.memory = torch.nn.Parameter(torch.zeros(4))

    def forward(self, value):
        attention = self.memory.softmax(0).view(1, 1, 4).expand(len(value), 2, 4)
        return self.conv(value).sigmoid(), attention


def test_end_to_end_training_exports_traceable_scores_and_validation_checkpoint(tmp_path, monkeypatch):
    config = load_config(PROJECT_ROOT / "configs/asoca_revision_legacy_inputs.json")
    for label, name in ((0, "normal"), (1, "diseased")):
        root = tmp_path / name
        root.mkdir()
        for index in range(8 if label == 0 else 4):
            value = np.full((2, 4, 4, 4), -100 + label * 200 + index, dtype=np.float32)
            with (root / f"{index}.pkl").open("wb") as handle:
                pickle.dump(value, handle)
        config[f"{'normal' if label == 0 else 'diseased'}_data_root"] = str(root)
    config.update({"save_path": str(tmp_path / "outputs"), "split_manifest": str(tmp_path / "splits.json"),
                   "k_folds": 2, "validation_normal_count": 1, "training_seeds": [1], "epochs": 2,
                   "num_workers": 0, "expected_patch_shape": [4, 4, 4], "bootstrap_samples": 10})
    manifest = prepare_manifest(config)
    assert manifest == prepare_manifest(config)
    monkeypatch.setattr("main_files.core.model_trainer.build_model", lambda config: TinyAutoencoder())
    output = run_experiment(config, manifest, "cpu", selected_folds=[1], selected_seeds=[1])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    fold_dir = output / "fold_1/seed_1"
    checkpoint = torch.load(fold_dir / "best_model.pt", weights_only=False, map_location="cpu")
    assert set(checkpoint["train_patient_ids"]) == set(manifest["folds"][0]["train"])
    assert set(checkpoint["validation_patient_ids"]) == set(manifest["folds"][0]["validation"])
    import csv
    with (fold_dir / "patient_scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r["patient_id"] for r in rows if r["role"] == "test"} == set(
        manifest["folds"][0]["test_normal"] + manifest["folds"][0]["test_diseased"])
    assert (fold_dir / "patch_scores.csv").is_file()
    manifest["patients"][0]["size_bytes"] += 1
    (tmp_path / "splits.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed"):
        prepare_manifest(config)
