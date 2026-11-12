import csv
import hashlib
import pickle

import numpy as np
import pytest
import torch

from main_files.core.external import load_external_records, score_external
from main_files.core.protocol import load_config, PROJECT_ROOT


def cohort(tmp_path):
    rows = []
    for i, label in enumerate([0, 0, 1, 1]):
        path = tmp_path / f"{i}.pkl"
        with path.open("wb") as handle:
            pickle.dump(np.full((2, 4, 4, 4), -300 + label * 700, np.float32), handle)
        rows.append({"patient_id": f"External:{i}", "label": label, "revision_role": "external_test", "path": str(path)})
    index = tmp_path / "external.csv"
    with index.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return index


def test_external_index_frozen_hash_count_and_roles(tmp_path):
    index = cohort(tmp_path)
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    assert len(load_external_records(index, digest, 4)) == 4
    with pytest.raises(ValueError, match="changed"):
        load_external_records(index, "wrong", 4)
    with pytest.raises(ValueError, match="incomplete"):
        load_external_records(index, digest, 800)
    index.write_text(index.read_text().replace("external_test", "train", 1))
    with pytest.raises(ValueError, match="role"):
        load_external_records(index)


def test_external_stream_exports_all_patients_and_uses_internal_threshold(tmp_path):
    class ZeroModel(torch.nn.Module):
        def forward(self, value):
            return torch.zeros_like(value), torch.ones(len(value), 1, 1)

    config = load_config(PROJECT_ROOT / "configs/asoca_revision.json")
    config.update({"expected_patch_shape": [4, 4, 4], "bootstrap_samples": 10})
    validation = [{"patient_id": f"ASOCA:{i}", "label": 0, **{k: float(i) for k in ("max", "mean", "top_k", "percentile")}}
                  for i in (1, 2, 3)]
    result = score_external(ZeroModel(), load_external_records(cohort(tmp_path)), validation, config,
                            torch.device("cpu"), tmp_path, {"fold": 1, "seed": 1})
    metrics = result["aggregation_results"]["max"]
    assert (result["patient_count"], result["normal_count"], result["diseased_count"]) == (4, 2, 2)
    assert metrics["threshold"] == pytest.approx(2.9)
    assert metrics["auroc"] == metrics["ap"] == 1
    with (tmp_path / "external_patient_scores_raw.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and {row["role"] for row in rows} == {"external_test"}
    assert len({row["patient_id"] for row in rows}) == 4
