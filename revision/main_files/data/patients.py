"""Read existing local patient pickle files without duplicating the dataset."""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class LocalArrayUnpickler(pickle.Unpickler):
    """Read trusted NumPy-2 preprocessing arrays in the NumPy-1 training env.

    NumPy 2 pickles refer to numpy._core; NumPy 1 has the same array helpers
    under numpy.core. This is an explicit namespace bridge, not a data change.
    """

    def find_class(self, module, name):
        if int(np.__version__.split(".")[0]) < 2 and module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


class PatientPatchDataset(Dataset):
    def __init__(self, patients, hu_window, expected_shape):
        self.records = list(patients)
        self.volumes = []
        self.items = []
        lower, upper = hu_window
        for patient_index, record in enumerate(self.records):
            # Only trusted local preprocessing artifacts are accepted.
            with open(record["path"], "rb") as handle:
                array = np.asarray(LocalArrayUnpickler(handle).load(), dtype=np.float32)
            if array.ndim != 4 or tuple(array.shape[1:]) != tuple(expected_shape) or len(array) == 0:
                raise ValueError(f"Unexpected patch shape for {record['patient_id']}: {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite HU values in {record['patient_id']}")
            array = (np.clip(array, lower, upper) - lower) / (upper - lower)
            volume = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(1)
            self.volumes.append(volume)
            self.items.extend((patient_index, i) for i in range(len(volume)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        patient_index, patch_index = self.items[index]
        return self.volumes[patient_index][patch_index]
