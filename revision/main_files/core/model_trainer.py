"""Train only on normal training patients; select checkpoints on normal validation."""

import csv
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from main_files.model import MemAE_VSSM3D
from main_files.data.patients import PatientPatchDataset
from .evaluation import score_patients


logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def memory_entropy(attention):
    # Attention is [batch, latent tokens, memory slots]. Sum over slots.
    return -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1).mean()


def snapshot_state(model):
    # state_dict().copy() shares tensor storage and does not freeze a checkpoint.
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_model(config):
    return MemAE_VSSM3D(in_chans=config["in_chans"], dims=config["dims"], depths=config["depths"],
                       ssm_d_state=config["ssm_d_state"], mem_dim=config["mem_dim"],
                       shrink_thres=config["shrink_thres"])


def train_one_fold(train_patients, validation_patients, config, device, seed, output_dir):
    if any(row["label"] != 0 for row in train_patients + validation_patients):
        raise ValueError("Only normal subjects may enter training and validation")
    if {row["patient_id"] for row in train_patients} & {row["patient_id"] for row in validation_patients}:
        raise ValueError("Training and validation patients overlap")
    seed_everything(seed)
    dataset = PatientPatchDataset(train_patients, config["hu_window"], config["expected_patch_shape"])
    validation_dataset = PatientPatchDataset(validation_patients, config["hu_window"], config["expected_patch_shape"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        num_workers=config["num_workers"], pin_memory=device.type == "cuda",
                        worker_init_fn=seed_worker, generator=generator,
                        persistent_workers=config["num_workers"] > 0)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["eta_min"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.csv"
    best_loss, best_epoch, stale, best_state = float("inf"), None, 0, None
    started = time.perf_counter()
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_mse", "train_memory_entropy", "zero_memory_row_fraction",
                                                   "validation_patient_mean_mse", "lr", "seconds"])
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            epoch_start = time.perf_counter()
            model.train()
            total_loss, total_mse, total_entropy, count = 0.0, 0.0, 0.0, 0
            zero_memory_rows, memory_rows = 0, 0
            lr = optimizer.param_groups[0]["lr"]
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                reconstruction, attention = model(batch)
                if reconstruction.shape != batch.shape:
                    raise ValueError("Model output does not match input patch shape")
                mse = (reconstruction - batch).square().mean()
                entropy = memory_entropy(attention)
                zero_memory_rows += int((attention.detach().sum(-1) == 0).sum())
                memory_rows += attention.numel() // attention.shape[-1]
                loss = mse + config["entropy_loss_weight"] * entropy
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                size = len(batch)
                total_loss += loss.item() * size
                total_mse += mse.item() * size
                total_entropy += entropy.item() * size
                count += size
            scheduler.step()
            validation_records, _ = score_patients(model, validation_dataset, config, device)
            val_loss = float(np.mean([row["mean_patch_mse"] for row in validation_records]))
            if not np.isfinite(val_loss):
                raise ValueError("Non-finite validation loss")
            if val_loss < best_loss:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = snapshot_state(model)
                torch.save({"model_state_dict": best_state, "epoch": best_epoch, "config": config,
                            "training_seed": seed, "validation_patient_mean_mse": best_loss,
                            "train_patient_ids": [row["patient_id"] for row in train_patients],
                            "validation_patient_ids": [row["patient_id"] for row in validation_patients]},
                           output_dir / "best_model.pt")
            else:
                stale += 1
            record = {"epoch": epoch, "train_loss": total_loss / count, "train_mse": total_mse / count,
                      "train_memory_entropy": total_entropy / count, "validation_patient_mean_mse": val_loss,
                      "zero_memory_row_fraction": zero_memory_rows / memory_rows,
                      "lr": lr, "seconds": time.perf_counter() - epoch_start}
            writer.writerow(record)
            handle.flush()
            logger.info("seed=%s epoch=%d train_mse=%.6f val_mse=%.6f best_epoch=%s time=%.1fs",
                        seed, epoch, record["train_mse"], val_loss, best_epoch, record["seconds"])
            if stale >= config["early_stopping_patience"]:
                break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    validation_records, validation_patches = score_patients(model, validation_dataset, config, device)
    return model, validation_records, validation_patches, {
        "best_epoch": best_epoch, "epochs_completed": epoch, "training_seed": seed,
        "validation_patient_mean_mse": best_loss, "training_seconds": time.perf_counter() - started,
        "train_patient_count": len(train_patients), "validation_patient_count": len(validation_patients),
        "train_patch_count": len(dataset), "validation_patch_count": len(validation_dataset),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_version": config.get("model_version", "unspecified"),
        "early_stopping_source": "normal validation patient mean MSE only",
    }
