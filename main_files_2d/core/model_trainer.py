"""
Model training utilities
=======================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import wandb
import os
from tqdm import tqdm
from typing import Tuple, List, Dict, Any
import logging
from omegaconf import DictConfig

from data.ImageDataset2D import ImageDataset2D
from model_2d import MemAE_VSSM2D
from core.model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Enhanced model training with comprehensive logging"""

    @staticmethod
    def train_one_fold(fold_idx: int, train_files: List[str], normal_test_files: List[str],
                      diseased_files: List[str], data_roots: Dict[str, str],
                      config: DictConfig, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
        """Train model for one fold with detailed logging"""

        fold_start_time = time.time()

        logger.info("=" * 60)
        logger.info(f"Fold {fold_idx + 1}/{config.k_folds} Training Started")
        logger.info("=" * 60)
        logger.info(f"Train files: {len(train_files)}")
        logger.info(f"Normal test files: {len(normal_test_files)}")
        logger.info(f"Diseased files: {len(diseased_files)}")

        # Data transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x + 300) / 700),
        ])

        # Training dataset
        train_dataset = ImageDataset2D(
            data_root=data_roots['normal'],
            pkl_files=train_files,
            transform=transform
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False
        )

        # Model initialization
        model = MemAE_VSSM2D(
            in_chans=config.in_chans,
            dims=config.dims,
            depths=config.depths,
            ssm_d_state=config.ssm_d_state,
            mem_dim=config.mem_dim
        ).to(device)

        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Optimizer and scheduler
        recon_loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.eta_min
        )

        # Early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0

        # Training history
        train_history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_entropy_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        # Validation dataloader
        normal_test_dataset = ImageDataset2D(
            data_root=data_roots['normal'],
            pkl_files=normal_test_files,
            transform=transform
        )
        normal_test_dataloader = DataLoader(
            normal_test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        # Training loop
        for epoch in range(config.epochs):
            epoch_start_time = time.time()
            model.train()
            total_loss = 0
            total_recon_loss = 0
            total_entropy_loss = 0

            pbar = tqdm(
                train_dataloader,
                desc=f"Fold {fold_idx + 1} Epoch {epoch + 1}/{config.epochs}",
                leave=False
            )

            for i, (item_idx, data) in enumerate(pbar):
                data = data.to(device, non_blocking=True)
                optimizer.zero_grad()

                recon_x, attention = model(data)
                recon_loss = recon_loss_func(recon_x, data)
                entropy_loss = -torch.mean(torch.sum(attention * torch.log(attention + 1e-12), dim=1))

                loss = recon_loss + config.entropy_loss_weight * entropy_loss
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_entropy_loss += entropy_loss.item()

                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.6f}",
                    'Recon': f"{recon_loss.item():.6f}",
                    'Entropy': f"{entropy_loss.item():.6f}",
                    'LR': f"{scheduler.get_last_lr()[0]:.2e}"
                })

            # Calculate epoch averages
            avg_train_loss = total_loss / len(train_dataloader)
            avg_recon_loss = total_recon_loss / len(train_dataloader)
            avg_entropy_loss = total_entropy_loss / len(train_dataloader)
            current_lr = scheduler.get_last_lr()[0]

            scheduler.step()

            # Validation
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for item_idx, data in normal_test_dataloader:
                    data = data.to(device, non_blocking=True)
                    recon_x, _ = model(data)
                    val_loss += recon_loss_func(recon_x, data).item()

            val_loss /= len(normal_test_dataloader)

            # Save training history
            train_history['train_loss'].append(avg_train_loss)
            train_history['train_recon_loss'].append(avg_recon_loss)
            train_history['train_entropy_loss'].append(avg_entropy_loss)
            train_history['val_loss'].append(val_loss)
            train_history['learning_rate'].append(current_lr)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_improve += 1

            # Detailed logging every 10 epochs
            if epoch % 10 == 0 or epoch == config.epochs - 1:
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Fold {fold_idx + 1} Epoch {epoch + 1}/{config.epochs}:")
                logger.info(f"  Train Loss: {avg_train_loss:.6f} (Recon: {avg_recon_loss:.6f}, Entropy: {avg_entropy_loss:.6f})")
                logger.info(f"  Val Loss: {val_loss:.6f}")
                logger.info(f"  Learning Rate: {current_lr:.2e}")
                logger.info(f"  Epoch Time: {epoch_time:.2f}s")
                logger.info(f"  Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch + 1})")

            # W&B logging
            wandb.log({
                f"Fold_{fold_idx + 1}/Train_Loss": avg_train_loss,
                f"Fold_{fold_idx + 1}/Train_Recon_Loss": avg_recon_loss,
                f"Fold_{fold_idx + 1}/Train_Entropy_Loss": avg_entropy_loss,
                f"Fold_{fold_idx + 1}/Val_Loss": val_loss,
                f"Fold_{fold_idx + 1}/Learning_Rate": current_lr,
                f"Fold_{fold_idx + 1}/Epochs_No_Improve": epochs_no_improve,
                "epoch": epoch
            })

            if epochs_no_improve >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1} (Best epoch: {best_epoch + 1})")
                break

        # Restore best model
        model.load_state_dict(best_model_state)

        # Final evaluation - FP-based patient-level only
        from core.fp_based_evaluator import FPBasedEvaluator

        # FP-based patient-level evaluation
        results = FPBasedEvaluator.fp_based_patient_evaluation(
            model, normal_test_files, diseased_files, data_roots, transform, device, config, fold_idx
        )

        # Log FP-constrained results
        FPBasedEvaluator.log_fp_results(results['fp_constrained_results'], fold_idx)

        logger.info(f"Patient-level AUC: {results['patient_auc']:.4f}")
        logger.info(f"Normal patients: {results['normal_patient_count']}, Diseased patients: {results['diseased_patient_count']}")

        # Visualization code removed per user request
        # Only keep essential results tracking

        # Add training information
        results['train_history'] = train_history
        results['best_epoch'] = best_epoch + 1
        results['total_epochs'] = epoch + 1
        results['fold_training_time'] = time.time() - fold_start_time

        logger.info(f"Fold {fold_idx + 1} completed in {results['fold_training_time']:.2f}s")
        logger.info(f"Best model from epoch {best_epoch + 1}")

        return model, results