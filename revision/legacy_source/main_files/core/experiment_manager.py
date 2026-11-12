"""
Main experiment orchestrator
===========================
"""

import os
import json
import yaml
import torch
import wandb
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from typing import Dict, List, Any
import logging
from omegaconf import OmegaConf, DictConfig

from core.model_trainer import ModelTrainer
from utils.statistical_analyzer import StatisticalAnalyzer
from utils.visualizer import Visualizer
from utils.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Main experiment orchestrator"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.device = self._setup_device()
        self.save_paths = self._setup_directories()

    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device

    def _setup_directories(self) -> Dict[str, str]:
        """Setup experiment directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(self.config.save_path) / f"{self.config.run_name}_{timestamp}"

        paths = {
            'main': str(save_dir),
            'plots': str(save_dir / "plots"),
            'models': str(save_dir / "models"),
            'logs': str(save_dir / "logs"),
            'configs': str(save_dir / "configs")
        }

        for path in paths.values():
            os.makedirs(path, exist_ok=True)

        logger.info(f"Experiment directory: {paths['main']}")
        return paths

    def run_experiment(self):
        """Run complete k-fold cross validation experiment"""

        # Initialize W&B
        wandb.init(
            project="MemAE-VSSM3D-Production-KFold",
            config=OmegaConf.to_container(self.config, resolve=True),
            name=self.config.run_name,
            dir=self.save_paths['logs']
        )

        # Save configuration
        self._save_configuration()

        # Prepare data
        data_info = self._prepare_data()

        # Run k-fold cross validation
        fold_results = self._run_kfold_cv(data_info)

        # Analyze results
        self._analyze_and_save_results(fold_results)

        # Create comprehensive patient overview
        from utils.results_saver import create_comprehensive_patient_overview
        from utils.anomaly_patch_saver import AnomalyPatchSaver

        create_comprehensive_patient_overview("./fold_results")

        # Create anomaly patch overview
        AnomalyPatchSaver.create_anomaly_overview("./fold_results", self.config.k_folds)

        # Cleanup
        wandb.finish()
        logger.info("Experiment completed successfully! 🎉")
        logger.info("Volume-wise results and patient plots saved in ./fold_results/")
        logger.info("Patient overview saved in ./fold_results/patient_overview/")
        logger.info("Anomalous patches saved in ./fold_results/fold_X_anomalous_patches/")
        logger.info("Anomaly overview saved in ./fold_results/anomaly_overview/")
        logger.info("Extreme score visualizations saved in ./fold_results/fold_X_extreme_scores/")

    def _save_configuration(self):
        """Save experiment configuration"""
        config_dict = OmegaConf.to_container(self.config, resolve=True)

        # Save as YAML
        with open(os.path.join(self.save_paths['configs'], 'experiment_config.yaml'), 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        # Save as JSON for compatibility
        with open(os.path.join(self.save_paths['configs'], 'experiment_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info("Configuration saved")

    def _prepare_data(self) -> Dict[str, Any]:
        """Prepare and validate data"""
        data_roots = {
            'normal': self.config.normal_data_root,
            'diseased': self.config.diseased_data_root
        }

        # Validate data directories
        for key, path in data_roots.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{key.capitalize()} data directory not found: {path}")

        # Get file lists
        normal_files = sorted([f for f in os.listdir(data_roots['normal']) if f.endswith('.pkl')])
        diseased_files = sorted([f for f in os.listdir(data_roots['diseased']) if f.endswith('.pkl')])

        logger.info(f"Total Normal files: {len(normal_files)}")
        logger.info(f"Total Diseased files: {len(diseased_files)}")

        if len(normal_files) < self.config.k_folds:
            raise ValueError(f"Normal files ({len(normal_files)}) should be >= k_folds ({self.config.k_folds})")

        return {
            'data_roots': data_roots,
            'normal_files': normal_files,
            'diseased_files': diseased_files
        }

    def _run_kfold_cv(self, data_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute k-fold cross validation"""

        # Setup k-fold splitter
        kfold = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=self.config.random_seed)
        normal_indices = np.arange(len(data_info['normal_files']))

        fold_results = []

        logger.info("=" * 80)
        logger.info(f"STARTING {self.config.k_folds}-FOLD CROSS VALIDATION")
        logger.info("=" * 80)

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(normal_indices)):
            train_files = [data_info['normal_files'][i] for i in train_idx]
            normal_test_files = [data_info['normal_files'][i] for i in test_idx]

            # Train model
            model, results = ModelTrainer.train_one_fold(
                fold_idx, train_files, normal_test_files, data_info['diseased_files'],
                data_info['data_roots'], self.config, self.device
            )

            fold_results.append(results)

            # Save model
            model_save_path = os.path.join(self.save_paths['models'], f'fold_{fold_idx + 1}_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold_idx': fold_idx,
                'config': OmegaConf.to_container(self.config, resolve=True),
                'results': {k: v for k, v in results.items() if k not in ['normal_scores', 'diseased_scores', 'all_scores', 'all_labels', 'predictions']}
            }, model_save_path)

            # Log fold results
            self._log_fold_results(fold_idx, results)

        return fold_results

    def _log_fold_results(self, fold_idx: int, results: Dict[str, Any]):
        """Log results for individual fold - FP-based patient level only"""

        # Patient-level AUC
        patient_auc = results['patient_auc']
        wandb.log({f"Fold_{fold_idx + 1}/Patient_AUC": patient_auc})

        # FP-constrained results
        fp_metrics = {}
        for fp_key, fp_result in results['fp_constrained_results'].items():
            fp_val = fp_key.split('_')[1]
            fp_metrics.update({
                f"Fold_{fold_idx + 1}/FP{fp_val}_Accuracy": fp_result['accuracy'],
                f"Fold_{fold_idx + 1}/FP{fp_val}_Precision": fp_result['precision'],
                f"Fold_{fold_idx + 1}/FP{fp_val}_Recall": fp_result['recall'],
                f"Fold_{fold_idx + 1}/FP{fp_val}_Specificity": fp_result['specificity'],
                f"Fold_{fold_idx + 1}/FP{fp_val}_F1": fp_result['f1_score'],
                f"Fold_{fold_idx + 1}/FP{fp_val}_Threshold": fp_result['threshold'],
                f"Fold_{fold_idx + 1}/FP{fp_val}_TP": fp_result['tp'],
                f"Fold_{fold_idx + 1}/FP{fp_val}_FN": fp_result['fn'],
            })

        # Common metrics
        common_metrics = {
            f"Fold_{fold_idx + 1}/Patient_Count_Normal": results['normal_patient_count'],
            f"Fold_{fold_idx + 1}/Patient_Count_Diseased": results['diseased_patient_count'],
            f"Fold_{fold_idx + 1}/Training_Time": results.get('fold_training_time', 0),
            f"Fold_{fold_idx + 1}/Best_Epoch": results.get('best_epoch', 0),
        }

        # Combine all metrics
        fold_metrics = {**fp_metrics, **common_metrics}

        wandb.log(fold_metrics)

        logger.info("=" * 60)
        logger.info(f"FOLD {fold_idx + 1} RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Patient-level AUC: {patient_auc:.4f}")
        logger.info(f"Normal Patients: {results['normal_patient_count']}")
        logger.info(f"Diseased Patients: {results['diseased_patient_count']}")
        logger.info("")
        logger.info("FP-Constrained Results Summary:")

        # Best performing FP constraints
        for fp_key in ['FP_0', 'FP_1', 'FP_2']:
            if fp_key in results['fp_constrained_results']:
                fp_result = results['fp_constrained_results'][fp_key]
                fp_val = fp_key.split('_')[1]
                logger.info(f"  FP={fp_val}: Acc={fp_result['accuracy']:.3f}, Rec={fp_result['recall']:.3f}, "
                           f"Spec={fp_result['specificity']:.3f}, F1={fp_result['f1_score']:.3f}")

        logger.info("")
        logger.info(f"Training Time: {results.get('fold_training_time', 0):.1f}s")
        logger.info(f"Best Epoch: {results.get('best_epoch', 0)}/{results.get('total_epochs', 0)}")

    def _analyze_and_save_results(self, fold_results: List[Dict[str, Any]]):
        """Analyze and save comprehensive results"""

        # Statistical analysis
        statistical_summary = StatisticalAnalyzer.calculate_statistical_summary(fold_results, self.config)

        # Generate visualizations
        Visualizer.plot_comprehensive_results(fold_results, self.config, self.save_paths['plots'])

        # Generate reports (if report generator supports new format)
        try:
            ReportGenerator.generate_detailed_report(
                fold_results, statistical_summary, self.config, self.save_paths['main']
            )
        except Exception as e:
            logger.warning(f"Report generation skipped: {e}")

        # Save complete results
        self._save_complete_results(fold_results, statistical_summary)

        # Log final metrics to W&B
        self._log_final_metrics(statistical_summary)

        # Upload plots to W&B
        self._upload_plots_to_wandb()

    def _save_complete_results(self, fold_results: List[Dict[str, Any]],
                               statistical_summary: Dict[str, Any]):
        """Save complete experiment results"""

        # Prepare results for JSON serialization
        results_summary = {
            'experiment_config': OmegaConf.to_container(self.config, resolve=True),
            'statistical_summary': statistical_summary,
            'individual_fold_results': [],
            'total_experiment_time': sum([r.get('fold_training_time', 0) for r in fold_results])
        }

        # Process each fold result for JSON compatibility
        for i, result in enumerate(fold_results):
            fold_summary = {
                'fold_id': i + 1,
                'patient_metrics': {
                    'patient_auc': result.get('patient_auc', 0.0),
                    'normal_patient_count': result.get('normal_patient_count', 0),
                    'diseased_patient_count': result.get('diseased_patient_count', 0),
                },
                'fp_constrained_results': result.get('fp_constrained_results', {}),
                'training_info': {
                    'best_epoch': result.get('best_epoch', 0),
                    'total_epochs': result.get('total_epochs', 0),
                    'training_time': result.get('fold_training_time', 0)
                }
            }

            # Add volume-level metrics if they exist (for backward compatibility)
            if 'auc' in result:
                fold_summary['volume_metrics'] = {
                    'auc': result.get('auc', 0.0),
                    'accuracy': result.get('accuracy', 0.0),
                    'precision': result.get('precision', 0.0),
                    'recall': result.get('recall', 0.0),
                    'f1_score': result.get('f1_score', 0.0),
                    'sensitivity': result.get('sensitivity', 0.0),
                    'specificity': result.get('specificity', 0.0),
                    'threshold': result.get('threshold', 0.0),
                }

                fold_summary['confusion_matrix'] = {
                    'true_positives': result.get('true_positives', 0),
                    'false_positives': result.get('false_positives', 0),
                    'true_negatives': result.get('true_negatives', 0),
                    'false_negatives': result.get('false_negatives', 0),
                }

                fold_summary['error_statistics'] = {
                    'normal_mean_error': result.get('normal_mean_error', 0.0),
                    'normal_std_error': result.get('normal_std_error', 0.0),
                    'diseased_mean_error': result.get('diseased_mean_error', 0.0),
                    'diseased_std_error': result.get('diseased_std_error', 0.0),
                }

                fold_summary['percentiles'] = {
                    'normal': result.get('normal_percentiles', {}),
                    'diseased': result.get('diseased_percentiles', {})
                }

            results_summary['individual_fold_results'].append(fold_summary)

        # Save main results
        with open(os.path.join(self.save_paths['main'], 'complete_results.json'), 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

        # Save raw scores separately if they exist
        raw_scores = {}
        for i, result in enumerate(fold_results):
            fold_scores = {}

            if 'normal_scores' in result:
                fold_scores['normal_scores'] = result['normal_scores'].tolist()
            if 'diseased_scores' in result:
                fold_scores['diseased_scores'] = result['diseased_scores'].tolist()
            if 'all_scores' in result:
                fold_scores['all_scores'] = result['all_scores'].tolist()
            if 'all_labels' in result:
                fold_scores['all_labels'] = result['all_labels'].tolist()
            if 'predictions' in result:
                fold_scores['predictions'] = result['predictions'].tolist()

            if fold_scores:  # Only add if there are scores
                raw_scores[f'fold_{i + 1}'] = fold_scores

        if raw_scores:  # Only save if there's data
            with open(os.path.join(self.save_paths['main'], 'raw_scores.json'), 'w') as f:
                json.dump(raw_scores, f, indent=2)

        logger.info("Complete results saved")

    def _log_final_metrics(self, statistical_summary: Dict[str, Any]):
        """Log final summary metrics to W&B - supports both patient and volume level"""

        final_metrics = {}

        # Patient-level AUC (primary metric)
        if 'patient_auc' in statistical_summary:
            stats = statistical_summary['patient_auc']
            final_metrics["Final/Patient_AUC_Mean"] = stats['mean']
            final_metrics["Final/Patient_AUC_Std"] = stats['std']
            final_metrics["Final/Patient_AUC_CI_Lower"] = stats['ci_lower']
            final_metrics["Final/Patient_AUC_CI_Upper"] = stats['ci_upper']

            logger.info("=" * 60)
            logger.info("FINAL RESULTS SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Patient-level AUC: {stats['mean']:.4f} ± {stats['std']:.4f} "
                        f"(95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])")

        # FP-constrained results
        if 'fp_constrained_summary' in statistical_summary:
            logger.info("")
            logger.info("FP-Constrained Results (Mean ± Std):")

            for fp_key, fp_stats in statistical_summary['fp_constrained_summary'].items():
                fp_val = fp_key.split('_')[1]

                for metric_name, metric_stats in fp_stats.items():
                    metric_key = f"Final/{fp_key}_{metric_name}"
                    final_metrics[f"{metric_key}_Mean"] = metric_stats['mean']
                    final_metrics[f"{metric_key}_Std"] = metric_stats['std']

                # Log summary for this FP constraint
                if 'accuracy' in fp_stats and 'recall' in fp_stats:
                    logger.info(
                        f"  FP={fp_val}: Acc={fp_stats['accuracy']['mean']:.3f}±{fp_stats['accuracy']['std']:.3f}, "
                        f"Rec={fp_stats['recall']['mean']:.3f}±{fp_stats['recall']['std']:.3f}, "
                        f"Spec={fp_stats['specificity']['mean']:.3f}±{fp_stats['specificity']['std']:.3f}")

        # Volume-level metrics (if available)
        volume_metrics_display = {
            'auc': 'Volume_AUC',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1-Score',
            'specificity': 'Specificity'
        }

        volume_metrics_found = False
        for metric_key, metric_name in volume_metrics_display.items():
            if metric_key in statistical_summary:
                if not volume_metrics_found:
                    logger.info("")
                    logger.info("Volume-level Metrics:")
                    volume_metrics_found = True

                stats = statistical_summary[metric_key]
                final_metrics[f"Final/{metric_name}_Mean"] = stats['mean']
                final_metrics[f"Final/{metric_name}_Std"] = stats['std']
                final_metrics[f"Final/{metric_name}_CI_Lower"] = stats['ci_lower']
                final_metrics[f"Final/{metric_name}_CI_Upper"] = stats['ci_upper']

                logger.info(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                            f"(95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])")

        logger.info("=" * 60)

        # Log all metrics to W&B
        if final_metrics:
            wandb.log(final_metrics)

    def _upload_plots_to_wandb(self):
        """Upload generated plots to W&B"""

        plot_files = {
            "Results/Comprehensive_Metrics": os.path.join(self.save_paths['plots'], 'comprehensive_metrics.png'),
            "Results/Confusion_Matrices": os.path.join(self.save_paths['plots'], 'confusion_matrices.png'),
            "Results/Detailed_Analysis": os.path.join(self.save_paths['plots'], 'detailed_analysis.png'),
            "Results/ROC_Curves": os.path.join(self.save_paths['plots'], 'roc_curves.png')
        }

        wandb_images = {}
        for name, path in plot_files.items():
            if os.path.exists(path):
                wandb_images[name] = wandb.Image(path)

        # Upload individual ROC curves
        roc_dir = os.path.join(self.save_paths['plots'], 'individual_roc_curves')
        if os.path.exists(roc_dir):
            for roc_file in sorted(os.listdir(roc_dir)):
                if roc_file.endswith('.png'):
                    fold_name = roc_file.replace('_roc_curve.png', '').replace('_', ' ').title()
                    wandb_images[f"ROC_Curves/{fold_name}"] = wandb.Image(
                        os.path.join(roc_dir, roc_file)
                    )

        if wandb_images:
            wandb.log(wandb_images)