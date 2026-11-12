#!/usr/bin/env python3
"""Summarize scalar diagnostics; never load images or modify checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def gradients(traces):
    names = ('final_recon_conv', 'memory.readout', 'memory.query', 'stem')
    return {name: statistics.mean(next(f for f in row['features'] if f['name'] == name)
                                 ['task_gradient']['rms'] for row in traces) for name in names}


def intervention_summary(rows):
    result = {}
    for name in sorted({r['intervention'] for r in rows}):
        subset = [r for r in rows if r['intervention'] == name]
        baseline = statistics.mean(r['baseline_mse'] for r in subset)
        altered = statistics.mean(r['intervened_mse'] for r in subset)
        result[name] = {
            'patch_count': len(subset), 'baseline_mean_mse': baseline,
            'intervened_mean_mse': altered, 'ratio_of_mean_mse': altered / baseline,
            'relative_mse_change_percent': 100 * (altered / baseline - 1),
            'mean_reconstruction_relative_rms': statistics.mean(
                r['differences']['reconstruction']['relative_rms'] for r in subset),
            'bitwise_equal_reconstructions': sum(r['differences']['reconstruction']['bitwise_equal'] for r in subset),
            'all_noop_checks_exact': all(r['noop_bitwise_equal'] for r in subset),
        }
    return result


def summarize(root):
    manifest = read(root / 'manifest.json')
    old_root = Path(manifest['parent_ssm_repair'])
    inputs, parent, initial = {}, [], []

    def load(path):
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read(path)

    for path in sorted((root / 'parent_checkpoint_audit').glob('S*.json')):
        d = load(path)
        parent.append({'job': d['job'], 'checkpoint_sha256': d['checkpoint_sha256'],
                       'state_and_rng_unchanged': d['state_and_rng_unchanged'],
                       'gradient_rms_mean_3patch': gradients(d['gradient_traces']),
                       'interventions_45patch': intervention_summary(d['recurrence_interventions_45patch'])})
    for job in manifest['jobs']:
        path = root / job['id'] / 'verification_probe.json'
        if path.exists():
            d = load(path)['hybrid']
            initial.append({'job': job['id'], 'gradient_rms_mean_3patch': gradients(d['gradient_traces']),
                            'interventions_3patch': intervention_summary(d['recurrence_interventions'])})
    first = {r['job']: r['gradient_rms_mean_3patch'] for r in initial}
    gain = {name: first['V1_s3_decoder_fanin_seed42'][name] / first['V0_s3_replay_seed42'][name]
            for name in first['V0_s3_replay_seed42']}
    old_smoke = {r['job']: r for r in read(old_root / 'verification.json')['rows']}
    new_smoke = {r['job']: r for r in load(root / 'verification.json')['rows']}
    smoke_match = {str(seed): old_smoke[f'S3_all_ssm_no_decay_seed{seed}']['loss_after_3_updates'] ==
                             new_smoke[f'V0_s3_replay_seed{seed}']['loss_after_3_updates']
                   for seed in manifest['config']['seeds']}
    result = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Normal TRAIN mechanisms; parent fixed epoch200. No disease, test, AUROC/AP or model selection.',
        'aggregation': 'Each intervention uses mean voxel MSE over 45 equal-size patches (3 per 15 patients); '
                       'reported ratio is ratio of means, not mean patch ratios. Gradients average 3 patches only.',
        'parent_checkpoints': parent, 'initialization_probes': initial,
        'initial_fanin_to_base_gradient_rms_ratio_seed42': gain,
        's3_replay_smoke_loss_exact': smoke_match, 'input_sha256': inputs,
        'limits': ['Inference ablation measures current reliance, not benefit after retraining.',
                   'A gradient norm increase is not evidence of improved convergence or anomaly detection.',
                   'Paired seeds and shared patients are not independent clinical samples.',
                   'Frozen corrected model with legacy-derived TRAIN inputs, entropy0 and T_max1000 is a mechanism protocol.'],
    }
    path = root / 'pretraining_diagnostic_summary.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    print(summarize(args.root.resolve()))
