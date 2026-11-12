"""Summarize fixed epoch-200 completed jobs without touching the running queue."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root/'manifest.json').read_text())
    rows, timeline, evidence, initial_gradients = [], [], {}, []
    for job in manifest['jobs']:
        if job['status'] != 'complete':
            continue
        directory = root/job['id']
        final_path = directory/'probe_epoch_200.json'
        full_path = directory/'full_normal_training_epoch_200.json'
        final = json.loads(final_path.read_text())
        full = json.loads(full_path.read_text())
        ssm = final['ssm']
        assert final['epoch'] == full['epoch'] == 200
        assert full['patches'] == 956 and len(ssm['functional']) == 18
        for path in (final_path, full_path):
            evidence[str(path)] = digest(path)
        dt = [p for p in ssm['parameters'] if p['name'].endswith('dt_proj.weight')]
        xp = [p for p in ssm['parameters'] if p['name'].endswith('x_proj.weight')]
        ratios = [f['raw_recurrent_over_Du_rms'] for f in ssm['functional']]
        rows.append({'job': job['id'], 'variant': job['variant'], 'seed': job['seed'],
            'epoch': 200, 'full_train_mse': full['mse'],
            'query_rank99': full['feature_covariance']['query']['rank99'],
            'readout_rank99': full['feature_covariance']['readout']['rank99'],
            'memory_rank99': full['memory_rank99'],
            'largest_duplicate_neighbourhood': full['largest_duplicate_neighbourhood_l2_1e_minus6'],
            'global_effective_slots': full['global_effective_slots'],
            'local_effective_slots': full['effective_slots_exp_mean_query_entropy'],
            'dt_collapsed_blocks': sum(p['nonzero_subnormal'] == p['numel'] for p in dt),
            'dt_l2_range': [min(p['l2_float64'] for p in dt), max(p['l2_float64'] for p in dt)],
            'xproj_l2_range': [min(p['l2_float64'] for p in xp), max(p['l2_float64'] for p in xp)],
            'bias_only_observations': sum(f['delta_equals_bias_only'] for f in ssm['functional']),
            'zero_raw_recurrent_observations': sum(f['raw_recurrent_rms'] == 0. for f in ssm['functional']),
            'raw_recurrent_over_Du_min_median_max': [min(ratios), statistics.median(ratios), max(ratios)],
            'delta_input_contribution_maxabs': max(f['delta_input_contribution_maxabs'] for f in ssm['functional']),
            'dt_task_gradient_l2_max': max(p['task_grad_l2'] for p in dt)})
        for path in sorted(directory.glob('probe_epoch_*.json')):
            d = json.loads(path.read_text()); s = d['ssm']
            dt = [p for p in s['parameters'] if p['name'].endswith('dt_proj.weight')]
            xp = [p for p in s['parameters'] if p['name'].endswith('x_proj.weight')]
            timeline.append({'job': job['id'], 'epoch': d['epoch'],
                'bias_only_observations': sum(f['delta_equals_bias_only'] for f in s['functional']),
                'dt_l2_min_max': [min(p['l2_float64'] for p in dt), max(p['l2_float64'] for p in dt)],
                'xproj_l2_min_max': [min(p['l2_float64'] for p in xp), max(p['l2_float64'] for p in xp)]})
            if job['variant'] == 'S0_control' and d['epoch'] in (0, 1, 3, 5, 10, 20):
                for kind, parameters in [('dt_weight', dt), ('xproj_weight', xp)]:
                    initial_gradients.append({'job': job['id'], 'epoch': d['epoch'], 'kind': kind,
                        'clip_scale': s['clip_scale'], 'parameters': [{
                            'name': p['name'], 'task_grad_l2': p['task_grad_l2'],
                            'decay_l2': p['decay_times_parameter_l2'],
                            'decay_to_clipped_task_gradient_ratio': p['decay_times_parameter_l2']/(p['task_grad_l2']*s['clip_scale']) if p['task_grad_l2'] else None
                        } for p in parameters]})
    baselines = {r['seed']: r for r in rows if r['variant'] == 'S0_control'}
    for row in rows:
        row['train_mse_ratio_to_S0_same_seed'] = row['full_train_mse']/baselines[row['seed']]['full_train_mse']
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
        'queue_status_at_read': manifest['status'],
        'job_counts_at_read': {s: sum(j['status'] == s for j in manifest['jobs']) for s in ['complete', 'running', 'queued', 'failed']},
        'scope': 'Partial, fixed epoch200 complete jobs only. Full956 normal TRAIN reconstruction/ranks; SSM function on 3 fixed TRAIN patches x6blocks. No validation/disease/test evaluation.',
        'rows': rows, 'timeline': timeline, 'early_gradient_probes': initial_gradients,
        'input_evidence_sha256': evidence,
        'interpretation_limits': ['S2 jointly removes x_proj/dt_proj/A_log/D decay; it does not isolate the minimum required subset.',
          'Raw recurrent/Du ratio is before normalization/gating, not final predictive importance.',
          'Preserved dynamics and lower normal TRAIN MSE do not establish disease discrimination or held-out generalization.',
          'S4 changes decay across non-memory parameters; Adam/AdamW with the same decay number are not matched regularization strengths.',
          'All arms use the corrected model but frozen legacy TRAIN inputs, entropy0 and memory decay0. This is not final original-method k-fold evaluation.']}
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['| 조건 | seed | 정상 TRAIN MSE | S0 대비 | Δ bias-only /18 | 재귀/Du 중앙값 | query rank99 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['variant']} | {row['seed']} | {row['full_train_mse']:.8f} | {row['train_mse_ratio_to_S0_same_seed']:.3f}× | {row['bias_only_observations']} | {row['raw_recurrent_over_Du_min_median_max'][1]:.4g} | {row['query_rank99']} |")
    (output/'table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output': str(output), 'counts': result['job_counts_at_read'], 'rows': len(rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.output.resolve())
