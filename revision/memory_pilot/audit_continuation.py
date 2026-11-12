#!/usr/bin/env python3
"""Read-only audit of completed continuation checkpoints and stored TRAIN probes."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(1)
    manifest = read(root/'manifest.json')
    assert manifest['status'] == 'complete'
    for name, expected in read(root/'source_hashes.json').items():
        assert digest(root/name) == expected, name
    summary = {'status': 'complete', 'audited_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'Epoch-100 checkpoints and stored all-956-normal-TRAIN diagnostics; no new forward evaluation, disease/test access or training.',
               'source_hashes_verified': True, 'started_utc': manifest['started_utc'],
               'finished_utc': manifest['finished_utc'], 'rows': [], 'paired': []}
    summary['wall_minutes'] = (datetime.fromisoformat(manifest['finished_utc'])-datetime.fromisoformat(manifest['started_utc'])).total_seconds()/60
    for job in manifest['jobs']:
        directory = root/job['id']
        state = read(directory/'run_state.json')
        history = read(directory/'history.json')
        assert job['status'] == 'complete' and job['returncode'] == 0
        assert state['completed_epochs'] == 100 and state['total_optimizer_steps'] == 47800
        assert [h['epoch'] for h in history] == list(range(1, 101))
        assert all(h['patches'] == 956 and h['optimizer_steps'] == 478 for h in history)
        assert all(math.isfinite(h[k]) for h in history for k in ('mse', 'loss', 'legacy_entropy'))
        ck = torch.load(directory/'epoch_100.pt', map_location='cpu', weights_only=False)
        assert ck['epoch'] == 100 and ck['seed'] == job['seed'] and ck['variant'] == job['variant']
        assert ck['parent_checkpoint_sha256'] == job['parent_checkpoint_sha256']
        assert ck['initial_state_hash'] == manifest['initial_state_hashes'][str(job['seed'])]
        assert ck['batch_order_hash'] == manifest['batch_order_hashes'][str(job['seed'])]
        assert ck['scheduler_state_dict']['T_max'] == 1000 and ck['scheduler_state_dict']['last_epoch'] == 100
        assert {int(s['step']) for s in ck['optimizer_state_dict']['state'].values()} == {47800}
        assert ck['config']['mem_dim'] == 300 and ck['intervention'] == manifest['variants'][job['variant']]
        assert set(ck['rng_state']) == {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'data_loader'}
        memory = ck['model_state_dict']['mem_rep.memory'].double()
        assert memory.shape == (300, 128)
        distances = torch.cdist(memory, memory, compute_mode='donot_use_mm_for_euclid_dist')
        largest = int((distances <= 1e-6).sum(1).max())
        full = read(directory/'full_normal_training_epoch_100.json')
        assert full['epoch'] == 100 and full['patches'] == 956 and len(full['patients']) == 15
        assert largest == full['largest_duplicate_neighbourhood_l2_1e_minus6']
        assert sum(p['patches'] for p in full['patients']) == 956
        weighted = sum(p['mse']*p['patches'] for p in full['patients'])/956
        assert math.isclose(weighted, full['mse'], rel_tol=1e-12, abs_tol=1e-15)
        assert math.isclose(math.exp(full['mean_query_entropy']), full['effective_slots_exp_mean_query_entropy'], rel_tol=1e-12)
        assert math.isclose(sum(full['global_slot_mass']), 1., abs_tol=1e-6)
        trajectory = []
        for epoch in manifest['config']['probe_epochs']:
            q = read(directory/f'probe_epoch_{epoch:03d}.json')
            assert q['epoch'] == epoch and q['patches'] == 45
            trajectory.append({'epoch': epoch, 'mse': q['mse'],
                               'effective_slots': q['effective_slots_exp_mean_query_entropy'],
                               'largest_duplicate_neighbourhood': q['largest_duplicate_neighbourhood_l2_1e_minus6'],
                               'readout_rank99': q['feature_covariance']['readout']['rank99'],
                               'memory_norm_median': q['memory_norm_min_median_max'][1]})
        row = {'job': job['id'], 'variant': job['variant'], 'seed': job['seed'],
               'checkpoint_sha256': digest(directory/'epoch_100.pt'),
               'history_and_optimizer_steps_verified': True, 'memory_duplication_recomputed': True,
               'stored_patient_mse_aggregation_verified': True,
               'mse': full['mse'], 'effective_slots': full['effective_slots_exp_mean_query_entropy'],
               'largest_duplicate_neighbourhood': largest, 'memory_rank99': full['memory_rank99'],
               'readout_rank99': full['feature_covariance']['readout']['rank99'],
               'memory_norm_median': full['memory_norm_min_median_max'][1],
               'argmax_distinct_slots': full['argmax_distinct_slots'],
               'most_common_argmax_fraction': full['most_common_argmax_fraction'],
               'duplicate_neighbourhoods_by_l2_threshold': {str(t): int((distances <= t).sum(1).max()) for t in (1e-6, 1e-5, 1e-4, 1e-3)},
               'fixed45_trajectory': trajectory}
        summary['rows'].append(row)
        del ck
    for seed in manifest['config']['seeds']:
        a = next(r for r in summary['rows'] if r['seed'] == seed and r['variant'] == 'M0_baseline')
        b = next(r for r in summary['rows'] if r['seed'] == seed and r['variant'] == 'M1_entropy_off')
        summary['paired'].append({'seed': seed, 'normal_training_mse_reduction_percent': 100*(1-b['mse']/a['mse'])})
    out = root/'completed_audit.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({**summary, 'rows': [{k: v for k, v in r.items() if k != 'fixed45_trajectory'} for r in summary['rows']]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
