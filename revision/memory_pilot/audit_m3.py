#!/usr/bin/env python3
"""Verify completed M3 checkpoints and compare stored normal-TRAIN metrics to M1."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('frozen_m3', root/'run.py')
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    base = run.helper(root/'base_run.py', 'base_helper')
    base.verify_files(root)
    manifest = read(root/'manifest.json')
    assert manifest['status'] == 'complete'
    result = {'status': 'complete', 'audited_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Read-only CPU checkpoint and stored normal-TRAIN metric audit; no new forward evaluation or training.',
              'started_utc': manifest['started_utc'], 'finished_utc': manifest['finished_utc'],
              'source_hashes_verified': True, 'rows': []}
    result['wall_minutes'] = (datetime.fromisoformat(manifest['finished_utc'])-datetime.fromisoformat(manifest['started_utc'])).total_seconds()/60
    for job in manifest['jobs']:
        seed = job['seed']; directory = root/job['id']
        assert job['status'] == 'complete' and job['returncode'] == 0
        state = read(directory/'run_state.json'); history = read(directory/'history.json')
        assert state['completed_epochs'] == 100 and state['total_optimizer_steps'] == 47800
        assert [h['epoch'] for h in history] == list(range(1, 101))
        assert all(h['patches'] == 956 and h['optimizer_steps'] == 478 and h['loss'] == h['mse'] for h in history)
        assert all(math.isfinite(h[k]) for h in history for k in ('mse', 'loss', 'legacy_entropy'))
        ck = torch.load(directory/'epoch_100.pt', map_location='cpu', weights_only=False)
        assert ck['epoch'] == 100 and ck['seed'] == seed and ck['variant'] == job['variant']
        assert ck['intervention'] == {'entropy_weight': 0., 'memory_decay': 0.}
        assert ck['initial_state_hash'] == manifest['initial_state_hashes'][str(seed)]
        assert ck['batch_order_hash'] == manifest['batch_order_hashes'][str(seed)]
        assert ck['scheduler_state_dict']['T_max'] == 1000 and ck['scheduler_state_dict']['last_epoch'] == 100
        optimizer = ck['optimizer_state_dict']; groups = optimizer['param_groups']
        assert len(groups) == 2 and groups[0]['weight_decay'] == 1e-5 and groups[1]['weight_decay'] == 0
        assert len(groups[1]['params']) == 1
        assert {int(s['step']) for s in optimizer['state'].values()} == {47800}
        assert set(ck['rng_state']) == {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'data_loader'}
        memory = ck['model_state_dict']['mem_rep.memory']
        assert memory.shape == (300, 128)
        full = read(directory/'full_normal_training_epoch_100.json')
        assert full['epoch'] == 100 and full['patches'] == 956 and len(full['patients']) == 15
        recomputed = run.geometry(memory, torch)
        assert recomputed == full['memory_geometry']
        assert full['largest_duplicate_neighbourhood_l2_1e_minus6'] == recomputed['max_neighbourhood_including_self_by_absolute_l2']['1e-06']
        assert sum(p['patches'] for p in full['patients']) == 956
        weighted = sum(p['mse']*p['patches'] for p in full['patients'])/956
        assert math.isclose(weighted, full['mse'], rel_tol=1e-12)
        assert math.isclose(math.exp(full['mean_query_entropy']), full['effective_slots_exp_mean_query_entropy'], rel_tol=1e-12)
        refdir = root/'control_reference'
        control = read(refdir/f'M1_seed{seed}_full_normal_training_epoch_100.json')
        reference_geometry = read(refdir/f'M1_seed{seed}_geometry_epoch_100.json')
        assert base.digest(Path(job['control_checkpoint'])) == job['control_checkpoint_sha256']
        initial = torch.load(root/f'initial_seed_{seed}.pt', map_location='cpu', weights_only=False)['model_state_dict']
        assert base.state_digest(initial) == ck['initial_state_hash']
        initial_memory = initial['mem_rep.memory'].double(); current = memory.double()
        directional_cosine = torch.nn.functional.cosine_similarity(initial_memory, current, dim=1)
        movement = (current-initial_memory).norm(dim=1)
        trajectory = []
        for epoch in manifest['config']['probe_epochs']:
            q = read(directory/f'probe_epoch_{epoch:03d}.json')
            assert q['epoch'] == epoch and q['patches'] == 45
            trajectory.append({'epoch': epoch, 'mse': q['mse'], 'effective_slots': q['effective_slots_exp_mean_query_entropy'],
                               'memory_norm_median': q['memory_norm_min_median_max'][1],
                               'readout_rank99': q['feature_covariance']['readout']['rank99'],
                               'absolute_l2_1e_minus3_max_neighbourhood': q['memory_geometry']['max_neighbourhood_including_self_by_absolute_l2']['0.001'],
                               'relative_l2_0point1_max_neighbourhood': q['memory_geometry']['max_neighbourhood_including_self_by_relative_l2']['0.1']})
        def compact(q):
            return {'mse': q['mse'], 'effective_slots': q['effective_slots_exp_mean_query_entropy'],
                    'memory_norm_median': q['memory_norm_min_median_max'][1],
                    'memory_rank99': q['memory_rank99'], 'readout_rank99': q['feature_covariance']['readout']['rank99'],
                    'argmax_distinct_slots': q['argmax_distinct_slots'], 'global_effective_slots': q['global_effective_slots']}
        result['rows'].append({'seed': seed, 'job': job['id'], 'checkpoint_sha256': base.digest(directory/'epoch_100.pt'),
                               'history_optimizer_groups_steps_scheduler_verified': True, 'geometry_recomputed': True,
                               'patient_mse_aggregation_verified': True, 'm1': compact(control), 'm3': compact(full),
                               'm3_over_m1_mse_ratio': full['mse']/control['mse'],
                               'm3_vs_m1_mse_percent_change': 100*(full['mse']/control['mse']-1),
                               'm1_geometry': reference_geometry, 'm3_geometry': recomputed,
                               'initial_to_final_memory_row_movement_min_median_max': [float(movement.min()), float(movement.median()), float(movement.max())],
                               'initial_to_final_memory_direction_cosine_min_median_max': [float(directional_cosine.min()), float(directional_cosine.median()), float(directional_cosine.max())],
                               'fixed45_trajectory': trajectory})
        del ck, initial
    base.write_json(root/'completed_audit.json', result)
    print(json.dumps({**result, 'rows': [{k: v for k, v in row.items() if k not in ('fixed45_trajectory', 'm1_geometry', 'm3_geometry')} for row in result['rows']]}, indent=2))


if __name__ == '__main__':
    main()
