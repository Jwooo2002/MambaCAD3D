#!/usr/bin/env python3
"""Audit finished pilot checkpoints using only the original normal TRAIN split."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('frozen_pilot', root/'run.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    from probe import measure
    run.verify_files(root)
    np, torch, cls = run.imports(root)
    manifest = json.loads((root/'manifest.json').read_text())
    cfg = manifest['config']
    assert manifest['status'] == 'complete'
    data, fixed_indices = run.load_inputs(root, np, torch)
    all_indices = {}
    offset = 0
    for entry in json.loads((root/'input_manifest.json').read_text()):
        all_indices[entry['patient']] = list(range(offset, offset+entry['patches']))
        offset += entry['patches']
    assert offset == len(data) == 956
    report = {'started_utc': run.now(), 'scope': 'All 956 normal TRAINING patches from the same 15 patients. No validation, disease, or test access. Read-only epoch-20 checkpoint audit.',
              'source_hashes_verified': True, 'rows': []}
    for job in manifest['jobs']:
        before = time.monotonic()
        path = root/job['id']
        assert job['status'] == 'complete' and job['returncode'] == 0
        state = json.loads((path/'run_state.json').read_text())
        history = json.loads((path/'history.json').read_text())
        assert state['completed_epochs'] == 20
        assert [h['epoch'] for h in history] == list(range(1, 21))
        assert all(h['patches'] == 956 and h['optimizer_steps'] == 478 for h in history)
        initial = torch.load(path/'epoch_00.pt', map_location='cpu', weights_only=False)
        expected = manifest['initial_state_hashes'][str(job['seed'])]
        assert run.state_digest(initial['model_state_dict']) == expected
        del initial
        checkpoint = torch.load(path/'epoch_20.pt', map_location='cpu', weights_only=False)
        assert checkpoint['initial_state_hash'] == expected
        assert checkpoint['batch_order_hash'] == manifest['batch_order_hashes'][str(job['seed'])]
        assert checkpoint['scheduler_state_dict']['T_max'] == 1000
        assert checkpoint['scheduler_state_dict']['last_epoch'] == 20
        model = run.instantiate(cls, cfg).cuda()
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        weights_before = run.state_digest(model.state_dict())
        options = manifest['variants'][job['variant']]
        fixed = measure(model, data, fixed_indices, options['entropy_weight'], options['memory_decay'])
        saved = json.loads((path/'probe_epoch_20.json').read_text())
        keys = ('mse', 'effective_slots_exp_mean_query_entropy', 'global_effective_slots', 'mean_top1_mass')
        errors = {k: abs(fixed[k]-saved[k]) for k in keys}
        assert all(np.isclose(fixed[k], saved[k], rtol=1e-6, atol=1e-9) for k in keys), errors
        final = measure(model, data, all_indices, options['entropy_weight'], options['memory_decay'])
        assert run.state_digest(model.state_dict()) == weights_before
        final['probe_scope'] = report['scope']
        final['patient_macro_mse'] = float(np.mean([p['mse'] for p in final['patients']]))
        p = np.array(final['global_slot_mass'])
        cumulative = np.cumsum(np.sort(p/p.sum())[::-1])
        final['slots_for_95pct_global_mass'] = int(np.searchsorted(cumulative, .95))+1
        final['slots_for_99pct_global_mass'] = int(np.searchsorted(cumulative, .99))+1
        row = {'job': job['id'], 'variant': job['variant'], 'seed': job['seed'],
               'verified_initial_state_and_batch_order': True, 'verified_9560_optimizer_steps': True,
               'recomputed_fixed_probe_absolute_errors': errors, 'weights_unchanged_by_audit': True,
               'checkpoint_sha256': run.digest(path/'epoch_20.pt'),
               'full_normal_training': final, 'seconds': time.monotonic()-before}
        report['rows'].append(row)
        run.write_json(root/'completed_audit.json', report)
        print(json.dumps({'job': job['id'], 'mse': final['mse'], 'effective_slots': final['effective_slots_exp_mean_query_entropy'],
                          'readout_rank99': final['feature_covariance']['readout']['rank99'], 'seconds': row['seconds']}), flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
    report.update(status='complete', finished_utc=run.now())
    run.write_json(root/'completed_audit.json', report)


if __name__ == '__main__':
    main()
