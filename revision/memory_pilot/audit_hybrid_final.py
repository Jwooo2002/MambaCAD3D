#!/usr/bin/env python3
"""After training ends, audit all final models on the same45normal TRAIN patches."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(root, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'complete', 'Wait for the finite training queue to finish'
    assert len(manifest['jobs']) == 12 and all(j['status'] == 'complete' and j['returncode'] == 0 for j in manifest['jobs'])
    busy = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    assert not busy, 'Selected GPU is occupied; do not interfere with training'
    runner = helper(root / 'run.py', 'frozen_hybrid_runner')
    base = helper(root / 'base_run.py', 'frozen_base')
    base.verify_files(root)
    np, torch, implementation = runner.engine_for(root).imports(root)
    probe = helper(root / 'ssm_probe.py', 'frozen_hybrid_probe')
    data, patients = base.load_inputs(root, np, torch)
    indices = [index for values in patients.values() for index in values]
    assert len(patients) == 15 and len(indices) == 45
    output = root / 'final_checkpoint_audit'
    output.mkdir(exist_ok=True)
    records = []
    for job in manifest['jobs']:
        checkpoint = root / job['id'] / 'epoch_200.pt'
        expected_hash = base.digest(checkpoint)
        target = output / f"{job['id']}.json"
        if target.exists():
            record = json.loads(target.read_text())
            assert record['checkpoint_sha256'] == expected_hash and record['state_and_rng_unchanged']
            assert len(record['interventions']) == 180 and record['job'] == job['id']
            records.append(str(target))
            continue
        _, _, implementation = runner.engine_for(root, job['variant']).imports(root)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['epoch'] == 200 and saved['variant'] == job['variant'] and saved['seed'] == job['seed']
        model = base.instantiate(implementation.MemAE_VSSM3D, manifest['config']).cuda()
        model.load_state_dict(saved['model_state_dict'])
        model.eval()
        before = base.state_digest(model.state_dict())
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        interventions = probe.recurrent_dependence(model, data, indices, implementation, stagewise=True)
        assert len(interventions) == 180
        assert before == base.state_digest(model.state_dict()) and expected_hash == base.digest(checkpoint)
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert all(row['noop_bitwise_equal'] for row in interventions)
        if manifest['variants'][job['variant']]['direct_only']:
            assert all(row['differences']['reconstruction']['bitwise_equal'] for row in interventions)
        record = {'job': job['id'], 'variant': job['variant'], 'seed': job['seed'],
                  'checkpoint_sha256': expected_hash, 'epoch': 200, 'patch_indices': indices,
                  'state_and_rng_unchanged': True, 'checked_utc': base.now(), 'interventions': interventions,
                  'scope': 'Same45normal TRAIN patches as parent audit. All/stage recurrence-off; no test data.'}
        base.write_json(target, record)
        records.append(str(target))
        print(json.dumps({'audited': job['id'], 'patches': 45, 'interventions': 180}), flush=True)
        del model, saved, record, interventions
        torch.cuda.empty_cache()
    base.write_json(output / 'summary.json', {'status': 'complete', 'records': records, 'jobs': 12,
                    'patches_per_model': 45, 'checked_utc': base.now(),
                    'scope': 'Inference dependence after fixed200epochs, not clinical performance or retraining benefit.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    audit(args.root.resolve(), args.gpu)
