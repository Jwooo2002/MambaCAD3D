#!/usr/bin/env python3
"""One prespecified, read-only epoch-100 comparison on previously exposed data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parent.parent
ARCH_KEYS = ('in_chans', 'dims', 'depths', 'ssm_d_state', 'mem_dim')
VARIANTS = ('M0_baseline', 'M1_entropy_off', 'M3_entropy_and_memory_decay_off')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare(root):
    import numpy as np
    if root.exists():
        raise RuntimeError('Refusing to overwrite existing evaluation')
    continuation = PROJECT / 'outputs/memory_collapse_continuation_20261110T075000Z'
    m3 = PROJECT / 'outputs/memory_m3_pilot_20261110T123000Z'
    baseline = PROJECT / 'outputs/legacy_retrain_20261109T064401Z'
    split_path = PROJECT / 'outputs/legacy_candidate_sweep_20261109/split_manifest.json'
    split = read(split_path)[0]
    assert split['fold'] == 1
    cfg = read(continuation / 'config.json')
    assert all(cfg[k] == read(m3 / 'config.json')[k] for k in ARCH_KEYS)
    assert cfg['batch_size'] == 2
    source_hashes = {}
    for old_root in (continuation, m3):
        for name, expected in read(old_root / 'source_hashes.json').items():
            assert digest(old_root / name) == expected, name
        actual = {str(p.relative_to(old_root / 'source_snapshot/main_files')): digest(p)
                  for p in (old_root / 'source_snapshot/main_files').rglob('*.py')}
        if source_hashes:
            assert actual == source_hashes, 'Legacy model sources differ'
        source_hashes = actual
        assert {p['patient'] for p in read(old_root / 'input_manifest.json')} == set(split['train'])
    assert len(set(split['train'])) == 15
    assert len(set(split['normal_test_and_validation'])) == 5
    assert len(set(split['diseased_test'])) == 20
    assert set(split['train']).isdisjoint(split['normal_test_and_validation'])
    basecfg = read(baseline / 'effective_config.json')
    inputs = []
    for label, key, root_key in ((0, 'normal_test_and_validation', 'normal_data_root'),
                                 (1, 'diseased_test', 'diseased_data_root')):
        for filename in split[key]:
            path = Path(basecfg[root_key]) / filename
            before = digest(path)
            with path.open('rb') as f:
                array = np.stack(pickle.load(f)).astype(np.float32)
            assert len(array) > 0 and array.shape[1:] == (64, 32, 32), array.shape
            assert np.isfinite(array).all() and digest(path) == before
            inputs.append({'id': f'{label}/{filename}', 'filename': filename, 'label': label,
                           'path': str(path), 'patches': len(array), 'sha256': before})
    jobs = []
    for seed in (42, 123):
        for variant in VARIANTS:
            old_root = m3 if variant == VARIANTS[2] else continuation
            old_job = old_root / f'{variant}_seed{seed}'
            checkpoint = old_job / 'epoch_100.pt'
            probe = old_job / 'probe_epoch_100.json'
            jobs.append({'id': old_job.name, 'variant': variant, 'seed': seed,
                         'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                         'probe': str(probe), 'probe_sha256': digest(probe),
                         'config': str(old_root / 'config.json'),
                         'config_sha256': digest(old_root / 'config.json')})
    root.mkdir(parents=True)
    shutil.copytree(continuation / 'source_snapshot/main_files', root / 'source_snapshot/main_files',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__, root / 'evaluate_fixed_checkpoints.py')
    shutil.copy2(continuation / 'input_manifest.json', root / 'training_input_manifest.json')
    write(root / 'input_manifest.json', inputs)
    write(root / 'config.json', {k: cfg[k] for k in ARCH_KEYS})
    write(root / 'protocol.json', {
        'created_utc': now(), 'models': list(VARIANTS), 'seeds': [42, 123], 'epoch': 100,
        'fold': 1, 'patients': {'normal_nontraining': 5, 'diseased': 20},
        'training_patients': 15, 'normalization': 'float32 (x+300)/700; no clipping',
        'patch_shape': [64, 32, 32], 'batch_size': 2, 'all_patches': True,
        'patch_score': 'mean squared reconstruction error over every voxel',
        'patient_score': 'maximum over all patches', 'primary_metric': 'AUROC',
        'secondary': ['average_precision', 'classwise_patient_max_score_distribution'],
        'higher_score': 'diseased', 'positive_label': 1, 'jobs': jobs,
        'source_python_sha256': source_hashes, 'original_split': str(split_path),
        'original_split_sha256': digest(split_path), 'training_files': split['train'],
        'scope': 'Exploratory comparison on previously evaluated subjects; not fresh validation/test.',
        'prohibited': ['training', 'checkpoint selection', 'aggregation search', 'best-seed selection',
                       'memory intervention', 'final model adoption'],
    })
    frozen = {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()}
    write(root / 'frozen_hashes.json', frozen)
    write(root / 'status.json', {'status': 'prepared', 'prepared_utc': now(), 'jobs': 6})
    print(json.dumps({'prepared': str(root), 'patients': len(inputs),
                      'patches': sum(e['patches'] for e in inputs)}), flush=True)


def verify_frozen(root):
    for name, expected in read(root / 'frozen_hashes.json').items():
        assert digest(root / name) == expected, name


def worker(root, seed, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(root / 'source_snapshot/main_files'))
    from model import MemAE_VSSM3D
    from data.VideoDataset import VideoDataset
    from torchvision import transforms
    verify_frozen(root)
    protocol = read(root / 'protocol.json')
    cfg = read(root / 'config.json')
    inputs = read(root / 'input_manifest.json')
    status_path = root / f'worker_seed{seed}.json'
    status = {'status': 'running', 'started_utc': now(), 'seed': seed,
              'gpu': gpu, 'pid': os.getpid(), 'completed_jobs': []}
    write(status_path, status)
    try:
        tensors = {}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x+300)/700)])
        for entry in inputs:
            assert digest(entry['path']) == entry['sha256']
            with open(entry['path'], 'rb') as f:
                array = np.stack(pickle.load(f)).astype(np.float32)
            data = (torch.from_numpy(array).unsqueeze(1) + 300) / 700
            assert len(data) == entry['patches'] and torch.isfinite(data).all()
            # Validate first/last patches of every subject against the original dataset.
            legacy = VideoDataset(str(Path(entry['path']).parent), transform=transform,
                                  pkl_files=[entry['filename']])
            assert torch.equal(data[0], legacy[0][1])
            assert torch.equal(data[-1], legacy[len(legacy)-1][1])
            assert digest(entry['path']) == entry['sha256']
            tensors[entry['id']] = data
        probes = []
        for entry in read(root / 'training_input_manifest.json'):
            assert digest(entry['path']) == entry['sha256']
            with open(entry['path'], 'rb') as f:
                clips = pickle.load(f)
            indices = np.unique(np.linspace(0, len(clips)-1, min(3, len(clips)), dtype=int))
            data = torch.from_numpy(np.stack([clips[i] for i in indices]).astype(np.float32)).unsqueeze(1)
            probes.append((entry['patient'], (data+300)/700))
        for job in [j for j in protocol['jobs'] if j['seed'] == seed]:
            status.update(job=job['id'], patients_done=0, phase='checkpoint_and_probe_verification')
            write(status_path, status)
            started = time.monotonic()
            assert digest(job['checkpoint']) == job['checkpoint_sha256']
            assert digest(job['probe']) == job['probe_sha256']
            assert digest(job['config']) == job['config_sha256']
            checkpoint = torch.load(job['checkpoint'], map_location='cpu', weights_only=False)
            assert checkpoint['epoch'] == 100 and checkpoint['seed'] == seed
            assert checkpoint['variant'] == job['variant']
            assert all(checkpoint['config'][k] == cfg[k] for k in ARCH_KEYS)
            model = MemAE_VSSM3D(**cfg).cuda().eval()
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            before = state_digest(model)
            saved_probe = read(job['probe'])
            saved_by_patient = {p['patient']: p for p in saved_probe['patients']}
            probe_sum = 0.
            count = 0
            max_probe_delta = 0.
            with torch.no_grad():
                for patient, data in probes:
                    patient_sum = 0.
                    for start in range(0, len(data), 2):
                        x = data[start:start+2].cuda()
                        recon, _ = model(x)
                        errors = (recon-x).square().flatten(1).mean(1)
                        patient_sum += errors.sum().item()
                    observed = patient_sum/len(data)
                    expected = saved_by_patient[patient]['mse']
                    assert np.isclose(observed, expected, rtol=1e-5, atol=1e-9), (patient, observed, expected)
                    max_probe_delta = max(max_probe_delta, abs(observed-expected))
                    probe_sum += patient_sum
                    count += len(data)
                assert count == saved_probe['patches'] == 45
                assert np.isclose(probe_sum/count, saved_probe['mse'], rtol=1e-5, atol=1e-9)
                rows = []
                for entry in inputs:
                    status.update(phase='all_patch_inference', patients_done=len(rows))
                    write(status_path, status)
                    data = tensors[entry['id']]
                    scores = []
                    for start in range(0, len(data), 2):
                        x = data[start:start+2].cuda()
                        recon, _ = model(x)
                        errors = (recon-x).square().flatten(1).mean(1)
                        assert torch.isfinite(errors).all()
                        scores.extend(errors.cpu().tolist())
                    assert len(scores) == entry['patches']
                    rows.append({'id': entry['id'], 'label': entry['label'], 'patches': len(scores),
                                 'patch_mse_in_original_order': scores, 'patient_score_max': max(scores),
                                 'patient_mean_mse_diagnostic': float(np.mean(scores))})
            assert before == state_digest(model), 'Inference altered model state'
            assert digest(job['checkpoint']) == job['checkpoint_sha256']
            labels = [r['label'] for r in rows]
            scores = [r['patient_score_max'] for r in rows]
            result = {'job': job, 'epoch': 100, 'patient_aggregation': 'max',
                      'AUROC': float(roc_auc_score(labels, scores)),
                      'AP': float(average_precision_score(labels, scores)),
                      'patients': rows, 'model_state_sha256_before_and_after': before,
                      'preflight': {'original_loader_first_last_each_subject_exact': True,
                                    'saved_training_probe_45_patch_mse_reproduced': True,
                                    'max_patient_probe_mse_abs_difference': max_probe_delta},
                      'completed_utc': now(), 'duration_seconds': time.monotonic()-started}
            write(root / f"{job['id']}.json", result)
            status['completed_jobs'].append(job['id'])
            print(json.dumps({'seed': seed, 'job': job['id'], 'status': 'completed',
                              'seconds': result['duration_seconds']}), flush=True)
            del model, checkpoint
            torch.cuda.empty_cache()
        status.update(status='completed', completed_utc=now(), patients_done=25)
        write(status_path, status)
    except BaseException as exc:
        status.update(status='failed', failed_utc=now(), error=repr(exc))
        write(status_path, status)
        traceback.print_exc()
        raise


def run(root):
    verify_frozen(root)
    state = read(root / 'status.json')
    assert state['status'] == 'prepared', 'Refusing duplicate inference launch'
    state.update(status='running', started_utc=now(), pid=os.getpid())
    write(root / 'status.json', state)
    processes = []
    for seed, gpu in ((42, 0), (123, 1)):
        log = (root / f'worker_seed{seed}.log').open('w')
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
                              '--root', str(root), '--seed', str(seed), '--gpu', str(gpu)],
                             stdout=log, stderr=subprocess.STDOUT)
        processes.append((p, log))
    codes = [p.wait() for p, _ in processes]
    for _, log in processes:
        log.close()
    state.update(status='inference_completed' if codes == [0, 0] else 'failed',
                 finished_utc=now(), worker_exit_codes=codes)
    write(root / 'status.json', state)
    print(json.dumps(state), flush=True)
    if any(codes):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root.resolve())
    elif args.action == 'run':
        run(args.root.resolve())
    else:
        worker(args.root.resolve(), args.seed, args.gpu)
