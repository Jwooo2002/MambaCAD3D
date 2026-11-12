#!/usr/bin/env python3
"""Frozen, finite nested-holdout search. Training uses normal patients only."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

VARIANTS = {
    'P0_reference': {'weight_decay': 1e-5, 'entropy_loss_weight': 2e-4},
    'P1_no_decay': {'weight_decay': 0., 'entropy_loss_weight': 2e-4},
    'P2_weak_entropy': {'weight_decay': 0., 'entropy_loss_weight': 2e-5},
    'P3_no_entropy': {'weight_decay': 0., 'entropy_loss_weight': 0.},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str((str(value.dtype), tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def imports(root):
    sys.path.insert(0, str(root / 'source_snapshot'))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from main_files.core.model_trainer import build_model, memory_entropy, seed_everything
    from main_files.core.evaluation import score_patients
    from main_files.core.metrics import patient_metrics
    from main_files.data.patients import PatientPatchDataset
    return np, torch, build_model, memory_entropy, seed_everything, score_patients, patient_metrics, PatientPatchDataset


def validate_splits(patients, folds):
    by_id = {p['patient_id']: p for p in patients}
    assert len(by_id) == len(patients) == 40
    all_tests = []
    for f in folds:
        roles = [f[k] for k in ('inner_train', 'inner_normal', 'inner_diseased', 'outer_test')]
        assert all(len(r) == len(set(r)) for r in roles)
        assert not any(set(a) & set(b) for i, a in enumerate(roles) for b in roles[i+1:])
        assert set().union(*map(set, roles)) == set(by_id)
        assert len(f['inner_train']) == 12 and len(f['inner_normal']) == 3
        assert len(f['inner_diseased']) == 15 and len(f['outer_test']) == 10
        assert all(by_id[p]['label'] == 0 for p in f['inner_train'] + f['inner_normal'])
        assert all(by_id[p]['label'] == 1 for p in f['inner_diseased'])
        assert sum(by_id[p]['label'] for p in f['outer_test']) == 5
        assert sorted(f['outer_train']) == sorted(f['inner_train'] + f['inner_normal'])
        assert not set(f['outer_train']) & set(f['outer_test'])
        all_tests += f['outer_test']
    assert sorted(all_tests) == sorted(by_id)


def prepare(root, project, epochs):
    if root.exists():
        raise RuntimeError('Refusing to overwrite experiment')
    root.mkdir(parents=True)
    shutil.copytree(project/'main_files', root/'source_snapshot/main_files',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, root/'run.py')
    shutil.copy2(Path(__file__).with_name('analyze.py'), root/'analyze.py')
    source = json.loads((project/'splits/asoca_ordered_mm_v2_4fold.json').read_text())
    cfg = json.loads((project/'configs/asoca_revision.json').read_text())
    cfg.update(epochs=epochs, num_workers=0, training_seeds=[42, 123, 2024],
               early_stopping_patience=epochs+1, bootstrap_samples=2000)
    # The external cohort is not accessed by this first internal search batch.
    for key in list(cfg):
        if 'external' in key:
            del cfg[key]
    write_json(root/'config.json', cfg)
    import numpy as np
    from sklearn.model_selection import KFold
    disease = sorted(p['patient_id'] for p in source['patients'] if p['label'] == 1)
    disease_folds = list(KFold(n_splits=4, shuffle=True, random_state=20261112).split(disease))
    folds = []
    for old, (dev, test) in zip(source['folds'], disease_folds):
        folds.append({'fold': old['fold'], 'inner_train': old['train'], 'inner_normal': old['validation'],
                      'inner_diseased': [disease[i] for i in dev],
                      'outer_train': sorted(old['train'] + old['validation']),
                      'outer_test': old['test_normal'] + [disease[i] for i in test]})
    validate_splits(source['patients'], folds)
    manifest = {'created_utc': now(), 'patients': source['patients'], 'folds': folds,
                'normal_split_source': str(project/'splits/asoca_ordered_mm_v2_4fold.json'),
                'disease_split_seed': 20261112, 'historical_exposure':
                'All ASOCA subjects were used historically. This is internal revalidation, not a pristine test cohort.'}
    write_json(root/'splits.json', manifest)
    np, torch, build_model, _, seed_all, _, _, Dataset = imports(root)
    # Verify byte-level provenance and shapes before any disease inference.
    for p in manifest['patients']:
        assert digest(p['path']) == p['patch_sha256'], p['patient_id']
        data = Dataset([p], cfg['hu_window'], cfg['expected_patch_shape'])
        p['patch_count'] = len(data)
    write_json(root/'splits.json', manifest)
    by_id = {p['patient_id']: p for p in manifest['patients']}
    pairs = {}
    for stage, seeds in [('inner', [42, 123]), ('outer', [42, 123, 2024])]:
        for f in folds:
            ids = f['inner_train'] if stage == 'inner' else f['outer_train']
            count = sum(by_id[p]['patch_count'] for p in ids)
            for seed in seeds:
                key = f'{stage}_f{f["fold"]}_s{seed}'
                actual_seed = (100000 if stage == 'inner' else 200000) + 1000*f['fold'] + seed
                seed_all(actual_seed)
                model = build_model(cfg)
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(state, root/f'{key}_initial.pt')
                gen = torch.Generator().manual_seed(actual_seed + 1000003)
                orders = [torch.randperm(count, generator=gen).tolist() for _ in range(epochs)]
                assert all(sorted(o) == list(range(count)) for o in orders)
                write_json(root/f'{key}_orders.json', orders)
                pairs[key] = {'actual_seed': actual_seed, 'initial_state_sha256': state_digest(state),
                              'train_patch_count': count}
    write_json(root/'pairs.json', pairs)
    plan = {'status': 'prepared', 'created_utc': now(), 'variants': VARIANTS,
            'epochs': epochs, 'inner_seeds': [42, 123], 'outer_seeds': [42, 123, 2024],
            'outer_folds': 4, 'max_training_runs': 56, 'max_wall_hours': 48,
            'architecture': 'Unchanged corrected grouped-SSM + sparse memory300 + kernel3 CNN decoder, 2,109,241 parameters.',
            'input': 'Previously frozen ordered_mm_v2 physical patches; clipped HU [-300,400], scaled to [0,1]. All patches retained.',
            'selection': 'Within each outer fold, mean inner AUROC over two seeds. Eligible when mean AP >= P0 mean AP - 0.02. Tie: AP then declared variant order. Fixed final epoch only.',
            'inner_validation': '3 normal + 15 diseased patients; weight training uses 12 normal patients. This is label-assisted hyperparameter selection.',
            'outer_test': '5 normal + 5 diseased, each patient once per seed; refit on all 15 outer-development normals. No outer test before all four fold selections are locked.',
            'primary_metric': 'max-patch patient AUROC; AP secondary with prevalence recorded; other aggregations descriptive only.',
            'training': 'Adam, lr1e-4, cosine T_max=epochs eta_min1e-6, batch2, clip1.0, fixed epochs, no early stopping, same initialization/orders within each stage/fold/seed.',
            'reporting': 'All candidates, folds and seeds; paired patient bootstrap retaining seeds within patient and resampling within fold/class. Conditional fitted-model uncertainty, not all retraining uncertainty.',
            'interpretation': 'Evaluate the nested selection procedure, not a post-hoc single winning configuration. Existing historical exposure persists. No automatic final paper adoption.',
            'exclusions': 'No architecture, sampling, cosine-memory, memory-size, initialization, augmentation, score aggregation or external-cohort tuning in this batch.'}
    write_json(root/'plan.json', plan)
    jobs = []
    for f in folds:
        for variant in VARIANTS:
            for seed in plan['inner_seeds']:
                jobs.append(make_job('inner', f['fold'], seed, variant))
    write_json(root/'manifest.json', {'status': 'prepared', 'created_utc': now(), 'jobs': jobs})
    frozen = {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    write_json(root/'frozen_hashes.json', frozen)
    print(json.dumps({'prepared': str(root), 'inner_jobs': len(jobs), 'epochs': epochs}), flush=True)


def make_job(stage, fold, seed, variant):
    return {'id': f'{stage}_f{fold}_s{seed}_{variant}', 'stage': stage, 'fold': fold,
            'seed': seed, 'variant': variant, 'status': 'queued'}


def verify_files(root):
    for name, expected in json.loads((root/'frozen_hashes.json').read_text()).items():
        assert digest(root/name) == expected, name


def preflight(root, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    verify_files(root)
    np, torch, build, entropy, seed_all, score, metrics, Dataset = imports(root)
    from main_files import csms6s
    assert torch.cuda.is_available()
    assert any((csms6s.WITH_SELECTIVESCAN_OFLEX, csms6s.WITH_SELECTIVESCAN_CORE, csms6s.WITH_SELECTIVESCAN_MAMBA))
    cfg = json.loads((root/'config.json').read_text())
    splits = json.loads((root/'splits.json').read_text())
    validate_splits(splits['patients'], splits['folds'])
    by_id = {p['patient_id']: p for p in splits['patients']}
    data = Dataset([by_id[p] for p in splits['folds'][0]['inner_train']], cfg['hu_window'], cfg['expected_patch_shape'])
    state = torch.load(root/'inner_f1_s42_initial.pt', map_location='cpu', weights_only=True)
    x = torch.stack([data[i] for i in range(2)]).cuda()
    rows = []
    for variant, options in VARIANTS.items():
        seed_all(101042)
        model = build(cfg).cuda()
        model.load_state_dict(state)
        assert sum(p.numel() for p in model.parameters()) == 2109241
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=options['weight_decay'])
        times = []
        for step in range(12):
            torch.cuda.synchronize()
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            recon, att = model(x)
            loss = (recon-x).square().mean() + options['entropy_loss_weight']*entropy(att)
            assert recon.shape == x.shape and torch.isfinite(loss) and (att.sum(-1) > 0).all()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            torch.cuda.synchronize()
            if step >= 2:
                times.append(time.perf_counter()-start)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            expected = model(x)[0].clone()
        clone = build(cfg).cuda()
        clone.load_state_dict({k: v.detach().cpu().clone() for k,v in model.state_dict().items()})
        clone.eval()
        with torch.no_grad():
            assert torch.equal(expected, clone(x)[0])
        rows.append({'variant': variant, 'step_seconds_median': float(np.median(times)),
                     'finite_loss': float(loss.detach()), 'checkpoint_roundtrip_exact': True})
        del model, opt, clone, loss, recon, att, expected
        torch.cuda.empty_cache()
    pairs = json.loads((root/'pairs.json').read_text())
    inner_steps = sum(math.ceil(p['train_patch_count']/2) for k,p in pairs.items() if k.startswith('inner')) * 4
    outer_steps = sum(math.ceil(p['train_patch_count']/2) for k,p in pairs.items() if k.startswith('outer')) * 2
    estimate = (inner_steps+outer_steps)*cfg['epochs']*max(r['step_seconds_median'] for r in rows)/2/3600
    result = {'status': 'passed', 'checked_utc': now(), 'gpu': gpu, 'gpu_name': torch.cuda.get_device_name(0),
              'rows': rows, 'estimated_two_gpu_training_hours': estimate,
              'estimate_with_30_percent_margin_hours': estimate*1.3,
              'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
              'disease_inference_performed': False}
    write_json(root/f'preflight_gpu{gpu}.json', result)
    print(json.dumps(result), flush=True)


def train(root, job_id, gpu, deadline):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    verify_files(root)
    np, torch, build, entropy, seed_all, score, metrics, Dataset = imports(root)
    manifest = json.loads((root/'manifest.json').read_text())
    job = next(j for j in manifest['jobs'] if j['id'] == job_id)
    cfg = json.loads((root/'config.json').read_text())
    cfg.update(VARIANTS[job['variant']])
    split = json.loads((root/'splits.json').read_text())
    f = next(f for f in split['folds'] if f['fold'] == job['fold'])
    by_id = {p['patient_id']: p for p in split['patients']}
    train_ids = f['inner_train'] if job['stage'] == 'inner' else f['outer_train']
    eval_ids = f['inner_normal']+f['inner_diseased'] if job['stage'] == 'inner' else f['outer_test']
    assert not set(train_ids) & set(eval_ids)
    assert all(by_id[p]['label'] == 0 for p in train_ids)
    if job['stage'] == 'outer':
        assert (root/'selections.json').is_file()
        selections = json.loads((root/'selections.json').read_text())
        assert len(selections['folds']) == 4
        selected = selections['folds'][str(job['fold'])]['selected_variant']
        assert job['variant'] in (selected, 'P0_reference')
    out = root/job_id
    out.mkdir(exist_ok=False)
    pair_key = f'{job["stage"]}_f{job["fold"]}_s{job["seed"]}'
    pair = json.loads((root/'pairs.json').read_text())[pair_key]
    seed_all(pair['actual_seed'])
    for pid in train_ids:
        assert digest(by_id[pid]['path']) == by_id[pid]['patch_sha256']
    data = Dataset([by_id[p] for p in train_ids], cfg['hu_window'], cfg['expected_patch_shape'])
    train_tensor = torch.cat(data.volumes).pin_memory()
    assert len(data) == pair['train_patch_count']
    orders = json.loads((root/f'{pair_key}_orders.json').read_text())
    model = build(cfg).cuda()
    model.load_state_dict(torch.load(root/f'{pair_key}_initial.pt', map_location='cpu', weights_only=True))
    assert state_digest(model.state_dict()) == pair['initial_state_sha256']
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=cfg['eta_min'])
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    state = {**job, 'status': 'training', 'started_utc': now(), 'pid': os.getpid(), 'gpu': gpu,
             'train_patient_ids': train_ids, 'evaluation_patient_ids': eval_ids,
             'actual_seed': pair['actual_seed'], 'initial_state_sha256': pair['initial_state_sha256'],
             'orders_sha256': digest(root/f'{pair_key}_orders.json'), 'config': cfg}
    write_json(out/'state.json', state)
    fields = ['epoch','train_mse','entropy','loss','effective_slots_local','effective_slots_global',
              'max_attention','zero_memory_fraction','lr','seconds']
    start = time.perf_counter()
    def save(name, epoch, completed_updates):
        temp = out/(name+'.tmp')
        torch.save({'model_state_dict': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch,
                    'completed_updates': completed_updates, 'job': job, 'config': cfg,
                    'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}, temp)
        temp.replace(out/name)
    updates = 0
    with (out/'history.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch, order in enumerate(orders, 1):
            epoch_start = time.perf_counter()
            model.train()
            sums = np.zeros(7)
            global_mass = torch.zeros(cfg['mem_dim'], device='cuda')
            lr = opt.param_groups[0]['lr']
            for pos in range(0, len(order), cfg['batch_size']):
                if stopped[0] or time.time() >= deadline or (root/'STOP').exists():
                    save('interrupted.pt', epoch-1, updates)
                    state.update(status='interrupted', finished_utc=now(), completed_epochs=epoch-1,
                                 interrupted_epoch=epoch, completed_updates=updates)
                    write_json(out/'state.json', state)
                    return
                indices = order[pos:pos+cfg['batch_size']]
                x = train_tensor[indices].cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                reconstruction, att = model(x)
                mse = (reconstruction-x).square().mean()
                ent = entropy(att)
                loss = mse + cfg['entropy_loss_weight']*ent
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                updates += 1
                with torch.no_grad():
                    local_entropy = -(att*att.clamp_min(1e-12).log()).sum(-1)
                    vals = torch.stack([mse, ent, loss, local_entropy.exp().mean(),
                                        att.max(-1).values.mean(), (att.sum(-1)==0).float().mean()])
                    sums[:6] += vals.detach().cpu().numpy()*len(x)
                    sums[6] += len(x)
                    global_mass += att.detach().sum((0,1))
            scheduler.step()
            sums[:6] /= sums[6]
            mass = global_mass/global_mass.sum().clamp_min(1e-12)
            row = dict(zip(['train_mse','entropy','loss','effective_slots_local','max_attention','zero_memory_fraction'],map(float,sums[:6])))
            row.update(epoch=epoch, effective_slots_global=float((-(mass*mass.clamp_min(1e-12).log()).sum()).exp()),
                       lr=lr, seconds=time.perf_counter()-epoch_start)
            writer.writerow(row); handle.flush()
            state.update(completed_epochs=epoch, completed_updates=updates, latest=row, heartbeat_utc=now())
            write_json(out/'state.json', state)
            if epoch % 50 == 0 or epoch == cfg['epochs']:
                save(f'epoch_{epoch}.pt', epoch, updates)
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps({'job': job_id, **row}), flush=True)
    # Disease/outer inputs are loaded only after the final checkpoint is immutable.
    checkpoint = out/f'epoch_{cfg["epochs"]}.pt'
    checkpoint_hash = digest(checkpoint)
    before = state_digest(model.state_dict())
    for pid in eval_ids:
        assert digest(by_id[pid]['path']) == by_id[pid]['patch_sha256']
    eval_data = Dataset([by_id[p] for p in eval_ids], cfg['hu_window'], cfg['expected_patch_shape'])
    state.update(status='scoring', training_seconds=time.perf_counter()-start, scoring_started_utc=now())
    write_json(out/'state.json', state)
    write_json(out/'evaluation_access.json', {'accessed_utc': now(), 'stage': job['stage'],
               'patient_ids': eval_ids, 'checkpoint_sha256': checkpoint_hash,
               'selection_sha256': digest(root/'selections.json') if job['stage']=='outer' else None})
    patient_rows, patch_rows = score(model, eval_data, cfg, torch.device('cuda'), warmup=True)
    assert state_digest(model.state_dict()) == before and digest(checkpoint) == checkpoint_hash
    for filename, rows in [('patient_scores.csv',patient_rows), ('patch_scores.csv',patch_rows)]:
        with (out/filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    results = {a: metrics([r['label'] for r in patient_rows], [r[a] for r in patient_rows],
                          patient_ids=[r['patient_id'] for r in patient_rows])
               for a in ('max','mean','top_k','percentile')}
    write_json(out/'results.json', {'job': job, 'metrics': results, 'patients': patient_rows,
                'checkpoint_sha256': checkpoint_hash, 'scored_patches':len(patch_rows),
                'final_state_sha256':before, 'inference_state_unchanged':True})
    state.update(status='complete', finished_utc=now(), elapsed_seconds=time.perf_counter()-start,
                 checkpoint_sha256=checkpoint_hash, metrics=results['max'])
    write_json(out/'state.json', state)
    print(json.dumps({'completed': job_id, 'metrics':results['max']}), flush=True)


def choose(rows):
    summaries = {}
    for variant in VARIANTS:
        matched = [r for r in rows if r['variant']==variant]
        assert sorted(r['seed'] for r in matched) == [42,123]
        summaries[variant] = {m:sum(r[m] for r in matched)/2 for m in ('auroc','ap')}
    guard = summaries['P0_reference']['ap'] - .02
    eligible = [v for v in VARIANTS if summaries[v]['ap'] >= guard - 1e-12]
    winner = min(eligible, key=lambda v:(-round(summaries[v]['auroc'],12),-round(summaries[v]['ap'],12),list(VARIANTS).index(v)))
    return {'selected_variant':winner, 'candidate_means':summaries, 'ap_minimum':guard, 'eligible':eligible}


def lock_selections(root, manifest):
    assert len(manifest['jobs']) == 32 and all(j['status']=='complete' for j in manifest['jobs'])
    selections = {'locked_utc':now(), 'folds':{}, 'rule':json.loads((root/'plan.json').read_text())['selection']}
    for fold in range(1,5):
        rows = []
        for j in manifest['jobs']:
            if j['fold'] == fold:
                result = json.loads((root/j['id']/'results.json').read_text())
                rows.append({'variant':j['variant'],'seed':j['seed'],**result['metrics']['max']})
        selections['folds'][str(fold)] = choose(rows)
    assert not (root/'selections.json').exists()
    write_json(root/'selections.json', selections)
    for fold in range(1,5):
        selected = selections['folds'][str(fold)]['selected_variant']
        for seed in [42,123,2024]:
            for variant in dict.fromkeys(['P0_reference',selected]):
                manifest['jobs'].append(make_job('outer', fold, seed, variant))
    manifest['selections_sha256'] = digest(root/'selections.json')


def summarize(root):
    manifest = json.loads((root/'manifest.json').read_text())
    completed, running = [], []
    for j in manifest['jobs']:
        p = root/j['id']/'state.json'
        if p.exists():
            state = json.loads(p.read_text())
            if state['status']=='complete':
                completed.append({k:j[k] for k in ('id','stage','fold','seed','variant')} | state['metrics'])
            else:
                running.append({'id':j['id'],'status':state['status'],'epochs':state.get('completed_epochs',0),
                                'latest':state.get('latest')})
    result = {'updated_utc':now(),'status':manifest['status'],'completed':completed,'active_or_interrupted':running,
              'planned_jobs':len(manifest['jobs']), 'deadline_utc':manifest.get('deadline_utc'),
              'warning':'Inner scores select hyperparameters. Outer scores evaluate the locked procedure; no next-batch tuning from an alleged untouched test.'}
    write_json(root/'progress.json', result)
    print(json.dumps({'status':result['status'],'complete':len(completed),'planned':len(manifest['jobs']),
                      'active':[(r['id'],r['epochs']) for r in running]}), flush=True)


def queue(root):
    verify_files(root)
    for gpu in ('0','1'):
        preflight_result = json.loads((root/f'preflight_gpu{gpu}.json').read_text())
        assert preflight_result['status']=='passed'
        assert preflight_result['estimate_with_30_percent_margin_hours'] <= 48
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['status'] == 'prepared'
    deadline = time.time()+48*3600
    manifest.update(status='running',started_utc=now(),queue_pid=os.getpid(),deadline_unix=deadline,
                    deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat())
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_:stopped.__setitem__(0,True))
    signal.signal(signal.SIGINT, lambda *_:stopped.__setitem__(0,True))
    running = {}
    stop_started = None
    write_json(root/'manifest.json', manifest)
    try:
        while True:
            for gpu,(proc,job,handle) in list(running.items()):
                if proc.poll() is not None:
                    handle.close()
                    state_path = root/job['id']/'state.json'
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                    job.update(status=state.get('status','failed'),exit_code=proc.returncode,finished_utc=now())
                    if proc.returncode != 0 or job['status'] != 'complete':
                        job['status'] = 'interrupted' if state.get('status')=='interrupted' else 'failed'
                        manifest['status'] = 'failed' if job['status']=='failed' else 'interim_stopped'
                        stopped[0] = True
                    del running[gpu]
            stop_requested = stopped[0] or (root/'STOP').exists() or time.time() >= deadline
            if stop_requested:
                if stop_started is None:
                    stop_started = time.time()
                if manifest['status']=='running':
                    manifest['status']='interim_stopped'
                for proc,job,handle in running.values():
                    if time.time()-stop_started < 90:
                        proc.terminate()
                    else:
                        proc.kill()
                if not running:
                    break
            else:
                pending = [j for j in manifest['jobs'] if j['status']=='queued']
                if not pending and not running:
                    if not (root/'selections.json').exists():
                        lock_selections(root, manifest)
                        write_json(root/'manifest.json', manifest)
                        pending = [j for j in manifest['jobs'] if j['status']=='queued']
                    else:
                        manifest['status']='complete'
                        break
                for gpu in ('0','1'):
                    if gpu in running or not pending:
                        continue
                    # Existing other projects' GPU work is never stopped.
                    usage = subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                        '--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True).strip()
                    if int(usage) > 1000:
                        continue
                    job = pending.pop(0)
                    handle = (root/(job['id']+'.log')).open('w')
                    cmd = [sys.executable,'-u',str(root/'run.py'),'train','--root',str(root),
                           '--job',job['id'],'--gpu',gpu,'--deadline',str(deadline)]
                    proc = subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,cwd=root)
                    job.update(status='running',pid=proc.pid,gpu=gpu,started_utc=now())
                    running[gpu]=(proc,job,handle)
                    print(json.dumps({'started':job['id'],'gpu':gpu,'pid':proc.pid}),flush=True)
            manifest['heartbeat_utc']=now()
            write_json(root/'manifest.json',manifest)
            summarize(root)
            time.sleep(20)
    finally:
        for proc,job,handle in running.values():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.wait()
            handle.close()
        manifest['finished_utc']=now()
        write_json(root/'manifest.json',manifest)
        summarize(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','preflight','train','queue','summarize'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--gpu',default='0')
    p.add_argument('--job')
    p.add_argument('--deadline',type=float,default=float('inf'))
    args=p.parse_args();root=args.root.resolve()
    if args.command=='prepare':prepare(root,args.project.resolve(),args.epochs)
    elif args.command=='preflight':preflight(root,args.gpu)
    elif args.command=='train':train(root,args.job,args.gpu,args.deadline)
    elif args.command=='queue':queue(root)
    else:summarize(root)


if __name__=='__main__':
    main()
