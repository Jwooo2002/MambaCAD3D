#!/usr/bin/env python3
"""Finite paired SSM repair experiment; corrected model, normal TRAIN only."""
import argparse
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

VARIANTS = {
    'S0_control': {'optimizer': 'Adam', 'exclude': 'memory', 'dt_bias_init': False},
    'S1_dt_no_decay': {'optimizer': 'Adam', 'exclude': 'dt', 'dt_bias_init': False},
    'S2_dynamics_no_decay': {'optimizer': 'Adam', 'exclude': 'dynamics', 'dt_bias_init': False},
    'S3_all_ssm_no_decay': {'optimizer': 'Adam', 'exclude': 'ssm', 'dt_bias_init': False},
    'S4_adamw': {'optimizer': 'AdamW', 'exclude': 'memory', 'dt_bias_init': False},
    'S5_dt_bias_init': {'optimizer': 'Adam', 'exclude': 'memory', 'dt_bias_init': True},
    'S6_dynamics_no_decay_dt_init': {'optimizer': 'Adam', 'exclude': 'dynamics', 'dt_bias_init': True},
    'S7_all_ssm_no_decay_dt_init': {'optimizer': 'Adam', 'exclude': 'ssm', 'dt_bias_init': True},
}
PROBES = [0, 1, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200]
SAVES = [0, 20, 50, 100, 150, 200]


def helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def imports(root):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(root/'source_snapshot'))
    from main_files import model
    return np, torch, model


def excluded_names(model, option):
    names = {'mem_rep.memory'}
    for prefix, module in model.named_modules():
        if type(module).__name__ != 'SS3D':
            continue
        for name, _ in module.named_parameters():
            if (option == 'ssm' or (option == 'dt' and name.startswith('dt_proj.')) or
                (option == 'dynamics' and (name in ('A_log', 'D') or name.startswith(('x_proj.', 'dt_proj.'))))):
                names.add(prefix+'.'+name)
    return names


def optimizer_for(model, variant, cfg, torch):
    option = VARIANTS[variant]
    excluded = excluded_names(model, option['exclude'])
    groups = []
    for is_excluded in (False, True):
        params = [(n, p) for n, p in model.named_parameters() if (n in excluded) == is_excluded]
        groups.append({'params': [p for n, p in params], 'param_names': [n for n, p in params],
                       'weight_decay': 0. if is_excluded else 1e-5})
    return getattr(torch.optim, option['optimizer'])(groups, lr=cfg['lr'])


def prepare(root, parent, project):
    if root.exists():
        raise RuntimeError('Refusing to overwrite experiment')
    base = helper(parent/'base_run.py', 'base')
    base.verify_files(parent)
    old = json.loads((parent/'manifest.json').read_text())
    assert old['status'] == 'complete'
    root.mkdir(parents=True)
    for source, dest in [(Path(__file__), 'run.py'), (Path(__file__).with_name('ssm_repair_probe.py'), 'ssm_probe.py'),
                         (parent/'base_run.py', 'base_run.py'), (parent/'probe.py', 'probe.py')]:
        shutil.copy2(source, root/dest)
    source = root/'source_snapshot/main_files'
    source.mkdir(parents=True)
    (source/'__init__.py').write_text('')
    for name in ('model.py', 'csms6s.py', 'utils_scan.py'):
        shutil.copy2(project/'main_files'/name, source/name)
    for name in ('input_manifest.json', 'training_split.json'):
        shutil.copy2(parent/name, root/name)
    cfg = dict(old['config'])
    cfg.update(epochs=200, probe_epochs=PROBES, checkpoint_epochs=SAVES, entropy_weight=0.,
               memory_decay=0., shrink_thres=.0025,
               trainer_differences='New common corrected-model baseline, fresh paired initial states, '
               'full-coverage precomputed 200-epoch orders; same frozen legacy TRAIN inputs; '
               'M3 diagnostic background (entropy=0, memory decay=0); no disease/validation/test selection.',
               rng_comparison_note='All arms share seed-specific base weights and batch orders. '
               'Initialization arms differ only in six dt_proj.bias tensors. CUDA backward may have rounding nondeterminism.',
               model_note='Current corrected direction packing + true hard shrink + kernel3 down/up; '
               'not a matched continuation of legacy M3, and not the full corrected data pipeline.',
               scheduler_note='Cosine T_max=1000 retained across every arm to isolate SSM interventions; no scheduler search.')
    base.write_json(root/'config.json', cfg)
    np, torch, implementation = imports(root)
    hashes, jobs, order_hashes = {}, [], {}
    inputs = json.loads((root/'input_manifest.json').read_text())
    total = sum(x['patches'] for x in inputs)
    assert total == 956 and len(inputs) == 15
    for seed in cfg['seeds']:
        base.set_seed(seed, np, torch)
        model = base.instantiate(implementation.MemAE_VSSM3D, cfg)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({'model_state_dict': state, 'seed': seed}, root/f'initial_seed_{seed}.pt')
        hashes[f'base_{seed}'] = base.state_digest(state)
        init_rng = torch.Generator().manual_seed(seed+5000003)
        changed = []
        for name, module in model.named_modules():
            if isinstance(module, implementation.SS3D):
                dt = torch.exp(torch.rand(module.dt_proj.bias.shape, generator=init_rng)*(math.log(.1)-math.log(.001))+math.log(.001))
                with torch.no_grad():
                    module.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
                changed.append(name+'.dt_proj.bias')
        initialized = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        assert len(changed) == 6
        assert {k for k in state if not torch.equal(state[k], initialized[k])} == set(changed)
        torch.save({'model_state_dict': initialized, 'seed': seed, 'changed_keys': changed}, root/f'initial_dt_seed_{seed}.pt')
        hashes[f'dt_{seed}'] = base.state_digest(initialized)
        gen = torch.Generator().manual_seed(seed+1000003)
        orders = [torch.randperm(total, generator=gen).tolist() for _ in range(cfg['epochs'])]
        assert orders[:100] == json.loads((parent/f'orders_seed_{seed}.json').read_text())
        base.write_json(root/f'orders_seed_{seed}.json', orders)
        order_hashes[str(seed)] = base.digest(root/f'orders_seed_{seed}.json')
    for variant, options in VARIANTS.items():
        for seed in cfg['seeds']:
            name = f'{variant}_seed{seed}'
            (root/name).mkdir()
            filename = f"initial_{'dt_' if options['dt_bias_init'] else ''}seed_{seed}.pt"
            jobs.append({'id': name, 'variant': variant, 'seed': seed, 'status': 'queued', 'initial_file': filename,
                         'initial_state_hash': hashes[f"{'dt' if options['dt_bias_init'] else 'base'}_{seed}"]})
    frozen = {str(p.relative_to(root)): base.digest(p) for p in root.rglob('*') if p.is_file()}
    base.write_json(root/'source_hashes.json', frozen)
    base.write_json(root/'manifest.json', {'status': 'prepared', 'created_utc': base.now(),
        'parent_pilot': str(parent), 'legacy_sweep': old['legacy_sweep'], 'config': cfg, 'variants': VARIANTS,
        'jobs': jobs, 'training_patches': total, 'total_epochs': 3200, 'initial_state_hashes': hashes,
        'batch_order_hashes': order_hashes, 'scope': 'Normal TRAIN mechanism experiment, not final k-fold performance.',
        'gpu_policy': 'Two idle GPUs, one job per GPU, variant pairs first, finite queue; no new launches after failure.',
        'runtime_prior': {'epoch_seconds': 34., 'estimated_wall_hours_two_gpus': 3200*34/2/3600,
                          'source': str(parent), 'requested_minimum_wall_hours': 10},
        'init_definition': 'Only dt_proj.bias reset: softplus(bias) log-uniform [0.001,0.1]. Existing dt weight distribution preserved.',
        'comparisons': ['S1-S0', 'S2-S1', 'S3-S2', 'S4-S0', 'S5-S0', 'S6-S2', 'S7-S3', 'S6-S5', 'S7-S6'],
        'decision_rule': 'Report every seed and fixed epoch. Assess input-dependent delta, task gradients, recurrent output, '
        'memory geometry/feature ranks and normal reconstruction together. Preserved norms alone are not success; '
        'no automatic winner, AUROC claim or final model adoption.'})
    print(json.dumps({'prepared': str(root), 'jobs': len(jobs), 'total_epochs': 3200}), flush=True)


def verify(root, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    base = helper(root/'base_run.py', 'base')
    base.verify_files(root)
    np, torch, implementation = imports(root)
    probe = helper(root/'ssm_probe.py', 'ssm_probe')
    manifest = json.loads((root/'manifest.json').read_text())
    cfg = manifest['config']
    data, probes = base.load_inputs(root, np, torch)
    indices = [v[len(v)//2] for v in list(probes.values())[:3]]
    rows = []
    for seed in cfg['seeds']:
        orders = json.loads((root/f'orders_seed_{seed}.json').read_text())
        assert len(orders) == 200 and all(sorted(x) == list(range(len(data))) for x in orders)
    for job in manifest['jobs']:
        model = base.instantiate(implementation.MemAE_VSSM3D, cfg).cuda()
        initial = torch.load(root/job['initial_file'], map_location='cpu', weights_only=False)['model_state_dict']
        model.load_state_dict(initial)
        assert base.state_digest(model.state_dict()) == job['initial_state_hash']
        opt = optimizer_for(model, job['variant'], cfg, torch)
        all_params = [p for g in opt.param_groups for p in g['params']]
        assert len(all_params) == len({id(p) for p in all_params}) == len(list(model.parameters()))
        for g in opt.param_groups:
            assert len(g['params']) == len(g['param_names'])
            assert all(g['weight_decay'] == (0. if n in excluded_names(model, VARIANTS[job['variant']]['exclude']) else 1e-5) for n in g['param_names'])
        for m in model.modules():
            if hasattr(m, 'drop_prob'): assert m.drop_prob == 0.
        if VARIANTS[job['variant']]['dt_bias_init']:
            for n, m in model.named_modules():
                if isinstance(m, implementation.SS3D):
                    dt = torch.nn.functional.softplus(m.dt_proj.bias)
                    assert float(dt.min()) >= .001-1e-7 and float(dt.max()) <= .1+1e-7
        x = data[:2].cuda()
        model.eval()
        with torch.no_grad():
            expected = model(x)[0].clone()
        before = base.state_digest(model.state_dict())
        rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        if job['seed'] == 42:
            observed = probe.measure_ssm(model, opt, data, indices, implementation)
            assert before == base.state_digest(model.state_dict())
            assert torch.equal(rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
            assert torch.equal(expected, model(x)[0].detach())
            base.write_json(root/job['id']/'verification_ssm_probe.json', observed)
        model.train()
        loss, seconds = None, time.monotonic()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            recon, attention = model(x)
            assert recon.shape == x.shape and torch.isfinite(attention).all()
            loss = (recon-x).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        torch.cuda.synchronize()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert base.state_digest(model.state_dict()) != before
        assert {int(s['step']) for s in opt.state.values()} == {3}
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=cfg['eta_min'])
        buffer = io.BytesIO()
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location='cpu', weights_only=False)
        clone = base.instantiate(implementation.MemAE_VSSM3D, cfg).cuda()
        clone.load_state_dict(saved['model'])
        clone_opt = optimizer_for(clone, job['variant'], cfg, torch)
        clone_opt.load_state_dict(saved['optimizer'])
        assert base.state_digest(clone.state_dict()) == base.state_digest(model.state_dict())
        for p, q in zip(model.parameters(), clone.parameters()):
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                assert torch.equal(opt.state[p][key].cpu(), clone_opt.state[q][key].cpu())
        model.eval(); clone.eval()
        with torch.no_grad(): assert torch.equal(model(x)[0], clone(x)[0])
        row = {'job': job['id'], 'passed': True, 'loss_after_3_updates': float(loss.detach()),
               'three_update_seconds': time.monotonic()-seconds,
               'probe_preserves_state_rng_and_output': job['seed'] == 42,
               'optimizer_groups': [{'weight_decay': g['weight_decay'], 'param_names': g['param_names']} for g in opt.param_groups]}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'optimizer_groups'}), flush=True)
        del model, clone, opt, clone_opt, initial, saved, buffer, recon, loss, attention
        torch.cuda.empty_cache()
    base.write_json(root/'verification.json', {'status': 'passed', 'checked_utc': base.now(),
                    'full_200_epoch_orders_verified': True, 'rows': rows})


def train(root, job_id, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    base = helper(root/'base_run.py', 'base')
    base.verify_files(root)
    np, torch, implementation = imports(root)
    mem_probe = helper(root/'probe.py', 'mem_probe')
    ssm_probe = helper(root/'ssm_probe.py', 'ssm_probe')
    manifest = json.loads((root/'manifest.json').read_text())
    cfg = manifest['config']
    job = next(j for j in manifest['jobs'] if j['id'] == job_id)
    output = root/job_id
    state = {'status': 'running', 'phase': 'loading', 'started_utc': base.now(), 'pid': os.getpid(),
             'gpu': gpu, 'variant': job['variant'], 'seed': job['seed'], 'epoch': 0, 'completed_epochs': 0,
             'total_epochs': cfg['epochs']}
    base.write_json(output/'run_state.json', state)
    try:
        base.set_seed(job['seed'], np, torch)
        data, probes = base.load_inputs(root, np, torch)
        indices = [v[len(v)//2] for v in list(probes.values())[:3]]
        model = base.instantiate(implementation.MemAE_VSSM3D, cfg).cuda()
        initial = torch.load(root/job['initial_file'], map_location='cpu', weights_only=False)['model_state_dict']
        model.load_state_dict(initial)
        assert base.state_digest(model.state_dict()) == job['initial_state_hash']
        del initial
        opt = optimizer_for(model, job['variant'], cfg, torch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['scheduler_T_max'], eta_min=cfg['eta_min'])
        orders = json.loads((root/f"orders_seed_{job['seed']}.json").read_text())
        base.set_seed(job['seed']+2000003, np, torch)
        loader_rng = torch.Generator().manual_seed(job['seed']+3000003)
        history = []

        def probe_and_save(epoch):
            state.update(phase='normal_training_probe', epoch=epoch)
            base.write_json(output/'run_state.json', state)
            before = base.state_digest(model.state_dict())
            with torch.random.fork_rng(devices=[0]):
                memory = mem_probe.measure(model, data, probes, 0., 0.)
                ssm = ssm_probe.measure_ssm(model, opt, data, indices, implementation)
            assert before == base.state_digest(model.state_dict())
            base.write_json(output/f'probe_epoch_{epoch:03d}.json', {'epoch': epoch, 'checked_utc': base.now(), 'memory': memory, 'ssm': ssm})
            if epoch in SAVES:
                saved = {'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                         'config': cfg, 'seed': job['seed'], 'epoch': epoch, 'variant': job['variant'],
                         'intervention': VARIANTS[job['variant']], 'initial_state_hash': job['initial_state_hash'],
                         'batch_order_hash': manifest['batch_order_hashes'][str(job['seed'])],
                         'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                       'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state(),
                                       'data_loader': loader_rng.get_state()}}
                tmp = output/f'epoch_{epoch:03d}.pt.tmp'
                torch.save(saved, tmp); tmp.replace(output/f'epoch_{epoch:03d}.pt')
            print(json.dumps({'probe_epoch': epoch, 'mse': memory['mse'],
                'bias_only_observations': sum(r['delta_equals_bias_only'] for r in ssm['functional']),
                'observations': len(ssm['functional'])}), flush=True)

        probe_and_save(0)
        for epoch, order in enumerate(orders, 1):
            batches = [order[i:i+cfg['batch_size']] for i in range(0, len(order), cfg['batch_size'])]
            state.update(epoch=epoch, phase='training', steps_completed=0, steps_per_epoch=len(batches))
            base.write_json(output/'run_state.json', state)
            model.train(); started = time.monotonic(); total = 0.; count = 0
            loader = torch.utils.data.DataLoader(data, batch_sampler=batches, num_workers=cfg['num_workers'], pin_memory=True, generator=loader_rng)
            for number, x in enumerate(loader, 1):
                x = x.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                recon, _ = model(x)
                loss = (recon-x).square().mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                total += float(loss.detach())*len(x); count += len(x)
                if number == 1 or number % 100 == 0:
                    state.update(steps_completed=number, heartbeat_utc=base.now())
                    base.write_json(output/'run_state.json', state)
            torch.cuda.synchronize(); seconds = time.monotonic()-started
            lr = opt.param_groups[0]['lr']; scheduler.step()
            assert count == len(data) and scheduler.last_epoch == epoch
            history.append({'epoch': epoch, 'mse': total/count, 'lr': lr, 'training_seconds': seconds,
                            'patches': count, 'optimizer_steps': len(batches)})
            base.write_json(output/'history.json', history)
            print(json.dumps(history[-1]), flush=True)
            state.update(completed_epochs=epoch, last_epoch_seconds=seconds, heartbeat_utc=base.now())
            base.write_json(output/'run_state.json', state)
            if epoch in PROBES: probe_and_save(epoch)
        state.update(phase='full_normal_training_audit'); base.write_json(output/'run_state.json', state)
        full = {}; offset = 0
        for row in json.loads((root/'input_manifest.json').read_text()):
            full[row['patient']] = list(range(offset, offset+row['patches'])); offset += row['patches']
        before = base.state_digest(model.state_dict())
        record = mem_probe.measure(model, data, full, 0., 0.)
        assert before == base.state_digest(model.state_dict())
        record.update(epoch=200, checked_utc=base.now(), probe_scope='All 956 normal TRAIN patches; no validation/test')
        base.write_json(output/'full_normal_training_epoch_200.json', record)
        assert {int(s['step']) for s in opt.state.values()} == {95600}
        state.update(status='complete', phase='complete', completed_epochs=200, finished_utc=base.now())
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=base.now())
        raise
    finally:
        base.write_json(output/'run_state.json', state)


def summarize(root):
    base = helper(root/'base_run.py', 'base')
    manifest = json.loads((root/'manifest.json').read_text())
    rows = []
    for job in manifest['jobs']:
        row = {'job': job['id'], 'variant': job['variant'], 'seed': job['seed'], 'status': job['status']}
        path = root/job['id']/'probe_epoch_200.json'
        if path.exists():
            result = json.loads(path.read_text()); memory, ssm = result['memory'], result['ssm']
            dt = [p for p in ssm['parameters'] if p['name'].endswith('dt_proj.weight')]
            row.update(normal_probe_mse=memory['mse'], query_rank99=memory['feature_covariance']['query']['rank99'],
                       memory_rank99=memory['memory_rank99'], global_effective_slots=memory['global_effective_slots'],
                       dt_weight_min_block_l2=min(p['l2_float64'] for p in dt),
                       bias_only_observations=sum(r['delta_equals_bias_only'] for r in ssm['functional']))
        rows.append(row)
    base.write_json(root/'results_summary.json', {'status': manifest['status'], 'updated_utc': base.now(), 'jobs': rows,
        'scope': 'Normal TRAIN diagnostics. No held-out disease evaluation or winner selection.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'verify', 'train', 'queue', 'summarize'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--project', type=Path)
    p.add_argument('--gpu', default='0')
    p.add_argument('--job')
    args = p.parse_args(); root = args.root.resolve()
    if args.command == 'prepare': prepare(root, args.parent.resolve(), args.project.resolve())
    elif args.command == 'verify': verify(root, args.gpu)
    elif args.command == 'train': train(root, args.job, args.gpu)
    elif args.command == 'summarize': summarize(root)
    else:
        helper(root/'base_run.py', 'base').queue(root)
        summarize(root)


if __name__ == '__main__':
    main()
