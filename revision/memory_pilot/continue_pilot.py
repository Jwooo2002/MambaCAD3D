#!/usr/bin/env python3
"""Finite continuation of paired M0/M1 pilots from epoch 20 to epoch 100."""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

VARIANTS = ('M0_baseline', 'M1_entropy_off')
PROBES = [20, 25, 30, 40, 50, 60, 75, 100]


def helper(path):
    spec = importlib.util.spec_from_file_location('frozen_pilot_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(root, parent):
    if root.exists():
        raise RuntimeError('Refusing to overwrite an existing continuation')
    old = helper(parent/'run.py')
    old.verify_files(parent)
    prior = json.loads((parent/'manifest.json').read_text())
    assert prior['status'] == 'complete'
    root.mkdir(parents=True)
    shutil.copy2(__file__, root/'run.py')
    shutil.copy2(parent/'run.py', root/'base_run.py')
    shutil.copy2(parent/'probe.py', root/'probe.py')
    shutil.copytree(parent/'source_snapshot', root/'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('input_manifest.json', 'training_split.json'):
        shutil.copy2(parent/name, root/name)
    base = helper(root/'base_run.py')
    np, torch, cls = base.imports(root)
    cfg = dict(prior['config'])
    cfg.update(epochs=100, resume_epoch=20, probe_epochs=PROBES,
               trainer_differences='Continue original model/Adam/scheduler states from epoch 20; paired permutation stream extended to 100; fixed normal-training-only probes; no validation/test/early stopping.',
               rng_resume_note='Parent checkpoint did not save RNG state. Forward/update has no active stochastic layers; this is checked in preflight. New phase RNG is fixed per seed; explicit batch permutations extend the original stream. Do not claim bitwise equivalence to an uninterrupted 100-epoch run.')
    assert cfg['scheduler_T_max'] == 1000 and cfg['mem_dim'] == 300
    base.write_json(root/'config.json', cfg)
    order_hashes = {}
    for seed in cfg['seeds']:
        generator = torch.Generator().manual_seed(seed+1000003)
        orders = [torch.randperm(prior['training_patches'], generator=generator).tolist() for _ in range(100)]
        assert orders[:20] == json.loads((parent/f'orders_seed_{seed}.json').read_text())
        assert all(sorted(order) == list(range(prior['training_patches'])) for order in orders)
        base.write_json(root/f'orders_seed_{seed}.json', orders)
        order_hashes[str(seed)] = base.digest(root/f'orders_seed_{seed}.json')
    jobs = []
    for seed in cfg['seeds']:
        for variant in VARIANTS:
            name = f'{variant}_seed{seed}'
            source = parent/name
            destination = root/name
            destination.mkdir()
            state = json.loads((source/'run_state.json').read_text())
            assert state['status'] == 'complete' and state['completed_epochs'] == 20
            for old_name, new_name in [('epoch_20.pt', 'resume_epoch_20.pt'),
                                       ('history.json', 'parent_history.json'),
                                       ('probe_epoch_20.json', 'parent_probe_epoch_20.json')]:
                shutil.copy2(source/old_name, destination/new_name)
            saved = torch.load(destination/'resume_epoch_20.pt', map_location='cpu', weights_only=False)
            assert saved['epoch'] == 20 and saved['seed'] == seed and saved['variant'] == variant
            assert saved['initial_state_hash'] == prior['initial_state_hashes'][str(seed)]
            assert saved['batch_order_hash'] == prior['batch_order_hashes'][str(seed)]
            assert saved['scheduler_state_dict']['last_epoch'] == 20
            assert {int(s['step']) for s in saved['optimizer_state_dict']['state'].values()} == {9560}
            jobs.append({'id': name, 'variant': variant, 'seed': seed, 'status': 'queued',
                         'parent_checkpoint': str(source/'epoch_20.pt'),
                         'parent_checkpoint_sha256': base.digest(destination/'resume_epoch_20.pt'),
                         'resume_model_state_hash': base.state_digest(saved['model_state_dict'])})
    hashes = {str(p.relative_to(root)): base.digest(p) for p in root.rglob('*') if p.is_file()}
    base.write_json(root/'source_hashes.json', hashes)
    base.write_json(root/'manifest.json', {'status': 'prepared', 'created_utc': base.now(),
                    'parent_pilot': str(parent), 'baseline': prior['baseline'], 'legacy_sweep': prior['legacy_sweep'],
                    'config': cfg, 'variants': {v: prior['variants'][v] for v in VARIANTS},
                    'initial_state_hashes': prior['initial_state_hashes'],
                    'parent_batch_order_hashes': prior['batch_order_hashes'], 'batch_order_hashes': order_hashes,
                    'training_patches': prior['training_patches'], 'jobs': jobs,
                    'total_additional_epochs': 320, 'total_epochs_per_job': 100,
                    'scope': 'Normal-training-only mechanism continuation; no validation/test/disease evaluation or final model adoption.',
                    'gpu_policy': 'One job per idle GPU 0/1; four finite jobs; stop new launches after a failure.'})
    print(json.dumps({'prepared': str(root), 'jobs': 4, 'resume_epoch': 20, 'target_epoch': 100}), flush=True)


def restore(checkpoint, cfg, variant, base, torch, cls):
    model = base.instantiate(cls, cfg).cuda()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer = base.optimizer_for(model, variant, cfg, torch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['scheduler_T_max'], eta_min=cfg['eta_min'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return model, optimizer, scheduler


def step(model, optimizer, x, weight, torch):
    optimizer.zero_grad(set_to_none=True)
    recon, attention = model(x)
    mse = (recon-x).square().mean()
    entropy = -(attention*torch.log(attention+1e-12)).sum(1).mean()
    loss = mse+weight*entropy
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return tuple(float(v.detach()) for v in (loss, mse, entropy))


def assert_same(a, b, torch):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_same(a[k], b[k], torch)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_same(x, y, torch)
    else:
        assert a == b, (a, b)


def compare_numeric(a, b, torch, *, rtol, atol):
    """Bound floating update differences; metadata and counters stay exact."""
    stats = {'max_abs_difference': 0., 'differing_tensors': 0, 'rtol': rtol, 'atol': atol}

    def visit(x, y, path):
        if isinstance(x, torch.Tensor):
            x, y = x.cpu(), y.cpu()
            assert x.dtype == y.dtype and x.shape == y.shape, path
            if not x.is_floating_point() or path.endswith('.step'):
                assert torch.equal(x, y), path
            else:
                delta = float((x-y).abs().max()) if x.numel() else 0.
                stats['max_abs_difference'] = max(stats['max_abs_difference'], delta)
                stats['differing_tensors'] += int(not torch.equal(x, y))
                assert torch.allclose(x, y, rtol=rtol, atol=atol), (path, delta)
        elif isinstance(x, dict):
            assert x.keys() == y.keys(), path
            for k in x: visit(x[k], y[k], f'{path}.{k}')
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y), path
            for k, (u, v) in enumerate(zip(x, y)): visit(u, v, f'{path}.{k}')
        else:
            assert x == y, (path, x, y)

    visit(a, b, '')
    return stats


def verify(root, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    base = helper(root/'base_run.py')
    base.verify_files(root)
    np, torch, cls = base.imports(root)
    sys.path.insert(0, str(root))
    from probe import measure
    manifest = json.loads((root/'manifest.json').read_text()); cfg = manifest['config']
    data, probes = base.load_inputs(root, np, torch)
    rows = []
    for job in manifest['jobs']:
        checkpoint = torch.load(root/job['id']/'resume_epoch_20.pt', map_location='cpu', weights_only=False)
        model, optimizer, scheduler = restore(checkpoint, cfg, job['variant'], base, torch, cls)
        assert base.state_digest(model.state_dict()) == job['resume_model_state_hash']
        assert_same(optimizer.state_dict(), checkpoint['optimizer_state_dict'], torch)
        assert_same(scheduler.state_dict(), checkpoint['scheduler_state_dict'], torch)
        for module in model.modules():
            if hasattr(module, 'drop_prob'): assert module.drop_prob == 0.
            if isinstance(module, torch.nn.modules.dropout._DropoutNd): assert module.p == 0.
        options = manifest['variants'][job['variant']]
        probe = measure(model, data, probes, options['entropy_weight'], options['memory_decay'])
        prior = json.loads((root/job['id']/'parent_probe_epoch_20.json').read_text())
        for key in ('mse', 'effective_slots_exp_mean_query_entropy', 'global_effective_slots'):
            assert np.isclose(probe[key], prior[key], rtol=1e-6, atol=1e-9)
        assert base.state_digest(model.state_dict()) == job['resume_model_state_hash']
        orders = json.loads((root/f"orders_seed_{job['seed']}.json").read_text())
        assert len(orders) == 100 and all(sorted(o) == list(range(len(data))) for o in orders)
        model.train()
        cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state().clone()
        step(model, optimizer, data[orders[20][:2]].cuda(), options['entropy_weight'], torch)
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert {int(s['step']) for s in optimizer.state.values()} == {9561}
        buffer = io.BytesIO()
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()}, buffer)
        buffer.seek(0)
        roundtrip = torch.load(buffer, map_location='cpu', weights_only=False)
        model2, optimizer2, scheduler2 = restore(roundtrip, cfg, job['variant'], base, torch, cls)
        assert_same(model.state_dict(), model2.state_dict(), torch)
        assert_same(optimizer.state_dict(), optimizer2.state_dict(), torch)
        assert_same(scheduler.state_dict(), scheduler2.state_dict(), torch)
        # Independently deserialize to avoid shared CPU Adam step tensors.
        buffer.seek(0)
        control = torch.load(buffer, map_location='cpu', weights_only=False)
        model3, optimizer3, scheduler3 = restore(control, cfg, job['variant'], base, torch, cls)
        assert_same(model2.state_dict(), model3.state_dict(), torch)
        assert_same(optimizer2.state_dict(), optimizer3.state_dict(), torch)
        x = data[orders[20][2:4]].cuda()
        with torch.no_grad():
            assert_same(model(x), model2(x), torch)
        losses = [step(m, o, x, options['entropy_weight'], torch)
                  for m, o in ((model, optimizer), (model2, optimizer2), (model3, optimizer3))]
        assert losses[0] == losses[1] == losses[2], losses
        # The frozen GPU backward path has rounding-level nondeterminism even
        # between two independent reloads. Require exact restoration/forward,
        # bounded subsequent updates, and report the independent control too.
        update_stats = compare_numeric(model.state_dict(), model2.state_dict(), torch, rtol=1e-6, atol=1e-8)
        optimizer_stats = compare_numeric(optimizer.state_dict(), optimizer2.state_dict(), torch, rtol=1e-5, atol=1e-8)
        control_stats = compare_numeric(model2.state_dict(), model3.state_dict(), torch, rtol=1e-6, atol=1e-8)
        control_optimizer_stats = compare_numeric(optimizer2.state_dict(), optimizer3.state_dict(), torch, rtol=1e-5, atol=1e-8)
        scheduler.step(); scheduler2.step()
        assert_same(scheduler.state_dict(), scheduler2.state_dict(), torch)
        rows.append({'job': job['id'], 'weights_optimizer_scheduler_exactly_restored': True,
                     'epoch20_probe_reproduced': True, 'next_update_consumes_no_rng': True,
                     'save_reload_state_and_forward_exact': True,
                     'save_reload_next_update': update_stats, 'save_reload_optimizer_update': optimizer_stats,
                     'independent_reload_control_update': control_stats,
                     'independent_reload_control_optimizer_update': control_optimizer_stats,
                     'all_100_epoch_orders_full_coverage': True})
        print(json.dumps({'verified_job': job['id'], 'update': update_stats, 'control': control_stats}), flush=True)
        del model, optimizer, scheduler, model2, optimizer2, scheduler2, model3, optimizer3, scheduler3
        del checkpoint, roundtrip, control
        torch.cuda.empty_cache()
    base.write_json(root/'verification.json', {'status': 'passed', 'checked_utc': base.now(), 'rows': rows,
                    'rng_limitation': cfg['rng_resume_note']})
    print(json.dumps({'verification': 'passed', 'jobs': len(rows)}), flush=True)


def train(root, job_id, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    base = helper(root/'base_run.py'); base.verify_files(root)
    np, torch, cls = base.imports(root)
    sys.path.insert(0, str(root))
    from probe import measure
    manifest = json.loads((root/'manifest.json').read_text()); cfg = manifest['config']
    job = next(j for j in manifest['jobs'] if j['id'] == job_id)
    output = root/job_id; options = manifest['variants'][job['variant']]
    state = {'status': 'running', 'started_utc': base.now(), 'pid': os.getpid(), 'gpu': gpu,
             'variant': job['variant'], 'seed': job['seed'], 'epoch': 20, 'completed_epochs': 20,
             'resume_epoch': 20, 'total_epochs': 100, 'phase': 'restoring'}
    base.write_json(output/'run_state.json', state)
    try:
        data, probes = base.load_inputs(root, np, torch)
        checkpoint = torch.load(output/'resume_epoch_20.pt', map_location='cpu', weights_only=False)
        model, optimizer, scheduler = restore(checkpoint, cfg, job['variant'], base, torch, cls)
        assert base.state_digest(model.state_dict()) == job['resume_model_state_hash']
        assert_same(optimizer.state_dict(), checkpoint['optimizer_state_dict'], torch)
        assert_same(scheduler.state_dict(), checkpoint['scheduler_state_dict'], torch)
        assert scheduler.last_epoch == 20
        del checkpoint
        orders = json.loads((root/f"orders_seed_{job['seed']}.json").read_text())
        history = json.loads((output/'parent_history.json').read_text())
        assert [h['epoch'] for h in history] == list(range(1, 21))
        base.set_seed(job['seed']+2000003, np, torch)
        data_rng = torch.Generator().manual_seed(job['seed']+4000003)

        def probe_and_save(epoch):
            state.update(phase='normal_training_probe', epoch=epoch)
            base.write_json(output/'run_state.json', state)
            with torch.random.fork_rng(devices=[0]):
                record = measure(model, data, probes, options['entropy_weight'], options['memory_decay'])
            record.update(epoch=epoch, checked_utc=base.now())
            base.write_json(output/f'probe_epoch_{epoch:03d}.json', record)
            saved = {'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                     'seed': job['seed'], 'epoch': epoch, 'variant': job['variant'], 'config': cfg,
                     'intervention': options, 'initial_state_hash': manifest['initial_state_hashes'][str(job['seed'])],
                     'batch_order_hash': manifest['batch_order_hashes'][str(job['seed'])],
                     'parent_checkpoint_sha256': job['parent_checkpoint_sha256'],
                     'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                   'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state(),
                                   'data_loader': data_rng.get_state()}}
            tmp = output/f'epoch_{epoch:03d}.pt.tmp'
            torch.save(saved, tmp); tmp.replace(output/f'epoch_{epoch:03d}.pt')
            print(json.dumps({'probe_epoch': epoch, 'mse': record['mse'],
                              'effective_slots': record['effective_slots_exp_mean_query_entropy'],
                              'duplicates': record['largest_duplicate_neighbourhood_l2_1e_minus6']}), flush=True)

        probe_and_save(20)
        for epoch in range(21, 101):
            order = orders[epoch-1]
            batches = [order[i:i+cfg['batch_size']] for i in range(0, len(order), cfg['batch_size'])]
            state.update(epoch=epoch, phase='training', steps_completed=0, steps_per_epoch=len(batches))
            base.write_json(output/'run_state.json', state)
            before = time.monotonic(); model.train()
            loader = torch.utils.data.DataLoader(data, batch_sampler=batches, num_workers=cfg['num_workers'],
                                                pin_memory=True, generator=data_rng)
            sums = [0., 0., 0.]; count = 0
            for step_number, x in enumerate(loader, 1):
                x = x.cuda(non_blocking=True)
                values = step(model, optimizer, x, options['entropy_weight'], torch)
                for i, v in enumerate(values): sums[i] += v*len(x)
                count += len(x)
                if step_number == 1 or step_number%100 == 0:
                    state.update(steps_completed=step_number, heartbeat_utc=base.now())
                    base.write_json(output/'run_state.json', state)
            torch.cuda.synchronize(); seconds = time.monotonic()-before
            lr = optimizer.param_groups[0]['lr']; scheduler.step()
            assert count == 956 and scheduler.last_epoch == epoch
            history.append({'epoch': epoch, 'loss': sums[0]/count, 'mse': sums[1]/count,
                            'legacy_entropy': sums[2]/count, 'lr': lr, 'training_seconds': seconds,
                            'patches': count, 'optimizer_steps': len(batches)})
            base.write_json(output/'history.json', history)
            print(json.dumps(history[-1]), flush=True)
            state.update(completed_epochs=epoch, last_epoch_seconds=seconds, heartbeat_utc=base.now())
            base.write_json(output/'run_state.json', state)
            if epoch in PROBES: probe_and_save(epoch)
        state.update(phase='full_normal_training_audit')
        base.write_json(output/'run_state.json', state)
        full_indices = {}; offset = 0
        for entry in json.loads((root/'input_manifest.json').read_text()):
            full_indices[entry['patient']] = list(range(offset, offset+entry['patches']))
            offset += entry['patches']
        weights_before = base.state_digest(model.state_dict())
        full = measure(model, data, full_indices, options['entropy_weight'], options['memory_decay'])
        assert base.state_digest(model.state_dict()) == weights_before
        full.update(epoch=100, checked_utc=base.now(), probe_scope='All 956 NORMAL TRAINING patches, fixed epoch-100 checkpoint; no validation/test access.')
        base.write_json(output/'full_normal_training_epoch_100.json', full)
        assert {int(s['step']) for s in optimizer.state.values()} == {47800}
        state.update(status='complete', phase='complete', finished_utc=base.now(), completed_epochs=100,
                     steps_completed=478, total_optimizer_steps=47800)
    except BaseException as error:
        state.update(status='failed', finished_utc=base.now(), error=f'{type(error).__name__}: {error}')
        raise
    finally:
        base.write_json(output/'run_state.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'queue', 'train'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--job')
    args = parser.parse_args(); root = args.root.resolve()
    if args.command == 'prepare': prepare(root, args.parent.resolve())
    elif args.command == 'verify': verify(root, args.gpu)
    elif args.command == 'train': train(root, args.job, args.gpu)
    else: helper(root/'base_run.py').queue(root)


if __name__ == '__main__':
    main()
