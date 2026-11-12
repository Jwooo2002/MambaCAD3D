#!/usr/bin/env python3
"""Finite M3 pilot: entropy off plus memory-only decay off, paired to M1."""
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

VARIANT = 'M3_entropy_and_memory_decay_off'
OPTIONS = {'entropy_weight': 0., 'memory_decay': 0.}
PROBES = [0, 1, 3, 5, 10, 20, 25, 30, 40, 50, 60, 75, 100]


def helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def optimizer_for(model, cfg, torch):
    memory = model.mem_rep.memory
    other = [p for p in model.parameters() if p is not memory]
    return torch.optim.Adam([{'params': other, 'weight_decay': 1e-5},
                             {'params': [memory], 'weight_decay': 0.}], lr=cfg['lr'])


def geometry(memory, torch):
    m = memory.detach().cpu().double()
    norms = m.norm(dim=1)
    distances = torch.cdist(m, m, compute_mode='donot_use_mm_for_euclid_dist')
    relative = distances/((norms[:, None]+norms[None, :])/2).clamp_min(1e-12)
    mask = torch.eye(len(m), dtype=torch.bool)
    nearest = distances.masked_fill(mask, float('inf')).min(1).values
    near_relative = relative.masked_fill(mask, float('inf')).min(1).values
    return {'relative_distance_definition': 'L2 distance divided by mean of the two row norms, denominator clamped at 1e-12',
            'max_neighbourhood_including_self_by_absolute_l2': {str(t): int((distances <= t).sum(1).max()) for t in (1e-6, 1e-5, 1e-4, 1e-3)},
            'max_neighbourhood_including_self_by_relative_l2': {str(t): int((relative <= t).sum(1).max()) for t in (1e-3, 1e-2, .1)},
            'nearest_other_absolute_l2_min_median_max': [float(nearest.min()), float(nearest.median()), float(nearest.max())],
            'nearest_other_relative_l2_min_median_max': [float(near_relative.min()), float(near_relative.median()), float(near_relative.max())]}


def prepare(root, parent, control):
    if root.exists():
        raise RuntimeError('Refusing to overwrite an existing M3 run')
    base = helper(parent/'run.py', 'parent_helper')
    base.verify_files(parent)
    base.verify_files(control)
    old = json.loads((parent/'manifest.json').read_text())
    reference = json.loads((control/'manifest.json').read_text())
    assert old['status'] == reference['status'] == 'complete'
    assert Path(reference['parent_pilot']) == parent
    root.mkdir(parents=True)
    shutil.copy2(__file__, root/'run.py')
    shutil.copy2(parent/'run.py', root/'base_run.py')
    shutil.copy2(control/'run.py', root/'training_utils.py')
    shutil.copy2(parent/'probe.py', root/'probe.py')
    shutil.copytree(parent/'source_snapshot', root/'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('input_manifest.json', 'training_split.json'):
        shutil.copy2(parent/name, root/name)
    np, torch, cls = base.imports(root)
    cfg = dict(old['config'])
    cfg.update(epochs=100, probe_epochs=PROBES,
               trainer_differences='Fresh M3 run from saved paired initialization; exact 100-epoch M1 order stream; memory-only Adam decay zero and entropy zero; no validation/test/early stopping.',
               rng_comparison_note='Match fixed phase RNG and loader seeds, including the M1 epoch-20 phase boundary. No active stochastic forward layers. Prior checkpoints lacked boundary RNG; GPU backward has rounding-level nondeterminism; no bitwise uninterrupted-run equivalence claimed.')
    assert cfg['scheduler_T_max'] == 1000 and cfg['mem_dim'] == 300
    base.write_json(root/'config.json', cfg)
    refs = root/'control_reference'; refs.mkdir()
    jobs = []
    for seed in cfg['seeds']:
        for name, source in [(f'initial_seed_{seed}.pt', parent), (f'orders_seed_{seed}.json', control)]:
            shutil.copy2(source/name, root/name)
        initial = torch.load(root/f'initial_seed_{seed}.pt', map_location='cpu', weights_only=False)
        assert base.state_digest(initial['model_state_dict']) == old['initial_state_hashes'][str(seed)]
        orders = json.loads((root/f'orders_seed_{seed}.json').read_text())
        assert len(orders) == 100 and all(sorted(o) == list(range(956)) for o in orders)
        assert orders[:20] == json.loads((parent/f'orders_seed_{seed}.json').read_text())
        ref0 = parent/f'M1_entropy_off_seed{seed}'/'probe_epoch_00.json'
        ref100 = control/f'M1_entropy_off_seed{seed}'/'full_normal_training_epoch_100.json'
        shutil.copy2(ref0, refs/f'M1_seed{seed}_probe_epoch_000.json')
        shutil.copy2(ref100, refs/f'M1_seed{seed}_full_normal_training_epoch_100.json')
        checkpoint_path = control/f'M1_entropy_off_seed{seed}'/'epoch_100.pt'
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        base.write_json(refs/f'M1_seed{seed}_geometry_epoch_100.json', geometry(ck['model_state_dict']['mem_rep.memory'], torch))
        name = f'{VARIANT}_seed{seed}'; (root/name).mkdir()
        jobs.append({'id': name, 'variant': VARIANT, 'seed': seed, 'status': 'queued',
                     'control_checkpoint': str(checkpoint_path), 'control_checkpoint_sha256': base.digest(checkpoint_path)})
        del ck, initial
    hashes = {str(p.relative_to(root)): base.digest(p) for p in root.rglob('*') if p.is_file()}
    base.write_json(root/'source_hashes.json', hashes)
    base.write_json(root/'manifest.json', {'status': 'prepared', 'created_utc': base.now(),
                    'parent_pilot': str(parent), 'control_continuation': str(control),
                    'baseline': old['baseline'], 'legacy_sweep': old['legacy_sweep'],
                    'config': cfg, 'variants': {VARIANT: OPTIONS}, 'jobs': jobs,
                    'initial_state_hashes': old['initial_state_hashes'],
                    'batch_order_hashes': reference['batch_order_hashes'],
                    'training_patches': 956, 'total_epochs': 200,
                    'scope': 'Two fresh 100-epoch M3 normal-training-only diagnostic jobs. No disease/validation/test evaluation or final adoption.',
                    'gpu_policy': 'One job per idle GPU 0/1; two finite jobs; stop new launches on a failure.'})
    print(json.dumps({'prepared': str(root), 'jobs': 2, 'epochs_each': 100}), flush=True)


def verify(root, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    base = helper(root/'base_run.py', 'base_helper')
    util = helper(root/'training_utils.py', 'training_helper')
    base.verify_files(root)
    np, torch, cls = base.imports(root)
    sys.path.insert(0, str(root))
    from probe import measure
    manifest = json.loads((root/'manifest.json').read_text()); cfg = manifest['config']
    data, probes = base.load_inputs(root, np, torch)
    rows = []
    for job in manifest['jobs']:
        seed = job['seed']
        initial = torch.load(root/f'initial_seed_{seed}.pt', map_location='cpu', weights_only=False)['model_state_dict']
        model = base.instantiate(cls, cfg).cuda(); model.load_state_dict(initial)
        ref = base.instantiate(cls, cfg).cuda(); ref.load_state_dict(initial)
        assert base.state_digest(model.state_dict()) == manifest['initial_state_hashes'][str(seed)]
        for module in model.modules():
            if hasattr(module, 'drop_prob'): assert module.drop_prob == 0.
            if isinstance(module, torch.nn.modules.dropout._DropoutNd): assert module.p == 0.
        opt = optimizer_for(model, cfg, torch)
        refopt = base.optimizer_for(ref, 'M1_entropy_off', cfg, torch)
        covered = [p for group in opt.param_groups for p in group['params']]
        assert len(covered) == len({id(p) for p in covered}) == len(list(model.parameters()))
        for group in opt.param_groups:
            for p in group['params']: assert group['weight_decay'] == (0. if p is model.mem_rep.memory else 1e-5)
        assert not opt.state
        before = base.state_digest(model.state_dict())
        cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state().clone()
        probe = measure(model, data, probes, 0., 0.)
        assert before == base.state_digest(model.state_dict())
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        prior = json.loads((root/'control_reference'/f'M1_seed{seed}_probe_epoch_000.json').read_text())
        for k in ('mse', 'effective_slots_exp_mean_query_entropy', 'global_effective_slots'):
            assert np.isclose(probe[k], prior[k], rtol=1e-6, atol=1e-9), k
        orders = json.loads((root/f'orders_seed_{seed}.json').read_text())
        assert len(orders) == 100 and all(sorted(o) == list(range(len(data))) for o in orders)
        assert base.digest(root/f'orders_seed_{seed}.json') == manifest['batch_order_hashes'][str(seed)]
        x = data[orders[0][:2]].cuda()
        with torch.no_grad(): util.assert_same(model(x), ref(x), torch)
        model.zero_grad(set_to_none=True)
        recon, attention = model(x)
        mse = (recon-x).square().mean()
        entropy = -(attention*torch.log(attention+1e-12)).sum(1).mean()
        loss = mse+0.*entropy
        assert torch.equal(loss, mse) and torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        # Shared gradients isolate the parameter-group intervention from CUDA
        # backward rounding differences already documented in the M1 pilot.
        for p, q in zip(model.parameters(), ref.parameters()): q.grad = p.grad.detach().clone()
        opt.step(); refopt.step()
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert all(torch.isfinite(p).all() for p in model.parameters())
        nonmemory = max(float((p-ref.state_dict()[k]).abs().max()) for k, p in model.state_dict().items() if k != 'mem_rep.memory')
        assert nonmemory <= 1e-7, nonmemory
        memory_difference = float((model.mem_rep.memory-ref.mem_rep.memory).abs().max())
        assert memory_difference > 0
        assert {int(v['step']) for v in opt.state.values()} == {1}
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=cfg['eta_min'])
        buf = io.BytesIO(); torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict()}, buf)
        buf.seek(0); saved = torch.load(buf, map_location='cpu', weights_only=False)
        ref.load_state_dict(saved['model']); reloaded = optimizer_for(ref, cfg, torch)
        resumed = torch.optim.lr_scheduler.CosineAnnealingLR(reloaded, T_max=1000, eta_min=cfg['eta_min'])
        reloaded.load_state_dict(saved['optimizer']); resumed.load_state_dict(saved['scheduler'])
        util.assert_same(opt.state_dict(), reloaded.state_dict(), torch)
        util.assert_same(model.state_dict(), ref.state_dict(), torch)
        util.assert_same(scheduler.state_dict(), resumed.state_dict(), torch)
        rows.append({'seed': seed, 'initial_state_and_output_match_M1': True,
                     'initial_probe_matches_M1': True, 'probe_preserves_state_and_rng': True,
                     'only_memory_decay_removed': True, 'nonmemory_update_max_difference_shared_gradients': nonmemory,
                     'memory_update_max_difference_shared_gradients': memory_difference,
                     'finite_update_and_no_rng_consumption': True, 'checkpoint_roundtrip_exact': True,
                     'all_100_orders_match_M1_and_have_full_coverage': True})
        print(json.dumps(rows[-1]), flush=True)
        del model, ref, opt, refopt, reloaded, scheduler, resumed, initial, saved
        torch.cuda.empty_cache()
    base.write_json(root/'verification.json', {'status': 'passed', 'checked_utc': base.now(), 'rows': rows})


def train(root, job_id, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    base = helper(root/'base_run.py', 'base_helper')
    util = helper(root/'training_utils.py', 'training_helper')
    base.verify_files(root)
    np, torch, cls = base.imports(root)
    sys.path.insert(0, str(root))
    from probe import measure
    manifest = json.loads((root/'manifest.json').read_text()); cfg = manifest['config']
    job = next(j for j in manifest['jobs'] if j['id'] == job_id); output = root/job_id; seed = job['seed']
    state = {'status': 'running', 'phase': 'loading', 'started_utc': base.now(), 'pid': os.getpid(),
             'gpu': gpu, 'variant': VARIANT, 'seed': seed, 'epoch': 0, 'completed_epochs': 0, 'total_epochs': 100}
    base.write_json(output/'run_state.json', state)
    try:
        base.set_seed(seed, np, torch)
        data, probes = base.load_inputs(root, np, torch)
        model = base.instantiate(cls, cfg).cuda()
        initial = torch.load(root/f'initial_seed_{seed}.pt', map_location='cpu', weights_only=False)['model_state_dict']
        model.load_state_dict(initial)
        assert base.state_digest(model.state_dict()) == manifest['initial_state_hashes'][str(seed)]
        del initial
        optimizer = optimizer_for(model, cfg, torch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=cfg['eta_min'])
        orders = json.loads((root/f'orders_seed_{seed}.json').read_text())
        base.set_seed(seed+2000003, np, torch)
        data_rng = torch.Generator().manual_seed(seed+3000003)
        history = []

        def probe_and_save(epoch):
            state.update(phase='normal_training_probe', epoch=epoch)
            base.write_json(output/'run_state.json', state)
            weights = base.state_digest(model.state_dict())
            with torch.random.fork_rng(devices=[0]):
                record = measure(model, data, probes, 0., 0.)
            assert weights == base.state_digest(model.state_dict())
            record.update(epoch=epoch, checked_utc=base.now(), memory_geometry=geometry(model.mem_rep.memory, torch))
            base.write_json(output/f'probe_epoch_{epoch:03d}.json', record)
            saved = {'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                     'seed': seed, 'epoch': epoch, 'variant': VARIANT, 'config': cfg, 'intervention': OPTIONS,
                     'initial_state_hash': manifest['initial_state_hashes'][str(seed)],
                     'batch_order_hash': manifest['batch_order_hashes'][str(seed)],
                     'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                   'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state(),
                                   'data_loader': data_rng.get_state()}}
            tmp = output/f'epoch_{epoch:03d}.pt.tmp'; torch.save(saved, tmp); tmp.replace(output/f'epoch_{epoch:03d}.pt')
            print(json.dumps({'probe_epoch': epoch, 'mse': record['mse'],
                              'effective_slots': record['effective_slots_exp_mean_query_entropy'],
                              'duplicates': record['largest_duplicate_neighbourhood_l2_1e_minus6']}), flush=True)

        probe_and_save(0)
        for epoch, order in enumerate(orders, 1):
            if epoch == 21:
                base.set_seed(seed+2000003, np, torch)
                data_rng = torch.Generator().manual_seed(seed+4000003)
            batches = [order[i:i+cfg['batch_size']] for i in range(0, len(order), cfg['batch_size'])]
            state.update(epoch=epoch, phase='training', steps_completed=0, steps_per_epoch=len(batches))
            base.write_json(output/'run_state.json', state)
            before = time.monotonic(); model.train()
            loader = torch.utils.data.DataLoader(data, batch_sampler=batches, num_workers=cfg['num_workers'], pin_memory=True, generator=data_rng)
            sums = [0., 0., 0.]; count = 0
            for number, x in enumerate(loader, 1):
                x = x.cuda(non_blocking=True)
                values = util.step(model, optimizer, x, 0., torch)
                for i, value in enumerate(values): sums[i] += value*len(x)
                count += len(x)
                if number == 1 or number%100 == 0:
                    state.update(steps_completed=number, heartbeat_utc=base.now())
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
        state.update(phase='full_normal_training_audit'); base.write_json(output/'run_state.json', state)
        full_indices = {}; offset = 0
        for entry in json.loads((root/'input_manifest.json').read_text()):
            full_indices[entry['patient']] = list(range(offset, offset+entry['patches'])); offset += entry['patches']
        before = base.state_digest(model.state_dict())
        full = measure(model, data, full_indices, 0., 0.)
        assert before == base.state_digest(model.state_dict())
        full.update(epoch=100, checked_utc=base.now(), memory_geometry=geometry(model.mem_rep.memory, torch),
                    probe_scope='All 956 NORMAL TRAINING patches; fixed epoch 100; no validation/test access.')
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
    parser.add_argument('--control', type=Path)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--job')
    args = parser.parse_args(); root = args.root.resolve()
    if args.command == 'prepare': prepare(root, args.parent.resolve(), args.control.resolve())
    elif args.command == 'verify': verify(root, args.gpu)
    elif args.command == 'train': train(root, args.job, args.gpu)
    else: helper(root/'base_run.py', 'base_helper').queue(root)


if __name__ == '__main__':
    main()
