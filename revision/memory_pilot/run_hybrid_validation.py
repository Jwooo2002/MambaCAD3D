#!/usr/bin/env python3
"""Paired normal-TRAIN validation of SSM recurrence and CNN initialization."""
import argparse
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil


VARIANTS = {
    'V0_s3_replay': {'optimizer': 'Adam', 'exclude': 'ssm', 'decoder_fanin': False, 'direct_only': False},
    'V1_s3_decoder_fanin': {'optimizer': 'Adam', 'exclude': 'ssm', 'decoder_fanin': True, 'direct_only': False},
    'V2_adam_zero_decay': {'optimizer': 'Adam', 'exclude': 'all', 'decoder_fanin': False, 'direct_only': False},
    'V3_adamw_explicit_groups': {'optimizer': 'AdamW', 'exclude': 'explicit', 'decoder_fanin': False, 'direct_only': False},
    'V4_s3_direct_only': {'optimizer': 'Adam', 'exclude': 'ssm', 'decoder_fanin': False, 'direct_only': True},
    'V5_s3_fanin_direct_only': {'optimizer': 'Adam', 'exclude': 'ssm', 'decoder_fanin': True, 'direct_only': True},
}


def helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def excluded_names(model, option):
    if option == 'all':
        return {n for n, _ in model.named_parameters()}
    excluded = {'mem_rep.memory'}
    for name, p in model.named_parameters():
        if option == 'explicit' and (p.ndim == 1 or name.endswith('.bias') or name.endswith('.A_log')):
            excluded.add(name)
    if option == 'ssm':
        for prefix, module in model.named_modules():
            if type(module).__name__ == 'SS3D':
                excluded.update(prefix+'.'+n for n, _ in module.named_parameters())
    return excluded


def optimizer_for(model, variant, cfg, torch):
    option = VARIANTS[variant]
    excluded = excluded_names(model, option['exclude'])
    groups = []
    for flag in (False, True):
        params = [(n, p) for n, p in model.named_parameters() if (n in excluded) == flag]
        if params:
            groups.append({'params': [p for _, p in params], 'param_names': [n for n, _ in params],
                           'weight_decay': 0. if flag else 1e-5})
    return getattr(torch.optim, option['optimizer'])(groups, lr=cfg['lr'])


def engine_for(root, variant=None):
    engine = helper(root/'repair_run.py', 'repair_engine')
    engine.VARIANTS = VARIANTS
    engine.optimizer_for = optimizer_for
    original_imports = engine.imports

    def imports(path):
        np, torch, implementation = original_imports(path)
        if not hasattr(implementation, '_hybrid_original_scan'):
            implementation._hybrid_original_scan = implementation.selective_scan_fn
        implementation.selective_scan_fn = implementation._hybrid_original_scan
        if variant is not None and VARIANTS[variant]['direct_only']:
            def direct_scan(**kwargs):
                return kwargs['u'] * kwargs['D'][None, :, None]
            implementation.selective_scan_fn = direct_scan
        return np, torch, implementation

    engine.imports = imports
    return engine


def prepare(root, parent):
    if root.exists():
        raise RuntimeError('Refusing to overwrite experiment')
    base = helper(parent/'base_run.py', 'base')
    base.verify_files(parent)
    old = json.loads((parent/'manifest.json').read_text())
    root.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    for source, target in [(Path(__file__), 'run.py'), (here/'hybrid_probe.py', 'ssm_probe.py'),
                           (parent/'run.py', 'repair_run.py'), (parent/'base_run.py', 'base_run.py'),
                           (parent/'probe.py', 'probe.py')]:
        shutil.copy2(source, root/target)
    # The recurrence-off controls disconnect dt/x_proj/A. Report that explicitly.
    basic = (parent/'ssm_probe.py').read_text()
    old_grad = "grads = torch.autograd.grad(mse, [p for _, p in named])"
    assert basic.count(old_grad) == 1
    basic = basic.replace(old_grad, "raw_grads = torch.autograd.grad(mse, [p for _, p in named], allow_unused=True)\n        grads = [torch.zeros_like(p) if g is None else g for (_, p), g in zip(named, raw_grads)]\n        disconnected = {name for (name, _), g in zip(named, raw_grads) if g is None}")
    basic = basic.replace("'task_grad_l2': norm(grad),", "'task_gradient_connected': name not in disconnected, 'task_grad_l2': norm(grad),")
    (root/'basic_ssm_probe.py').write_text(basic)
    shutil.copytree(parent/'source_snapshot', root/'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('input_manifest.json', 'training_split.json'):
        shutil.copy2(parent/name, root/name)
    cfg = dict(old['config'])
    cfg.update(trainer_differences='Same frozen corrected model, normal TRAIN data, 200-epoch orders and '
               'base initialization as the SSM repair parent. Decoder-only initialization and recurrence-off '
               'controls are prespecified; no validation/disease/test selection.',
               rng_comparison_note='Exact copied parent initial state and orders. Fan-in arms rescale only nine '
               'decoder weight tensors, preserving their random directions; no other initial weights change.',
               mechanism_scope='normal TRAIN only; not final k-fold performance')
    base.write_json(root/'config.json', cfg)
    engine = engine_for(root)
    np, torch, implementation = engine.imports(root)
    states, initializer_rows = {}, []
    for seed in cfg['seeds']:
        for name in (f'initial_seed_{seed}.pt', f'orders_seed_{seed}.json'):
            shutil.copy2(parent/name, root/name)
        original = torch.load(root/f'initial_seed_{seed}.pt', map_location='cpu', weights_only=False)['model_state_dict']
        model = base.instantiate(implementation.MemAE_VSSM3D, cfg)
        model.load_state_dict(original)
        changed = []
        for name, module in model.named_modules():
            if not (name.startswith('decoder_stages.') or name == 'final_recon_conv'):
                continue
            if not isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
                continue
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            gain = math.sqrt(2.) if isinstance(module, torch.nn.Conv3d) and name != 'final_recon_conv' else 1.
            scale = gain/math.sqrt(fan_in)/.02
            with torch.no_grad():
                module.weight.mul_(scale)
            changed.append(name+'.weight')
            initializer_rows.append({'seed': seed, 'name': name+'.weight', 'fan_in_tensor_convention': fan_in,
                                     'gain': gain, 'scale_from_original': scale})
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        assert len(changed) == 9
        assert {k for k in state if not torch.equal(state[k], original[k])} == set(changed)
        torch.save({'model_state_dict': state, 'seed': seed, 'changed_keys': changed}, root/f'initial_fanin_seed_{seed}.pt')
        states[f'base_{seed}'] = base.state_digest(original)
        states[f'fanin_{seed}'] = base.state_digest(state)
    jobs = []
    for variant, options in VARIANTS.items():
        for seed in cfg['seeds']:
            name = f'{variant}_seed{seed}'
            (root/name).mkdir()
            tag = 'fanin' if options['decoder_fanin'] else 'base'
            initial = f"initial_{'fanin_' if options['decoder_fanin'] else ''}seed_{seed}.pt"
            jobs.append({'id': name, 'variant': variant, 'seed': seed, 'status': 'queued',
                         'initial_file': initial, 'initial_state_hash': states[f'{tag}_{seed}']})
    base.write_json(root/'initialization.json', {'rows': initializer_rows,
        'definition': 'Preserve initial random directions; hidden decoder Conv3d gain sqrt(2), '
        'ConvTranspose3d and output Conv3d gain1; tensor fan-in convention; biases and all encoder/memory weights unchanged. '
        'He-inspired candidate, not an exact U-Mamba/GELU-optimal initialization claim.'})
    frozen = {str(p.relative_to(root)): base.digest(p) for p in root.rglob('*') if p.is_file()}
    base.write_json(root/'source_hashes.json', frozen)
    base.write_json(root/'manifest.json', {'status': 'prepared', 'created_utc': base.now(),
        'parent_ssm_repair': str(parent), 'legacy_sweep': old['legacy_sweep'], 'config': cfg,
        'variants': VARIANTS, 'jobs': jobs, 'training_patches': old['training_patches'], 'total_epochs': 2400,
        'initial_state_hashes': states,
        'batch_order_hashes': {str(s): base.digest(root/f'orders_seed_{s}.json') for s in cfg['seeds']},
        'scope': 'Normal TRAIN mechanism validation. No external/disease/test access or automatic model adoption.',
        'comparisons': ['V0 vs parent S3 replay', 'V1-V0 decoder initialization', 'V4-V0 recurrence at baseline init',
                        'V5-V1 recurrence at fan-in init', 'V5-V4 initialization without recurrence',
                        'V2 vs parent S4 Adam0/AdamW1e-5', 'V3 vs parent S4 explicit grouping'],
        'decision_rule': 'Report all two-seed fixed-epoch results, task gradients and recurrent dependence; '
        'do not select a winner from normal TRAIN MSE or call it AUROC/AP improvement.',
        'gpu_policy': 'One training job per idle GPU; existing jobs finish first. Stop new launches on failure.',
        'runtime_prior': {'epoch_seconds': 34., 'estimated_wall_hours_two_gpus': 2400*34/2/3600}})
    print(json.dumps({'prepared': str(root), 'jobs': len(jobs), 'total_epochs': 2400}), flush=True)


def verify(root, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    base = helper(root/'base_run.py', 'base')
    base.verify_files(root)
    manifest = json.loads((root/'manifest.json').read_text())
    cfg = manifest['config']
    engine = engine_for(root)
    np, torch, implementation = engine.imports(root)
    probe = helper(root/'ssm_probe.py', 'hybrid_probe')
    data, patients = base.load_inputs(root, np, torch)
    indices = [v[len(v)//2] for v in list(patients.values())[:3]]
    for seed in cfg['seeds']:
        orders = json.loads((root/f'orders_seed_{seed}.json').read_text())
        assert len(orders) == 200 and all(sorted(o) == list(range(len(data))) for o in orders)
        assert base.digest(root/f'orders_seed_{seed}.json') == base.digest(Path(manifest['parent_ssm_repair'])/f'orders_seed_{seed}.json')
    rows = []
    for job in manifest['jobs']:
        engine = engine_for(root, job['variant'])
        _, _, implementation = engine.imports(root)
        model = base.instantiate(implementation.MemAE_VSSM3D, cfg).cuda()
        model.load_state_dict(torch.load(root/job['initial_file'], map_location='cpu', weights_only=False)['model_state_dict'])
        assert model.mem_rep.shrink_thres == cfg['shrink_thres']
        assert base.state_digest(model.state_dict()) == job['initial_state_hash']
        opt = optimizer_for(model, job['variant'], cfg, torch)
        params = [p for g in opt.param_groups for p in g['params']]
        assert len(params) == len({id(p) for p in params}) == len(list(model.parameters()))
        excluded = excluded_names(model, VARIANTS[job['variant']]['exclude'])
        assert all(g['weight_decay'] == (0. if n in excluded else 1e-5) for g in opt.param_groups for n in g['param_names'])
        x = data[:2].cuda()
        model.eval()
        with torch.no_grad():
            before_output = model(x)[0].clone()
        before = base.state_digest(model.state_dict())
        rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        if job['seed'] == 42:
            observed = probe.measure_ssm(model, opt, data, indices, implementation)
            assert before == base.state_digest(model.state_dict())
            assert torch.equal(rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
            with torch.no_grad():
                assert torch.equal(before_output, model(x)[0])
            assert all(p.grad is None for p in model.parameters())
            if VARIANTS[job['variant']]['direct_only']:
                assert all(r['differences']['reconstruction']['bitwise_equal'] for r in observed['hybrid']['recurrence_interventions'])
                assert all(not r['task_gradient_connected'] for r in observed['parameters'] if r['name'].endswith('dt_proj.weight'))
            base.write_json(root/job['id']/'verification_probe.json', observed)
        model.train()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            reconstruction, attention = model(x)
            loss = (reconstruction-x).square().mean()
            assert reconstruction.shape == x.shape and torch.isfinite(loss) and torch.isfinite(attention).all()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert base.state_digest(model.state_dict()) != before
        assert {int(s['step']) for s in opt.state.values()} == {3}
        buffer = io.BytesIO()
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location='cpu', weights_only=False)
        clone = base.instantiate(implementation.MemAE_VSSM3D, cfg).cuda()
        clone.load_state_dict(saved['model'])
        clone_opt = optimizer_for(clone, job['variant'], cfg, torch)
        clone_opt.load_state_dict(saved['optimizer'])
        assert base.state_digest(clone.state_dict()) == base.state_digest(model.state_dict())
        for p, q in zip(model.parameters(), clone.parameters()):
            assert set(opt.state.get(p, {})) == set(clone_opt.state.get(q, {}))
            for key, value in opt.state.get(p, {}).items():
                assert torch.equal(value.cpu(), clone_opt.state[q][key].cpu())
        model.eval(); clone.eval()
        with torch.no_grad():
            assert torch.equal(model(x)[0], clone(x)[0])
        row = {'job': job['id'], 'passed': True, 'loss_after_3_updates': float(loss.detach()),
               'probe_invariance_checked': job['seed'] == 42,
               'optimizer_groups': [{'weight_decay': g['weight_decay'], 'param_names': g['param_names']} for g in opt.param_groups]}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'optimizer_groups'}), flush=True)
        del model, clone, opt, clone_opt, saved, buffer, reconstruction, attention, loss
        torch.cuda.empty_cache()
    base.write_json(root/'verification.json', {'status': 'passed', 'checked_utc': base.now(),
                    'full_orders_match_parent': True, 'rows': rows})


def audit_parent(root, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    base = helper(root/'base_run.py', 'base')
    base.verify_files(root)
    manifest = json.loads((root/'manifest.json').read_text())
    parent = Path(manifest['parent_ssm_repair'])
    engine = engine_for(root)
    np, torch, implementation = engine.imports(root)
    probe = helper(root/'ssm_probe.py', 'hybrid_probe')
    data, patients = base.load_inputs(root, np, torch)
    gradient_indices = [v[len(v)//2] for v in list(patients.values())[:3]]
    reliance_indices = [i for indices in patients.values() for i in indices]
    output = root/'parent_checkpoint_audit'
    output.mkdir(exist_ok=True)
    records = []
    for variant in ('S0_control', 'S3_all_ssm_no_decay', 'S4_adamw'):
        for seed in manifest['config']['seeds']:
            job = f'{variant}_seed{seed}'
            checkpoint = parent/job/'epoch_200.pt'
            assert json.loads((parent/job/'run_state.json').read_text())['status'] == 'complete'
            destination = output/f'{job}.json'
            if destination.exists():
                raise RuntimeError('Refusing to overwrite checkpoint audit')
            checkpoint_hash = base.digest(checkpoint)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model = base.instantiate(implementation.MemAE_VSSM3D, manifest['config']).cuda()
            model.load_state_dict(saved['model_state_dict'])
            model.eval()
            before = base.state_digest(model.state_dict())
            rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            record = probe.measure_hybrid(model, data, gradient_indices, implementation)
            record['recurrence_interventions_45patch'] = probe.recurrent_dependence(model, data, reliance_indices, implementation, stagewise=True)
            assert before == base.state_digest(model.state_dict()) and checkpoint_hash == base.digest(checkpoint)
            assert torch.equal(rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
            record.update(job=job, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                          checked_utc=base.now(), state_and_rng_unchanged=True)
            base.write_json(destination, record)
            records.append(str(destination))
            print(json.dumps({'audited': job, 'gradient_patches': len(gradient_indices), 'intervention_patches': len(reliance_indices)}), flush=True)
            del model, saved, record
            torch.cuda.empty_cache()
    base.write_json(output/'summary.json', {'status': 'complete', 'records': records,
                    'scope': 'Fixed normal TRAIN only; 45patch intervention and 3patch gradient traces; not model selection.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'verify', 'audit-parent', 'train', 'queue', 'summarize'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--gpu', default='0')
    p.add_argument('--job')
    args = p.parse_args()
    root = args.root.resolve()
    if args.command == 'prepare':
        prepare(root, args.parent.resolve())
    elif args.command == 'verify':
        verify(root, args.gpu)
    elif args.command == 'audit-parent':
        audit_parent(root, args.gpu)
    elif args.command == 'train':
        manifest = json.loads((root/'manifest.json').read_text())
        variant = next(j['variant'] for j in manifest['jobs'] if j['id'] == args.job)
        engine_for(root, variant).train(root, args.job, args.gpu)
    elif args.command == 'summarize':
        engine_for(root).summarize(root)
    else:
        helper(root/'base_run.py', 'base').queue(root)
        engine_for(root).summarize(root)


if __name__ == '__main__':
    main()
