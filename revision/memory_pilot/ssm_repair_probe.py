"""Read-only SSM diagnostics on fixed normal TRAIN patches."""
import math


def measure_ssm(model, optimizer, data, indices, implementation):
    import torch
    import torch.nn.functional as F

    def norm(x):
        return float(x.detach().double().norm())

    def rms(x):
        return float(x.detach().double().square().mean().sqrt())

    modules = {n: m for n, m in model.named_modules() if isinstance(m, implementation.SS3D)}
    decay = {id(p): g['weight_decay'] for g in optimizer.param_groups for p in g['params']}
    was_training = model.training
    model.eval()
    handles, functional, parameters = [], [], []
    active = {}
    original_scan = implementation.selective_scan_fn

    def observed_scan(**kw):
        out = original_scan(**kw)
        direct = kw['u'] * kw['D'][None, :, None]
        delta = kw['delta'].detach()
        functional[-1].update(
            delta_min=float(delta.min()), delta_mean=float(delta.mean()), delta_max=float(delta.max()),
            raw_recurrent_rms=rms(out-direct), raw_Du_rms=rms(direct),
            raw_recurrent_over_Du_rms=rms(out-direct)/max(rms(direct), 1e-30),
            slowest_one_token_carry_mean=float(torch.exp(delta * kw['A'].max(-1).values[None, :, None]).mean()))
        return out

    try:
        for name, module in modules.items():
            def pre(m, args, name=name):
                active['module'] = name
            handles.append(module.register_forward_pre_hook(pre))

            def delta_hook(m, args, out, name=name):
                bias = m.bias[None, :, None].expand_as(out)
                dt = F.softplus(out)
                functional.append({'patch_index': active['patch_index'], 'module': name,
                    'pre_softplus_equals_bias': torch.equal(out, bias),
                    'delta_equals_bias_only': torch.equal(dt, F.softplus(bias)),
                    'pre_softplus_input_contribution_maxabs': float((out-bias).abs().max()),
                    'delta_input_contribution_maxabs': float((dt-F.softplus(bias)).abs().max()),
                    'delta_token_std_rms': rms(dt.double().std(-1, correction=0))})
            handles.append(module.dt_proj.register_forward_hook(delta_hook))
        implementation.selective_scan_fn = observed_scan
        with torch.no_grad():
            for index in indices:
                active['patch_index'] = index
                model(data[[index]].cuda())
    finally:
        implementation.selective_scan_fn = original_scan
        for handle in handles:
            handle.remove()

    try:
        # Full parameter gradient norm is required to match trainer clipping.
        named = list(model.named_parameters())
        x = data[[indices[0]]].cuda()
        recon, _ = model(x)
        mse = F.mse_loss(recon, x)
        grads = torch.autograd.grad(mse, [p for _, p in named])
        total_norm = math.sqrt(sum(norm(g)**2 for g in grads))
        clip_scale = min(1., 1./(total_norm+1e-6))
        for (name, p), grad in zip(named, grads):
            if not any(name.startswith(n+'.') for n in modules):
                continue
            v = p.detach().double()
            tiny = torch.finfo(p.dtype).tiny
            moments = optimizer.state.get(p, {})
            parameters.append({'name': name, 'numel': p.numel(), 'l2_float64': norm(p),
                'maxabs': float(v.abs().max()), 'nonzero': int((v != 0).sum()),
                'nonzero_subnormal': int(((v != 0) & (v.abs() < tiny)).sum()),
                'task_grad_l2': norm(grad), 'task_grad_after_clip_l2': norm(grad)*clip_scale,
                'weight_decay': decay[id(p)], 'decay_times_parameter_l2': norm(p)*decay[id(p)],
                'adam_exp_avg_l2': norm(moments['exp_avg']) if 'exp_avg' in moments else None,
                'adam_exp_avg_sq_l2': norm(moments['exp_avg_sq']) if 'exp_avg_sq' in moments else None})
        rates = {name: {'exp_A_log_min': float(m.A_log.detach().exp().min()),
                        'exp_A_log_max': float(m.A_log.detach().exp().max()),
                        'D_l2': norm(m.D)} for name, m in modules.items()}
        assert len(functional) == len(indices)*len(modules)
        assert all(math.isfinite(r['l2_float64']) and math.isfinite(r['task_grad_l2']) for r in parameters)
        return {'scope': 'Fixed middle patch from first three NORMAL TRAIN patients; no test data',
            'patch_indices': indices, 'functional': functional, 'parameters': parameters,
            'rates': rates, 'gradient_probe_mse': float(mse.detach()),
            'all_parameter_task_gradient_l2': total_norm, 'clip_scale': clip_scale,
            'optimizer': type(optimizer).__name__,
            'decay_note': 'For Adam, decay*p is added after task-gradient clipping and before moments. '
                          'For AdamW it is a decoupled parameter shrinkage rate, not a gradient; '
                          'its actual direct parameter step is lr*decay*p. These are not equal-strength regularizers.'}
    finally:
        model.train(was_training)
