"""Scalar-only diagnostics of decoder/memory gradients and recurrent dependence."""
from contextlib import contextmanager
import importlib.util
import math
from pathlib import Path


def _stats(tensor):
    if tensor is None:
        return {'connected': False, 'rms': 0., 'l2': 0., 'maxabs': 0.}
    value = tensor.detach().double()
    return {'connected': True, 'rms': float(value.square().mean().sqrt()),
            'l2': float(value.norm()), 'maxabs': float(value.abs().max())}


def gradient_trace(model, x, implementation):
    """Autograd.grad leaves parameter .grad buffers untouched."""
    import torch
    features, handles = {}, []

    def capture(name):
        def hook(module, inputs, output):
            if name == 'memory':
                features['memory.query'] = inputs[0]
                features['memory.readout'] = output[0]
            else:
                features[name] = output
        return hook

    for name, module in model.named_modules():
        if (name == 'stem' or name == 'final_recon_conv' or
                (name.startswith('encoder_stages.') and name.count('.') == 1) or
                (name.startswith('decoder_stages.') and name.count('.') == 2) or
                isinstance(module, implementation.SS3D)):
            handles.append(module.register_forward_hook(capture(name)))
        elif name == 'mem_rep':
            handles.append(module.register_forward_hook(capture('memory')))
    try:
        reconstruction, _ = model(x)
        loss = (reconstruction-x).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite gradient-probe loss')
        tensors = list({id(v): v for v in features.values()}.values())
        named = list(model.named_parameters())
        grads = torch.autograd.grad(loss, tensors+[p for _, p in named], allow_unused=True)
        feature_grads = {id(v): g for v, g in zip(tensors, grads[:len(tensors)])}
        rows = [{'name': name, 'shape': list(value.shape),
                 'activation': _stats(value), 'task_gradient': _stats(feature_grads[id(value)])}
                for name, value in features.items()]
        parameters = [{'name': name, 'numel': p.numel(), 'value': _stats(p),
                       'task_gradient': _stats(g)}
                      for (name, p), g in zip(named, grads[len(tensors):])]
        return {'mse': float(loss.detach()), 'features': rows, 'parameters': parameters,
                'total_task_gradient_l2': math.sqrt(sum(r['task_gradient']['l2']**2 for r in parameters)),
                'scope': 'MSE task gradient before optimizer decay and clipping; RMS normalizes tensor element counts.'}
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def recurrent_scale(model, implementation, scale, selected=None):
    """Change only raw scan recurrence; preserve D*u and every surrounding path."""
    original = implementation.selective_scan_fn
    active, handles = {}, []
    modules = {n: m for n, m in model.named_modules() if isinstance(m, implementation.SS3D)}
    selected = set(modules) if selected is None else set(selected)
    if not selected.issubset(modules):
        raise ValueError('Unknown SSM intervention target')
    for name, module in modules.items():
        def before(module, args, name=name):
            active['name'] = name
        handles.append(module.register_forward_pre_hook(before))

    def scan(**kwargs):
        if active['name'] not in selected or scale == 1.:
            return original(**kwargs)
        direct = kwargs['u'] * kwargs['D'][None, :, None]
        if scale == 0.:
            return direct
        return direct + scale*(original(**kwargs)-direct)

    implementation.selective_scan_fn = scan
    try:
        yield
    finally:
        implementation.selective_scan_fn = original
        for handle in handles:
            handle.remove()


def _outputs(model, x):
    query = model.encode(x)
    readout, attention = model.mem_rep(query)
    return {'query': query, 'readout': readout, 'attention': attention,
            'reconstruction': model.decode(readout)}


def recurrent_dependence(model, data, indices, implementation, stagewise=False):
    import torch
    modules = [n for n, m in model.named_modules() if isinstance(m, implementation.SS3D)]
    interventions = {'all_recurrence_off': modules}
    if stagewise:
        for stage in range(len(model.encoder_stages)):
            interventions[f'stage_{stage}_recurrence_off'] = [n for n in modules if n.startswith(f'encoder_stages.{stage}.')]
    rows = []
    with torch.no_grad():
        for index in indices:
            x = data[[index]].cuda()
            baseline = _outputs(model, x)
            baseline_mse = float((baseline['reconstruction']-x).square().mean())
            with recurrent_scale(model, implementation, 1.):
                noop = _outputs(model, x)
            assert all(torch.equal(noop[k], baseline[k]) for k in baseline), 'No-op intervention changed output'
            for name, selected in interventions.items():
                with recurrent_scale(model, implementation, 0., selected):
                    altered = _outputs(model, x)
                mse = float((altered['reconstruction']-x).square().mean())
                row = {'patch_index': index, 'intervention': name, 'selected_modules': selected,
                       'baseline_mse': baseline_mse, 'intervened_mse': mse,
                       'mse_ratio': mse/baseline_mse if baseline_mse > 0 else None,
                       'noop_bitwise_equal': True, 'differences': {}}
                for key in baseline:
                    diff, reference = _stats(altered[key]-baseline[key]), _stats(baseline[key])
                    diff['relative_rms'] = diff['rms']/reference['rms'] if reference['rms'] > 0 else None
                    diff['bitwise_equal'] = torch.equal(altered[key], baseline[key])
                    row['differences'][key] = diff
                assert math.isfinite(mse)
                rows.append(row)
    return rows


def measure_hybrid(model, data, indices, implementation, stagewise=False):
    import torch
    was_training = model.training
    model.eval()
    try:
        gradients = []
        for index in indices:
            row = gradient_trace(model, data[[index]].cuda(), implementation)
            row['patch_index'] = index
            gradients.append(row)
        dependence = recurrent_dependence(model, data, indices, implementation, stagewise)
        return {'patch_indices': indices, 'gradient_traces': gradients, 'recurrence_interventions': dependence,
                'interpretation': 'Fixed NORMAL TRAIN patches. Inference intervention measures current dependence, '
                'not the benefit of recurrence after retraining. No diagnosis or AUROC/AP inference.'}
    finally:
        model.train(was_training)


def measure_ssm(model, optimizer, data, indices, implementation):
    """Frozen experiment entry point, adding diagnostics to the existing probe."""
    path = Path(__file__).with_name('basic_ssm_probe.py')
    spec = importlib.util.spec_from_file_location('basic_ssm_probe', path)
    basic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(basic)
    result = basic.measure_ssm(model, optimizer, data, indices, implementation)
    result['hybrid'] = measure_hybrid(model, data, indices, implementation)
    return result
