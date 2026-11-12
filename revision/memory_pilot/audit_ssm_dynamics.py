#!/usr/bin/env python3
"""Read-only SSM diagnostics on three prespecified NORMAL TRAIN patches."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')


def main(root, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    project = Path(__file__).resolve().parent.parent
    saved_eval = project / 'outputs/memory_fixed_epoch100_eval_20261110T150000Z'
    source = saved_eval / 'source_snapshot/main_files'
    sys.path.insert(0, str(source))
    import model as implementation
    from utils_scan import single_axis_spiral_scan
    protocol = read(saved_eval / 'protocol.json')
    for name, expected in protocol['source_python_sha256'].items():
        assert digest(source / name) == expected
    assert not root.exists(), 'Refusing overwrite'
    root.mkdir(parents=True)
    shutil.copy2(__file__, root / Path(__file__).name)
    training = read(saved_eval / 'training_input_manifest.json')[:3]
    inputs, selection = [], []
    for entry in training:
        assert digest(entry['path']) == entry['sha256']
        with open(entry['path'], 'rb') as f:
            clips = pickle.load(f)
        index = (len(clips)-1)//2
        x = (torch.from_numpy(np.asarray(clips[index], dtype=np.float32))[None, None]+300)/700
        assert x.shape == (1, 1, 64, 32, 32) and torch.isfinite(x).all()
        inputs.append(x.cuda())
        selection.append(dict(path=entry['path'], sha256=entry['sha256'], patch_index=index))
    write(root / 'protocol.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'Read-only, three NORMAL TRAIN patients, one fixed middle patch each; no test subjects or scoring.',
          'selection': selection, 'jobs': protocol['jobs'], 'source': str(source),
          'source_hashes': protocol['source_python_sha256'],
          'delta_summary': 'All channels and 512 equally spaced sequence tokens.',
          'state_carry_summary': '512 fixed pairs at same (y,x) in adjacent z planes in each direction; slowest A mode per channel.',
          'limitation': 'Direct single-layer state carry, not full network receptive field or feature attribution.'})

    def describe(x):
        x = x.detach().double().flatten()
        q = torch.quantile(x, torch.tensor([0., .05, .5, .95, 1.], device=x.device, dtype=x.dtype))
        return dict(zip(('min', 'p05', 'median', 'p95', 'max'), map(float, q)))

    def rms(x):
        return float(x.detach().double().square().mean().sqrt())

    def state_hash(model):
        h = hashlib.sha256()
        for name, value in model.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    rows = []
    original_scan = implementation.selective_scan_fn
    for job in protocol['jobs']:
        assert digest(job['checkpoint']) == job['checkpoint_sha256']
        checkpoint = torch.load(job['checkpoint'], map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        model = implementation.MemAE_VSSM3D(**{k: cfg[k] for k in ('in_chans','dims','depths','ssm_d_state','mem_dim')}).cuda().eval()
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        before = state_hash(model)
        active, records, hooks = {}, [], []
        for name, module in model.named_modules():
            if isinstance(module, implementation.SS3D):
                def pre(m, args, name=name):
                    active.update(name=name, shape=list(args[0].shape))
                hooks.append(module.register_forward_pre_hook(pre))
            elif isinstance(module, implementation.VSSBlock3D):
                def post(m, args, out, name=name):
                    records[-1]['block_residual_branch_over_input_rms'] = rms(out-args[0])/max(rms(args[0]), 1e-30)
                hooks.append(module.register_forward_hook(post))

        def observed_scan(**kwargs):
            out = original_scan(**kwargs)
            u, delta, A, D = [kwargs[k] for k in ('u', 'delta', 'A', 'D')]
            assert kwargs['delta_softplus'] is False and kwargs['delta_bias'] is None
            _, _, z, h, w = active['shape']
            channels = u.shape[1]//2
            token_index = torch.linspace(0, u.shape[-1]-1, min(512,u.shape[-1]), device=u.device).long().unique()
            dt = delta[..., token_index]
            slowest_rate = (-A).min(-1).values
            direct = u*D[None, :, None]
            recurrent = out-direct
            local_log10_carry = -dt*slowest_rate[None, :, None]/np.log(10.)
            # Trace exactly corresponding spatial voxels in adjacent planes through both scan permutations.
            ids = torch.arange(z*h*w, device=u.device).reshape(1, 1, z, h, w)
            sequences = single_axis_spiral_scan(ids)[0, :, 0]
            inverse = torch.argsort(sequences, dim=-1)
            pair = torch.linspace(0, (z-1)*h*w-1, min(512,(z-1)*h*w), device=u.device).long().unique()
            prefix = delta.double().cumsum(-1).reshape(1,2,channels,z*h*w)
            plane_carry, plane_distances = [], []
            for direction in (0,1):
                a, b = inverse[direction, pair], inverse[direction, pair+h*w]
                lo, hi = torch.minimum(a,b), torch.maximum(a,b)
                # Product from earlier h_lo to later h_hi uses delta[lo+1:hi+1].
                total_dt = prefix[:,direction,:,hi]-prefix[:,direction,:,lo]
                rate = slowest_rate.reshape(2,channels)[direction]
                plane_carry.append(-total_dt*rate[None,:,None]/np.log(10.))
                plane_distances.append(hi-lo)
            records.append({'patch_index': active['patch_index'], 'module': active['name'],
                            'input_shape': active['shape'], 'delta': describe(dt),
                            'A_decay_rate': describe(-A),
                            'slowest_state_one_step_log10_carry': describe(local_log10_carry),
                            'adjacent_z_same_xy_scan_distance': describe(torch.cat(plane_distances)),
                            'adjacent_z_slowest_state_log10_carry': describe(torch.cat(plane_carry)),
                            'raw_recurrent_over_Du_rms': rms(recurrent)/max(rms(direct),1e-30),
                            'raw_Du_rms': rms(direct), 'raw_recurrent_rms': rms(recurrent)})
            return out

        equality = []
        with torch.no_grad():
            for index, x in enumerate(inputs):
                # Baseline forward has metadata hooks but records are populated only when observed_scan runs.
                for hook in hooks:
                    hook.remove()
                hooks = []
                reference, ref_att = model(x)
                for name, module in model.named_modules():
                    if isinstance(module, implementation.SS3D):
                        def pre(m,args,name=name): active.update(name=name,shape=list(args[0].shape))
                        hooks.append(module.register_forward_pre_hook(pre))
                    elif isinstance(module, implementation.VSSBlock3D):
                        def post(m,args,out): records[-1]['block_residual_branch_over_input_rms']=rms(out-args[0])/max(rms(args[0]),1e-30)
                        hooks.append(module.register_forward_hook(post))
                active['patch_index'] = index
                implementation.selective_scan_fn = observed_scan
                observed, att = model(x)
                implementation.selective_scan_fn = original_scan
                assert torch.equal(reference, observed) and torch.equal(ref_att, att)
                equality.append(True)
        for hook in hooks:
            hook.remove()
        assert len(records) == 18 and before == state_hash(model)
        assert digest(job['checkpoint']) == job['checkpoint_sha256']
        row = {'job': job['id'], 'checkpoint_sha256':job['checkpoint_sha256'],
               'weights_and_outputs_unchanged': all(equality), 'state_sha256': before, 'ssm_records': records}
        rows.append(row)
        write(root / f"{job['id']}.json", row)
        print(json.dumps({'job':job['id'],'status':'completed','ssm_observations':len(records)}), flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
    write(root / 'results.json', {'completed_utc':datetime.now(timezone.utc).isoformat(),
          'status':'passed','jobs':rows,'scope':'3 normal training patches; no training or disease/test evaluation',
          'hook_outputs_and_weights_unchanged':True})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    main(args.root.resolve(),args.gpu)
