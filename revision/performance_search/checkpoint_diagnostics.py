#!/usr/bin/env python3
"""CPU-only weight diagnostics, with no inference, selection or checkpoint changes."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root, checkpoint):
    torch.set_num_threads(1)
    checkpoint_hash=sha(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    job=saved['job']
    pair=f'{job["stage"]}_f{job["fold"]}_s{job["seed"]}'
    initial=torch.load(root/f'{pair}_initial.pt',map_location='cpu',weights_only=True)
    state=saved['model_state_dict']
    assert saved['epoch'] in (50,100,150,200)
    assert set(state)==set(initial)
    assert all(torch.isfinite(v).all() for v in state.values())
    memory=state['mem_rep.memory'].double()
    norm=memory.norm(dim=1)
    centered=memory-memory.mean(0,keepdim=True)
    singular=torch.linalg.svdvals(centered)
    energy=singular.square()
    probabilities=energy/energy.sum().clamp_min(1e-300)
    effective_rank=float((-(probabilities*probabilities.clamp_min(1e-300).log()).sum()).exp()) if float(energy.sum())>0 else 0.
    unit=memory/norm[:,None].clamp_min(1e-300)
    cosines=(unit@unit.T)[torch.triu(torch.ones(len(memory),len(memory),dtype=torch.bool),diagonal=1)]
    params=[]
    for name,value in state.items():
        if any(part in name for part in ('.x_proj.','.dt_proj.','.A_log','.D')):
            assert value.dtype==torch.float32
            v=value.double()
            row={'name':name,'l2':float(v.norm()),'initial_l2':float(initial[name].double().norm()),
                 'change_l2':float((v-initial[name].double()).norm()),
                 'zero_fraction':float((value==0).float().mean()),
                 'float32_subnormal_fraction':float(((value.abs()<torch.finfo(value.dtype).tiny)&(value!=0)).float().mean())}
            if name.endswith('dt_proj.bias'):
                row['softplus_bias_median']=float(torch.nn.functional.softplus(v).median())
            params.append(row)
    result={'job':job['id'],'epoch':saved['epoch'],'completed_updates':saved['completed_updates'],
            'checkpoint_sha256':checkpoint_hash,'scope':'CPU parameter-only diagnostics; no patient inputs, inference or anomaly metrics; not a candidate-selection criterion.',
            'memory':{'norm_mean':float(norm.mean()),'norm_min':float(norm.min()),'norm_max':float(norm.max()),
                      'mean_pairwise_cosine':float(cosines.mean()),'min_pairwise_cosine':float(cosines.min()),
                      'max_pairwise_cosine':float(cosines.max()),'centered_energy_effective_rank':effective_rank,
                      'centered_frobenius_norm':float(centered.norm()),
                      'warning':'Diverse memory weights do not prove diverse reads, recurrent usefulness or anomaly detection.'},
            'ssm_parameters':params}
    assert sha(checkpoint)==checkpoint_hash
    out=root/'weight_diagnostics';out.mkdir(exist_ok=True)
    dest=out/f'{job["id"]}_epoch{saved["epoch"]}.json'
    if dest.exists():
        assert json.loads(dest.read_text())==result
    else:
        dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'job':job['id'],'epoch':saved['epoch'],'memory':result['memory'],
                      'subnormal_ssm_parameters':[p['name'] for p in params if p['float32_subnormal_fraction']>0]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args();analyze(args.root.resolve(),args.checkpoint.resolve())
