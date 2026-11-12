#!/usr/bin/env python3
"""Synthetic CPU boundary diagnosis; no patient data, training, or candidate change."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_class(path):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MemModule')
    scope={'torch':torch,'nn':nn,'F':F}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['MemModule']


def diagnose(root, project):
    torch.set_num_threads(1)
    torch.manual_seed(20261112)
    sources={'corrected':root/'source_snapshot/main_files/model.py',
             'legacy':project/'legacy_reproduction/source/model.py'}
    source_hashes={name:digest(path) for name,path in sources.items()}
    classes={name:memory_class(path) for name,path in sources.items()}
    bank=F.normalize(torch.randn(2048,128,dtype=torch.float64),dim=1)
    directions=F.normalize(torch.randn(16,128,dtype=torch.float64),dim=1)
    probe=F.normalize(torch.randn(128,dtype=torch.float64),dim=0).view(1,128,1,1,1)
    records=[]
    for size in (300,2000,2048):
        for mode,cls in classes.items():
            for norm in (0.,1.,10.):
                module=cls(size,128,shrink_thres=.0025).double()
                with torch.no_grad():module.memory.copy_(bank[:size])
                query=(directions*norm).T.reshape(1,128,1,1,16).contiguous().requires_grad_(True)
                output,attention=module(query)
                (output*probe).sum().backward()
                zeros=(attention.sum(-1)==0)
                record={'slots':size,'implementation':mode,'query_l2_per_token':norm,
                    'tokens':16,'zero_read_fraction':float(zeros.double().mean()),
                    'mean_active_slots':float((attention>0).sum(-1).double().mean()),
                    'output_l2':float(output.detach().norm()),
                    'query_gradient_l2':float(query.grad.norm()),
                    'memory_gradient_l2':float(module.memory.grad.norm())}
                if norm==0:
                    record['uniform_softmax_weight']=1/size
                    record['threshold_times_slots']=.0025*size
                    expected_dead=mode=='corrected' and size>400
                    assert bool(zeros.all())==expected_dead
                    if expected_dead:
                        assert record['query_gradient_l2']==record['memory_gradient_l2']==0
                    else:
                        assert record['query_gradient_l2']>0 and record['memory_gradient_l2']>0
                    if mode=='legacy':
                        repeated=attention[:,:1,:].expand(1,1024,size)
                        old=-(repeated*(repeated+1e-12).log()).sum(1).mean()
                        corrected=-(repeated*(repeated+1e-12).log()).sum(2).mean()
                        assert torch.isclose(old,corrected*1024/size,rtol=1e-12,atol=1e-12)
                        record['legacy_to_corrected_entropy_ratio_at_1024_tokens']=float((old/corrected).detach())
                records.append(record)
    assert source_hashes=={name:digest(path) for name,path in sources.items()}
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Synthetic CPU queries with controlled norms and a shared unit-norm memory bank; not actual encoder activations, not historical failure reproduction, no patient data or optimizer updates.',
        'seed':20261112,'dtype':'float64','threshold':.0025,'feature_dim':128,
        'source_paths':{name:str(path) for name,path in sources.items()},'source_sha256':source_hashes,
        'script_sha256':digest(Path(__file__)),
        'gradient_probe':'Fixed linear scalar functional of memory readout; measures local differentiability, not actual reconstruction training gradient.',
        'historical_size_caveat':'Preserved constructor default is2000, user recalls2048; the exact historical failed run has not been located.',
        'current_frozen_queue_changed':False,'records':records}
    dest=root/'memory_size_boundary_diagnostic';dest.mkdir(exist_ok=True)
    (dest/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (dest/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'result_path':str(dest/'results.json'),'zero_query_cases':[r for r in records if r['query_l2_per_token']==0]}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    diagnose(args.root.resolve(),Path(__file__).resolve().parents[1])
