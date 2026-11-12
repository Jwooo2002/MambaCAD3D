#!/usr/bin/env python3
"""Read-only normal-training feature rank and memory gradient diagnosis."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='1')
    p.add_argument('--sweep', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    import numpy as np
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = args.sweep.resolve()
    manifest = json.loads((root/'sweep_manifest.json').read_text())
    sys.path.insert(0, str(Path(manifest['jobs'][0]['output'])/'source_snapshot/main_files'))
    import candidate_hooks
    from model import MemAE_VSSM3D
    splits = json.loads((root/'split_manifest.json').read_text())
    jobs = [('R0', Path(manifest['baseline']))] + [(j['variant'], Path(j['output'])) for j in manifest['jobs'] if j['variant']=='R3_cosine']
    rows = []
    for name, folder in jobs:
        cfg = json.loads((folder/'effective_config.json').read_text())
        options = json.loads((folder/'variant.json').read_text()) if name!='R0' else {}
        candidate_hooks.configure(options, folder)
        state = json.loads((folder/'run_state.json').read_text())
        for fold in range(1, 5):
            saved = torch.load(Path(state['legacy_results_directory'])/'models'/f'fold_{fold}_model.pt',map_location='cpu',weights_only=False)
            model = MemAE_VSSM3D(**{k:cfg[k] for k in ('in_chans','dims','depths','ssm_d_state','mem_dim')}).cuda().eval()
            model.load_state_dict(saved['model_state_dict'],strict=True)
            moments = {k:{'sum':torch.zeros(128,dtype=torch.float64,device='cuda'),'outer':torch.zeros(128,128,dtype=torch.float64,device='cuda'),'n':0} for k in ['encoder_query','memory_readout']}
            first = None
            with torch.no_grad():
                for filename in splits[fold-1]['train']:
                    with (Path(cfg['normal_data_root'])/filename).open('rb') as handle:
                        data=pickle.load(handle)
                    indices=np.unique(np.linspace(0,len(data)-1,3,dtype=int))
                    x=(torch.from_numpy(np.stack([data[i] for i in indices]).astype(np.float32)).unsqueeze(1).cuda()+300)/700
                    if first is None: first=x[:1]
                    z=model.encode(x);readout,_=model.mem_rep(z)
                    for key,v in [('encoder_query',z),('memory_readout',readout)]:
                        flat=v.flatten(2).transpose(1,2).reshape(-1,128).double()
                        mom=moments[key];mom['sum']+=flat.sum(0);mom['outer']+=flat.T@flat;mom['n']+=len(flat)
            ranks={}
            for key,mom in moments.items():
                mean=mom['sum']/mom['n'];cov=mom['outer']/mom['n']-mean[:,None]*mean[None,:]
                e=torch.linalg.eigvalsh(cov).flip(0).clamp_min(0);fractions=e/e.sum()
                ranks[key]={'rank99':int((fractions.cumsum(0)<.99).sum())+1,'top1_fraction':float(fractions[0]),'top4_fraction':float(fractions[:4].sum()),'tokens':mom['n']}
            output,att=model(first)
            mse=F.mse_loss(output,first)
            penalty=cfg['entropy_loss_weight']*-(att*torch.log(att+1e-12)).sum(1).mean()
            mem=model.mem_rep.memory
            g_recon=torch.autograd.grad(mse,mem,retain_graph=True)[0]
            g_entropy=torch.autograd.grad(penalty,mem,retain_graph=True)[0]
            (mse+penalty).backward()
            preclip=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            norm=mem.detach().norm(dim=1);ref=mem.detach()[norm.argsort()[150]]
            duplicated=(mem.detach()-ref).norm(dim=1)<=1e-6
            groups={}
            for group,mask in [('duplicate_cluster',duplicated),('other_slots',~duplicated)]:
                def values(g):return [float(v) for v in g[mask].norm(dim=1).quantile(torch.tensor([.5,1.],device='cuda'))]
                groups[group]={'slots':int(mask.sum()),'recon_gradient_median_max':values(g_recon),'entropy_gradient_median_max':values(g_entropy),'clipped_total_gradient_median_max':values(mem.grad),'adam_decay_term_median_max':values(1e-5*mem.detach())}
            row={'id':name,'fold':fold,'normal_training_only':True,'rank':ranks,'first_normal_training_patch_gradients':groups,'preclip_total_gradient_norm':float(preclip)}
            rows.append(row)
            args.output.write_text(json.dumps({'scope':'Saved-model local gradients and global centered feature covariance on fixed normal training patches. These are not historical training trajectories or causal estimates.','rows':rows},indent=2)+'\n')
            print(json.dumps(row),flush=True)
            del model,saved


if __name__=='__main__':main()
