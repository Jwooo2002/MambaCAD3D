#!/usr/bin/env python3
"""Prepare, verify and execute a finite paired memory-mechanism pilot."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time

VARIANTS = {'M0_baseline': {'entropy_weight': .0002, 'memory_decay': 1e-5},
            'M1_entropy_off': {'entropy_weight': 0., 'memory_decay': 1e-5},
            'M2_memory_decay_off': {'entropy_weight': .0002, 'memory_decay': 0.}}
PROBE_EPOCHS = [0, 1, 3, 5, 10, 20]


def now(): return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''): h.update(block)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def imports(root):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(root/'source_snapshot/main_files'))
    from model import MemAE_VSSM3D
    return np, torch, MemAE_VSSM3D


def set_seed(seed, np, torch):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def instantiate(cls, cfg):
    return cls(**{k:cfg[k] for k in ('in_chans','dims','depths','ssm_d_state','mem_dim')})


def load_inputs(root, np, torch):
    manifest = json.loads((root/'input_manifest.json').read_text())
    arrays = []; probe_indices = {}; start = 0
    for entry in manifest:
        path = Path(entry['path'])
        if digest(path) != entry['sha256']: raise RuntimeError(f'Input changed: {path.name}')
        with path.open('rb') as handle: clips = pickle.load(handle)
        array = np.stack(clips).astype(np.float32)
        assert len(array) == entry['patches']
        arrays.append(array)
        indices = np.unique(np.linspace(0, len(array)-1, min(3,len(array)), dtype=int))
        probe_indices[path.name] = [start+int(i) for i in indices]
        start += len(array)
    data = (torch.from_numpy(np.concatenate(arrays)).unsqueeze(1)+300)/700
    assert torch.isfinite(data).all()
    return data, probe_indices


def optimizer_for(model, variant, cfg, torch):
    options = VARIANTS[variant]
    if variant != 'M2_memory_decay_off':
        return torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    mem = model.mem_rep.memory
    other = [p for p in model.parameters() if p is not mem]
    return torch.optim.Adam([{'params':other,'weight_decay':1e-5},
                             {'params':[mem],'weight_decay':0.}], lr=cfg['lr'])


def prepare(root, baseline, legacy_sweep):
    if root.exists(): raise RuntimeError('Output already exists; refusing overwrite')
    root.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    shutil.copy2(__file__, root/'run.py'); shutil.copy2(here/'probe.py', root/'probe.py')
    source = baseline/'source_snapshot/main_files'
    old_hashes = json.loads((baseline/'source_hashes.json').read_text())
    for name, expected in old_hashes.items():
        assert digest(source/name) == expected, name
    shutil.copytree(source, root/'source_snapshot/main_files', ignore=shutil.ignore_patterns('__pycache__'))
    basecfg = json.loads((baseline/'effective_config.json').read_text())
    cfg = {k:basecfg[k] for k in ('in_chans','dims','depths','ssm_d_state','mem_dim','lr','eta_min','batch_size')}
    cfg.update(epochs=20, scheduler_T_max=basecfg['epochs'], probe_epochs=PROBE_EPOCHS,
               num_workers=1, fold=1, seeds=[42,123], normal_training_only=True,
               early_stopping=False, checkpoint_selection='fixed epoch; no validation or test selection',
               data_normalization='float32 (x+300)/700; no clipping; matches frozen legacy transform',
               trainer_differences='Fixed paired initial weights and precomputed full-coverage batch orders; 20 epochs; no validation/test/early stopping; read-only training probes; preloaded normalized inputs')
    split = json.loads((legacy_sweep/'split_manifest.json').read_text())[0]
    assert set(split['train']).isdisjoint(split['normal_test_and_validation'])
    np, torch, cls = imports(root)
    inputs = []; total = 0
    for name in split['train']:
        path = Path(basecfg['normal_data_root'])/name
        with path.open('rb') as handle: count = len(pickle.load(handle))
        inputs.append({'path':str(path),'patient':name,'patches':count,'sha256':digest(path)})
        total += count
    assert len(inputs)==15
    write_json(root/'input_manifest.json',inputs)
    write_json(root/'training_split.json',{'fold':1,'train':split['train'],
               'scope':'Only these normal training files will be loaded. All validation/test/external data excluded.'})
    write_json(root/'config.json',cfg)
    initials = {}; order_hashes = {}
    for seed in cfg['seeds']:
        set_seed(seed,np,torch); model=instantiate(cls,cfg)
        state={k:v.detach().clone().cpu() for k,v in model.state_dict().items()}
        torch.save({'model_state_dict':state,'seed':seed},root/f'initial_seed_{seed}.pt')
        initials[str(seed)]=state_digest(state)
        generator=torch.Generator().manual_seed(seed+1000003)
        orders=[torch.randperm(total,generator=generator).tolist() for _ in range(cfg['epochs'])]
        assert all(sorted(order)==list(range(total)) for order in orders)
        write_json(root/f'orders_seed_{seed}.json',orders)
        order_hashes[str(seed)]=digest(root/f'orders_seed_{seed}.json')
    jobs=[]
    for seed in cfg['seeds']:
        for variant in VARIANTS:
            name=f'{variant}_seed{seed}'; (root/name).mkdir()
            jobs.append({'id':name,'variant':variant,'seed':seed,'status':'queued'})
    hashes={str(p.relative_to(root)):digest(p) for p in (root/'source_snapshot').rglob('*.py')}
    for name in ['run.py','probe.py','config.json','training_split.json','input_manifest.json']:
        hashes[name]=digest(root/name)
    for seed in cfg['seeds']:
        for name in [f'initial_seed_{seed}.pt',f'orders_seed_{seed}.json']:hashes[name]=digest(root/name)
    write_json(root/'source_hashes.json',hashes)
    write_json(root/'manifest.json',{'status':'prepared','created_utc':now(),'baseline':str(baseline),
               'legacy_sweep':str(legacy_sweep),'config':cfg,'variants':VARIANTS,'jobs':jobs,
               'initial_state_hashes':initials,'batch_order_hashes':order_hashes,'training_patches':total,
               'total_epochs':120,'scope':'Normal-training-only mechanism pilot; not k-fold AUROC/AP evaluation',
               'gpu_policy':'GPU 1 when idle; GPU 0 only after old sweep completes and GPU is idle'})
    print(json.dumps({'prepared':str(root),'jobs':6,'epochs_each':20,'patches':total}),flush=True)


def verify_files(root):
    for name, expected in json.loads((root/'source_hashes.json').read_text()).items():
        if digest(root/name)!=expected: raise RuntimeError(f'Frozen source/config changed: {name}')


def verify(root, gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    verify_files(root)
    np,torch,cls=imports(root)
    cfg=json.loads((root/'config.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    data,probes=load_inputs(root,np,torch)
    # Compare optimized loading with original frame-by-frame legacy preprocessing.
    from data.VideoDataset import VideoDataset
    from torchvision import transforms
    entry=json.loads((root/'input_manifest.json').read_text())[0]
    transform=transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x:(x+300)/700)])
    old=VideoDataset(str(Path(entry['path']).parent),transform=transform,pkl_files=[entry['patient']])
    assert torch.equal(data[0],old[0][1]) and torch.equal(data[len(old)-1],old[len(old)-1][1])
    seed=42; initial=torch.load(root/f'initial_seed_{seed}.pt',map_location='cpu',weights_only=False)['model_state_dict']
    assert state_digest(initial)==manifest['initial_state_hashes'][str(seed)]
    outputs={};updated={};losses={}
    for variant in VARIANTS:
        set_seed(seed,np,torch); model=instantiate(cls,cfg).cuda();model.load_state_dict(initial)
        optimizer=optimizer_for(model,variant,cfg,torch)
        covered=[p for g in optimizer.param_groups for p in g['params']]
        assert len({id(p) for p in covered})==len(list(model.parameters()))
        for group in optimizer.param_groups:
            for parameter in group['params']:
                wanted=VARIANTS[variant]['memory_decay'] if parameter is model.mem_rep.memory else 1e-5
                assert group['weight_decay']==wanted
        recon,att=model(data[:2].cuda());outputs[variant]=recon.detach().cpu()
        mse=(recon-data[:2].cuda()).square().mean();entropy=-(att*torch.log(att+1e-12)).sum(1).mean()
        loss=mse+VARIANTS[variant]['entropy_weight']*entropy
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();assert all(torch.isfinite(p).all() for p in model.parameters())
        updated[variant]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        losses[variant]={'loss':float(loss),'mse':float(mse),'legacy_entropy':float(entropy),'gradient_norm':float(grad)}
    assert all(torch.equal(outputs['M0_baseline'],v) for v in outputs.values())
    nonmemory_diff=max(float((updated['M0_baseline'][k]-updated['M2_memory_decay_off'][k]).abs().max()) for k in initial if k!='mem_rep.memory')
    assert nonmemory_diff<=1e-7
    assert not torch.equal(updated['M0_baseline']['mem_rep.memory'],updated['M2_memory_decay_off']['mem_rep.memory'])
    assert losses['M1_entropy_off']['loss']==losses['M1_entropy_off']['mse']
    from probe import measure
    initial_before=state_digest(model.state_dict());rng=torch.get_rng_state().clone();rngcuda=torch.cuda.get_rng_state().clone()
    result=measure(model,data,probes,.0002,0.)
    assert initial_before==state_digest(model.state_dict())
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(rngcuda,torch.cuda.get_rng_state())
    for seed in cfg['seeds']:
        orders=json.loads((root/f'orders_seed_{seed}.json').read_text())
        assert len(orders)==20 and all(sorted(o)==list(range(len(data))) for o in orders)
    assert eligible_gpus(set(),False,{'0':False,'1':False})==['1']
    assert eligible_gpus({'1'},True,{'0':False,'1':True})==['0']
    assert eligible_gpus(set(),True,{'0':True,'1':True})==[]
    write_json(root/'verification.json',{'status':'passed','checked_utc':now(),
               'input_matches_legacy_exactly':True,'initial_outputs_identical':True,
               'optimizer_groups_verified':True,'all_variants_finite_update':True,
               'nonmemory_one_step_diff_decay_control':nonmemory_diff,
               'probe_preserves_weights_and_rng':True,'every_epoch_full_patch_coverage':True,
               'scheduler_gate_verified':True,'probe_patches':result['patches'],'one_step':losses})
    print(json.dumps({'verification':'passed','probe_patches':result['patches']}),flush=True)


def train(root, job_id, gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    verify_files(root)
    np,torch,cls=imports(root)
    cfg=json.loads((root/'config.json').read_text()); manifest=json.loads((root/'manifest.json').read_text())
    job=next(j for j in manifest['jobs'] if j['id']==job_id); output=root/job_id
    seed=job['seed'];variant=job['variant']; options=VARIANTS[variant]
    state={'status':'running','started_utc':now(),'pid':os.getpid(),'gpu':gpu,'seed':seed,
           'variant':variant,'epoch':0,'total_epochs':cfg['epochs'],'phase':'loading'}
    write_json(output/'run_state.json',state)
    try:
        set_seed(seed,np,torch);data,probes=load_inputs(root,np,torch)
        model=instantiate(cls,cfg).cuda()
        initial=torch.load(root/f'initial_seed_{seed}.pt',map_location='cpu',weights_only=False)['model_state_dict']
        model.load_state_dict(initial)
        assert state_digest(model.state_dict())==manifest['initial_state_hashes'][str(seed)]
        orders=json.loads((root/f'orders_seed_{seed}.json').read_text())
        assert digest(root/f'orders_seed_{seed}.json')==manifest['batch_order_hashes'][str(seed)]
        optimizer=optimizer_for(model,variant,cfg,torch)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['scheduler_T_max'],eta_min=cfg['eta_min'])
        set_seed(seed+2000003,np,torch)
        data_rng=torch.Generator().manual_seed(seed+3000003)
        from probe import measure
        history=[]
        def probe_and_save(epoch):
            state.update(phase='normal_training_probe',epoch=epoch);write_json(output/'run_state.json',state)
            before=time.monotonic()
            with torch.random.fork_rng(devices=[0]):
                record=measure(model,data,probes,options['entropy_weight'],options['memory_decay'])
            record.update(epoch=epoch,checked_utc=now(),probe_seconds=time.monotonic()-before)
            write_json(output/f'probe_epoch_{epoch:02d}.json',record)
            saved={'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                   'seed':seed,'epoch':epoch,'variant':variant,'config':cfg,'intervention':options,
                   'initial_state_hash':manifest['initial_state_hashes'][str(seed)],
                   'batch_order_hash':manifest['batch_order_hashes'][str(seed)]}
            if epoch==cfg['epochs']:
                saved.update(optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict())
            tmp=output/f'epoch_{epoch:02d}.pt.tmp';torch.save(saved,tmp);tmp.replace(output/f'epoch_{epoch:02d}.pt')
            print(json.dumps({'probe_epoch':epoch,'mse':record['mse'],'effective_slots':record['effective_slots_exp_mean_query_entropy'],
                              'duplicates':record['largest_duplicate_neighbourhood_l2_1e_minus6']}),flush=True)
        probe_and_save(0)
        for epoch,order in enumerate(orders,1):
            state.update(epoch=epoch,phase='training',steps_completed=0);write_json(output/'run_state.json',state)
            before=time.monotonic();model.train()
            batches=[order[i:i+cfg['batch_size']] for i in range(0,len(order),cfg['batch_size'])]
            loader=torch.utils.data.DataLoader(data,batch_sampler=batches,num_workers=cfg['num_workers'],
                                               pin_memory=True,generator=data_rng)
            sums=[0.,0.,0.];count=0
            for step,x in enumerate(loader,1):
                x=x.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
                recon,att=model(x);mse=(recon-x).square().mean();entropy=-(att*torch.log(att+1e-12)).sum(1).mean()
                loss=mse+options['entropy_weight']*entropy
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                for i,v in enumerate((loss,mse,entropy)):sums[i]+=float(v.detach())*len(x)
                count+=len(x)
                if step==1 or step%100==0:
                    state.update(steps_completed=step,steps_per_epoch=len(batches),heartbeat_utc=now());write_json(output/'run_state.json',state)
            torch.cuda.synchronize();seconds=time.monotonic()-before
            lr=optimizer.param_groups[0]['lr'];scheduler.step()
            record={'epoch':epoch,'loss':sums[0]/count,'mse':sums[1]/count,'legacy_entropy':sums[2]/count,
                    'lr':lr,'training_seconds':seconds,'patches':count,'optimizer_steps':len(batches)}
            assert count==len(data)
            history.append(record);write_json(output/'history.json',history)
            print(json.dumps(record),flush=True)
            state.update(completed_epochs=epoch,last_epoch_seconds=seconds,heartbeat_utc=now());write_json(output/'run_state.json',state)
            if epoch in PROBE_EPOCHS:probe_and_save(epoch)
        state.update(status='complete',phase='complete',finished_utc=now(),epoch=cfg['epochs'])
    except BaseException as error:
        state.update(status='failed',finished_utc=now(),error=f'{type(error).__name__}: {error}')
        raise
    finally:write_json(output/'run_state.json',state)


def eligible_gpus(active, old_complete, busy):
    return [g for g in ('1','0') if g not in active and not busy[g] and (g!='0' or old_complete)]


def queue(root):
    assert json.loads((root/'verification.json').read_text())['status']=='passed'
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((root/'manifest.json').read_text())
        if state['status']!='prepared':raise RuntimeError('Queue already started; refusing duplicate launch')
        state.update(status='running',started_utc=now(),queue_pid=os.getpid())
        processes={}
        while True:
            for job in state['jobs']:
                if job['status']=='running' and processes[job['id']].poll() is not None:
                    path=root/job['id']/'run_state.json'
                    result=json.loads(path.read_text()) if path.exists() else {}
                    job.update(status='complete' if result.get('status')=='complete' and processes[job['id']].returncode==0 else 'failed',
                               returncode=processes[job['id']].returncode,finished_utc=now())
                    if job['status']=='failed':job['error']=result.get('error','Exited without complete state')
                    print(json.dumps({'job':job['id'],'status':job['status']}),flush=True)
            active={j['gpu'] for j in state['jobs'] if j['status']=='running'}
            queued=[j for j in state['jobs'] if j['status']=='queued'];failed=any(j['status']=='failed' for j in state['jobs'])
            if not active and (failed or not queued):
                state.update(status='failed' if failed else 'complete',finished_utc=now());write_json(root/'manifest.json',state);return
            old=json.loads((Path(state['legacy_sweep'])/'sweep_manifest.json').read_text())
            old_complete=old['status']=='complete'
            state['old_sweep_complete']=old_complete
            busy={g:True for g in ('0','1')}
            if queued and not failed:
                for gpu in ('0','1'):
                    if gpu in active or (gpu=='0' and not old_complete):continue
                    try:
                        r=subprocess.run(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
                        busy[gpu]=bool(r.stdout.strip())
                    except (subprocess.SubprocessError,OSError) as error:state['gpu_check_error']=str(error)
                for gpu in eligible_gpus(active,old_complete,busy):
                    if not queued:break
                    job=queued.pop(0);out=root/job['id']
                    with (out/'training.log').open('xb') as log:
                        process=subprocess.Popen([sys.executable,'-B','-u',str(root/'run.py'),'train','--root',str(root),
                                                  '--job',job['id'],'--gpu',gpu],cwd=root,stdin=subprocess.DEVNULL,
                                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    processes[job['id']]=process
                    job.update(status='running',gpu=gpu,pid=process.pid,started_utc=now())
                    print(json.dumps({'launched':job['id'],'gpu':gpu,'pid':process.pid}),flush=True)
            state.update(heartbeat_utc=now(),new_jobs_paused_after_failure=failed)
            write_json(root/'manifest.json',state);time.sleep(10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','verify','train','queue'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--legacy-sweep',type=Path)
    parser.add_argument('--gpu',default='1')
    parser.add_argument('--job')
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='prepare':prepare(root,args.baseline.resolve(),args.legacy_sweep.resolve())
    elif args.command=='verify':verify(root,args.gpu)
    elif args.command=='train':train(root,args.job,args.gpu)
    else:queue(root)


if __name__=='__main__':main()
