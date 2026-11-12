#!/usr/bin/env python3
"""Read-only experiment audit and bounded wait on the actual queue PID."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time


def read(path):
    return json.loads(path.read_text())


def process(pid, root, job=None):
    directory=Path('/proc')/str(pid)
    try:
        cmd=directory.joinpath('cmdline').read_bytes().split(b'\0')
        args=[s.decode() for s in cmd if s]
        raw=directory.joinpath('stat').read_text()
    except FileNotFoundError:
        return {'pid':pid,'present':False,'live':False}
    stat=raw[raw.rfind(')')+2:].split()
    expected=root/'run.py'
    if job is None:
        manifest=read(root/'manifest.json')
        if 'queue_entrypoint' in manifest:
            expected=Path(manifest['queue_entrypoint'])
            assert expected.parent==root
            assert hashlib.sha256(expected.read_bytes()).hexdigest()==manifest['queue_entrypoint_sha256']
    assert str(expected) in args and str(root) in args, (pid,args)
    if job is not None:
        assert args[args.index('--job')+1]==job
    return {'pid':pid,'present':True,'live':stat[0]!='Z','state':stat[0],'ppid':int(stat[1]),
            'start_ticks':int(stat[19]),'arguments':args}


def snapshot(root, verify=False):
    manifest=read(root/'manifest.json')
    cfg=read(root/'config.json')
    pairs=read(root/'pairs.json')
    split=read(root/'splits.json')
    if verify:
        for name,expected in read(root/'frozen_hashes.json').items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    queue=process(manifest['queue_pid'],root)
    rows=[];gpu_step_times={};remaining_steps=0
    for job in manifest['jobs']:
        pair=pairs[f'{job["stage"]}_f{job["fold"]}_s{job["seed"]}']
        steps=math.ceil(pair['train_patch_count']/cfg['batch_size'])
        path=root/job['id']/'state.json'
        completed=0
        if path.exists():
            state=read(path)
            f=next(f for f in split['folds'] if f['fold']==job['fold'])
            expected_train=f['inner_train'] if job['stage']=='inner' else f['outer_train']
            expected_eval=f['inner_normal']+f['inner_diseased'] if job['stage']=='inner' else f['outer_test']
            assert state['train_patient_ids']==expected_train
            assert state['evaluation_patient_ids']==expected_eval
            assert not set(expected_train)&set(expected_eval)
            assert state['initial_state_sha256']==pair['initial_state_sha256']
            assert state['actual_seed']==pair['actual_seed']
            completed=state.get('completed_epochs',0)
            assert 0<=completed<=cfg['epochs']
            if state['status'] in ('training','scoring','complete'):
                assert state.get('completed_updates',0)==completed*steps
            text=(root/job['id']/'history.csv').read_text()
            text=text[:text.rfind('\n')+1]
            history=list(csv.DictReader(io.StringIO(text)))[:completed]
            assert len(history)==completed
            assert [int(r['epoch']) for r in history]==list(range(1,completed+1))
            assert all(math.isfinite(float(v)) for r in history for v in r.values())
            worker=process(state['pid'],root,job['id']) if state['status'] in ('training','scoring') else None
            if worker and worker['present']:
                assert worker['ppid']==manifest['queue_pid']
            recent=history[-10:]
            seconds=statistics.median(float(r['seconds']) for r in recent) if recent else None
            if seconds and worker and worker['present']:
                gpu_step_times[state['gpu']]=seconds/steps
            rows.append({'job':job['id'],'stage':job['stage'],'seed':job['seed'],
                         'status':state['status'],'completed_epochs':completed,'total_epochs':cfg['epochs'],
                         'worker':worker,'recent_epoch_median_seconds':seconds,
                         'training_remaining_minutes':seconds*(cfg['epochs']-completed)/60 if seconds else None,
                         'latest':state.get('latest'), 'all_history_values_finite':True,
                         'patient_roles_and_initial_state_match':True})
        remaining_steps+=(cfg['epochs']-completed)*steps
    if not (root/'selections.json').exists():
        remaining_steps+=sum(math.ceil(p['train_patch_count']/cfg['batch_size'])*cfg['epochs']*2
                             for key,p in pairs.items() if key.startswith('outer'))
    wall_hours=remaining_steps/sum(1/t for t in gpu_step_times.values())/3600 if len(gpu_step_times)==2 else None
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'queue_status':manifest['status'],
            'queue_process':queue,'frozen_files_verified':verify,'jobs':rows,
            'remaining_training_hours_upper_job_count':wall_hours,
            'remaining_hours_with_30_percent_margin':wall_hours*1.3 if wall_hours else None,
            'timing_scope':'Recent throughput; assumes two available GPUs and at most24 outer refits. Not a guarantee.',
            'disease_access_count':len(list(root.glob('*/evaluation_access.json'))),
            'deadline_utc':manifest['deadline_utc']}
    dest=root/'runtime_audit';dest.mkdir(exist_ok=True)
    temp=dest/'latest.json.tmp';temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(dest/'latest.json')
    with (dest/'observations.jsonl').open('a') as handle:
        handle.write(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps({'checked_utc':result['checked_utc'],'queue_live':queue['live'],'status':manifest['status'],
          'jobs':[{k:r[k] for k in ('job','status','completed_epochs','training_remaining_minutes')} for r in rows],
          'remaining_hours':wall_hours,'disease_access_count':result['disease_access_count']}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--verify-frozen',action='store_true')
    p.add_argument('--wait-seconds',type=float,default=0)
    args=p.parse_args();root=args.root.resolve()
    assert 0<=args.wait_seconds<=55
    start=snapshot(root,args.verify_frozen)
    if args.wait_seconds and start['queue_process']['live']:
        until=time.monotonic()+args.wait_seconds
        while time.monotonic()<until:
            time.sleep(min(5,max(0,until-time.monotonic())))
            current=process(start['queue_process']['pid'],root)
            if not current['live'] or current['start_ticks']!=start['queue_process']['start_ticks']:
                break
        snapshot(root)
