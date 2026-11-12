#!/usr/bin/env python3
"""Archive interrupted attempts and reuse frozen scheduling with its original deadline."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def load_frozen(root):
    spec=importlib.util.spec_from_file_location('frozen_performance_runner',root/'run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_files(root)
    return module


def pid_live(pid):
    try:
        stat=(Path('/proc')/str(pid)/'stat').read_text()
    except FileNotFoundError:
        return False
    return stat[stat.rfind(')')+2:].split()[0]!='Z'


def prepare(root):
    frozen=load_frozen(root);manifest=read(root/'manifest.json')
    assert manifest['status']=='running'
    assert time.time()<manifest['deadline_unix']
    assert not (root/'STOP').exists()
    assert not pid_live(manifest['queue_pid']),'Existing queue is live'
    interrupted=[]
    for job in manifest['jobs']:
        out=root/job['id']
        if job['status']=='complete':
            assert read(out/'state.json')['status']=='complete'
            assert (out/'results.json').is_file()
        elif job['status']=='queued':
            assert not out.exists(),job['id']
        else:
            assert job['status']=='running',job
            assert not pid_live(job['pid']),'Existing worker is live'
            state=read(out/'state.json')
            assert state['status']=='training'
            assert not (out/'results.json').exists() and not (out/'evaluation_access.json').exists()
            files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
            interrupted.append((job,state,files))
    assert interrupted,'Nothing to recover'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=root/'recovery_attempts'/stamp;archive.mkdir(parents=True,exist_ok=False)
    shutil.copy2(root/'manifest.json',archive/'original_manifest.json')
    if (root/'queue.log').exists():shutil.copy2(root/'queue.log',archive/'original_queue.log')
    entry=root/f'recovery_queue_{stamp}.py';shutil.copy2(Path(__file__),entry)
    records=[]
    for job,state,files in interrupted:
        original=root/job['id'];dest=archive/job['id']
        original.rename(dest)
        assert files=={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
        log=root/(job['id']+'.log')
        if log.exists():log.rename(archive/log.name)
        records.append({'job':job['id'],'last_completed_epochs':state.get('completed_epochs'),
                        'original_pid':job['pid'],'archived_files_sha256':files,
                        'restart_policy':'Frozen initialization and full200 epoch orders; discarded partial attempt never evaluated.'})
        fresh=frozen.make_job(job['stage'],job['fold'],job['seed'],job['variant'])
        job.clear();job.update(fresh)
    record={'recorded_utc':datetime.now(timezone.utc).isoformat(),
            'reason':'Host reboot; queue and workers confirmed absent before recovery.',
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'original_queue_pid':manifest['queue_pid'],'original_deadline_utc':manifest['deadline_utc'],
            'original_started_utc':manifest['started_utc'],'restarted_attempts':records,
            'entrypoint':str(entry),'entrypoint_sha256':sha(entry),'frozen_runner_sha256':sha(root/'run.py'),
            'training_and_selection_unchanged':True}
    write(archive/'recovery_record.json',record)
    manifest.update(status='recovery_ready',queue_entrypoint=str(entry),queue_entrypoint_sha256=sha(entry),
                    latest_recovery_record=str(archive/'recovery_record.json'))
    write(root/'manifest.json',manifest)
    frozen.verify_files(root)
    return record


def recovery_start(root):
    manifest=read(root/'manifest.json')
    assert manifest['status']=='recovery_ready'
    assert sha(manifest['queue_entrypoint'])==manifest['queue_entrypoint_sha256']
    deadline=manifest['deadline_unix']
    assert time.time()<deadline
    manifest.update(status='running',queue_pid=os.getpid(),resumed_utc=datetime.now(timezone.utc).isoformat())
    return manifest,deadline


def queue_ast(source):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='queue')
    stop=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
              and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='stopped')
    assert isinstance(node.body[0],ast.Expr) and node.body[0].value.func.id=='verify_files'
    assert isinstance(node.body[1],ast.For)
    # Keep frozen verification/preflight and every scheduling/selection/stop statement.
    # Only replace initial manifest setup, which otherwise grants a new48h deadline.
    node=copy.deepcopy(node)
    node.body=node.body[:2]+ast.parse('manifest, deadline = recovery_start(root)').body+node.body[stop:]
    return ast.fix_missing_locations(node)


def queue(root):
    with (root/'recovery_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=load_frozen(root)
        scope=dict(vars(frozen));scope['recovery_start']=recovery_start
        tree=ast.Module(body=[queue_ast((root/'run.py').read_text())],type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree),str(Path(__file__)),'exec'),scope)
        scope['queue'](root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','queue'])
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='prepare':print(json.dumps(prepare(root)),flush=True)
    else:queue(root)
