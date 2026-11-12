import ast
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('recovery',ROOT/'performance_search/recover_after_reboot.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def test_frozen_scheduling_and_selection_body_is_identical():
    source=(ROOT/'outputs/performance_nested_20261112T031500Z_200ep/run.py').read_text()
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='queue')
    new=r.queue_ast(source)
    def tail(node):
        start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
                   and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='stopped')
        return [ast.dump(n) for n in node.body[start:]]
    assert tail(new)==tail(original)
    assert [ast.dump(x) for x in new.body[:2]]==[ast.dump(x) for x in original.body[:2]]


def test_duplicate_recovery_queue_is_rejected(tmp_path):
    with (tmp_path/'recovery_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):r.queue(tmp_path)


def test_recovery_preserves_completed_results_and_original_deadline(tmp_path,monkeypatch):
    shutil.copy2(ROOT/'outputs/performance_nested_20261112T031500Z_200ep/run.py',tmp_path/'run.py')
    r.write(tmp_path/'frozen_hashes.json',{'run.py':r.sha(tmp_path/'run.py')})
    frozen=r.load_frozen(tmp_path)
    done=frozen.make_job('inner',1,42,'P0_reference');done.update(status='complete',pid=111)
    lost=frozen.make_job('inner',1,123,'P0_reference');lost.update(status='running',pid=222)
    pending=frozen.make_job('inner',1,42,'P1_no_decay')
    for job,state in ((done,'complete'),(lost,'training')):
        out=tmp_path/job['id'];out.mkdir()
        r.write(out/'state.json',{'status':state,'completed_epochs':200 if state=='complete' else 179})
        (out/'history.csv').write_text('preserved\n')
    (tmp_path/done['id']/'results.json').write_text('{"immutable":true}')
    (tmp_path/lost['id']/'epoch_150.pt').write_bytes(b'checkpoint-evidence')
    (tmp_path/(lost['id']+'.log')).write_text('old log\n')
    deadline=time.time()+10000
    manifest={'status':'running','queue_pid':333,'started_utc':'original-start','deadline_unix':deadline,
              'deadline_utc':'original-deadline','jobs':[done,lost,pending]}
    r.write(tmp_path/'manifest.json',manifest)
    monkeypatch.setattr(r,'pid_live',lambda _:True)
    with pytest.raises(AssertionError,match='queue is live'):r.prepare(tmp_path)
    assert (tmp_path/lost['id']).exists()
    monkeypatch.setattr(r,'pid_live',lambda _:False)
    record=r.prepare(tmp_path)
    updated=r.read(tmp_path/'manifest.json')
    assert updated['jobs'][0]==done
    assert (tmp_path/done['id']/'results.json').read_text()=='{"immutable":true}'
    assert updated['jobs'][1]['status']=='queued' and not (tmp_path/lost['id']).exists()
    assert updated['jobs'][2]==pending
    archive=Path(updated['latest_recovery_record']).parent
    assert (archive/lost['id']/'epoch_150.pt').read_bytes()==b'checkpoint-evidence'
    assert (archive/(lost['id']+'.log')).read_text()=='old log\n'
    active,returned_deadline=r.recovery_start(tmp_path)
    assert returned_deadline==deadline and active['deadline_unix']==deadline
    assert active['deadline_utc']=='original-deadline' and active['started_utc']=='original-start'
    assert record['original_queue_pid']==333
