#!/usr/bin/env python3
"""Observe one existing finite queue; never launch or restart training jobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def process_matches(pid, root, job=None, proc_root=Path('/proc')):
    try:
        command = (proc_root / str(pid) / 'cmdline').read_bytes().decode()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return str(root) in command and (job is None or job in command)


def observe_job(root, job, proc_root=Path('/proc')):
    live = process_matches(job['pid'], root, job['id'], proc_root)
    path = root / job['id'] / 'run_state.json'
    try:
        state = json.loads(path.read_text())
        error = None
    except FileNotFoundError:
        # Queue publishes PID before the child imports torch and writes its state.
        state, error = {}, 'state_not_created_yet'
    except json.JSONDecodeError as exc:
        state, error = {}, f'transient_state_read: {exc}'
    row = {'job': job['id'], 'pid': job['pid'], 'live': live,
           'completed_epochs': state.get('completed_epochs', 0),
           'phase': state.get('phase', 'starting' if live else 'awaiting_queue_reconciliation')}
    if error:
        row['state_read_note'] = error
    return row


def monitor(root, poll_seconds):
    while True:
        manifest = json.loads((root / 'manifest.json').read_text())
        queue_live = process_matches(manifest.get('queue_pid'), root)
        rows = [observe_job(root, job) for job in manifest['jobs'] if job['status'] == 'running']
        record = {'checked_utc': datetime.now(timezone.utc).isoformat(),
                  'queue_pid': manifest.get('queue_pid'), 'queue_live': queue_live,
                  'status': manifest['status'], 'complete': sum(j['status'] == 'complete' for j in manifest['jobs']),
                  'failed': sum(j['status'] == 'failed' for j in manifest['jobs']), 'running': rows}
        temporary = root / 'runtime_monitor.json.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(root / 'runtime_monitor.json')
        print(json.dumps(record), flush=True)
        if manifest['status'] == 'complete':
            return subprocess.run([sys.executable, '-B', str(root.parents[1] / 'memory_pilot/analyze_hybrid_validation.py'),
                                   '--root', str(root), '--output', str(root / 'analysis'), '--require-complete']).returncode
        if manifest['status'] in ('failed', 'stopped') or not queue_live:
            return 2
        time.sleep(poll_seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=45)
    args = parser.parse_args()
    assert 1 <= args.poll_seconds <= 60
    sys.exit(monitor(args.root.resolve(), args.poll_seconds))
