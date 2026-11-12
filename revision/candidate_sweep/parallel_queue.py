#!/usr/bin/env python3
"""Finite two-GPU scheduler that can adopt already-running candidate training."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return {"state": fields[0], "start_ticks": fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(pid, start_ticks):
    identity = process_identity(pid)
    return bool(identity and identity["state"] != "Z" and identity["start_ticks"] == str(start_ticks))


def completed_job_status(run_state, process_alive):
    if process_alive:
        return "running"
    return "complete" if run_state.get("status") == "complete" else "failed"


def available_gpus(active, baseline_complete, busy):
    return [gpu for gpu in ("0", "1")
            if gpu not in active and gpu not in busy and (gpu != "0" or baseline_complete)]


def baseline_ready(root, state):
    baseline = json.loads((Path(state["baseline"]) / "run_state.json").read_text())
    is_alive = alive(state["baseline_pid"], state["baseline_start_ticks"])
    ready = baseline.get("status") == "complete" and not is_alive
    state["baseline_gate"] = {"status": baseline.get("status"), "process_alive": is_alive,
                              "gpu_0_released": ready}
    return ready


def gpu_busy(gpu):
    result = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True,
                            text=True, timeout=10)
    return bool(result.stdout.strip())


def launch(root, job, gpu):
    output = Path(job["output"])
    config_path = output / "effective_config.json"
    config = json.loads(config_path.read_text())
    config["gpu_id"] = gpu
    write_json(config_path, config)
    with (output / "launcher.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, "-B", "-u", str(root / "run_candidates.py"),
                                    "run", "--output", str(output)], cwd=root,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    identity = process_identity(process.pid)
    if identity is None:
        raise RuntimeError("Candidate exited before its process identity was recorded")
    job.update(status="running", gpu=gpu, pid=process.pid, process_start_ticks=identity["start_ticks"],
               started_utc=now(), scheduler="parallel")
    return process


def run(root, interval=15):
    root = root.resolve()
    with (root / "parallel_queue.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "sweep_manifest.json"
        state = json.loads(path.read_text())
        if state.get("queue_mode") not in {"parallel_ready", "parallel"}:
            raise ValueError("Hand off the old queue before starting this scheduler")
        previous_pid = state.get("queue_pid")
        previous_start = state.get("queue_process_start_ticks")
        if previous_start and previous_pid != os.getpid() and alive(previous_pid, previous_start):
            raise RuntimeError("Another scheduler is still alive")
        for job in state["jobs"]:
            if job["status"] == "running" and not all(k in job for k in ("gpu", "process_start_ticks")):
                raise ValueError("Running jobs must have a verified GPU and process identity")
        state.update(queue_mode="parallel", status="running", queue_pid=os.getpid(),
                     queue_process_start_ticks=process_identity(os.getpid())["start_ticks"],
                     gpus=["0", "1"], parallel_started_utc=now())
        children = {}
        print(f"{now()} Adopted running jobs; GPU 0 waits for baseline completion.", flush=True)
        while True:
            for job in state["jobs"]:
                if job["status"] != "running":
                    continue
                if job["pid"] in children:
                    children[job["pid"]].poll()
                is_alive = alive(job["pid"], job["process_start_ticks"])
                result_path = Path(job["output"]) / "run_state.json"
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                status = completed_job_status(result, is_alive)
                if status != "running":
                    job.update(status=status, finished_utc=result.get("finished_utc", now()))
                    if status == "failed":
                        job["error"] = result.get("error", "Training process exited without a complete run state")
                    print(f"{now()} {job['variant']}: {status}", flush=True)
            active = {job["gpu"] for job in state["jobs"] if job["status"] == "running"}
            failed = any(job["status"] == "failed" for job in state["jobs"])
            queued = [job for job in state["jobs"] if job["status"] == "queued"]
            if not active and (failed or not queued):
                state.update(status="failed" if failed else "complete", finished_utc=now())
                write_json(path, state)
                return
            ready = baseline_ready(root, state)
            state["new_jobs_paused_after_failure"] = failed
            if not failed and queued:
                eligible = available_gpus(active, ready, set())
                for gpu in eligible:
                    if not queued:
                        break
                    try:
                        occupied = gpu_busy(gpu)
                    except (subprocess.SubprocessError, OSError) as error:
                        state["gpu_check_error"] = f"{type(error).__name__}: {error}"
                        continue
                    state.pop("gpu_check_error", None)
                    if occupied:
                        continue
                    job = queued.pop(0)
                    try:
                        child = launch(root, job, gpu)
                    except BaseException as error:
                        job.update(status="failed", error=f"Launch failed: {type(error).__name__}: {error}",
                                   finished_utc=now())
                        break
                    children[child.pid] = child
                    print(f"{now()} Started {job['variant']} on GPU {gpu}, pid {child.pid}", flush=True)
                    write_json(path, state)
            state["scheduler_heartbeat_utc"] = now()
            write_json(path, state)
            time.sleep(interval)


def self_test():
    assert available_gpus({"1"}, False, set()) == []
    assert available_gpus({"1"}, True, set()) == ["0"]
    assert available_gpus(set(), False, set()) == ["1"]
    assert available_gpus(set(), True, {"0"}) == ["1"]
    assert available_gpus({"0", "1"}, True, set()) == []
    assert completed_job_status({"status": "complete"}, True) == "running"
    assert completed_job_status({"status": "complete"}, False) == "complete"
    assert completed_job_status({"status": "running"}, False) == "failed"
    assert completed_job_status({"status": "failed"}, False) == "failed"
    identity = process_identity(os.getpid())
    assert alive(os.getpid(), identity["start_ticks"])
    assert not alive(os.getpid(), "-1")
    # Exercise identity-based liveness and completion after a real child exits.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"])
    start = process_identity(child.pid)["start_ticks"]
    assert alive(child.pid, start)
    child.wait()
    assert not alive(child.pid, start)
    # Run the real coordinator with tiny CPU stand-ins for training. Adopt one
    # existing job, then dispatch the remaining two without changing their seed.
    import tempfile
    global gpu_busy
    real_gpu_busy = gpu_busy
    try:
        gpu_busy = lambda gpu: False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_candidates.py").write_text(
                "import json,sys,time\nfrom pathlib import Path\n"
                "p=Path(sys.argv[sys.argv.index('--output')+1])\n"
                "(p/'run_state.json').write_text(json.dumps({'status':'running'}))\n"
                "time.sleep(0.2)\n"
                "t=p/'run_state.tmp'\n"
                "t.write_text(json.dumps({'status':'complete'}))\n"
                "t.replace(p/'run_state.json')\n")
            baseline = root / "baseline"
            baseline.mkdir()
            write_json(baseline / "run_state.json", {"status": "complete"})
            jobs = []
            for name in ("adopted", "next", "last"):
                output = root / name
                output.mkdir()
                write_json(output / "effective_config.json", {"gpu_id": "1", "random_seed": 42})
                jobs.append({"variant": name, "output": str(output), "status": "queued"})
            adopted = launch(root, jobs[0], "1")
            original_pid = adopted.pid
            write_json(root / "sweep_manifest.json", {
                "queue_mode": "parallel_ready", "baseline": str(baseline),
                "baseline_pid": os.getpid(), "baseline_start_ticks": "-1", "jobs": jobs})
            run(root, interval=0.02)
            adopted.wait()
            final = json.loads((root / "sweep_manifest.json").read_text())
            assert final["status"] == "complete"
            assert final["jobs"][0]["pid"] == original_pid
            assert final["jobs"][1]["gpu"] == "0"
            assert all(j["status"] == "complete" for j in final["jobs"])
            assert all(json.loads((Path(j["output"]) / "effective_config.json").read_text())["random_seed"] == 42
                       for j in final["jobs"])
    finally:
        gpu_busy = real_gpu_busy
    print("Passed: GPU gate, live-job adoption, two-device dispatch, completion/failure, seed preservation and PID reuse guard.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.root:
        run(args.root)
    else:
        parser.error("--root or --self-test is required")


if __name__ == "__main__":
    main()
