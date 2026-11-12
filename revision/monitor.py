#!/usr/bin/env python3
"""Read local pilot jobs without requiring the training environment or a GPU."""

import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def collect_status():
    active_path = ROOT / "active_runs.json"
    if not active_path.exists():
        return {"status": "no registered runs", "jobs": []}
    registration = json.loads(active_path.read_text())
    jobs, contexts = [], set()
    for job in registration["jobs"]:
        try:
            os.kill(job["pid"], 0)
            process_exists = True
        except ProcessLookupError:
            # A restricted execution environment may not see host-launched PIDs.
            process_exists = None
        except PermissionError:
            process_exists = None
        candidates = sorted((ROOT / "outputs").glob(job["run_name"] + "_*/run_config.json"))
        output = candidates[-1].parent if candidates else None
        summary_path = output / "summary.json" if output else None
        summary = json.loads(summary_path.read_text()) if summary_path and summary_path.exists() else {}
        status = job.get("status", summary.get("status", "no completed fold yet; inspect recent training log"))
        result = {"gpu": job["gpu"], "pid": job["pid"], "folds": job["folds"], "seeds": job["seeds"],
                  "process_exists": process_exists, "status": status,
                  "process_visibility_note": "A null value means this execution environment cannot verify the host PID; it does not establish that training stopped.",
                  "completed_fold_seed_runs": summary.get("completed_fold_seed_runs", 0),
                  "output_directory": str(output) if output else None, "log": job["log_path"]}
        log_path = Path(job["log_path"])
        if log_path.exists():
            result["seconds_since_log_update"] = round(time.time() - log_path.stat().st_mtime, 1)
            with log_path.open("rb") as handle:
                handle.seek(max(0, log_path.stat().st_size - 6000))
                result["recent_log"] = handle.read().decode(errors="replace").splitlines()[-5:]
        for row in summary.get("results", []):
            context = (row["fold"], row["seed"])
            if context in contexts:
                raise ValueError(f"Duplicate fold/seed across jobs: {context}")
            contexts.add(context)
        jobs.append(result)
    preprocessing = []
    for path in sorted((ROOT / "processed_data/ordered_mm_v2").glob("preprocessing_*.json")):
        if "sample" in path.stem:
            continue
        value = json.loads(path.read_text())
        preprocessing.append({"cohort": path.stem, "status": value["status"],
                              "completed": value["completed_patients"], "expected": value["expected_patients"],
                              "errors": value["errors"]})
    return {"phase": registration["phase"], "expected_fold_seed_runs": registration["expected_fold_seed_runs"],
            "completed_fold_seed_runs": len(contexts), "preprocessing": preprocessing, "jobs": jobs}


if __name__ == "__main__":
    print(json.dumps(collect_status(), indent=2, ensure_ascii=False))
