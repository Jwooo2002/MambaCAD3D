#!/usr/bin/env python3
"""Summarize local ImageCAS-X download and preparation progress."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "external_data/imagecas_x"


def collect_status():
    kaggle = ROOT / "provenance/kaggle_files.json"
    if not kaggle.exists():
        return {"status": "not started"}
    files = json.loads(kaggle.read_text())["datasetFiles"]
    files = [row for row in files if row["name"].endswith(".change2zip") or ".z0" in row["name"]]
    size = 934085540 + sum(row["totalBytes"] for row in files)
    states = []
    downloaded = 0
    for name in ["annotations"] + [row["name"] for row in files]:
        path = ROOT / "reports/download_state" / f"{name}.json"
        value = json.loads(path.read_text()) if path.exists() else {"status": "pending"}
        downloaded += value.get("downloaded_bytes", 0)
        states.append({"file": name, "status": value["status"],
                       "GB": round(value.get("downloaded_bytes", 0) / 1e9, 3)})
    status_path = ROOT / "reports/preparation_status.json"
    result = {"status": json.loads(status_path.read_text()) if status_path.exists() else {},
              "downloaded_GB_approx": round(downloaded / 1e9, 2), "total_GB_approx": round(size / 1e9, 2),
              "download_percent_approx": round(100 * downloaded / size, 1),
              "verified_archive_parts": sum(row["status"] == "verified" for row in states),
              "original_ct_volumes_extracted": len(list((ROOT / "raw/imagecas_volumes").rglob("*.img.nii.gz"))),
              "files": states}
    for filename in ("geometry_qc_summary.json",):
        path = ROOT / "reports" / filename
        if path.exists():
            result[filename.removesuffix(".json")] = json.loads(path.read_text())
    return result


if __name__ == "__main__":
    print(json.dumps(collect_status(), indent=2))
