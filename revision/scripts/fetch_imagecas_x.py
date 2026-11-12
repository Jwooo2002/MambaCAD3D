#!/usr/bin/env python3
"""Download pinned ImageCAS-X v3 labels and ImageCAS v3 CT archives; resume safely."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from urllib.parse import quote, urljoin
import zipfile

import requests


ROOT = Path(__file__).resolve().parents[1] / "external_data/imagecas_x"
TRANSPORT = Path("/tmp/vmamba_imagecas_x_transport")
PLACEMENT_LOCK = threading.Lock()
LOG = logging.getLogger(__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def download(url, target, state_path, expected_size=None, checksum=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if target.exists():
        if expected_size is not None and target.stat().st_size != expected_size:
            raise ValueError(f"Existing download has wrong size: {target}")
        if checksum:
            algorithm, wanted = checksum.split(":", 1)
            digest = hashlib.new(algorithm)
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                    digest.update(chunk)
            if digest.hexdigest() != wanted:
                raise ValueError(f"Checksum mismatch: {target}")
        return target
    state = {"url": url, "target": str(target), "status": "downloading"}
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            # Always request an explicit range, including on a fresh download.
            with requests.get(url, headers={"Range": f"bytes={offset}-"}, stream=True,
                              timeout=(20, 90)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match[1]) != offset:
                        raise ValueError("Server returned an inconsistent download range")
                    total = int(match[3])
                elif response.status_code == 200:
                    offset = 0
                    total = int(response.headers.get("Content-Length", 0)) or expected_size
                else:
                    raise ValueError(f"Unexpected HTTP status {response.status_code}")
                size, last_report = offset, 0.0
                with partial.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(4 * 1024**2):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        size += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            state.update({"downloaded_bytes": size, "total_bytes": total,
                                          "updated_utc": datetime.now(timezone.utc).isoformat()})
                            write_json(state_path, state)
                            last_report = now
                if total is not None and size != total:
                    raise ValueError(f"Incomplete download: {size} / {total}")
                if expected_size is not None and size != expected_size:
                    raise ValueError(f"Downloaded size differs from publisher metadata: {target.name}")
            if checksum:
                algorithm, wanted = checksum.split(":", 1)
                digest = hashlib.new(algorithm)
                with partial.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                        digest.update(chunk)
                if digest.hexdigest() != wanted:
                    raise ValueError(f"Checksum mismatch: {target.name}")
                state["verified_checksum"] = checksum
            partial.replace(target)
            state.update({"status": "downloaded", "downloaded_bytes": size, "total_bytes": size})
            write_json(state_path, state)
            LOG.info("Downloaded %s (%.2f GB)", target.name, size / 1e9)
            return target
        except (requests.RequestException, ValueError) as exc:
            # Do not log response redirect URLs (they may contain temporary signatures).
            state.update({"status": "retrying", "last_error_type": type(exc).__name__, "attempt": attempt + 1})
            write_json(state_path, state)
            if attempt == 5:
                raise RuntimeError(f"Download failed after retries: {target.name} ({type(exc).__name__})") from None
            LOG.warning("Retrying %s after %s", target.name, type(exc).__name__)
            time.sleep(min(30, 2 ** attempt))


def extract_labels(archive):
    destination = ROOT / "raw/annotations"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError("Archive path escapes extraction root")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unexpected symlink in annotation archive")
            zipped.extract(member, destination)
    LOG.info("Extracted and CRC-checked annotation archive")


def fetch_annotation(metadata):
    entry = next(row for row in metadata["files"] if row["key"] == "ImageCAS-X_dataset_v3.zip")
    target = ROOT / "downloads" / entry["key"]
    archive = download(entry["links"]["self"], target, ROOT / "reports/download_state/annotations.json",
                       entry["size"], entry["checksum"])
    extract_labels(archive)
    return "annotations"


def fetch_kaggle_file(entry):
    name, expected = entry["name"], entry["totalBytes"]
    raw = ROOT / "downloads/imagecas_v3" / name
    raw.parent.mkdir(parents=True, exist_ok=True)
    state_path = ROOT / "reports/download_state" / (name.replace("/", "_") + ".json")
    verified_path = raw.with_name(raw.name + ".verified.json")
    if raw.exists() and raw.stat().st_size == expected and verified_path.exists():
        return name
    url = ("https://www.kaggle.com/api/v1/datasets/download/xiaoweixumedicalai/imagecas/"
           + quote(name, safe="") + "?datasetVersionNumber=3")
    # Receive concurrent network streams on the local SSD. Serialize final writes
    # to the data disk so they do not compete with each other or archive extraction.
    TRANSPORT.mkdir(parents=True, exist_ok=True)
    wrapper = download(url, TRANSPORT / (raw.name + ".kaggle.zip"), state_path)
    with PLACEMENT_LOCK:
        return unwrap_kaggle_file(entry, wrapper, raw, state_path, verified_path, url)


def unwrap_kaggle_file(entry, wrapper, raw, state_path, verified_path, url):
    name, expected = entry["name"], entry["totalBytes"]
    digest = hashlib.sha256()
    temporary = raw.with_name(raw.name + ".unwrapping")
    with zipfile.ZipFile(wrapper) as zipped:
        entries = [member for member in zipped.infolist() if not member.is_dir()]
        if len(entries) != 1 or Path(entries[0].filename).name != name or entries[0].file_size != expected:
            raise ValueError(f"Unexpected Kaggle download wrapper: {name}")
        with zipped.open(entries[0]) as source, temporary.open("wb") as destination:
            for chunk in iter(lambda: source.read(8 * 1024**2), b""):
                destination.write(chunk)
                digest.update(chunk)
    if temporary.stat().st_size != expected:
        raise ValueError(f"Unwrapped file has incorrect size: {name}")
    temporary.replace(raw)
    verification = {"name": name, "source": url, "dataset_version": 3, "size_bytes": expected,
                    "sha256": digest.hexdigest(), "kaggle_wrapper_crc_checked": True}
    write_json(verified_path, verification)
    write_json(state_path, {**verification, "status": "verified", "downloaded_bytes": expected, "total_bytes": expected})
    wrapper.unlink()  # Only the redundant, server-generated single-file wrapper is removed.
    LOG.info("Verified original archive part %s", name)
    return name


def ensure_7zip():
    tool_dir = ROOT / "tools/7zip"
    binary = tool_dir / "7zz"
    if binary.exists():
        return binary
    response = requests.get("https://www.7-zip.org/download.html", timeout=40)
    response.raise_for_status()
    links = re.findall(r'href="([^"]+linux-x64\.tar\.xz)"', response.text, re.I)
    if not links:
        raise ValueError("Cannot find official Linux 7-Zip download")
    url = urljoin(response.url, links[0])
    archive = download(url, ROOT / "tools/7zip.tar.xz", ROOT / "reports/download_state/7zip.json")
    tool_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile() and Path(member.name).name in ("7zz", "License.txt", "readme.txt"):
                with tar.extractfile(member) as source, (tool_dir / Path(member.name).name).open("wb") as output:
                    shutil.copyfileobj(source, output)
    if not binary.exists():
        raise ValueError("Official archive did not contain 7zz")
    binary.chmod(0o755)
    write_json(tool_dir / "source.json", {"url": url, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    return binary


def extract_volume_group(group, binary):
    marker = ROOT / "reports" / f"extracted_{group}.json"
    if marker.exists():
        return
    parts_dir = ROOT / "downloads/imagecas_v3"
    central = parts_dir / f"{group}.change2zip"
    alias = parts_dir / f"{group}.zip"
    if not alias.exists():
        alias.symlink_to(central)
    destination = ROOT / "raw/imagecas_volumes"
    destination.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "reports" / f"extract_{group}.log"
    with log_path.open("w") as log:
        subprocess.run([str(binary), "x", str(alias), f"-o{destination}", "-y", "-bsp0", "-mmt=2",
                        "-ir!*.img.nii.gz"], stdout=log, stderr=subprocess.STDOUT, check=True, stdin=subprocess.DEVNULL)
    LOG.info("Extracted and CRC-checked CCTA volumes in group %s", group)
    write_json(marker, {"group": group, "status": "extracted", "source_crc_checked": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--annotations-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for folder in ("downloads", "provenance", "raw", "prepared", "reports/download_state"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    metadata = json.loads((ROOT / "provenance/zenodo_v3.json").read_text())
    files = json.loads((ROOT / "provenance/kaggle_files.json").read_text())["datasetFiles"]
    parts = [entry for entry in files if re.fullmatch(r"\d+-\d+\.(change2zip|z\d+)", entry["name"])]
    if len(parts) != 25:
        raise ValueError("Expected five groups with five archive parts each")
    write_json(ROOT / "reports/preparation_status.json", {"status": "downloading", "annotation_version": 3,
               "zenodo_record": 23894668, "imagecas_version": 3, "original_archive_bytes": sum(p["totalBytes"] for p in parts)})
    try:
        if args.annotations_only:
            fetch_annotation(metadata)
            return
        binary = ensure_7zip()
        groups = sorted({p["name"].split(".")[0] for p in parts}, key=lambda s: int(s.split("-")[0]))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_annotation, metadata)] + [pool.submit(fetch_kaggle_file, item) for item in parts]
            for future in as_completed(futures):
                future.result()
        # The source disk is rotational; defer extraction until network placement
        # completes to avoid mixing random writes with split-archive reads.
        for group in groups:
            extract_volume_group(group, binary)
        volumes = sorted((ROOT / "raw/imagecas_volumes").rglob("*.img.nii.gz"))
        if len(volumes) != 1000:
            raise ValueError(f"Expected 1000 original CCTA volumes, found {len(volumes)}")
        write_json(ROOT / "reports/preparation_status.json", {"status": "downloaded_and_extracted", "annotation_version": 3,
                   "zenodo_record": 23894668, "imagecas_version": 3, "original_volume_count": len(volumes),
                   "next_step": "Pair 800 included scans with descriptors, masks and centerlines, then validate physical coordinates."})
        subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_imagecas_x.py")),
                        "--validate-geometry", "--workers", "4"], check=True)
        write_json(ROOT / "reports/preparation_status.json", {"status": "prepared_raw_cohort", "annotation_version": 3,
                   "zenodo_record": 23894668, "imagecas_version": 3, "original_volume_count": len(volumes),
                   "included_subject_count": 800, "geometry_report": str(ROOT / "reports/geometry_qc_summary.json"),
                   "patch_conversion": "Pending reconciliation of ASOCA physical sampling parameters"})
    except BaseException as exc:
        write_json(ROOT / "reports/preparation_status.json", {"status": "failed", "error_type": type(exc).__name__,
                   "message": str(exc) if not isinstance(exc, requests.RequestException) else "HTTP request failed; signed redirect URLs omitted"})
        raise


if __name__ == "__main__":
    main()
