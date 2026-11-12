# Archive verification — 2026-11-12

- v1: 98 copied code/configuration files; revision: 131 copied code/configuration/documentation files. Every copied file matched its source SHA-256 when captured. Archive manifests retain the original absolute source path and checksum.
- All 160 copied Python files parsed successfully without rewriting their contents. The additional revision AGENTS.md is copied unchanged.
- Common private-key, GitHub-token, AWS-access-key and assigned-secret patterns were checked in copied text; no matches were found.
- Existing revision tests were run from this checkout in the VMamba environment, with GPUs hidden: `python -m pytest -q tests/test_revision_protocol.py tests/test_model_corrections.py tests/test_external_protocol.py`: **16 passed**, 2 existing PyTorch AMP deprecation warnings, 5.24 seconds.
- These tests cover synthetic model/protocol behavior and the small synthetic CPU training fixture. No full dataset training, original v1 experiment, external evaluation or historical training-trajectory reproduction was run for this archive.
- `git diff --check` reports existing whitespace in the archived originals; it was retained so source hashes remain unchanged.
- No raw/processed datasets, patient/patch mappings, patient/patch score exports, checkpoints, plots, runtime logs or process-state files are tracked. Old paths and code remain unmodified; the README documents portability and provenance limits.
