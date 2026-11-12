# MambaCAD-3D

Private source archive of the original MambaCAD-3D implementation and subsequent revision work.

## Versions

- `main` and tag `v1.0`: the original `MemAE_kfold_simplescan` code/configuration snapshot captured on 2026-11-12. Copied source files are byte-for-byte unchanged.
- `revision` branch: retains v1 at the repository root and adds the ongoing revision workspace under `revision/`. Candidate sweeps, memory pilots, performance-search code, tests and research notes are preserved there. This branch is work in progress, not a final v2 release.

The archive label `v1.0` identifies this preserved source snapshot. It does not certify that the snapshot exactly matches the original submitted manuscript or historical checkpoint architecture. See the revision reproduction notes for the known provenance limits.

## Original layout and use

- `main_files/`: original 3D model, training and evaluation code.
- `main_files_2d/`, `main_files/2d_style/`: original 2D comparison code.
- `utils/`, `scripts/`: shared utilities and experiment launch scripts.
- `config/`, `temp_configs/`, root configuration files: preserved original settings.
- `README_multiple_seeds.md`: original multi-seed runner instructions.
- `SOURCE_ARCHIVE.json`: source paths, sizes and SHA-256 checksums for every copied v1 file.

Use the existing `VMamba` environment, or prepare an environment with the imported Python dependencies and a compatible PyTorch/CUDA selective-scan backend. The selective-scan wrapper supports VMamba `selective_scan_cuda_oflex` / `selective_scan_cuda_core` and Mamba `selective_scan_cuda`; extension binaries are not archived here. `ENVIRONMENT_OBSERVED.json` records selected package versions from the current environment, not a historical lockfile.

For the original entry point, run from `main_files/` with a reviewed configuration and your own local data paths:

```bash
cd main_files
python main.py --config /absolute/path/to/config.yaml
```

The source and old configs intentionally retain their original paths and settings. Review them before running: some launchers contain machine-specific paths, and `config/shallow_8.yaml` contains shell-command text before its YAML content and needs a separate valid run configuration. Historical evaluation behavior is preserved rather than silently corrected in v1.

## Archive scope

No CT images, datasets, patient/patch mapping files, per-patient scores, model checkpoints, generated plots, logs, process-state JSON files or local credential/tool directories are included. Existing datasets and running experiments remain at their original locations. The ignore rules prevent common generated artifacts from entering later commits.

For the revision branch, read `revision/README.md` and its research notes. Update machine-specific data/output paths for the new checkout; historical absolute paths in scripts, manifests and documentation are provenance references, not bundled resources. Preprocessing needs additional dependencies such as SimpleITK and VTK, which are not installed in the observed VMamba training environment.

## Third-party code

The selective-scan implementation derives from [VMamba](https://github.com/MzeroMiko/VMamba). Its existing MIT license is preserved in `third_party/VMamba-LICENSE`. This archive does not add a new blanket license for the project's own code.
