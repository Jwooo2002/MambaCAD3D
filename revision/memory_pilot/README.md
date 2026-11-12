# Memory-collapse mechanism pilot

Authorized on 2026-11-10: M0 baseline, M1 entropy coefficient zero, and M2 memory-only weight decay zero; seeds 42 and 123; fold 1 normal training patients only; 20 fixed epochs each.

`active_run.json` points to the immutable execution snapshot. Each run preserves the kernel-2 legacy model, entropy axis, memory clamp, learning rate, batch size, and 1000-epoch cosine scheduler time scale. There is no early stopping, validation selection, disease/test evaluation, or resampling of training patches. Every epoch includes all 956 training patches. Per-seed initial weights and all batch permutations are saved and shared across variants.

The queue uses idle GPU 1. GPU 0 becomes eligible only after the previous candidate sweep reports completion and its compute process has exited. Six jobs are finite; a failed job prevents new launches. This queue does not support automatic restart/adoption: inspect existing PIDs before any manual recovery.

At epochs 0, 1, 3, 5, 10, and 20, read-only probes use the first, middle, and last patch of each normal training patient. They record attention concentration and aggregate usage, memory duplicate neighbourhood size, query/readout covariance rank, reconstruction MSE, memory norms, and separate reconstruction/entropy gradients. Probes preserve weights and random-number state. All training patches remain in optimization; the 45 fixed probe patches are only for diagnostics.

Inspect `manifest.json` for queue status and each job's `run_state.json`, `history.json`, `training.log`, and `probe_epoch_*.json` for progress. Epoch checkpoints are independent copies; the epoch-20 checkpoint also saves optimizer and scheduler state. The `verification.json` report documents exact legacy preprocessing parity, identical initial outputs, finite updates, optimizer groups, probe invariance, complete batch coverage, and GPU scheduling gates.

This pilot diagnoses early training mechanisms. It does not establish improved AUROC/AP, long-term stability, or the value of memory versus a separately trained no-memory autoencoder.

## Authorized continuation: epoch 20 to 100

On 2026-11-10 at 16:55 KST, the user-approved M0/M1 continuation started on two idle RTX 4090 GPUs. `active_continuation.json` points to its execution snapshot; `active_run.json` continues to identify the completed six-job pilot. Four finite jobs cover baseline and entropy-off at seeds 42/123, adding 80 epochs per job. There is no M2 continuation, new intervention, disease evaluation, or final model adoption.

`continue_pilot.py` copies each original epoch-20 checkpoint and restores model, Adam and cosine scheduler states exactly. The original permutation stream is extended to 100 epochs, with its first 20 permutations checked against the parent. Memory stays at 300 and scheduler `T_max` at 1000. Every epoch still uses all 956 normal-training patches. Probes/checkpoints are saved at epochs 20/25/30/40/50/60/75/100; epoch 100 additionally evaluates all normal-training patches. Checkpoints now include RNG states.

The parent did not save RNG state. The resumed phase fixes its RNG per seed, and preflight verifies that a training update consumes no RNG and no active dropout is present. This is not a claim of bitwise equivalence to uninterrupted training. All four jobs passed exact restoration and forward-output checks. An initial strict next-update equality check exposed rounding differences; independent reload controls reproduced them. Final verification records bounded update differences (maximum model element difference 1.86e-9 to 3.73e-9), exact optimizer step counters, and unchanged scheduling. The legacy GPU computation is preserved.

The continuation queue exits after all four jobs complete and pauses new launches on a failure. It is a finite run, not a monitoring service. Inspect its `manifest.json` and each job's `run_state.json`, `history.json`, and `training.log`; never relaunch an active queue.

## M3: entropy off and memory-only decay off

User approved on 2026-11-10; launched at 21:30 KST. `active_m3.json` identifies two fresh 100-epoch jobs, seeds 42/123. `run_m3.py` copies the completed pilot's initial weights and the continuation's exact 100-epoch orders. It freezes the same legacy source, data and learning-rate schedule. Adam decay is zero only for `model.mem_rep.memory`; all other parameters retain 1e-5. Entropy coefficient is zero. The existing completed M1 runs are the controls; they are not retrained in this queue.

Both seeds passed checks for exact initial model/output parity, epoch-zero probe reproduction, correct optimizer coverage/decay groups, full patch coverage and exact order hashes, finite training updates, RNG-invariant probes and forward/update, and checkpoint roundtrip. With shared reconstruction gradients, M1/M3 nonmemory updates were exactly equal and memory updates differed. The epoch-21 RNG/loader phase boundary is matched to the prior M1 continuation, with the existing CUDA rounding and prior RNG-state limitations retained in the manifest.

Each job uses all 956 normal TRAIN patches per epoch, with no validation/test access or selection. Probes at 0/1/3/5/10/20/25/30/40/50/60/75/100 preserve weights and save optimizer, scheduler and RNG states. Alongside the same M1 diagnostics, probes add absolute and norm-relative memory-distance summaries. Frozen M1 endpoint geometry is stored in `control_reference`. Epoch 100 additionally evaluates all normal TRAIN patches. Two finite jobs exit on completion; failure prevents new launches. No staged initialization, warm-up, memory-size change, disease evaluation or final adoption is included.
