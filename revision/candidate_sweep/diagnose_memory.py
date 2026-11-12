#!/usr/bin/env python3
"""Read-only probes of saved models; never changes training or evaluation scores."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = args.sweep.resolve()
    sweep = json.loads((root / "sweep_manifest.json").read_text())
    source = Path(sweep["jobs"][0]["output"]) / "source_snapshot/main_files"
    sys.path.insert(0, str(source))
    import candidate_hooks
    from model import MemAE_VSSM3D
    splits = json.loads((root / "split_manifest.json").read_text())
    baseline = Path(sweep["baseline"])
    config = json.loads((baseline / "effective_config.json").read_text())
    cache = {}

    def samples(label, name):
        key = (label, name)
        if key not in cache:
            data_root = config["diseased_data_root" if label else "normal_data_root"]
            with (Path(data_root) / name).open("rb") as handle:
                clips = pickle.load(handle)
            indices = np.unique(np.linspace(0, len(clips) - 1, min(3, len(clips)), dtype=int))
            # Match original evaluation batches (size 2), including each selected
            # patch's neighbour, to avoid batch-size-dependent GPU rounding.
            groups = []
            for start in sorted({(int(i) // 2) * 2 for i in indices}):
                original_indices = list(range(start, min(start + 2, len(clips))))
                array = np.stack([np.asarray(clips[i], dtype=np.float32) for i in original_indices])
                groups.append(((torch.from_numpy(array).unsqueeze(1) + 300) / 700, original_indices))
            cache[key] = groups
        return cache[key]

    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Read-only diagnosis on fixed original evaluation batches containing first/middle/last patches per test patient. Full-test saved metrics are unchanged. This sample is for mechanism inspection, not new performance claims.",
              "rows": []}
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text())
    finished = {(r["id"], r["fold"]) for r in report["rows"]}
    jobs = [("R0", baseline)] + [(j["variant"], Path(j["output"])) for j in sweep["jobs"]]
    # All jobs have identical architecture. Only opt-in forward behavior differs.
    for name, output in jobs:
        state = json.loads((output / "run_state.json").read_text())
        cfg = json.loads((output / "effective_config.json").read_text())
        options = json.loads((output / "variant.json").read_text()) if name != "R0" else {}
        for fold in range(1, 5):
            if (name, fold) in finished:
                continue
            checkpoint = Path(state["legacy_results_directory"]) / "models" / f"fold_{fold}_model.pt"
            if not checkpoint.exists():
                continue
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            candidate_hooks.configure(options, output)
            model = MemAE_VSSM3D(**{k: cfg[k] for k in ("in_chans", "dims", "depths", "ssm_d_state", "mem_dim")})
            model.load_state_dict(saved["model_state_dict"], strict=True)
            model.cuda().eval()
            token_count = patch_count = 0
            scalars = {k: 0.0 for k in ("entropy", "effective_slots", "top1", "logit_sd", "query_norm", "recon_mse")}
            attention_sum = torch.zeros(cfg["mem_dim"], dtype=torch.float64)
            winners = torch.zeros(cfg["mem_dim"], dtype=torch.float64)
            recon_sum = recon_square_sum = input_sum = input_square_sum = None
            readout_sum = None
            stored_score_errors = []
            dt_diagnostics = []
            handles = []
            for layer_name, module in model.named_modules():
                if not hasattr(module, "dt_proj"):
                    continue

                def hook(projection, inputs, result, ss=module, ln=layer_name):
                    delta = F.softplus(result.detach()[..., ::max(1, result.shape[-1] // 128)])
                    decay = torch.exp(-delta * ss.A_log.detach()[:, 0].exp()[None, :, None])
                    dt_diagnostics.append({"layer": ln, "delta_mean": delta.mean().item(),
                                           "delta_median": delta.median().item(),
                                           "slowest_state_retention_mean": decay.mean().item()})

                handles.append(module.dt_proj.register_forward_hook(hook))
            split = splits[fold - 1]
            for label, files in ((0, split["normal_test_and_validation"]), (1, split["diseased_test"])):
                for filename in files:
                    for batch, indices in samples(label, filename):
                        x = batch.cuda()
                        with torch.no_grad():
                            z = model.encode(x)
                            for handle in handles:
                                handle.remove()
                            handles.clear()
                            readout, attention = model.mem_rep(z)
                            reconstruction = model.decode(readout)
                            flat_attention = attention.reshape(-1, cfg["mem_dim"])
                            entropy = -(flat_attention * flat_attention.clamp_min(1e-30).log()).sum(-1)
                            query = z.flatten(2).transpose(1, 2)
                            logits = candidate_hooks.memory_logits(query, model.mem_rep.memory)
                            scalars["entropy"] += entropy.sum().item()
                            scalars["effective_slots"] += entropy.exp().sum().item()
                            scalars["top1"] += flat_attention.max(-1).values.sum().item()
                            scalars["logit_sd"] += logits.std(-1).sum().item()
                            scalars["query_norm"] += query.norm(dim=-1).sum().item()
                            attention_sum += flat_attention.sum(0).double().cpu()
                            winners += torch.bincount(flat_attention.argmax(-1), minlength=cfg["mem_dim"]).double().cpu()
                            readout_total = readout.permute(0, 2, 3, 4, 1).reshape(-1, z.shape[1]).sum(0).double().cpu()
                            readout_sum = readout_total if readout_sum is None else readout_sum + readout_total
                            token_count += len(flat_attention)
                            errors = (reconstruction - x).square().flatten(1).mean(1)
                            scalars["recon_mse"] += errors.sum().item()
                            key = "diseased_patient_scores" if label else "normal_patient_scores"
                            expected = np.asarray(saved["results"][key][Path(filename).stem])[indices]
                            stored_score_errors.extend(np.abs(errors.cpu().numpy() - expected).tolist())
                            assert np.allclose(errors.cpu().numpy(), expected, rtol=1e-4, atol=1e-7), (name, fold, filename, indices, expected.tolist(), errors.cpu().numpy().tolist())
                            r = reconstruction.double().cpu().flatten(1)
                            inp = x.double().cpu().flatten(1)
                            recon_sum = r.sum(0) if recon_sum is None else recon_sum + r.sum(0)
                            recon_square_sum = r.square().sum(0) if recon_square_sum is None else recon_square_sum + r.square().sum(0)
                            input_sum = inp.sum(0) if input_sum is None else input_sum + inp.sum(0)
                            input_square_sum = inp.square().sum(0) if input_square_sum is None else input_square_sum + inp.square().sum(0)
                            patch_count += len(x)
            for key in scalars:
                scalars[key] /= patch_count if key == "recon_mse" else token_count
            global_attention = attention_sum / token_count
            global_h = -(global_attention * global_attention.clamp_min(1e-30).log()).sum().item()
            readout_mean = (readout_sum / token_count).float().cuda()
            with torch.no_grad():
                constant_z = readout_mean[None, :, None, None, None].expand(1, -1, *z.shape[2:]).contiguous()
                constant_recon = model.decode(constant_z).flatten().double().cpu()
                constant_delta = (recon_square_sum / patch_count - 2 * constant_recon * recon_sum / patch_count + constant_recon.square()).mean().item()
                reconstruction_var = (recon_square_sum / patch_count - (recon_sum / patch_count).square()).mean().item()
                input_var = (input_square_sum / patch_count - (input_sum / patch_count).square()).mean().item()
                memory = model.mem_rep.memory.detach()
                keys = F.normalize(memory, dim=-1)
                off_diagonal = (keys @ keys.T)[~torch.eye(len(memory), device=memory.device, dtype=torch.bool)]
                memory_norm_mean = memory.norm(dim=-1).mean().item()
                memory_cosine_mean = off_diagonal.mean().item()
            # Separate gradient magnitudes; loss-term magnitude alone is not gradient dominance.
            train_sample, _ = samples(0, split["train"][0])[0]
            x_train = train_sample[:1].cuda()
            z_train = model.encode(x_train)
            readout, att = model.mem_rep(z_train)
            reconstruction = model.decode(readout)
            mse = F.mse_loss(reconstruction, x_train)
            penalty = cfg["entropy_loss_weight"] * -(att * torch.log(att + 1e-12)).sum(1).mean()
            reconstruction_grads = torch.autograd.grad(mse, (z_train, model.mem_rep.memory), retain_graph=True)
            entropy_grads = torch.autograd.grad(penalty, (z_train, model.mem_rep.memory))
            grad_ratios = {key: e.norm().item() / max(r.norm().item(), 1e-30)
                           for key, r, e in zip(("query", "memory"), reconstruction_grads, entropy_grads)}
            row = {"id": name, "fold": fold, "patches_probed": patch_count, **scalars,
                   "global_attention_effective_slots": float(np.exp(global_h)),
                   "argmax_distinct_slots": int((winners > 0).sum()),
                   "most_common_argmax_fraction": float(winners.max() / token_count),
                   "global_attention_max_mass": float(global_attention.max()),
                   "memory_norm_mean": memory_norm_mean, "memory_key_cosine_offdiag_mean": memory_cosine_mean,
                   "reconstruction_variance_over_input_variance": reconstruction_var / max(input_var, 1e-30),
                   "constant_readout_output_difference_over_recon_error": constant_delta / scalars["recon_mse"],
                   "sample_scores_max_abs_error_vs_saved": max(stored_score_errors),
                   "entropy_to_reconstruction_gradient_norm_ratio": grad_ratios,
                   "dt": dt_diagnostics}
            report["rows"].append(row)
            report["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({k: row[k] for k in ("id", "fold", "effective_slots", "top1", "argmax_distinct_slots", "most_common_argmax_fraction", "constant_readout_output_difference_over_recon_error")}), flush=True)
            del model, saved, z_train, reconstruction, reconstruction_grads, entropy_grads, mse, penalty
            torch.cuda.empty_cache()
    report["status"] = "complete"
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
