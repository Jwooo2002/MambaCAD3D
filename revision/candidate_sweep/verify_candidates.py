#!/usr/bin/env python3
"""Check isolation, baseline equivalence, candidate gradients and sampler mass."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    args = parser.parse_args()
    root = args.sweep.resolve()
    manifest = json.loads((root / "sweep_manifest.json").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = manifest["gpu"]
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    source = Path(manifest["jobs"][0]["output"]) / "source_snapshot/main_files"
    sys.path.insert(0, str(source))
    import candidate_hooks as hooks
    from data.VideoDataset import VideoDataset
    original = load_module("original_model", Path(manifest["baseline"]) / "source_snapshot/main_files/model.py")
    candidate = load_module("candidate_model", source / "model.py")
    config = json.loads((Path(manifest["baseline"]) / "effective_config.json").read_text())
    kwargs = {k: config[k] for k in ("in_chans", "depths", "dims", "ssm_d_state", "mem_dim")}
    torch.manual_seed(42)
    baseline = original.MemAE_VSSM3D(**kwargs)
    baseline_state = {k: v.clone() for k, v in baseline.state_dict().items()}
    hooks.configure({}, root)
    torch.manual_seed(42)
    unchanged = candidate.MemAE_VSSM3D(**kwargs)
    assert all(torch.equal(v, unchanged.state_dict()[k]) for k, v in baseline_state.items())
    assert sum(p.numel() for p in unchanged.parameters()) == 1720121
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x + 300) / 700)])
    split = json.loads((root / "split_manifest.json").read_text())[0]
    data = VideoDataset(config["normal_data_root"], transform, [split["train"][0]])
    batch = torch.stack([data[0][1], data[1][1]]).cuda()
    baseline.cuda().eval()
    unchanged.cuda().eval()
    with torch.no_grad():
        expected, expected_attention = baseline(batch)
        actual, actual_attention = unchanged(batch)
    assert torch.equal(expected, actual), (expected - actual).abs().max().item()
    assert torch.equal(expected_attention, actual_attention)
    del baseline, unchanged, expected, actual, expected_attention, actual_attention
    torch.cuda.empty_cache()

    # Every patient has equal total probability, despite unequal patch counts.
    sampler = hooks.sampling_from_counts([1, 9, 40])
    assert sampler.num_samples == 50 and sampler.replacement
    assert torch.allclose(torch.stack([sampler.weights[:1].sum(), sampler.weights[1:10].sum(),
                                      sampler.weights[10:].sum()]), torch.ones(3, dtype=torch.double))
    hooks.configure({"cosine": True}, root)
    q, m = torch.randn(2, 5, 7), torch.randn(11, 7)
    assert torch.allclose(hooks.memory_logits(q, m), hooks.memory_logits(q * 3, m * 4), atol=1e-6)
    report = {"status": "passed", "baseline_parameter_count": 1720121,
              "baseline_state_and_forward_exact": True, "sampler_equal_patient_mass": True,
              "cosine_scale_invariance": True, "batch_shape": list(batch.shape), "variants": []}
    for job in manifest["jobs"]:
        output = Path(job["output"])
        options = json.loads((output / "variant.json").read_text())
        hooks.configure(options, output)
        torch.manual_seed(42)
        model = candidate.MemAE_VSSM3D(**kwargs)
        changed = [k for k, v in model.state_dict().items() if not torch.equal(v, baseline_state[k])]
        if options.get("explicit_dt"):
            assert changed and all("dt_proj" in k for k in changed)
            for module in model.modules():
                if hasattr(module, "dt_proj"):
                    dt = F.softplus(module.dt_proj.bias)
                    assert dt.min() >= 0.001 - 1e-7 and dt.max() <= 0.1 + 1e-7
        else:
            assert changed == []
        model.cuda().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
        reconstruction, attention = model(batch)
        assert reconstruction.shape == batch.shape
        assert torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=1e-5)
        reconstruction_loss = F.mse_loss(reconstruction, batch)
        entropy = -(attention * torch.log(attention + 1e-12)).sum(1).mean()
        loss = reconstruction_loss + options.get("entropy_loss_weight", config["entropy_loss_weight"]) * entropy
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        report["variants"].append({"variant": job["variant"], "loss": float(loss.detach()),
                                   "finite_forward_backward_update": True, "changed_initial_parameters": changed})
        del model, optimizer, reconstruction, attention, loss, entropy, reconstruction_loss
        torch.cuda.empty_cache()
    for job in manifest["jobs"]:
        output = Path(job["output"])
        ec = json.loads((output / "effective_config.json").read_text())
        allowed = {"save_path", "run_name", "gpu_id", "entropy_loss_weight"}
        assert {k for k in config if config[k] != ec[k]} <= allowed
        import hashlib
        for name, expected in manifest["baseline_source_hashes"].items():
            if name not in {"model.py", "core/model_trainer.py", "core/experiment_manager.py"}:
                assert hashlib.sha256((output / "source_snapshot/main_files" / name).read_bytes()).hexdigest() == expected
    report["evaluation_sources_unchanged"] = True
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "variants"}))
    print("All six candidates: finite forward, backward and optimizer update.")


if __name__ == "__main__":
    main()
