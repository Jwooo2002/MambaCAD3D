#!/usr/bin/env python3
"""Frozen-model memory interventions on all existing test patches; no training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score, average_precision_score
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = args.sweep.resolve()
    manifest = json.loads((root / 'sweep_manifest.json').read_text())
    baseline = Path(manifest['baseline'])
    source = Path(manifest['jobs'][0]['output']) / 'source_snapshot/main_files'
    sys.path.insert(0, str(source))
    import candidate_hooks
    from model import MemAE_VSSM3D
    splits = json.loads((root / 'split_manifest.json').read_text())
    cfg = json.loads((baseline / 'effective_config.json').read_text())
    cache = {}

    def data(label, filename):
        key = (label, filename)
        if key not in cache:
            folder = cfg['diseased_data_root' if label else 'normal_data_root']
            with (Path(folder) / filename).open('rb') as handle:
                clips = pickle.load(handle)
            x = torch.from_numpy(np.stack(clips).astype(np.float32)).unsqueeze(1)
            cache[key] = (x + 300) / 700
        return cache[key]

    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Frozen-model interventions, all original test patches. No fitting, retraining, hyperparameter selection, or changes to original predictions. Constant train readout is estimated from fixed first/middle/last patches of normal training patients only. Compression keeps slot multiplicity as a log-count prior.',
              'status': 'running', 'rows': []}
    jobs = [('R0', baseline)] + [(j['variant'], Path(j['output'])) for j in manifest['jobs']
                                if j['variant'] in ('R3_cosine', 'R5_dt_lambda')]
    modes = ['original', 'uniform_attention', 'fixed_train_mean', 'shuffled_values',
             'hard_top1', 'compressed_duplicates']
    for name, output in jobs:
        state = json.loads((output / 'run_state.json').read_text())
        options = json.loads((output / 'variant.json').read_text()) if name != 'R0' else {}
        candidate_hooks.configure(options, output)
        for fold in range(1, 5):
            if name == 'R5_dt_lambda' and fold != 3:
                continue
            checkpoint = Path(state['legacy_results_directory']) / 'models' / f'fold_{fold}_model.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model = MemAE_VSSM3D(**{k: cfg[k] for k in ('in_chans', 'dims', 'depths', 'ssm_d_state', 'mem_dim')})
            model.load_state_dict(saved['model_state_dict'], strict=True)
            model.cuda().eval()
            memory = model.mem_rep.memory.detach()
            # Identify repeated rows from weights alone; tolerance fixed before scoring.
            norms = memory.norm(dim=1)
            center = memory[norms.argsort()[len(memory) // 2]]
            distance = (memory - center).norm(dim=1)
            repeated = distance <= 1e-6
            count = int(repeated.sum())
            representative = memory[repeated].mean(0, keepdim=True)
            unique_values = torch.cat((representative, memory[~repeated]), dim=0)
            multiplicity = torch.ones(len(unique_values), device='cuda')
            multiplicity[0] = count
            sv = torch.linalg.svdvals((memory - memory.mean(0)).double())
            energy = sv.square() / sv.square().sum()
            perm = torch.randperm(len(memory), generator=torch.Generator().manual_seed(741))
            shuffled = memory[perm.to(memory.device)]
            train_readout_sum = torch.zeros(memory.shape[1], dtype=torch.float64)
            train_tokens = 0
            with torch.no_grad():
                for filename in splits[fold - 1]['train']:
                    clips = data(0, filename)
                    indices = np.unique(np.linspace(0, len(clips)-1, min(3, len(clips)), dtype=int))
                    x = clips[indices].cuda()
                    z = model.encode(x)
                    readout, _ = model.mem_rep(z)
                    flat = readout.movedim(1, -1).reshape(-1, memory.shape[1])
                    train_readout_sum += flat.sum(0).double().cpu()
                    train_tokens += len(flat)
                shape = z.shape[2:]
                constants = {}
                for mode, vector in [('uniform_attention', memory.mean(0)),
                                     ('fixed_train_mean', (train_readout_sum / train_tokens).float().cuda())]:
                    fixed = vector[None, :, None, None, None].expand(1, -1, *shape).contiguous()
                    constants[mode] = model.decode(fixed)

                patients = []
                counts = {mode: [0., 0., 0., 0.] for mode in modes}  # normal sum/count, diseased sum/count
                output_difference = {mode: 0. for mode in modes}
                patch_count = 0
                max_saved_difference = 0.
                joint_permutation_difference = None
                compressed_readout_difference = 0.
                for label, files in ((0, splits[fold-1]['normal_test_and_validation']),
                                     (1, splits[fold-1]['diseased_test'])):
                    for filename in files:
                        clips = data(label, filename)
                        scores = {mode: [] for mode in modes}
                        expected = np.asarray(saved['results']['diseased_patient_scores' if label else 'normal_patient_scores'][Path(filename).stem])
                        for start in range(0, len(clips), 2):
                            x = clips[start:start+2].cuda()
                            z = model.encode(x)
                            readout, attention = model.mem_rep(z)
                            q = z.flatten(2).transpose(1, 2)
                            original = model.decode(readout)

                            def unpack(v):
                                return v.transpose(1, 2).reshape_as(readout)

                            if joint_permutation_difference is None:
                                logits = candidate_hooks.memory_logits(q, shuffled)
                                att = logits.softmax(-1)
                                att = F.normalize(F.hardtanh(att, 0., 1.), p=1, dim=2)
                                joint = unpack(att @ shuffled)
                                joint_permutation_difference = (joint-readout).abs().max().item()
                                assert torch.allclose(joint, readout, atol=1e-6, rtol=1e-4)

                            if count > 1:
                                logits = candidate_hooks.memory_logits(q, unique_values) + multiplicity.log()
                                compressed = unpack(logits.softmax(-1) @ unique_values)
                                compressed_readout_difference = max(compressed_readout_difference, (compressed-readout).abs().max().item())
                            else:
                                compressed = readout
                            reconstructions = {'original': original,
                                               'uniform_attention': constants['uniform_attention'].expand_as(x),
                                               'fixed_train_mean': constants['fixed_train_mean'].expand_as(x),
                                               'shuffled_values': model.decode(unpack(attention @ shuffled)),
                                               'hard_top1': model.decode(unpack(memory[attention.argmax(-1)])),
                                               'compressed_duplicates': model.decode(compressed) if count > 1 else original}
                            for mode, reconstruction in reconstructions.items():
                                errors = (reconstruction-x).square().flatten(1).mean(1)
                                assert torch.isfinite(errors).all()
                                values = errors.cpu().numpy()
                                scores[mode].extend(values.tolist())
                                counts[mode][2*label] += errors.sum().item()
                                counts[mode][2*label+1] += len(x)
                                output_difference[mode] += (reconstruction-original).square().flatten(1).mean(1).sum().item()
                                if mode == 'original':
                                    error = float(np.max(np.abs(values-expected[start:start+2])))
                                    max_saved_difference = max(max_saved_difference, error)
                                    assert np.allclose(values, expected[start:start+2], atol=1e-7, rtol=1e-4), (name, fold, filename, start, error)
                            patch_count += len(x)
                        patients.append({'patient_id': Path(filename).stem, 'label': label,
                                         'patch_count': len(clips),
                                         'scores': {mode: {'max': float(np.max(v)), 'mean': float(np.mean(v))} for mode, v in scores.items()}})
                metrics = {}
                y = [p['label'] for p in patients]
                for mode in modes:
                    metrics[mode] = {agg: {'auroc': float(roc_auc_score(y, [p['scores'][mode][agg] for p in patients])),
                                           'ap': float(average_precision_score(y, [p['scores'][mode][agg] for p in patients]))}
                                     for agg in ('max', 'mean')}
                    c = counts[mode]
                    metrics[mode].update(normal_mse=c[0]/c[1], diseased_mse=c[2]/c[3],
                                         output_mse_change=output_difference[mode]/patch_count)
                row = {'id': name, 'fold': fold, 'checkpoint': str(checkpoint), 'patches': patch_count,
                       'repeated_slots_abs_l2_tolerance_1e_minus6': count,
                       'compressed_slots': len(unique_values),
                       'repeated_max_l2_distance': float(distance[repeated].max()),
                       'centered_memory_rank99': int((energy.cumsum(0)<.99).sum())+1,
                       'centered_memory_top4_variance_fraction': float(energy[:4].sum()),
                       'original_max_abs_score_difference_vs_saved': max_saved_difference,
                       'joint_key_value_permutation_max_abs_readout_difference': joint_permutation_difference,
                       'compressed_max_abs_readout_difference': compressed_readout_difference,
                       'metrics': metrics, 'patients': patients}
                report['rows'].append(row)
                report['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                print(json.dumps({'id':name, 'fold':fold, 'duplicate_slots':count,
                                  'auroc':{k:v['max']['auroc'] for k,v in metrics.items()},
                                  'normal_mse_ratio':{k:v['normal_mse']/metrics['original']['normal_mse'] for k,v in metrics.items()}}), flush=True)
            del model, saved
            torch.cuda.empty_cache()
    report['status'] = 'complete'
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
