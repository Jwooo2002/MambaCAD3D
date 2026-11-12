#!/usr/bin/env python3
"""Recompute fixed-model patient metrics independently from saved patch scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=float)
    lo, q1, med, q3, hi = np.quantile(values, [0, .25, .5, .75, 1])
    return dict(n=len(values), min=float(lo), q1=float(q1), median=float(med),
                q3=float(q3), max=float(hi), mean=float(values.mean()))


def audit(root):
    status = read(root / 'status.json')
    assert status['status'] == 'inference_completed' and status['worker_exit_codes'] == [0, 0]
    protocol = read(root / 'protocol.json')
    inputs = read(root / 'input_manifest.json')
    for name, expected in read(root / 'frozen_hashes.json').items():
        assert digest(root / name) == expected, name
    assert digest(protocol['original_split']) == protocol['original_split_sha256']
    for entry in inputs:
        assert digest(entry['path']) == entry['sha256']
    assert len(inputs) == len({p['id'] for p in inputs}) == 25
    assert sum(p['label'] == 0 for p in inputs) == 5
    assert sum(p['label'] == 1 for p in inputs) == 20
    assert set(protocol['training_files']).isdisjoint(p['filename'] for p in inputs if p['label'] == 0)
    rows = []
    checks = []
    results = {}
    for job in protocol['jobs']:
        path = root / f"{job['id']}.json"
        result = read(path)
        results[(job['variant'], job['seed'])] = result
        assert result['job'] == job and result['epoch'] == 100 and result['patient_aggregation'] == 'max'
        assert digest(job['checkpoint']) == job['checkpoint_sha256']
        assert digest(job['probe']) == job['probe_sha256']
        assert digest(job['config']) == job['config_sha256']
        assert len(result['patients']) == len(inputs)
        labels, scores = [], []
        for patient, entry in zip(result['patients'], inputs):
            assert patient['id'] == entry['id'] and patient['label'] == entry['label']
            values = np.asarray(patient['patch_mse_in_original_order'])
            assert len(values) == patient['patches'] == entry['patches']
            assert np.isfinite(values).all() and (values >= 0).all()
            assert float(values.max()) == patient['patient_score_max']
            assert float(values.mean()) == patient['patient_mean_mse_diagnostic']
            labels.append(patient['label'])
            scores.append(float(values.max()))
        y, s = np.asarray(labels), np.asarray(scores)
        positive, negative = s[y == 1], s[y == 0]
        wins = int((positive[:, None] > negative).sum())
        ties = int((positive[:, None] == negative).sum())
        auc_manual = (wins + .5*ties) / (len(positive)*len(negative))
        # AP sums precision at each distinct score times the increment in recall.
        ap_manual = 0.
        tp = fp = 0
        for value in np.unique(s)[::-1]:
            group = y[s == value]
            added_tp = int(group.sum())
            tp += added_tp
            fp += len(group)-added_tp
            ap_manual += added_tp / len(positive) * tp / (tp+fp)
        assert np.isclose(auc_manual, roc_auc_score(y, s), rtol=0, atol=1e-12)
        assert np.isclose(ap_manual, average_precision_score(y, s), rtol=0, atol=1e-12)
        assert np.isclose(result['AUROC'], auc_manual, rtol=0, atol=1e-12)
        assert np.isclose(result['AP'], ap_manual, rtol=0, atol=1e-12)
        assert result['preflight']['original_loader_first_last_each_subject_exact']
        assert result['preflight']['saved_training_probe_45_patch_mse_reproduced']
        row = {'variant': job['variant'], 'seed': job['seed'], 'AUROC': auc_manual,
               'AP': ap_manual, 'disease_normal_pairwise_wins': wins, 'pairwise_ties': ties,
               'normal_patient_max_distribution': distribution(negative),
               'diseased_patient_max_distribution': distribution(positive)}
        for label, name in ((0, 'normal'), (1, 'diseased')):
            patients = [p for p in result['patients'] if p['label'] == label]
            row[f'{name}_patch_weighted_mean_mse_diagnostic'] = float(np.mean(
                [v for p in patients for v in p['patch_mse_in_original_order']]))
        rows.append(row)
        checks.append({'job': job['id'], 'raw_result_sha256': digest(path),
                       'patients': len(y), 'all_patches': sum(p['patches'] for p in result['patients']),
                       'checkpoint_sha256_unchanged': True, 'saved_probe_reproduced': True,
                       'patient_max_auc_ap_independently_recomputed': True,
                       'max_probe_mse_absolute_difference': result['preflight']['max_patient_probe_mse_abs_difference']})
    means = []
    for variant in protocol['models']:
        selected = [r for r in rows if r['variant'] == variant]
        assert [r['seed'] for r in selected] == [42, 123]
        means.append({'variant': variant, 'AUROC_mean': float(np.mean([r['AUROC'] for r in selected])),
                      'AP_mean': float(np.mean([r['AP'] for r in selected]))})
    deltas = []
    for seed in protocol['seeds']:
        selected = {r['variant'].split('_')[0]: r for r in rows if r['seed'] == seed}
        for candidate, reference in (('M1', 'M0'), ('M3', 'M0'), ('M3', 'M1')):
            deltas.append({'seed': seed, 'comparison': f'{candidate}-{reference}',
                           'AUROC_difference': selected[candidate]['AUROC']-selected[reference]['AUROC'],
                           'AP_difference': selected[candidate]['AP']-selected[reference]['AP']})
    summary = {'status': 'completed_and_audited', 'audited_utc': datetime.now(timezone.utc).isoformat(),
               'started_utc': status['started_utc'], 'inference_finished_utc': status['finished_utc'],
               'scope': protocol['scope'], 'fold': 1, 'epoch': 100, 'aggregation': 'max',
               'normal_patients': 5, 'diseased_patients': 20,
               'normal_patches': sum(p['patches'] for p in inputs if p['label'] == 0),
               'diseased_patches': sum(p['patches'] for p in inputs if p['label'] == 1),
               'positive_prevalence': .8, 'rows': rows, 'descriptive_seed_means': means,
               'paired_seed_deltas': deltas,
               'limitations': ['Single fold and two seeds; no winner selection or final adoption.',
                              'Previously evaluated subjects, not untouched validation/test.',
                              'Only five normal subjects; seeds reuse the same subjects.',
                              'AP must be interpreted with 80% disease prevalence.',
                              'Fixed epoch 100 is a diagnostic endpoint, not an optimum or full convergence claim.']}
    write(root / 'results_summary.json', summary)
    write(root / 'completed_audit.json', {'status': 'passed', 'audited_utc': summary['audited_utc'],
           'frozen_sources_protocol_inputs_verified': True, 'normal_training_eval_disjoint': True,
           'checks': checks})
    shutil.copy2(__file__, root / 'audit_fixed_evaluation.py')
    make_plot(root, protocol, results)
    print(json.dumps({'rows': rows, 'means': means, 'deltas': deltas}, indent=2), flush=True)


def make_plot(root, protocol, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, layout='constrained')
    for ax, seed in zip(axes, protocol['seeds']):
        for i, variant in enumerate(protocol['models']):
            result = results[(variant, seed)]
            for label, name, color, shift in ((0, 'Normal (n=5)', '#2166ac', -.15),
                                               (1, 'Diseased (n=20)', '#b2182b', .15)):
                values = [p['patient_score_max'] for p in result['patients'] if p['label'] == label]
                jitter = np.linspace(-.065, .065, len(values))
                ax.scatter(i+shift+jitter, values, s=28, alpha=.75, color=color,
                           label=name if i == 0 else None)
                ax.plot([i+shift-.08, i+shift+.08], [np.median(values)]*2, color=color, lw=2.2)
        ax.set(xticks=range(3), xticklabels=['M0', 'M1', 'M3'], title=f'Seed {seed}', yscale='log')
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8, loc='upper right')
    axes[0].set_ylabel('Patient max patch MSE (log scale)')
    fig.suptitle('Fixed epoch 100 / all 1,547 patches / reused-data exploratory evaluation', fontsize=11)
    fig.savefig(root / 'patient_score_distributions.png', dpi=180)
    fig.savefig(root / 'patient_score_distributions.pdf')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    audit(p.parse_args().root.resolve())
