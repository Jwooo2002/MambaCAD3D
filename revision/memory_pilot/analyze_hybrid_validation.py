#!/usr/bin/env python3
"""Read-only, fixed-epoch analysis of the prespecified hybrid validation suite."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from summarize_hybrid_audit import gradients, intervention_summary


EXPECTED = {
    'V0_s3_replay', 'V1_s3_decoder_fanin', 'V2_adam_zero_decay',
    'V3_adamw_explicit_groups', 'V4_s3_direct_only', 'V5_s3_fanin_direct_only',
}


def read(path):
    return json.loads(path.read_text())


def brief_probe(data):
    memory, ssm = data['memory'], data['ssm']
    dt = [p for p in ssm['parameters'] if p['name'].endswith('dt_proj.weight')]
    assert len(dt) == 6 and len(ssm['functional']) == 18
    row = {
        'epoch': data['epoch'], 'normal_45patch_mse': memory['mse'],
        'query_rank99_45patch': memory['feature_covariance']['query']['rank99'],
        'readout_rank99_45patch': memory['feature_covariance']['readout']['rank99'],
        'memory_rank99': memory['memory_rank99'],
        'global_effective_slots_45patch': memory['global_effective_slots'],
        'mean_top1_mass_45patch': memory['mean_top1_mass'],
        'dt_weight_l2_min': min(p['l2_float64'] for p in dt),
        'dt_gradient_connected_blocks': sum(p.get('task_gradient_connected', True) for p in dt),
        'dt_all_subnormal_blocks': sum(p['nonzero_subnormal'] == p['numel'] for p in dt),
        'delta_bias_only_observations': sum(p['delta_equals_bias_only'] for p in ssm['functional']),
        'delta_observations': len(ssm['functional']),
        'recurrence_over_Du_median': statistics.median(p['raw_recurrent_over_Du_rms'] for p in ssm['functional']),
        'recurrence_zero_observations': sum(p['raw_recurrent_rms'] == 0 for p in ssm['functional']),
    }
    if 'hybrid' in ssm:
        row['gradient_rms_mean_3patch'] = gradients(ssm['hybrid']['gradient_traces'])
        row['inference_intervention_3patch'] = intervention_summary(ssm['hybrid']['recurrence_interventions'])
    return row


def completed_result(directory, expected_seed, expected_variant, frozen_config):
    state = read(directory / 'run_state.json')
    if state['status'] != 'complete':
        return None
    history = read(directory / 'history.json')
    assert state['completed_epochs'] == frozen_config['epochs'] == 200
    assert state['seed'] == expected_seed and state['variant'] == expected_variant
    assert [r['epoch'] for r in history] == list(range(1, 201))
    assert all(r['patches'] == 956 and r['optimizer_steps'] == 478 for r in history)
    for row in history:
        expected_lr = frozen_config['eta_min'] + (frozen_config['lr']-frozen_config['eta_min']) * (
            1 + math.cos(math.pi*(row['epoch']-1)/frozen_config['scheduler_T_max']))/2
        assert math.isfinite(row['mse']) and math.isclose(row['lr'], expected_lr, rel_tol=1e-10, abs_tol=1e-14)
    for epoch in frozen_config['checkpoint_epochs']:
        path = directory / f'epoch_{epoch:03d}.pt'
        assert path.is_file() and path.stat().st_size > 0
    checkpoint_path = directory / 'epoch_200.pt'
    assert checkpoint_path.is_file()
    import torch
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert saved['epoch'] == 200 and saved['seed'] == expected_seed and saved['variant'] == expected_variant
    assert saved['config'] == frozen_config
    scheduler = saved['scheduler_state_dict']
    assert scheduler['last_epoch'] == 200 and scheduler['T_max'] == frozen_config['scheduler_T_max']
    assert scheduler['eta_min'] == frozen_config['eta_min']
    assert all(lr == frozen_config['lr'] for lr in scheduler['base_lrs'])
    assert [g['lr'] for g in saved['optimizer_state_dict']['param_groups']] == scheduler['_last_lr']
    assert all(torch.isfinite(value).all() for value in saved['model_state_dict'].values())
    assert {int(s['step']) for s in saved['optimizer_state_dict']['state'].values()} == {95600}
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    del saved
    for epoch in frozen_config['probe_epochs']:
        assert (directory / f'probe_epoch_{epoch:03d}.json').is_file()
    full = read(directory / 'full_normal_training_epoch_200.json')
    assert full['patches'] == 956 and len(full['patients']) == 15 and full['epoch'] == 200
    result = brief_probe(read(directory / 'probe_epoch_200.json'))
    result.update(full_train_956patch_mse=full['mse'],
                  full_train_query_rank99=full['feature_covariance']['query']['rank99'],
                  full_train_readout_rank99=full['feature_covariance']['readout']['rank99'],
                  full_train_global_effective_slots=full['global_effective_slots'],
                  total_training_seconds=sum(r['training_seconds'] for r in history),
                  checkpoint_sha256=checkpoint_sha256,
                  planned_checkpoint_files_present=frozen_config['checkpoint_epochs'],
                  full_lr_history_and_final_scheduler_verified=True,
                  checkpoint_metadata_finite_weights_and_steps_verified=True)
    return result


def analyze(root, destination, require_complete, require_final_audit=False):
    manifest = read(root / 'manifest.json')
    assert manifest['config'] == read(root / 'config.json')
    assert set(manifest['variants']) == EXPECTED
    assert {(j['variant'], j['seed']) for j in manifest['jobs']} == {(v, s) for v in EXPECTED for s in (42, 123)}
    assert len(manifest['jobs']) == 12
    # Detect source/input-definition mutation; mutable outputs are not frozen files.
    for name, expected in read(root / 'source_hashes.json').items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    jobs, final, trajectories, final_audits = [], {}, {}, []
    for job in manifest['jobs']:
        directory = root / job['id']
        state_path = directory / 'run_state.json'
        state = read(state_path) if state_path.exists() else {}
        rows = [brief_probe(read(path)) for path in sorted(directory.glob('probe_epoch_*.json'))]
        trajectories[job['id']] = rows
        record = {'job': job['id'], 'variant': job['variant'], 'seed': job['seed'],
                  'manifest_status': job['status'], 'run_status': state.get('status', 'not_started'),
                  'completed_epochs': state.get('completed_epochs', 0),
                  'latest_probe': rows[-1] if rows else None}
        if state.get('status') == 'complete':
            result = completed_result(directory, job['seed'], job['variant'], manifest['config'])
            assert result is not None
            if manifest['variants'][job['variant']]['direct_only']:
                assert result['dt_gradient_connected_blocks'] == 0
                assert result['recurrence_zero_observations'] == 18
                dependence = result['inference_intervention_3patch']['all_recurrence_off']
                assert dependence['bitwise_equal_reconstructions'] == dependence['patch_count'] == 3
            record['fixed_epoch200'] = result
            final[(job['variant'], job['seed'])] = result
            audit_path = root / 'final_checkpoint_audit' / f"{job['id']}.json"
            if audit_path.exists():
                audit = read(audit_path)
                assert audit['checkpoint_sha256'] == result['checkpoint_sha256']
                assert audit['job'] == job['id'] and audit['epoch'] == 200 and audit['state_and_rng_unchanged']
                assert len(audit['patch_indices']) == len(set(audit['patch_indices'])) == 45
                expected = {(index, name) for index in audit['patch_indices'] for name in (
                    'all_recurrence_off', 'stage_0_recurrence_off', 'stage_1_recurrence_off', 'stage_2_recurrence_off')}
                assert len(audit['interventions']) == 180
                assert {(r['patch_index'], r['intervention']) for r in audit['interventions']} == expected
                assert all(r['noop_bitwise_equal'] for r in audit['interventions'])
                if manifest['variants'][job['variant']]['direct_only']:
                    assert all(r['differences']['reconstruction']['bitwise_equal'] for r in audit['interventions'])
                record['final_inference_interventions_45patch'] = intervention_summary(audit['interventions'])
                final_audits.append(job['id'])
        jobs.append(record)

    # Use historical S3/S4 only as named historical references, never as a replacement for V0.
    parent = Path(manifest['parent_ssm_repair'])
    parent_manifest = read(parent / 'manifest.json')
    historical = {}
    for variant in ('S3_all_ssm_no_decay', 'S4_adamw'):
        for seed in (42, 123):
            record = completed_result(parent / f'{variant}_seed{seed}', seed, variant, parent_manifest['config'])
            assert record is not None
            historical[(variant, seed)] = record
    comparisons = [
        ('decoder_initialization_with_recurrence', 'V1_s3_decoder_fanin', 'V0_s3_replay'),
        ('decoder_initialization_without_recurrence', 'V5_s3_fanin_direct_only', 'V4_s3_direct_only'),
        ('recurrence_removed_baseline_initialization', 'V4_s3_direct_only', 'V0_s3_replay'),
        ('recurrence_removed_fanin_initialization', 'V5_s3_fanin_direct_only', 'V1_s3_decoder_fanin'),
        ('ssm_replay_vs_historical', 'V0_s3_replay', 'S3_all_ssm_no_decay'),
        ('adam_zero_vs_historical_adamw', 'V2_adam_zero_decay', 'S4_adamw'),
        ('explicit_groups_vs_historical_adamw', 'V3_adamw_explicit_groups', 'S4_adamw'),
    ]
    contrasts = []
    for name, treatment, control in comparisons:
        for seed in (42, 123):
            a, b = final.get((treatment, seed)), {**historical, **final}.get((control, seed))
            if a is None or b is None:
                continue
            av, bv = a['full_train_956patch_mse'], b['full_train_956patch_mse']
            contrasts.append({'comparison': name, 'seed': seed, 'treatment': treatment, 'control': control,
                              'mse_treatment': av, 'mse_control': bv, 'mse_difference': av-bv,
                              'mse_ratio': av/bv, 'mse_relative_change_percent': 100*(av/bv-1),
                              'historical_reference': control.startswith('S')})
    interactions = []
    for seed in (42, 123):
        names = ('V0_s3_replay', 'V1_s3_decoder_fanin', 'V4_s3_direct_only', 'V5_s3_fanin_direct_only')
        if all((name, seed) in final for name in names):
            v0, v1, v4, v5 = [final[(name, seed)]['full_train_956patch_mse'] for name in names]
            interactions.append({'seed': seed, 'definition': '(V5-V1)-(V4-V0)',
                                 'recurrence_removal_effect_fanin_minus_base_initialization_mse': (v5-v1)-(v4-v0),
                                 'interpretation': 'Descriptive interaction on raw MSE scale; no significance test.'})
    suite_complete = manifest['status'] == 'complete' and len(final) == 12 and all(
        j['status'] == 'complete' and j.get('returncode') == 0 for j in manifest['jobs'])
    audit_summary = root / 'final_checkpoint_audit' / 'summary.json'
    final_audit_complete = len(final_audits) == 12 and audit_summary.exists()
    if final_audit_complete:
        audit = read(audit_summary)
        assert audit['status'] == 'complete' and audit['jobs'] == 12 and audit['patches_per_model'] == 45
    result = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'suite_complete': suite_complete, 'completed_runs_verified': len(final), 'expected_runs': 12,
        'final45_audit_complete': final_audit_complete, 'final45_audits_verified': len(final_audits),
        'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_hashes_verified': True, 'jobs': jobs, 'trajectories': trajectories, 'fixed_epoch200_contrasts': contrasts,
        'factorial_interactions': interactions,
        'scope': 'Normal TRAIN mechanism verification, fixed epoch200. No clinical performance or winner selection.',
        'interpretation_rules': [
            'Partial trajectories are diagnostics, not fixed-epoch comparisons.',
            'Report both seeds, including opposite signs; no inferential CI from two seeds.',
            'Training MSE improvement, gradient size, memory rank and recurrent dependence answer distinct questions.',
            'Direct-only controls intentionally disconnect dt/x_proj/A; preserved weights do not imply active recurrence.',
            'Inference ablation on 3 fixed patches is local; parent audit uses 45 patches.',
            'Full TRAIN MSE uses956patches and cannot be substituted with the45patch probe.',
            'Historical S3/S4 are named references; exact replay is not assumed.',
            'AdamW direct shrink at these LR/decay values may round away in float32; inspect the isolated rounding check before attributing differences to decay grouping.',
        ],
    }
    rounding_check = root / 'adamw_shrink_rounding_check.json'
    if rounding_check.exists():
        result['isolated_adamw_shrink_check'] = read(rounding_check)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# CNN–memory–SSM 검증 결과', '',
             f"고정 epoch200 완료 확인: {len(final)}/12. 학습 전체 완료: {suite_complete}. 최종45patch 진단: {len(final_audits)}/12.", '',
             '정상 TRAIN 기전 진단이다. 중간 probe와 전체956patch 최종값을 구분하며 AUROC/AP 개선을 뜻하지 않는다.', '',
             '| 조건 | seed | 완료 epoch | 최종 전체 TRAIN MSE | Δ bias-only /18 | dt gradient 연결 /6 |',
             '| --- | --- | --- | --- | --- | --- |']
    for job in jobs:
        f = job.get('fixed_epoch200')
        cells = [f"{f['full_train_956patch_mse']:.8f}", str(f['delta_bias_only_observations']), str(f['dt_gradient_connected_blocks'])] if f else ['미완료', '—', '—']
        if f and manifest['variants'][job['variant']]['direct_only']:
            cells[1:] = ['재귀 제외', '0 (의도된 대조)']
        lines.append(f"| {job['variant']} | {job['seed']} | {job['completed_epochs']} | " + ' | '.join(cells) + ' |')
    lines.extend(['', 'V4/V5는 학습 처음부터 재귀를 제외한다. 계산된 Δ나 보존된 투영 가중치는 활성 재귀의 근거가 아니므로 해당 Δ 진단을 재귀 사용 조건과 같은 척도로 비교하지 않는다.'])
    if contrasts:
        lines.extend(['', '| 고정 비교: treatment − control | seed | 정상 MSE 차이 | 변화율 |', '| --- | --- | --- | --- |'])
        for c in contrasts:
            lines.append(f"| {c['treatment']} − {c['control']} | {c['seed']} | {c['mse_difference']:+.8f} | {c['mse_relative_change_percent']:+.2f}% |")
    (destination / 'analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output': str(destination), 'suite_complete': suite_complete, 'completed_runs': len(final)}))
    if require_complete and not suite_complete:
        raise SystemExit('Final verification rejected: suite remains incomplete')
    if require_final_audit and not final_audit_complete:
        raise SystemExit('Final verification rejected: 45patch inference audits remain incomplete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--require-final-audit', action='store_true')
    args = parser.parse_args()
    analyze(args.root.resolve(), args.output.resolve(), args.require_complete, args.require_final_audit)
