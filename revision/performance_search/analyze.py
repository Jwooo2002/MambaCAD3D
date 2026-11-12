#!/usr/bin/env python3
"""Report all runs; compare the locked selection procedure with its paired reference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def rank_metrics(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive, negative = scores[y == 1], scores[y == 0]
    auc = ((positive[:,None] > negative[None,:]) + .5*(positive[:,None] == negative[None,:])).mean()
    order = np.argsort(-scores, kind='stable')
    sorted_y, sorted_scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y)-1]
    cumulative = np.cumsum(sorted_y)[ends]
    ap = np.sum(np.diff(np.r_[0,cumulative]) * cumulative / (ends+1)) / y.sum()
    return np.array([auc,ap])


def audit_result(root, job, cfg, split):
    out = root/job['id']
    state = json.loads((out/'state.json').read_text())
    assert state['status']=='complete' and state['completed_epochs']==cfg['epochs']
    result = json.loads((out/'results.json').read_text())
    f = next(f for f in split['folds'] if f['fold']==job['fold'])
    expected = f['inner_normal']+f['inner_diseased'] if job['stage']=='inner' else f['outer_test']
    rows = result['patients']
    assert sorted(r['patient_id'] for r in rows)==sorted(expected)
    assert len(rows)==len(set(r['patient_id'] for r in rows))
    with (out/'patch_scores.csv').open() as handle:
        patches=list(csv.DictReader(handle))
    assert len(patches)==result['scored_patches']
    for row in rows:
        matched=[float(p['sse' if cfg['patch_score_reduction']=='sum' else 'mse']) for p in patches if p['patient_id']==row['patient_id']]
        assert len(matched)==row['patch_count']
        assert max(matched)==row['max']
    y=[r['label'] for r in rows]
    for aggregation in ('max','mean','top_k','percentile'):
        s=[r[aggregation] for r in rows]
        observed=result['metrics'][aggregation]
        assert np.allclose(rank_metrics(y,s), [observed['auroc'],observed['ap']],rtol=0,atol=1e-12)
    history=list(csv.DictReader((out/'history.csv').open()))
    assert [int(r['epoch']) for r in history]==list(range(1,cfg['epochs']+1))
    return result


def analyze(root, bootstrap=2000):
    manifest=json.loads((root/'manifest.json').read_text())
    cfg=json.loads((root/'config.json').read_text())
    split=json.loads((root/'splits.json').read_text())
    complete=[j for j in manifest['jobs'] if j['status']=='complete']
    results={j['id']:audit_result(root,j,cfg,split) for j in complete}
    report={'status':manifest['status'],'completed_runs':len(complete), 'planned_runs':len(manifest['jobs']),
            'all_completed_results_audited':True,'inner':[],'outer':[],
            'scope':'Patient-level internal revalidation of a nested holdout selection procedure. Historical ASOCA exposure remains.'}
    for j in complete:
        report[j['stage']].append({k:j[k] for k in ('id','fold','seed','variant')} | results[j['id']]['metrics']['max'])
    selections_path=root/'selections.json'
    if selections_path.exists():
        selections=json.loads(selections_path.read_text())
        assert hashlib.sha256(selections_path.read_bytes()).hexdigest()==manifest['selections_sha256']
        report['selections']=selections['folds']
        comparisons=[]
        fold_arrays=[]
        for f in range(1,5):
            chosen=selections['folds'][str(f)]['selected_variant']
            arrays=[]
            for seed in (42,123,2024):
                names=[f'outer_f{f}_s{seed}_{v}' for v in ('P0_reference',chosen)]
                if not all(name in results for name in names):
                    continue
                paired=[]
                for name in names:
                    rows=sorted(results[name]['patients'],key=lambda r:r['patient_id'])
                    paired.append(np.array([r['max'] for r in rows]))
                reference,selected=[results[n]['metrics']['max'] for n in names]
                comparisons.append({'fold':f,'seed':seed,'selected_variant':chosen,
                    'baseline_auroc':reference['auroc'],'selected_auroc':selected['auroc'],
                    'delta_auroc':selected['auroc']-reference['auroc'],
                    'baseline_ap':reference['ap'],'selected_ap':selected['ap'],
                    'delta_ap':selected['ap']-reference['ap']})
                arrays.append(paired)
                y=np.array([r['label'] for r in rows])
            if len(arrays)==3:
                fold_arrays.append((f,y,np.array(arrays))) # seed, arm, patient
        report['paired_outer_comparisons']=comparisons
        if len(comparisons)==12:
            report['mean_paired_difference']={m:float(np.mean([r['delta_'+m] for r in comparisons])) for m in ('auroc','ap')}
            report['seed_mean_differences']={str(s):{m:float(np.mean([r['delta_'+m] for r in comparisons if r['seed']==s])) for m in ('auroc','ap')} for s in (42,123,2024)}
            report['fold_mean_differences']={str(f):{m:float(np.mean([r['delta_'+m] for r in comparisons if r['fold']==f])) for m in ('auroc','ap')} for f in range(1,5)}
            if bootstrap:
                rng=np.random.default_rng(20261112)
                draws=[]
                for _ in range(bootstrap):
                    per_fold=[]
                    for f,y,scores in fold_arrays:
                        idx=np.concatenate([rng.choice(np.flatnonzero(y==c),5,replace=True) for c in (0,1)])
                        # Identical patient draws for both arms and all three seeds.
                        per_seed=[]
                        for s in range(3):
                            per_seed.append(rank_metrics(y[idx],scores[s,1,idx])-rank_metrics(y[idx],scores[s,0,idx]))
                        per_fold.append(np.mean(per_seed,axis=0))
                    draws.append(np.mean(per_fold,axis=0))
                ci=np.quantile(draws,[.025,.975],axis=0)
                report['paired_patient_bootstrap']={'resamples':bootstrap,'seed':20261112,
                    'method':'Class-stratified resampling within outer fold; each patient draw shared by all seeds and both arms; mean paired fold/seed metric.',
                    'scope':'Conditional on fitted models, locked selections, split and class counts; excludes model refitting and historical selection uncertainty.',
                    **{m:{'lower':float(ci[0,i]),'upper':float(ci[1,i])} for i,m in enumerate(('auroc','ap'))}}
    dest=root/'analysis'
    dest.mkdir(exist_ok=True)
    (dest/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# Frozen performance search results','',report['scope'],'',f"Complete {len(complete)}/{len(manifest['jobs'])}; status {manifest['status']}",'',
           '| Stage | Fold | Seed | Variant | AUROC | AP |','| --- | --- | --- | --- | ---: | ---: |']
    for stage in ('inner','outer'):
        for r in report[stage]:lines.append(f"| {stage} | {r['fold']} | {r['seed']} | {r['variant']} | {r['auroc']:.4f} | {r['ap']:.4f} |")
    if 'mean_paired_difference' in report:
        lines.extend(['',json.dumps({k:report[k] for k in ('mean_paired_difference','seed_mean_differences','fold_mean_differences')},indent=2)])
    (dest/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed':len(complete),'outer_pairs':len(report.get('paired_outer_comparisons',[])),
                      'mean_delta':report.get('mean_paired_difference'),'ci':report.get('paired_patient_bootstrap')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    args=p.parse_args();analyze(args.root.resolve(),args.bootstrap)
