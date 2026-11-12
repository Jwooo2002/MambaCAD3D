#!/usr/bin/env python3
"""Paired descriptive diagnostics; never alters candidate selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def pair_matrix(normal_scores,disease_scores):
    normal=np.asarray(normal_scores,dtype=float)
    disease=np.asarray(disease_scores,dtype=float)
    assert len(normal)>0 and len(disease)>0
    assert np.isfinite(normal).all() and np.isfinite(disease).all()
    return (disease[:,None]>normal[None,:]).astype(float)+.5*(disease[:,None]==normal[None,:])


def compare(root,baseline,candidate):
    states=[json.loads((root/j['id']/'state.json').read_text()) for j in (baseline,candidate)]
    for key in ('initial_state_sha256','orders_sha256','actual_seed','train_patient_ids','evaluation_patient_ids'):
        assert states[0][key]==states[1][key],key
    results=[json.loads((root/j['id']/'results.json').read_text()) for j in (baseline,candidate)]
    rows=[{r['patient_id']:r for r in result['patients']} for result in results]
    assert set(rows[0])==set(rows[1])
    assert all(rows[0][pid]['label']==rows[1][pid]['label'] for pid in rows[0])
    normal=sorted(pid for pid,r in rows[0].items() if r['label']==0)
    disease=sorted(pid for pid,r in rows[0].items() if r['label']==1)
    matrices=[pair_matrix([r[p]['max'] for p in normal],[r[p]['max'] for p in disease]) for r in rows]
    for matrix,result in zip(matrices,results):
        assert abs(float(matrix.mean())-result['metrics']['max']['auroc'])<1e-12
    delta=matrices[1]-matrices[0]
    record={'fold':candidate['fold'],'seed':candidate['seed'],'variant':candidate['variant'],
            'normal_count':len(normal),'diseased_count':len(disease),
            'primary':{'baseline_auroc':results[0]['metrics']['max']['auroc'],
                       'candidate_auroc':results[1]['metrics']['max']['auroc'],
                       'delta_auroc':float(delta.mean()),
                       'baseline_ap':results[0]['metrics']['max']['ap'],
                       'candidate_ap':results[1]['metrics']['max']['ap'],
                       'delta_ap':results[1]['metrics']['max']['ap']-results[0]['metrics']['max']['ap']},
            'pairs':{'total':int(delta.size),'improved':int((delta>0).sum()),
                     'worsened':int((delta<0).sum()),'unchanged':int((delta==0).sum()),
                     'tie_aware_pair_credit_delta':float(delta.sum())},
            'normal_patient_pair_credit':[{'patient_id':pid,
                         'baseline_correct_equivalents':float(matrices[0][:,i].sum()),
                         'candidate_correct_equivalents':float(matrices[1][:,i].sum()),
                         'delta':float(delta[:,i].sum())} for i,pid in enumerate(normal)],
            'secondary':{a:{m:results[1]['metrics'][a][m]-results[0]['metrics'][a][m] for m in ('auroc','ap')}
                         for a in ('mean','top_k','percentile')}}
    return record


def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text())
    complete=[j for j in manifest['jobs'] if j['stage']=='inner' and j['status']=='complete']
    by_key={(j['fold'],j['seed'],j['variant']):j for j in complete}
    pairs=[]
    for job in complete:
        if job['variant']=='P0_reference':continue
        key=(job['fold'],job['seed'],'P0_reference')
        if key in by_key:
            pairs.append(compare(root,by_key[key],job))
    means=[]
    for fold,variant in sorted({(p['fold'],p['variant']) for p in pairs}):
        part=[p for p in pairs if p['fold']==fold and p['variant']==variant]
        assert len(part)<=2
        complete_seeds=sorted(p['seed'] for p in part)==[42,123]
        means.append({'fold':fold,'variant':variant,'seed_count':len(part),'both_planned_seeds_complete':complete_seeds,
                      'mean_delta_auroc':float(np.mean([p['primary']['delta_auroc'] for p in part])),
                      'mean_delta_ap':float(np.mean([p['primary']['delta_ap'] for p in part]))})
    report={'updated_utc':datetime.now(timezone.utc).isoformat(),'scope':'Descriptive inner validation comparisons; not outer performance or independent patient CI. No selection-rule changes.',
            'comparisons':pairs,'fold_candidate_means':means}
    out=root/'analysis';out.mkdir(exist_ok=True)
    temp=out/'inner_paired.json.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(out/'inner_paired.json')
    print(json.dumps({'paired_runs':len(pairs),'fold_candidate_means':means}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();summarize(args.root.resolve())
