#!/usr/bin/env python3
"""Independently verify full patient/patch coverage and metrics after scoring."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda:handle.read(4*1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def check_scores(patients, rows, patches, cfg):
    by_id={p['patient_id']:p for p in patients}
    assert len(by_id)==len(patients)
    assert sorted(r['patient_id'] for r in rows)==sorted(by_id)
    assert {p['patient_id'] for p in patches}==set(by_id)
    expected_count=sum(p['patch_count'] for p in patients)
    assert len(patches)==expected_count
    volume_size=math.prod(cfg['expected_patch_shape'])
    for row in rows:
        patient=by_id[row['patient_id']]
        assert row['label']==patient['label']
        assert row['patch_count']==patient['patch_count']
        part=sorted([p for p in patches if p['patient_id']==row['patient_id']],key=lambda p:int(p['patch_index']))
        assert [int(p['patch_index']) for p in part]==list(range(patient['patch_count']))
        assert all(int(p['label'])==patient['label'] for p in part)
        mse=np.array([float(p['mse']) for p in part])
        sse=np.array([float(p['sse']) for p in part])
        assert np.isfinite(mse).all() and np.isfinite(sse).all()
        assert (mse>=0).all() and (sse>=0).all()
        assert np.allclose(sse,mse*volume_size,rtol=1e-6,atol=1e-8)
        scores=sse if cfg['patch_score_reduction']=='sum' else mse
        expected={'max':float(np.max(scores)),'mean':float(np.mean(scores)),
                  'top_k':float(np.mean(np.sort(scores)[-min(cfg['top_k'],len(scores)): ])),
                  'percentile':float(np.percentile(scores,cfg['aggregation_percentile']))}
        assert np.isclose(row['mean_patch_mse'],np.mean(mse),rtol=1e-12,atol=1e-12)
        for key,value in expected.items():
            assert np.isclose(row[key],value,rtol=1e-12,atol=1e-12),(row['patient_id'],key)
    labels=[r['label'] for r in rows]
    return {a:{'auroc':float(roc_auc_score(labels,[r[a] for r in rows])),
               'ap':float(average_precision_score(labels,[r[a] for r in rows]))}
            for a in ('max','mean','top_k','percentile')}


def verify(root, job):
    cfg=read(root/'config.json')
    split=read(root/'splits.json')
    patients={p['patient_id']:p for p in split['patients']}
    f=next(f for f in split['folds'] if f['fold']==job['fold'])
    train_ids=f['inner_train'] if job['stage']=='inner' else f['outer_train']
    eval_ids=f['inner_normal']+f['inner_diseased'] if job['stage']=='inner' else f['outer_test']
    assert not set(train_ids)&set(eval_ids)
    out=root/job['id']
    state=read(out/'state.json')
    result=read(out/'results.json')
    access=read(out/'evaluation_access.json')
    assert state['status']=='complete' and state['completed_epochs']==cfg['epochs']
    count=sum(patients[p]['patch_count'] for p in train_ids)
    assert state['completed_updates']==cfg['epochs']*math.ceil(count/cfg['batch_size'])
    assert state['train_patient_ids']==train_ids and state['evaluation_patient_ids']==eval_ids
    assert access['patient_ids']==eval_ids
    assert access['stage']==job['stage'] and result['inference_state_unchanged']
    checkpoint=out/f'epoch_{cfg["epochs"]}.pt'
    digest=sha(checkpoint)
    assert digest==state['checkpoint_sha256']==result['checkpoint_sha256']==access['checkpoint_sha256']
    if job['stage']=='outer':
        selections=read(root/'selections.json')
        assert access['selection_sha256']==sha(root/'selections.json')
        assert datetime.fromisoformat(access['accessed_utc'])>=datetime.fromisoformat(selections['locked_utc'])
        assert len(selections['folds'])==4
    with (out/'patch_scores.csv').open() as handle:
        patches=list(csv.DictReader(handle))
    assert len(patches)==result['scored_patches']
    computed=check_scores([patients[p] for p in eval_ids],result['patients'],patches,cfg)
    for a,metrics in computed.items():
        for key,value in metrics.items():
            assert np.isclose(value,result['metrics'][a][key],rtol=0,atol=1e-12)
    record={'status':'passed','checked_utc':datetime.now(timezone.utc).isoformat(),'job':job['id'],
            'patients':len(eval_ids),'patches':len(patches),'checkpoint_sha256':digest,
            'full_frozen_patch_counts_and_indices_verified':True,'patient_labels_verified':True,
            'all_aggregations_recomputed':True,'all_auroc_ap_recomputed':True,
            'scoring_state_unchanged':True,'analysis_script_sha256':sha(Path(__file__)),
            'metrics':computed}
    dest=root/'completed_audits';dest.mkdir(exist_ok=True)
    (dest/f'{job["id"]}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job')
    args=p.parse_args();root=args.root.resolve()
    manifest=read(root/'manifest.json')
    jobs=[j for j in manifest['jobs'] if j['status']=='complete' and (args.job is None or j['id']==args.job)]
    if args.job is not None:
        assert len(jobs)==1,'Requested job has not completed'
    for job in jobs:
        verify(root,job)
