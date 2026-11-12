#!/usr/bin/env python3
"""Post-hoc inner-fold diagnostic: fixed normal TRAIN mean reconstructions."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import spearmanr


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def diagnose(root,fold):
    torch.set_num_threads(1)
    sys.path.insert(0,str(root/'source_snapshot'))
    from main_files.data.patients import PatientPatchDataset
    from main_files.core.metrics import aggregate_patch_scores,patient_metrics
    cfg=json.loads((root/'config.json').read_text())
    split=json.loads((root/'splits.json').read_text())
    patients={p['patient_id']:p for p in split['patients']}
    f=next(f for f in split['folds'] if f['fold']==fold)
    train_ids=f['inner_train'];eval_ids=f['inner_normal']+f['inner_diseased']
    assert not set(train_ids)&set(eval_ids)
    assert all(patients[p]['label']==0 for p in train_ids)
    out=root/f'constant_reference_diagnostic_f{fold}'
    out.mkdir(exist_ok=False)
    write(out/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'stage':'inner_posthoc_diagnostic','fold':fold,'train_patient_ids':train_ids,
        'evaluation_patient_ids':eval_ids,'methods':['normal_train_scalar_mean','normal_train_voxel_mean'],
        'fit':'Patch-weighted mean of all normal training voxels, or per-voxel mean over all training patches.',
        'scope':'Motivated by completed P0 validation; supplemental diagnostic, not a candidate or model-selection input.',
        'no_outer_or_external_access':True,'no_optimizer_or_gpu':True,'score_direction':'higher reconstruction error indicates disease',
        'calculation_dtype':'float64 after the same frozen float32 HU preprocessing',
        'script_sha256':sha(Path(__file__))})
    total=np.zeros((1,*cfg['expected_patch_shape']),dtype=np.float64)
    count=0;sum_squares=0.
    for pid in train_ids:
        p=patients[pid]
        assert sha(Path(p['path']))==p['patch_sha256']
        dataset=PatientPatchDataset([p],cfg['hu_window'],cfg['expected_patch_shape'])
        x=dataset.volumes[0].numpy().astype(np.float64)
        assert len(x)==p['patch_count']
        total+=x.sum(0);sum_squares+=float(np.square(x).sum());count+=len(x)
    template=total/count
    scalar=float(template.mean())
    np.save(out/'normal_train_voxel_mean.npy',template)
    write(out/'fit.json',{'training_patients':len(train_ids),'training_patches':count,'scalar_mean':scalar,
                         'voxel_template_std':float(template.std()),
                         'training_population_variance':sum_squares/(count*template.size)-scalar**2,
                         'template_sha256':sha(out/'normal_train_voxel_mean.npy')})
    write(out/'evaluation_access.json',{'accessed_utc':datetime.now(timezone.utc).isoformat(),
            'stage':'inner_posthoc_diagnostic','patient_ids':eval_ids,
            'methods':['normal_train_scalar_mean','normal_train_voxel_mean'],
            'train_only_fit_completed':True,'does_not_enter_frozen_candidate_selection':True})
    observations={k:[] for k in ('normal_train_scalar_mean','normal_train_voxel_mean')}
    patches={k:[] for k in observations}
    for pid in eval_ids:
        p=patients[pid]
        assert sha(Path(p['path']))==p['patch_sha256']
        dataset=PatientPatchDataset([p],cfg['hu_window'],cfg['expected_patch_shape'])
        x=dataset.volumes[0].numpy().astype(np.float64)
        assert len(x)==p['patch_count']
        flat=x.reshape(len(x),-1)
        input_mean=flat.mean(1);input_variance=flat.var(1)
        for name,reconstruction in [('normal_train_scalar_mean',scalar),('normal_train_voxel_mean',template)]:
            sse=np.square(x-reconstruction).reshape(len(x),-1).sum(1)
            mse=sse/template.size
            if name=='normal_train_scalar_mean':
                assert np.allclose(mse,input_variance+(input_mean-scalar)**2,rtol=1e-12,atol=1e-12)
            scores=sse if cfg['patch_score_reduction']=='sum' else mse
            row={'patient_id':pid,'label':p['label'],'patch_count':len(x),
                 'mean_patch_mse':float(mse.mean()),'input_mean':float(input_mean.mean()),
                 'input_variance_mean':float(input_variance.mean()),
                 **{a:aggregate_patch_scores(scores,a,cfg['top_k'],cfg['aggregation_percentile']) for a in ('max','mean','top_k','percentile')}}
            observations[name].append(row)
            patches[name].extend({'patient_id':pid,'label':p['label'],'patch_index':i,'mse':float(mse[i]),
                                 'sse':float(sse[i]),'input_mean':float(input_mean[i]),'input_variance':float(input_variance[i])} for i in range(len(x)))
    report={'scope':'Inner fold1 post-hoc constant-reference diagnostic; no candidate selection or causal proof.',
            'fold':fold,'normal_count':len(f['inner_normal']),'disease_count':len(f['inner_diseased']),
            'patches':sum(patients[p]['patch_count'] for p in eval_ids),'methods':{}}
    for name,rows in observations.items():
        order=sorted(rows,key=lambda r:r['patient_id'])
        metrics={a:patient_metrics([r['label'] for r in order],[r[a] for r in order],patient_ids=[r['patient_id'] for r in order]) for a in ('max','mean','top_k','percentile')}
        comparisons=[]
        for seed in (42,123):
            model=json.loads((root/f'inner_f{fold}_s{seed}_P0_reference/results.json').read_text())
            predicted=sorted(model['patients'],key=lambda r:r['patient_id'])
            assert [r['patient_id'] for r in predicted]==[r['patient_id'] for r in order]
            comparisons.append({'P0_seed':seed,'max_score_spearman':float(spearmanr([r['max'] for r in order],[r['max'] for r in predicted]).statistic),
                'mean_score_spearman':float(spearmanr([r['mean'] for r in order],[r['mean'] for r in predicted]).statistic)})
        report['methods'][name]={'metrics':metrics,'comparisons_to_P0':comparisons,'patients':rows}
        for suffix,data in [('patients',rows),('patches',patches[name])]:
            with (out/f'{name}_{suffix}.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write(out/'results.json',report)
    print(json.dumps({name:{'metrics':v['metrics'],'comparisons_to_P0':v['comparisons_to_P0']} for name,v in report['methods'].items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--fold',type=int,default=1)
    args=p.parse_args();diagnose(args.root.resolve(),args.fold)
