import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('performance_run', ROOT/'performance_search/run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def synthetic_split():
    patients = [{'patient_id':f'{label}:{i}', 'label':label} for label in (0,1) for i in range(20)]
    folds=[]
    for f in range(4):
        test={f*5+i for i in range(5)}
        normal=[f'0:{i}' for i in range(20) if i not in test]
        folds.append({'fold':f+1,'inner_train':normal[3:],'inner_normal':normal[:3],
                      'inner_diseased':[f'1:{i}' for i in range(20) if i not in test],
                      'outer_train':normal,'outer_test':[f'{label}:{i}' for label in (0,1) for i in sorted(test)]})
    return patients,folds


def test_disjoint_patient_splits_and_once_only_outer_coverage():
    p,f=synthetic_split()
    run.validate_splits(p,f)
    bad=copy.deepcopy(f)
    bad[0]['inner_train'][0]=bad[0]['outer_test'][0]
    with pytest.raises(AssertionError):run.validate_splits(p,bad)
    bad=copy.deepcopy(f)
    bad[1]=copy.deepcopy(bad[0])
    with pytest.raises(AssertionError):run.validate_splits(p,bad)


def test_selection_uses_mean_seeds_and_ap_guardrail():
    values={'P0_reference':[(.6,.8),(.6,.8)],
            'P1_no_decay':[(.9,.75),(.9,.75)],
            'P2_weak_entropy':[(.7,.8),(.7,.8)],
            'P3_no_entropy':[(1.,.9),(.3,.8)]}
    rows=[dict(variant=v,seed=s,auroc=a,ap=p) for v,vs in values.items() for s,(a,p) in zip([42,123],vs)]
    selected=run.choose(rows)
    assert selected['selected_variant']=='P2_weak_entropy'
    assert 'P1_no_decay' not in selected['eligible']
    with pytest.raises(AssertionError):run.choose(rows[:-1])


def test_ties_prefer_declared_reference_and_outer_gate():
    rows=[dict(variant=v,seed=s,auroc=.7,ap=.8) for v in run.VARIANTS for s in [42,123]]
    assert run.choose(rows)['selected_variant']=='P0_reference'
    with pytest.raises(AssertionError):
        run.lock_selections(Path('/unused'), {'jobs':[]})


def test_all_fold_selections_lock_before_outer_jobs(tmp_path):
    (tmp_path/'plan.json').write_text(json.dumps({'selection':'test rule'}))
    jobs=[]
    for fold in range(1,5):
        for v in run.VARIANTS:
            for seed in [42,123]:
                j=run.make_job('inner',fold,seed,v)
                j['status']='complete';jobs.append(j)
                out=tmp_path/j['id'];out.mkdir()
                (out/'results.json').write_text(json.dumps({'metrics':{'max':{'auroc':.8 if v=='P2_weak_entropy' else .6,'ap':.8}}}))
    manifest={'jobs':jobs}
    manifest['jobs'][-1]['status']='running'
    with pytest.raises(AssertionError):run.lock_selections(tmp_path,manifest)
    assert not (tmp_path/'selections.json').exists()
    manifest['jobs'][-1]['status']='complete'
    run.lock_selections(tmp_path,manifest)
    assert len(manifest['jobs'])==56
    selections=json.loads((tmp_path/'selections.json').read_text())
    assert len(selections['folds'])==4
    assert {s['selected_variant'] for s in selections['folds'].values()}=={'P2_weak_entropy'}


def test_independent_rank_metrics_match_sklearn_with_ties():
    import numpy as np
    from sklearn.metrics import average_precision_score,roc_auc_score
    s=importlib.util.spec_from_file_location('performance_analyze',ROOT/'performance_search/analyze.py')
    a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    rng=np.random.default_rng(42)
    y=np.array([0]*5+[1]*5)
    for _ in range(100):
        scores=rng.integers(0,4,10).astype(float)
        assert np.allclose(a.rank_metrics(y,scores),[roc_auc_score(y,scores),average_precision_score(y,scores)],rtol=0,atol=1e-12)


def test_complete_report_with_reference_selected_has_zero_paired_ci(tmp_path):
    """A selected reference reuses its run; it must not invent a second sample."""
    import csv
    import hashlib
    from sklearn.metrics import average_precision_score,roc_auc_score
    spec=importlib.util.spec_from_file_location('performance_full_report',ROOT/'performance_search/analyze.py')
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    patients,folds=synthetic_split()
    by_id={p['patient_id']:p for p in patients}
    (tmp_path/'config.json').write_text(json.dumps({'epochs':2,'patch_score_reduction':'sum'}))
    (tmp_path/'splits.json').write_text(json.dumps({'patients':patients,'folds':folds}))
    selections={'folds':{str(f):{'selected_variant':'P0_reference'} for f in range(1,5)}}
    (tmp_path/'selections.json').write_text(json.dumps(selections))
    jobs=[]
    for f in folds:
        for seed in (42,123,2024):
            job=run.make_job('outer',f['fold'],seed,'P0_reference')
            job['status']='complete';jobs.append(job)
            out=tmp_path/job['id'];out.mkdir()
            rows=[]
            for i,pid in enumerate(f['outer_test']):
                value=float((i+seed)%4)
                rows.append({'patient_id':pid,'label':by_id[pid]['label'],'patch_count':1,
                             **{a:value for a in ('max','mean','top_k','percentile')}})
            labels=[r['label'] for r in rows];scores=[r['max'] for r in rows]
            metric={'auroc':roc_auc_score(labels,scores),'ap':average_precision_score(labels,scores)}
            (out/'results.json').write_text(json.dumps({'patients':rows,'scored_patches':len(rows),
                            'metrics':{a:metric for a in ('max','mean','top_k','percentile')}}))
            (out/'state.json').write_text(json.dumps({'status':'complete','completed_epochs':2}))
            (out/'history.csv').write_text('epoch\n1\n2\n')
            with (out/'patch_scores.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=['patient_id','sse'])
                writer.writeheader();writer.writerows({'patient_id':r['patient_id'],'sse':r['max']} for r in rows)
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'complete','jobs':jobs,
        'selections_sha256':hashlib.sha256((tmp_path/'selections.json').read_bytes()).hexdigest()}))
    analyzer.analyze(tmp_path,bootstrap=100)
    report=json.loads((tmp_path/'analysis/results.json').read_text())
    assert len(report['paired_outer_comparisons'])==12
    assert report['mean_paired_difference']=={'auroc':0.,'ap':0.}
    for m in ('auroc','ap'):
        assert report['paired_patient_bootstrap'][m]=={'lower':0.,'upper':0.}


def test_full_patch_audit_rejects_duplicates_drops_and_wrong_labels():
    import numpy as np
    s=importlib.util.spec_from_file_location('verify_complete',ROOT/'performance_search/verify_completed.py')
    v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
    cfg={'expected_patch_shape':[2,2,2],'patch_score_reduction':'sum','top_k':2,'aggregation_percentile':95}
    patients=[{'patient_id':str(i),'label':i,'patch_count':3} for i in (0,1)]
    patches=[];rows=[]
    for patient in patients:
        values=np.array([1,2,3],dtype=float)+patient['label']
        patches.extend({'patient_id':patient['patient_id'],'label':patient['label'],
                        'patch_index':i,'mse':value,'sse':value*8} for i,value in enumerate(values))
        rows.append({**patient,'mean_patch_mse':float(values.mean()),'max':float(values.max()*8),
                     'mean':float(values.mean()*8),'top_k':float(values[-2:].mean()*8),
                     'percentile':float(np.percentile(values*8,95))})
    v.check_scores(patients,rows,patches,cfg)
    with pytest.raises(AssertionError):v.check_scores(patients,rows,patches[:-1],cfg)
    duplicate=copy.deepcopy(patches);duplicate[-1]=copy.deepcopy(duplicate[-2])
    with pytest.raises(AssertionError):v.check_scores(patients,rows,duplicate,cfg)
    wrong=copy.deepcopy(rows);wrong[1]['label']=0
    with pytest.raises(AssertionError):v.check_scores(patients,wrong,patches,cfg)
    wrong=copy.deepcopy(rows);wrong[0]['top_k']+=.1
    with pytest.raises(AssertionError):v.check_scores(patients,wrong,patches,cfg)


def test_paired_rank_credit_counts_ties_and_inversions():
    import numpy as np
    spec=importlib.util.spec_from_file_location('inner_comparison',ROOT/'performance_search/compare_inner.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    baseline=m.pair_matrix([2.,4.],[1.,3.])
    candidate=m.pair_matrix([1.,2.],[3.,4.])
    assert baseline.mean()==.25
    assert candidate.mean()==1.
    assert (candidate-baseline).mean()==.75
    assert int(((candidate-baseline)>0).sum())==3
    assert np.array_equal(m.pair_matrix([1.,2.],[1.,2.]),[[.5,0.],[1.,.5]])
