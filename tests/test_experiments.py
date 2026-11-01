import copy
import json
from collections import Counter
from pathlib import Path
import pytest
from paper_rag.experiments.context import build_context, load_tasks
from paper_rag.experiments.evaluate import evaluate, normalize, score
from paper_rag.experiments.runner import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def task():
    return load_tasks(ROOT/'data/experiments/two_hop_demo.jsonl')[0]


def record(answer=None, condition='concentrated'):
    t=task()
    return {'run_id':condition,'task':t,'condition':condition,'rotation':0,'reverse':False,'status':'ok',
            'context':build_context(t,condition),
            'generation':{'answer':answer or json.dumps({'answer':t['answer'],'citations':['E1','E2']},ensure_ascii=False),
                          'input_tokens':100,'output_tokens':20,'elapsed_seconds':1.0,'gpu':None}}


def test_material_and_positions():
    t=task(); n=len(t['evidence'])
    positions={c:{e:[] for e in t['required_ids']} for c in ('concentrated','dispersed')}
    for reverse in (False,True):
        for rotation in range(n//2):
            a=build_context(t,'concentrated',rotation,reverse)
            b=build_context(t,'dispersed',rotation,reverse)
            assert a['material_hash']==b['material_hash']
            assert set(a['order'])==set(b['order'])
            assert abs(a['required_positions']['E1']-a['required_positions']['E2'])==1
            assert abs(b['required_positions']['E1']-b['required_positions']['E2'])==n//2
            for c,ctx in [('concentrated',a),('dispersed',b)]:
                for e in t['required_ids']:
                    positions[c][e].append(ctx['required_positions'][e])
    for c in positions.values():
        for p in c.values():
            assert Counter(p)==Counter(range(n))
    assert set(build_context(t,'necessary')['order'])==set(t['required_ids'])


def test_split_leakage_rejected(tmp_path):
    t=task(); duplicate=copy.deepcopy(t); duplicate.update(id='other',split='validation')
    p=tmp_path/'tasks.jsonl'; p.write_text('\n'.join(json.dumps(x) for x in (t,duplicate)))
    with pytest.raises(ValueError,match='leaks'):
        load_tasks(p)
    duplicate['fact_group']='different'
    p.write_text('\n'.join(json.dumps(x) for x in (t,duplicate)))
    with pytest.raises(ValueError,match='facts leak'):
        load_tasks(p)


def test_human_review_cannot_be_invented(tmp_path):
    t=task(); t['human_review']['status']='verified'
    p=tmp_path/'tasks.jsonl'; p.write_text(json.dumps(t))
    with pytest.raises(ValueError,match='reviewer'):
        load_tasks(p)


def test_answer_alias_format_and_support():
    r=record(); s=score(r)
    assert s['answer_correct'] and s['necessary_coverage']==1 and s['citation_support'] is None
    assert score(record(json.dumps({'answer':task()['answer']+'市','citations':['E1','unknown']})))['answer_correct']
    assert score(record('```json\n{}\n```'))['error_type']=='output_format_error'
    review={'reviewer':'human','reviewed_at':'2026-11-01','supported_citations':{'E1':True,'E2':False},'error_type':'association_error'}
    s=score(r,review)
    assert s['citation_support']==.5 and s['error_type']=='association_error'
    review['supported_citations'].pop('E2')
    with pytest.raises(ValueError,match='every cited'):
        score(r,review)
    assert normalize(' Ａ，b。 ')==normalize('ab')


def test_failures_and_pairs_are_traceable():
    a=record(); b=record(condition='dispersed'); b.update(status='error',error='Input budget exceeded')
    report=evaluate([a,b])
    assert report['summary'][1]['failures']==1
    assert report['paired_changes'][0]['complete_pair'] is False
    assert report['paired_changes'][0]['concentrated_correct'] is True
    assert report['paired_changes'][0]['dispersed_correct'] is False


def test_atomic_record_roundtrip(tmp_path):
    r=record(); p=tmp_path/'records'/'a.json'; atomic_json(p,r)
    assert json.loads(p.read_text())==r
    assert not list(p.parent.glob('*.tmp'))


def test_input_budget_never_truncates():
    from paper_rag.generation import validate_input_length
    validate_input_length(100,100,20,120)
    with pytest.raises(ValueError,match='truncation disabled'):
        validate_input_length(101,100,20,200)
    with pytest.raises(ValueError,match='truncation disabled'):
        validate_input_length(100,200,21,120)


def test_unreviewed_validation_is_refused_before_model_loading(tmp_path):
    from paper_rag.experiments.runner import run
    with pytest.raises(ValueError,match='Validation requires human'):
        run(ROOT/'data/experiments/two_hop_demo.jsonl',tmp_path,split='validation',allow_unreviewed=True)
    assert not list(tmp_path.iterdir())


def test_prompt_and_dataset_do_not_reveal_fixed_gold_citation_ids():
    from paper_rag.generation import TASKS
    assert 'E1' not in TASKS['experiment'] and 'E2' not in TASKS['experiment']
    tasks=load_tasks(ROOT/'data/experiments/two_hop_demo.jsonl')
    assert len({tuple(t['required_ids']) for t in tasks})>1


def test_unmeasured_demo_costs_stay_empty():
    r=record()
    for key in ('input_tokens','output_tokens','elapsed_seconds'):
        r['generation'][key]=None
    summary=evaluate([r])['summary'][0]
    assert summary['mean_input_tokens'] is None
    assert summary['mean_output_tokens'] is None
    assert summary['mean_seconds'] is None
    assert summary['max_peak_allocated_mb'] is None
