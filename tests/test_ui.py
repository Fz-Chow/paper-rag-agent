from pathlib import Path
from streamlit.testing.v1 import AppTest

UI=Path(__file__).resolve().parents[1]/'src/paper_rag/ui.py'


def test_experiment_views_without_api():
    app=AppTest.from_file(str(UI)).run(timeout=20)
    assert not app.exception
    app.sidebar.radio[0].set_value('实验对比').run()
    assert not app.exception
    assert any('无法读取结果' in x.value for x in app.error)
    app.sidebar.radio[0].set_value('失败案例').run()
    assert not app.exception


def test_saved_results_and_case_display(tmp_path):
    import json
    from paper_rag.experiments.context import build_context, load_tasks
    from paper_rag.experiments.runner import atomic_json
    root=UI.parents[2]
    task=load_tasks(root/'data/experiments/two_hop_demo.jsonl')[0]
    directory=tmp_path/'results'
    atomic_json(directory/'manifest.json',{'config':{'expected_records':3,'allow_unreviewed':True}})
    for condition in ('necessary','concentrated','dispersed'):
        run_id=f'display-{condition}'
        record={'run_id':run_id,'task':task,'condition':condition,'rotation':0,'reverse':False,
                'status':'ok','context':build_context(task,condition),
                'generation':{'answer':json.dumps({'answer':task['answer'],'citations':['E1','E2']},ensure_ascii=False),
                              'input_tokens':100,'output_tokens':20,'elapsed_seconds':1,'gpu':None}}
        atomic_json(directory/'records'/f'{run_id}.json',record)
    app=AppTest.from_file(str(UI)).run(timeout=20)
    app.sidebar.radio[0].set_value('实验对比').run()
    app.sidebar.text_input[0].set_value(str(directory)).run()
    assert not app.exception
    assert len(app.dataframe)==2
    app.sidebar.radio[0].set_value('失败案例').run()
    app.checkbox[0].set_value(False).run()
    assert not app.exception
    assert len(app.code)==3


def test_artificial_demo_has_explicit_labels_and_no_costs(tmp_path):
    import subprocess
    import sys
    import json
    root=UI.parents[2]
    output=tmp_path/'ui-demo'
    subprocess.run([sys.executable,str(root/'scripts/create_demo_results.py'),'--output',str(output)],check=True,capture_output=True,text=True)
    report=json.loads((output/'evaluation.json').read_text())
    assert len(report['details'])==14
    assert all(row['mean_seconds'] is None for row in report['summary'])
    app=AppTest.from_file(str(UI)).run(timeout=20)
    app.sidebar.radio[0].set_value('实验对比').run()
    app.sidebar.text_input[0].set_value(str(output)).run()
    assert not app.exception
    assert any('人工构造数据' in x.value for x in app.sidebar.warning)
    assert any('未调用模型' in x.value for x in app.warning)
