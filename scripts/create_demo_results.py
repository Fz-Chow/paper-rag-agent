"""Create clearly labeled, hand-authored UI examples. Never invokes a model."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paper_rag.experiments.context import build_context, load_tasks
from paper_rag.experiments.evaluate import evaluate
from paper_rag.experiments.runner import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/ui-demo'))
    args=parser.parse_args()
    out=args.output
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory must be empty; choose a new demo directory')
    root=Path(__file__).resolve().parents[1]
    task=load_tasks(root/'data/experiments/two_hop_demo.jsonl')[0]
    records=[]
    for reverse in (False,True):
        for rotation in range(len(task['evidence'])//2):
            conditions=['concentrated','dispersed']+(['necessary'] if rotation==0 else [])
            for condition in conditions:
                ctx=build_context(task,condition,rotation,reverse)
                wrong=condition=='dispersed' and rotation==0
                answer='北港' if wrong else task['answer']
                citations=['E1','E3'] if wrong else task['required_ids']
                run_id=f"{task['id']}--{condition}--r{rotation}--v{int(reverse)}"
                record={'run_id':run_id,'task':task,'condition':condition,'rotation':rotation,'reverse':reverse,
                    'status':'ok','presentation_only':True,'context':ctx,
                    'generation':{'answer':json.dumps({'answer':answer,'citations':citations},ensure_ascii=False),
                        'input_tokens':None,'output_tokens':None,'elapsed_seconds':None,'device':'not_measured','gpu':None}}
                atomic_json(out/'records'/f'{run_id}.json',record)
                records.append(record)
    atomic_json(out/'manifest.json',{'presentation_only':True,
        'notice':'人工构造的界面演示数据，未执行模型推理，不用于研究结果。',
        'config':{'expected_records':len(records),'allow_unreviewed':True,'split':'exploration','model':None}})
    atomic_json(out/'evaluation.json',evaluate(records))
    print(f'Created {len(records)} hand-authored UI examples in {out}; no model was called.')

if __name__=='__main__':
    main()
