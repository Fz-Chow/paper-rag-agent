"""Verify a completed real AMD generation smoke run, independent of answer accuracy."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paper_rag.experiments.evaluate import load_records


def verify(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    records=load_records(directory)
    if len(records)!=manifest['config']['expected_records']:
        raise ValueError('Incomplete run')
    pairs={}
    for r in records:
        if r['status']!='ok':
            raise ValueError(f"Failed run: {r['run_id']}; inspect its error field")
        g=r['generation']; gpu=g.get('gpu') or {}
        if g['device']!='cuda' or not gpu.get('hip_version') or gpu.get('peak_allocated_mb',0)<=0:
            raise ValueError(f"Missing real AMD inference metrics: {r['run_id']}")
        if not g['prompt'] or g['input_tokens']<=0 or g['output_tokens']<=0 or g['elapsed_seconds']<=0:
            raise ValueError('Missing generation telemetry')
        if g['prompt'] != r['context']['prompt'] or g['input_tokens'] != r['context']['input_tokens']:
            raise ValueError('Measured input does not match generated input')
        if r['condition']!='necessary':
            key=(r['task']['id'],r['rotation'],r['reverse'])
            pairs.setdefault(key,{})[r['condition']]=r
    for pair in pairs.values():
        a,b=pair['concentrated'],pair['dispersed']
        if a['context']['material_hash']!=b['context']['material_hash'] or a['generation']['input_tokens']!=b['generation']['input_tokens']:
            raise ValueError('Pair materials or input lengths differ')
    return {'records':len(records),'pairs':len(pairs),'amd_generation_verified':True,
            'note':'Runtime acceptance only; answer quality and human evidence review are separate.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.directory),ensure_ascii=False,indent=2))
