"""Inspect the actual runtime; optionally persist a JSON environment record."""
import argparse
import json
import os
import platform
import subprocess
from pathlib import Path


def inspect_runtime():
    info={'python':platform.python_version(),'platform':platform.platform(),
          'requested_device':os.getenv('PAPER_RAG_DEVICE','auto')}
    try:
        import torch
        info.update(torch=torch.__version__,hip=getattr(torch.version,'hip',None),gpu_available=torch.cuda.is_available())
        if info['gpu_available']:
            info['device_name']=torch.cuda.get_device_name()
            # A real tensor operation, not just visibility detection.
            x=torch.ones((32,32),device='cuda'); y=x@x
            torch.cuda.synchronize()
            info['gpu_tensor_check']=float(y[0,0].item())==32.0
    except ImportError:
        info.update(torch=None,hip=None,gpu_available=False)
    for command in ('rocminfo','rocm-smi'):
        try:
            r=subprocess.run([command],capture_output=True,text=True,timeout=15)
            info[command]={'returncode':r.returncode,'stdout':r.stdout[:12000],'stderr':r.stderr[:2000]}
        except (FileNotFoundError,subprocess.TimeoutExpired) as exc:
            info[command]={'unavailable':str(exc)}
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--require-amd',action='store_true')
    args=parser.parse_args()
    info=inspect_runtime(); text=json.dumps(info,ensure_ascii=False,indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+'\n')
    if args.require_amd and not (info.get('hip') and info.get('gpu_tensor_check')):
        parser.exit(1,'AMD HIP GPU tensor check failed; do not run a formal experiment.\n')

if __name__=='__main__':
    main()
