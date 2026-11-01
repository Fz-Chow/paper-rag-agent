#!/usr/bin/env python3
"""Run, resume, or evaluate an evidence layout experiment."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from paper_rag.experiments.runner import atomic_json, run
from paper_rag.experiments.evaluate import evaluate, load_records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("data/experiments/two_hop_demo.jsonl"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--split",choices=["exploration","validation"],default="exploration")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--allow-unreviewed",action="store_true")
    parser.add_argument("--require-gpu",action="store_true")
    parser.add_argument("--retry-errors",action="store_true")
    parser.add_argument("--evaluate-only",action="store_true")
    parser.add_argument("--reviews",type=Path,help="JSON mapping run_id to human citation review")
    args=parser.parse_args()
    if args.evaluate_only:
        records=load_records(args.output)
        if not records:
            parser.error("No result records")
        reviews=json.loads(args.reviews.read_text()) if args.reviews else None
        report=evaluate(records,reviews)
        atomic_json(args.output/"evaluation.json",report)
    else:
        if args.reviews:
            parser.error("--reviews requires --evaluate-only")
        report=run(args.dataset,args.output,args.split,args.limit,args.allow_unreviewed,args.require_gpu,args.seed,args.retry_errors)
    print(json.dumps(report["summary"],ensure_ascii=False,indent=2))
    if any(s["failures"] for s in report["summary"]):
        sys.exit(1)

if __name__ == "__main__":
    main()
