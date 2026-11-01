from __future__ import annotations
import hashlib
import json
import os
import random
import subprocess
import time
import traceback
from pathlib import Path
from .context import build_context, load_tasks, measure_context
from .evaluate import evaluate, load_records
from ..generation import LocalGenerator, environment, gpu_metrics

ROOT = Path(__file__).resolve().parents[3]

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def code_hash():
    files = sorted((ROOT/"src"/"paper_rag").rglob("*.py")) + sorted((ROOT/"scripts").glob("*.py"))
    return hashlib.sha256(b"".join(str(p.relative_to(ROOT)).encode()+p.read_bytes() for p in files)).hexdigest()


def run(dataset, output, split="exploration", limit=None, allow_unreviewed=False,
        require_gpu=False, seed=42, retry_errors=False):
    dataset, output = Path(dataset), Path(output)
    tasks = [t for t in load_tasks(dataset) if t["split"] == split]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("No tasks selected")
    if allow_unreviewed and split == "validation":
        raise ValueError("Validation requires human verification; no unreviewed override")
    if any(t["human_review"]["status"] != "verified" for t in tasks) and not allow_unreviewed:
        raise ValueError("Dataset needs human review; exploration smoke test may use --allow-unreviewed")
    if not os.getenv("PAPER_RAG_LLM_MODEL"):
        raise ValueError("Set PAPER_RAG_LLM_MODEL to one local model")
    output.mkdir(parents=True, exist_ok=True)
    lock = output/".running"
    # OS advisory lock releases automatically after interruption (Linux/macOS).
    import fcntl
    with lock.open("a") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(dataset, output, tasks, split, limit, allow_unreviewed, require_gpu, seed, retry_errors)


def _run_locked(dataset, output, tasks, split, limit, allow_unreviewed, require_gpu, seed, retry_errors):
    try:
        generator = LocalGenerator()
        if require_gpu and generator.device != "cuda":
            raise RuntimeError("GPU inference required; CPU run refused")
        meta = generator.metadata()
        # Free memory is transient and must not invalidate resume.
        meta["gpu"] = {k:v for k,v in (meta["gpu"] or {}).items() if k in {"device_name","hip_version"}}
        config = {"dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "code_sha256":code_hash(), "model":meta, "environment":environment(),
            "split":split,"limit":limit,"allow_unreviewed":allow_unreviewed,"require_gpu":require_gpu,"seed":seed,
            "task_ids":[t["id"] for t in tasks],
            "expected_records":sum(2*len(t["evidence"])+2 for t in tasks), "conditions":["necessary","concentrated","dispersed"],
            "schedule":"balanced adjacent/opposite slots x both necessary-evidence orders; seeded condition order"}
        manifest_path = output/"manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["config"] != config:
                raise ValueError("Resume refused: dataset/code/model/environment/config changed; use a new output directory")
        else:
            if list((output/"records").glob("*.json")):
                raise ValueError("Orphan records without manifest")
            manifest = {"config":config,"created_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
            atomic_json(manifest_path,manifest)
        for task in tasks:
            for reverse in (False,True):
                for rotation in range(len(task["evidence"])//2):
                    conditions = ["concentrated","dispersed"]
                    if rotation == 0:
                        conditions.append("necessary")
                    # Stable schedule even when preceding records already exist.
                    random.Random(f"{seed}:{task['id']}:{rotation}:{reverse}").shuffle(conditions)
                    contexts, raw_contexts, prep_error = {}, {}, None
                    try:
                        for condition in conditions:
                            raw_contexts[condition] = build_context(task,condition,rotation,reverse)
                            contexts[condition] = measure_context(raw_contexts[condition],task,generator)
                            generator.check_length(contexts[condition]["input_tokens"])
                        a,b = contexts["concentrated"],contexts["dispersed"]
                        if a["material_hash"] != b["material_hash"] or a["input_tokens"] != b["input_tokens"]:
                            raise ValueError("Pair rejected: concentrated/dispersed material or prompt token counts differ")
                    except Exception:
                        prep_error = traceback.format_exc()
                    for condition in conditions:
                        run_id = f"{task['id']}--{condition}--r{rotation}--v{int(reverse)}"
                        target = output/"records"/f"{run_id}.json"
                        if target.exists():
                            previous = json.loads(target.read_text())
                            if previous["status"] == "ok" or not retry_errors:
                                continue
                            history=output/"attempts"/f"{run_id}--{time.time_ns()}.json"
                            atomic_json(history,previous)
                        record = {"run_id":run_id,"task":task,"condition":condition,"rotation":rotation,"reverse":reverse,
                            "context":contexts.get(condition,raw_contexts.get(condition)),"status":"error",
                            "recorded_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
                        attempt_started = time.perf_counter()
                        try:
                            if prep_error and condition != "necessary":
                                raise ValueError(prep_error)
                            ctx = contexts.get(condition)
                            if ctx is None:
                                ctx = measure_context(build_context(task,condition,rotation,reverse),task,generator)
                                record["context"] = ctx
                            generated = generator.generate(task["question"],ctx["text"])
                            record.update(status="ok",generation=generated.to_dict())
                        except Exception:
                            record["error"] = traceback.format_exc()
                            record["failure_runtime"] = {"elapsed_seconds":time.perf_counter()-attempt_started,
                                "gpu":gpu_metrics(generator.device) if not prep_error else None}
                        atomic_json(target,record)
                        print(f"{run_id}: {record['status']}",flush=True)
        report=evaluate(load_records(output))
        atomic_json(output/"evaluation.json",report)
        return report
    except Exception:
        atomic_json(output/"startup_error.json",{"error":traceback.format_exc(),"environment":environment()})
        raise
