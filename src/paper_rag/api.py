from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core import PaperIndex, resolve_device


ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "data" / "papers"
index = PaperIndex.from_directory(PAPERS_DIR)
app = FastAPI(title="PaperRAG-Agent", version="0.2.0")
_generator = None
_generator_lock = threading.Lock()


class QueryRequest(BaseModel):
    question: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)


def _plan(question: str) -> dict:
    if re.search(r"比较|对比|区别|compare|difference|versus|\bvs\b", question, re.I):
        return {"task": "compare", "steps": ["search_papers", "compare_evidence", "answer_with_citations"]}
    if re.search(r"总结|概括|总结一下|summari[sz]e|overview", question, re.I):
        return {"task": "summarize", "steps": ["search_papers", "synthesize_evidence", "answer_with_citations"]}
    return {"task": "question_answering", "steps": ["search_papers", "answer_with_citations"]}


def _build_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {item['paper_title']}，第 {item['page']} 页\n{item['text']}"
        for i, item in enumerate(results, start=1)
    )


def _get_generator():
    global _generator
    model_name = os.getenv("PAPER_RAG_LLM_MODEL", "").strip()
    if not model_name:
        return None
    with _generator_lock:
        if _generator is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = resolve_device()
            _generator = ("loading", model_name)
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                ).to(device)
                model.eval()
                _generator = (tokenizer, model, device)
            except Exception:
                _generator = None
                raise
    return _generator


def _llm_answer(question: str, results: list[dict], task: str) -> str | None:
    loaded = _get_generator()
    if loaded is None:
        return None
    tokenizer, model, device = loaded
    instruction = {
        "compare": "比较证据中的方法、检索流程和适用场景。只比较证据实际支持的内容。",
        "summarize": "综合证据总结论文的研究问题、方法和主要结论。",
        "question_answering": "直接回答问题，并解释关键依据。",
    }[task]
    messages = [
        {
            "role": "system",
            "content": (
                "你是科研论文阅读助手。只根据给定证据回答，不得补充证据之外的事实。"
                "使用中文回答；每个关键结论都用 [1] 这样的编号引用证据。"
                "如果证据不足，明确说明不足。\n任务：" + instruction
            ),
        },
        {"role": "user", "content": f"问题：{question}\n\n论文证据：\n{_build_context(results)}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    max_new_tokens = int(os.getenv("PAPER_RAG_MAX_NEW_TOKENS", "512"))
    import torch

    with _generator_lock, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _gpu_metrics() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "device_name": torch.cuda.get_device_name(0),
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
            "free_mb": round(free_bytes / 1024**2, 1),
            "total_mb": round(total_bytes / 1024**2, 1),
            "hip_version": getattr(torch.version, "hip", None),
        }
    except Exception:
        return None


@app.get("/health")
def health() -> dict:
    active_device = index.device
    if os.getenv("PAPER_RAG_LLM_MODEL") and active_device == "cpu":
        active_device = resolve_device()
    return {
        "status": "ok",
        "chunks": len(index.chunks),
        "device": active_device,
        "retrieval_backend": index.retrieval_backend,
        "embedding_model": index.embedding_model_name or None,
        "reranker_model": index.reranker_model_name or None,
        "llm_model": os.getenv("PAPER_RAG_LLM_MODEL") or None,
    }


@app.post("/query")
def query(request: QueryRequest) -> dict:
    started = time.perf_counter()
    plan = _plan(request.question)
    try:
        results = index.search(request.question, request.top_k)
        answer = _llm_answer(request.question, results, plan["task"])
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference failed: {type(exc).__name__}: {exc}") from exc
    if answer is None:
        answer = "已检索到以下论文证据。设置 PAPER_RAG_LLM_MODEL 后可启用本地 GPU 综合回答。"
    return {
        "answer": answer,
        "sources": results,
        "agent": plan,
        "device": index.device if index.embedding_model or index.reranker else (
            resolve_device() if os.getenv("PAPER_RAG_LLM_MODEL") else "cpu"
        ),
        "retrieval_backend": index.retrieval_backend,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "gpu": _gpu_metrics(),
    }

