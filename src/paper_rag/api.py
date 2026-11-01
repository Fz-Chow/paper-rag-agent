from __future__ import annotations

import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core import PaperIndex, resolve_device
from .generation import get_generator


ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "data" / "papers"
index = PaperIndex.from_directory(PAPERS_DIR)
app = FastAPI(title="Evidence Layout Lab", version="0.3.0")


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
        generator = get_generator()
        generated = generator.generate(request.question, _build_context(results), plan["task"]) if generator else None
        answer = generated.answer if generated else None
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
        "gpu": generated.gpu if generated else None,
        "generation": generated.to_dict() if generated else None,
        "citation_check": {
            "cited_ids": re.findall(r"\[(\d+)\]", answer),
            "invalid_ids": [x for x in re.findall(r"\[(\d+)\]", answer) if not 1 <= int(x) <= len(results)],
            "support_status": "requires_human_review",
            "necessary_coverage": None,
        },
    }

