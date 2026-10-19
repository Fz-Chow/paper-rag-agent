from __future__ import annotations

import os
from pathlib import Path

import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .core import PaperIndex


ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "data" / "papers"
index = PaperIndex.from_directory(PAPERS_DIR)
app = FastAPI(title="PaperRAG-Agent", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)


def _build_context(results: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {r['paper_title']} p.{r['page']}\n{r['text']}" for i, r in enumerate(results, start=1))


def _llm_answer(question: str, results: list[dict]) -> str | None:
    if os.getenv("PAPER_RAG_LLM_PROVIDER", "none") == "none":
        return None
    base_url = os.getenv("PAPER_RAG_LLM_BASE_URL", "").rstrip("/")
    model = os.getenv("PAPER_RAG_LLM_MODEL", "")
    if not base_url or not model:
        return None
    prompt = f"Answer only from the evidence. Cite sources as [1], [2].\nQuestion: {question}\nEvidence:\n{_build_context(results)}"
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('PAPER_RAG_LLM_API_KEY', '')}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(index.chunks), "llm_provider": os.getenv("PAPER_RAG_LLM_PROVIDER", "none")}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    results = index.search(request.question, request.top_k)
    answer = _llm_answer(request.question, results)
    if answer is None:
        answer = "基线检索已找到以下证据片段；当前未配置 LLM，因此暂不生成综合回答。"
    return {"answer": answer, "sources": results}

