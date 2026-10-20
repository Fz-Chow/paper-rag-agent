from __future__ import annotations

import json
import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    chunk_id: str
    paper_title: str
    source_file: str
    page: int
    text: str


def resolve_device(requested: str | None = None) -> str:
    requested = (requested or os.getenv("PAPER_RAG_DEVICE", "auto")).lower()
    if requested == "cpu":
        return "cpu"
    if requested in {"cuda", "rocm"}:
        requested = "cuda"
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("PAPER_RAG_DEVICE must be auto, cpu, or rocm")
    try:
        import torch

        if torch.cuda.is_available() and requested != "cpu":
            return "cuda"
    except ImportError:
        pass
    if requested == "cuda":
        raise RuntimeError("GPU was requested, but torch.cuda.is_available() is false")
    return "cpu"


class PaperIndex:
    def __init__(self, chunks: list[Chunk], cache_dir: Path | None = None):
        self.chunks = chunks
        self.embedding_model_name = os.getenv("PAPER_RAG_EMBEDDING_MODEL", "").strip()
        self.reranker_model_name = os.getenv("PAPER_RAG_RERANKER_MODEL", "").strip()
        self.device = resolve_device() if self.embedding_model_name or self.reranker_model_name else "cpu"
        self.embedding_model = None
        self.reranker = None
        self.embeddings = None
        self.vectorizer = None
        self.matrix = None

        texts = [chunk.text for chunk in chunks]
        if self.embedding_model_name and texts:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
            cache_key = hashlib.sha256(
                (self.embedding_model_name + "\n" + "\n".join(f"{c.chunk_id}:{c.text}" for c in chunks)).encode("utf-8")
            ).hexdigest()
            cache_path = cache_dir / f"embeddings-{cache_key[:20]}.npz" if cache_dir else None
            if cache_path and cache_path.exists():
                with np.load(cache_path) as saved:
                    self.embeddings = saved["embeddings"]
            else:
                self.embeddings = self.embedding_model.encode(
                    texts,
                    batch_size=int(os.getenv("PAPER_RAG_EMBEDDING_BATCH_SIZE", "8")),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(cache_path, embeddings=self.embeddings)
        elif texts:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98)
            self.matrix = self.vectorizer.fit_transform(texts)

        if self.reranker_model_name:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(self.reranker_model_name, device=self.device)

    @classmethod
    def from_directory(cls, papers_dir: Path) -> "PaperIndex":
        manifest_path = papers_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks: list[Chunk] = []
        import pymupdf

        for item in manifest:
            pdf_path = papers_dir / item["file"]
            if not pdf_path.exists():
                continue
            with pymupdf.open(pdf_path) as document:
                for page_number, page in enumerate(document, start=1):
                    text = re.sub(r"\s+", " ", page.get_text("text")).strip()
                    paragraphs = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+", text) if len(p.strip()) >= 80]
                    if not paragraphs and text:
                        paragraphs = [text]
                    for index, paragraph in enumerate(paragraphs):
                        chunks.append(Chunk(f"{item['zotero_key']}-{page_number}-{index}", item["title"], item["file"], page_number, paragraph))
        return cls(chunks, cache_dir=papers_dir.parent / "index")

    @property
    def retrieval_backend(self) -> str:
        return "sentence-transformers" if self.embedding_model else "tfidf-cpu"

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.chunks:
            return []

        candidate_count = min(len(self.chunks), max(top_k * 4, top_k) if self.reranker else top_k)
        if self.embedding_model:
            query_vector = self.embedding_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
            scores = np.asarray(self.embeddings @ query_vector).ravel()
        else:
            if self.matrix is None or self.vectorizer is None:
                return []
            scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()

        indices = scores.argsort()[::-1][:candidate_count]
        results = [{**asdict(self.chunks[i]), "score": float(scores[i])} for i in indices if scores[i] > 0]
        if self.reranker and results:
            pairs = [(query, item["text"]) for item in results]
            rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
            for item, score in zip(results, rerank_scores):
                item["retrieval_score"] = round(item["score"], 4)
                item["score"] = round(float(score), 4)
            results.sort(key=lambda item: item["score"], reverse=True)
        else:
            for item in results:
                item["score"] = round(item["score"], 4)
        return results[:top_k]

