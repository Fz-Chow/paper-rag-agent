from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    chunk_id: str
    paper_title: str
    source_file: str
    page: int
    text: str


class PaperIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks]) if chunks else None

    @classmethod
    def from_directory(cls, papers_dir: Path) -> "PaperIndex":
        manifest_path = papers_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks: list[Chunk] = []
        import fitz

        for item in manifest:
            pdf_path = papers_dir / item["file"]
            if not pdf_path.exists():
                continue
            document = fitz.open(pdf_path)
            for page_number, page in enumerate(document, start=1):
                text = re.sub(r"\s+", " ", page.get_text("text")).strip()
                paragraphs = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+", text) if len(p.strip()) >= 80]
                if not paragraphs and text:
                    paragraphs = [text]
                for index, paragraph in enumerate(paragraphs):
                    chunks.append(Chunk(f"{item['zotero_key']}-{page_number}-{index}", item["title"], item["file"], page_number, paragraph))
        return cls(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.chunks or self.matrix is None:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [{**asdict(self.chunks[i]), "score": round(float(scores[i]), 4)} for i in indices if scores[i] > 0]

