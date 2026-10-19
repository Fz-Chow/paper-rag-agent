from pathlib import Path

from paper_rag.core import PaperIndex


def test_index_loads_zotero_papers():
    root = Path(__file__).resolve().parents[1]
    index = PaperIndex.from_directory(root / "data" / "papers")
    assert len(index.chunks) > 0
    results = index.search("retrieval augmented generation", top_k=3)
    assert results
    assert "paper_title" in results[0]

