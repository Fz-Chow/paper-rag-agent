from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_rag.core import PaperIndex

index = PaperIndex.from_directory(ROOT / "data" / "papers")
print(f"Indexed {len(index.chunks)} chunks from {len({c.source_file for c in index.chunks})} papers.")

