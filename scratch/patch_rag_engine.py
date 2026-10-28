import re
from pathlib import Path

p = Path(r"C:\Users\gnana\OneDrive\Desktop\5thyear\MINI PROJECT\Project\final\Agentic-RAG\rag_engine.py")
text = p.read_text(encoding="utf-8")

if "def get_retriever" not in text:
    marker = "return sorted(candidates, key=lambda x: (x.cross_encoder_prob, x.score), reverse=True)"
    addition = """

_GLOBAL_RETRIEVER: LegalRetriever | None = None


def get_retriever() -> LegalRetriever:
    \"\"\"Process-wide cached retriever instance.\"\"\"
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = LegalRetriever(load_corpus())
    return _GLOBAL_RETRIEVER


def reset_retriever() -> None:
    \"\"\"Clear cached retriever so newly ingested documents are immediately visible.\"\"\"
    global _GLOBAL_RETRIEVER
    _GLOBAL_RETRIEVER = None
"""
    if marker in text:
        text = text.replace(marker, marker + addition, 1)
        p.write_text(text, encoding="utf-8")
        print("Successfully patched rag_engine.py with get_retriever and reset_retriever!")
    else:
        print("Marker not found in rag_engine.py")
else:
    print("get_retriever already present in rag_engine.py")
