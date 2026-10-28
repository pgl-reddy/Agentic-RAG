from pathlib import Path

root = Path(__file__).resolve().parent.parent
rag_path = root / "rag_engine.py"

text = rag_path.read_text(encoding="utf-8")
old_part = '        data_dir / "pdf_legal_corpus.jsonl",\n    ):'
new_part = '        data_dir / "pdf_legal_corpus.jsonl",\n        data_dir / "indic_case_law_corpus.jsonl",\n    ):'

if old_part in text:
    text = text.replace(old_part, new_part, 1)
    rag_path.write_text(text, encoding="utf-8")
    print("Successfully patched rag_engine.py with indic_case_law_corpus.jsonl")
else:
    print("Could not find target in rag_engine.py")
