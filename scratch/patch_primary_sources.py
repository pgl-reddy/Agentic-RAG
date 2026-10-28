from pathlib import Path

root = Path(__file__).resolve().parent.parent
rag_path = root / "rag_engine.py"
text = rag_path.read_text(encoding="utf-8")

old_snippet = '        if e.source in PRIMARY_SOURCES or (e.url and e.url.startswith("local-pdf://"))'
new_snippet = '        if any(ps.lower() in e.source.lower() for ps in PRIMARY_SOURCES) or (e.url and e.url.startswith("local-pdf://"))'

if old_snippet in text:
    text = text.replace(old_snippet, new_snippet, 1)
    rag_path.write_text(text, encoding="utf-8")
    print("Successfully patched line 664 in rag_engine.py")
else:
    print("Old snippet not found!")
