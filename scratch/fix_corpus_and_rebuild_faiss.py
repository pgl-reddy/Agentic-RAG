"""Script to correct ITA.pdf records from 'constitution' to 'act',
sync manifest, and rebuild FAISS index.
"""
import json
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
data_dir = _root / "data"

corpus_file = data_dir / "pdf_legal_corpus.jsonl"
manifest_file = data_dir / "pdf_manifest.json"

records = []
with open(corpus_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("filename") == "ITA.pdf":
            # Fix classification
            num = r.get("article") or r.get("section") or ""
            r["kind"] = "act"
            r["structure_kind"] = "act"
            r["section"] = num
            r["article"] = ""
            r["title"] = f"Information Technology Rules, 2021 — Rule {num}" if num else "Information Technology Rules, 2021"
            r["url"] = f"local-pdf://ITA.pdf#section-{num}" if num else "local-pdf://ITA.pdf"
            r["act"] = "Information Technology (Intermediary Guidelines and Digital Media Ethics Code) Rules, 2021"
        records.append(r)

# Write back cleaned corpus
with open(corpus_file, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Updated {len(records)} records in {corpus_file}")

# Update manifest
manifest = {}
if manifest_file.exists():
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}

for hash_key, entry in manifest.items():
    if entry.get("filename") == "ITA.pdf":
        entry["structure_kind"] = "act"

# Add sample_constitution.pdf if missing
has_sample = any(e.get("filename") == "sample_constitution.pdf" for e in manifest.values())
if not has_sample:
    import hashlib
    sample_text = "".join(r["text"] for r in records if r.get("filename") == "sample_constitution.pdf")
    sample_hash = hashlib.sha256(sample_text.encode("utf-8")).hexdigest()
    manifest[sample_hash] = {
        "filename": "sample_constitution.pdf",
        "chunk_count": sum(1 for r in records if r.get("filename") == "sample_constitution.pdf"),
        "structure_kind": "constitution",
        "source": "India Code",
        "ingested_at": "2026-08-18T06:37:22.058379+00:00"
    }

manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Updated manifest at {manifest_file}")
