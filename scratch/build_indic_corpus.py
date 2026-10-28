"""Build authoritative Indian Supreme Court Case Law Corpus from IndicLegalQA dataset.
Aggregates 10,002 Q&A pairs across 1,211 unique Supreme Court cases into structured,
comprehensive judicial case digests for BM25 and FAISS dense vector retrieval.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import re
import sys

_root = Path(__file__).resolve().parent.parent
data_dir = _root / "data"

qa_path = data_dir / "indic_legal_qa_10k.json"
out_path = data_dir / "indic_case_law_corpus.jsonl"


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "")
    return s.strip()


def build_corpus():
    if not qa_path.exists():
        print(f"Error: {qa_path} not found.")
        sys.exit(1)

    print(f"Reading {qa_path}...")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    print(f"Loaded {len(qa_data):,} QA pairs.")

    cases: dict[str, list[dict]] = defaultdict(list)
    for row in qa_data:
        c_name = clean_text(row.get("case_name", ""))
        if c_name:
            cases[c_name].append(row)

    print(f"Grouped into {len(cases):,} unique Supreme Court cases.")

    out_records = []
    for idx, (case_name, pairs) in enumerate(cases.items(), 1):
        first_p = pairs[0]
        date = clean_text(first_p.get("judgment_date") or first_p.get("judgement_date") or "Supreme Court of India")
        ref_pdf = clean_text(first_p.get("reference_pdf") or "")

        # Build clean structured case digest
        lines = [
            f"Supreme Court of India: {case_name}",
            f"Judgment Date: {date}",
            "Jurisdiction: Supreme Court of India",
            "Authority Level: Primary Binding Judicial Precedent",
            "",
            "Judicial Holdings, Fact Findings, and Legal Principles:"
        ]

        for p in pairs:
            q = clean_text(p.get("question", ""))
            a = clean_text(p.get("answer", ""))
            if q and a:
                lines.append(f"- Aspect: {q}")
                lines.append(f"  Judicial Finding: {a}")

        full_text = "\n".join(lines)
        slug = re.sub(r"[^a-z0-9]+", "_", case_name.lower())[:45].strip("_")
        doc_id = f"indic_sc_{idx:04d}_{slug}"

        encoded_query = case_name.replace(" ", "+")
        record = {
            "id": doc_id,
            "title": f"Supreme Court of India: {case_name} ({date})",
            "source": "Supreme Court of India (IndicLegalQA Case Digest)",
            "url": f"https://indiankanoon.org/search/?formInput={encoded_query}",
            "text": full_text,
            "kind": "case_law",
            "dataset": "indic_legal_qa",
            "authority_level": "Primary Binding Case Law",
            "jurisdiction": "Supreme Court of India",
            "case_name": case_name,
            "judgment_date": date,
            "reference_pdf": ref_pdf,
            "ingested_at": "2026-10-27T00:00:00Z"
        }
        out_records.append(record)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(out_records):,} case digest records in {out_path}.")
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Corpus file size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    build_corpus()
