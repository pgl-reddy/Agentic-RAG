"""One-time migration: convert the existing JSONL corpora into a
persistent FAISS index (`data/faiss.index`) + metadata store
(`data/metadata.pkl`), without losing any information.

Reads every corpus file rag_engine.load_corpus() currently reads,
including the 12,163-record static corpus, the demo corpus, URL ingestions,
and PDF ingestions, so the FAISS index covers the same material the app
searches. Every field already on a record (including extras
like "ingested_at" or "filename" that PDF-ingested records carry) is kept;
only the fields required by the target schema are backfilled with an
empty default if a record doesn't already have them (this affects the
older URL-ingested records, which predate the extended metadata schema).

Usage:
    python migrate_to_faiss.py            # build/refresh from all corpora
    python migrate_to_faiss.py --rebuild  # wipe and rebuild from scratch

Safe to re-run: existing records (matched by their string "id") are never
re-embedded; only genuinely new records are added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rag_engine import load_corpus
from vector_store import FAISS_INDEX_PATH, METADATA_PATH, VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate JSONL legal corpora into a persistent FAISS index.")
    parser.add_argument("--rebuild", action="store_true", help="Delete any existing faiss.index/metadata.pkl first and rebuild from scratch.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embed this many chunks per batch (memory/throughput tradeoff).")
    args = parser.parse_args()

    if args.rebuild:
        for path in (FAISS_INDEX_PATH, METADATA_PATH):
            if path.exists():
                path.unlink()
                print(f"Removed existing {path}")

    corpus = load_corpus()
    if not corpus:
        raise SystemExit(
            "No records found in data/static_legal_corpus.jsonl, data/demo_legal_corpus.jsonl, "
            "data/legal_corpus.jsonl, or data/pdf_legal_corpus.jsonl — nothing to migrate."
        )
    print(f"Loaded {len(corpus)} chunks from the existing JSONL corpora.")

    store = VectorStore.load_or_create()
    print(f"Vector store currently holds {store.size} chunk(s) before migration.")

    added_total = 0
    for start in range(0, len(corpus), args.batch_size):
        batch = corpus[start : start + args.batch_size]
        # Don't re-save to disk after every small batch; save once at the end.
        added = store.add_records(batch, persist=False)
        added_total += added
        print(f"  processed {min(start + args.batch_size, len(corpus))}/{len(corpus)} "
              f"(added {added_total} new, skipped {min(start + args.batch_size, len(corpus)) - added_total} already-present)")

    store.save()
    print(f"\nDone. {added_total} new chunk(s) embedded and added.")
    print(f"Vector store now holds {store.size} chunk(s) total.")
    print(f"Saved to {FAISS_INDEX_PATH} and {METADATA_PATH}.")


if __name__ == "__main__":
    main()
