"""Ingest a single vetted legal web page into the local corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trusted_sources import document_records, extract_page, is_trusted_legal_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest one trusted Indian legal source URL.")
    parser.add_argument("url")
    parser.add_argument("--title", help="Override extracted title")
    parser.add_argument("--allow-domain", action="append", default=[], help="Explicitly vetted additional official domain")
    args = parser.parse_args()
    allowed = {domain.lower() for domain in args.allow_domain}
    if not is_trusted_legal_url(args.url, allowed):
        raise SystemExit("Rejected: URL is not in the trusted legal-source allowlist.")
    title, text = extract_page(args.url)
    records = document_records(args.url, args.title or title, text)
    if not records:
        raise SystemExit("No usable text was extracted; try an HTML content page rather than an index/PDF.")
    destination = Path(__file__).resolve().parent / "data" / "legal_corpus.jsonl"
    destination.parent.mkdir(exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Stored {len(records)} chunks in {destination}")
    try:
        from vector_store import get_vector_store
        added = get_vector_store().add_records(records, persist=True)
        if added:
            print(f"Indexed {added} new chunks into vector store.")
    except Exception as exc:
        print(f"Vector store indexing notice: {exc}")


if __name__ == "__main__":
    main()
