"""Trusted-source enforcement and conservative HTML-to-text ingestion."""
from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse
import hashlib
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

TRUSTED_EXACT = {
    "www.indiacode.nic.in", "indiacode.nic.in",
    "indiankanoon.org", "www.indiankanoon.org",
    "www.sci.gov.in", "sci.gov.in", "main.sci.gov.in", "registry.sci.gov.in",
    "delhihighcourt.nic.in", "www.delhihighcourt.nic.in",
    "bombayhighcourt.nic.in", "www.bombayhighcourt.nic.in",
    "calcuttahighcourt.gov.in", "www.calcuttahighcourt.gov.in",
    "allahabadhighcourt.in", "www.allahabadhighcourt.in",
    "hcmadras.tn.gov.in", "www.hcmadras.tn.gov.in",
    "mphc.gov.in", "www.mphc.gov.in",
    "hckerala.gov.in", "www.hckerala.gov.in",
    "patnahighcourt.gov.in", "www.patnahighcourt.gov.in",
    "jharkhandhighcourt.nic.in", "www.jharkhandhighcourt.nic.in",
    "orissahighcourt.nic.in", "www.orissahighcourt.nic.in",
    "cghighcourt.gov.in", "www.cghighcourt.gov.in",
    "highcourtchd.gov.in", "www.highcourtchd.gov.in",
    "ecourts.gov.in", "www.ecourts.gov.in", "services.ecourts.gov.in",
    "egazette.gov.in", "egazette.nic.in", "lawcommissionofindia.nic.in",
}


def is_trusted_legal_url(url: str, extra_domains: set[str] | None = None) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"https", "http"}:
        return False
    if host in TRUSTED_EXACT or host in (extra_domains or set()):
        return True
    if host.endswith(".hc.gov.in") or host.endswith(".ecourts.gov.in"):
        return True
    if (host.endswith(".gov.in") or host.endswith(".nic.in")) and (
        "highcourt" in host or "hcmadras" in host or host.startswith("hc")
    ):
        return True
    return False


def extract_page(url: str) -> tuple[str, str]:
    """Fetch one allowed page. Callers must validate the URL first."""
    response = requests.get(url, timeout=25, headers={"User-Agent": "NyayaRoute/1.0 research prototype"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    text = soup.get_text(" ", strip=True)
    return title, re.sub(r"\s+", " ", text)


def chunk_text(text: str, size: int = 900, overlap: int = 180) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size])
        if len(chunk) >= 180:
            chunks.append(chunk)
    # If document has substantial text (>= 40 chars) but is under 180 chars, preserve it as one chunk
    if not chunks and len(text.strip()) >= 40:
        chunks.append(text.strip())
    return chunks


def document_records(url: str, title: str, text: str) -> list[dict]:
    parsed = urlparse(url)
    source = "India Code" if "indiacode" in parsed.netloc else "Supreme Court of India" if "sci.gov.in" in parsed.netloc else "Indian Kanoon" if "indiankanoon" in parsed.netloc else parsed.netloc
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for index, chunk in enumerate(chunk_text(text)):
        digest = hashlib.sha256(f"{url}:{index}:{chunk}".encode()).hexdigest()[:16]
        records.append({"id": digest, "title": title, "url": url, "source": source, "text": chunk, "kind": "ingested legal material", "ingested_at": now})
    return records


def ingest_trusted_url(url: str, title_override: str | None = None) -> tuple[int, str]:
    """Fetch and append one explicitly supplied, allowed source URL."""
    if not is_trusted_legal_url(url):
        raise ValueError("Only India Code, Supreme Court, Indian Kanoon, and official High Court URLs are allowed.")
    title, text = extract_page(url)
    records = document_records(url, title_override or title, text)
    if not records:
        raise ValueError("No readable page text was extracted. Use a specific HTML legal document page, not a landing page or PDF.")
    destination = Path(__file__).resolve().parent / "data" / "legal_corpus.jsonl"
    destination.parent.mkdir(exist_ok=True)
    existing_ids = set()
    if destination.exists():
        existing_ids = {json.loads(line)["id"] for line in destination.read_text(encoding="utf-8").splitlines() if line.strip()}
    fresh = [record for record in records if record["id"] not in existing_ids]
    with destination.open("a", encoding="utf-8") as handle:
        for item in fresh:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if fresh:
        try:
            from vector_store import get_vector_store
            get_vector_store().add_records(fresh, persist=True)
        except Exception:
            pass
    return len(fresh), title_override or title

