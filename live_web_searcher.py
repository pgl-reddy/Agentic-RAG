"""Live legal web searcher for Indian Kanoon (Supreme Court, High Courts, and Acts).
Retrieves real-time judgments, citations, benches, and statutory excerpts.
"""

from __future__ import annotations

import concurrent.futures
import re
from typing import Any
import requests
from bs4 import BeautifulSoup
import trafilatura

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 NyayaRoute/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_SEARCH_CACHE: dict[tuple[str, int], list[dict[str, Any]]] = {}


def sanitize_search_query(query: str) -> str:
    """Prepare legal query for Indian Kanoon formInput."""
    # Clean up excess whitespace and non-alphanumeric noise
    cleaned = re.sub(r"[\"'\(\)\[\]]", " ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _fetch_single_doc(item: dict[str, Any], query_tokens: set[str]) -> dict[str, Any]:
    doc_id = item.get("doc_id")
    if not doc_id:
        return item
    clean_doc_url = f"https://indiankanoon.org/doc/{doc_id}/"
    try:
        doc_resp = requests.get(clean_doc_url, headers=DEFAULT_HEADERS, timeout=(1.8, 3.0))
        if doc_resp.status_code == 200:
            extracted = trafilatura.extract(doc_resp.text, include_comments=False)
            if extracted:
                lines = [l.strip() for l in extracted.split("\n") if len(l.strip()) > 70]
                substantive = [
                    l for l in lines
                    if not l.startswith("Case :") and not l.startswith("Applicant")
                    and not l.startswith("Counsel") and not l.startswith("Advocate")
                    and not l.startswith("HIGH COURT")
                ]
                best_paras = []
                for p in substantive:
                    p_tokens = set(re.findall(r"[a-zA-Z0-9]+", p.lower()))
                    overlap = len(query_tokens & p_tokens)
                    if overlap > 0:
                        best_paras.append((overlap, p))

                if best_paras:
                    best_paras.sort(key=lambda x: x[0], reverse=True)
                    top_texts = [p for _, p in best_paras[:3]]
                    doc_text = f"{item['title']}\n\n" + "\n\n".join(top_texts)
                    item["text"] = doc_text[:1500]
                else:
                    item["text"] = extracted[:1400].replace("\n\n", "\n").strip()
    except Exception:
        pass
    return item


def search_indian_kanoon(query: str, max_results: int = 4) -> list[dict[str, Any]]:
    """Search Indian Kanoon live for judgments and statutory provisions.
    
    Returns list of dicts with:
      - title: Case / Act title
      - url: Direct link to judgment or doc
      - source: 'Indian Kanoon'
      - text: Clean extracted excerpt (bench, facts, legal holding)
      - doc_id: Document ID if available
    """
    cleaned_query = sanitize_search_query(query)
    if not cleaned_query:
        return []

    cache_key = (cleaned_query.lower(), max_results)
    if cache_key in _SEARCH_CACHE:
        return _SEARCH_CACHE[cache_key]

    encoded = requests.utils.quote(cleaned_query)
    search_url = f"https://indiankanoon.org/search/?formInput={encoded}"
    
    results: list[dict[str, Any]] = []
    
    try:
        resp = requests.get(search_url, headers=DEFAULT_HEADERS, timeout=(3.0, 5.0))
        if resp.status_code != 200:
            return results

        soup = BeautifulSoup(resp.text, "html.parser")
        result_divs = soup.select(".result")

        prelim_results: list[dict[str, Any]] = []
        for div in result_divs[:max_results]:
            title_el = div.select_one(".result_title a")
            headline_el = div.select_one(".headline")
            if not title_el:
                continue

            title = title_el.text.strip()
            href = title_el.get("href", "")
            full_url = f"https://indiankanoon.org{href}" if href.startswith("/") else href
            snippet = headline_el.text.strip() if headline_el else ""

            # Robust doc_id extraction handling relative and absolute links
            doc_id = ""
            doc_match = re.search(r"/(?:doc|docfragment)/(\d+)", full_url)
            if doc_match:
                doc_id = doc_match.group(1)

            fallback_text = snippet if snippet else f"Judicial precedent from Indian Kanoon: {title}. Review the full judgment text for specific findings."

            item = {
                "title": title,
                "url": full_url,
                "source": "Indian Kanoon",
                "text": fallback_text,
                "doc_id": doc_id,
            }
            prelim_results.append(item)

        # Concurrently fetch substantive document excerpts
        q_tokens = set(re.findall(r"[a-zA-Z0-9]+", cleaned_query.lower()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(1, len(prelim_results)), 3)) as executor:
            list(executor.map(lambda it: _fetch_single_doc(it, q_tokens), prelim_results))

        results = prelim_results

        # Only cache successful non-empty results
        if results:
            _SEARCH_CACHE[cache_key] = results
            if len(_SEARCH_CACHE) > 128:
                _SEARCH_CACHE.pop(next(iter(_SEARCH_CACHE)))

    except Exception:
        # Graceful failure: return empty list on network or parse error
        return results

    return results
