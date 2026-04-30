"""
Evidence search for trusted-source matches.

Google Custom Search is used when GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX
are configured. If Google fails, SerpAPI is used as fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from urllib.error import HTTPError
from urllib.parse import quote_plus, urlencode, urlparse
from urllib.request import urlopen

from app.backend.config import (
    EVIDENCE_SEARCH_RESULTS,
    EVIDENCE_SEARCH_TIMEOUT,
    GOOGLE_SEARCH_API_KEY,
    GOOGLE_SEARCH_CX,
    SERPAPI_API_KEY,
    TRUSTED_EVIDENCE_DOMAINS,
)

try:
    import requests
except ImportError:
    requests = None


TOKEN_PATTERN = re.compile(r"[\u0980-\u09ffA-Za-z0-9]+")
MIN_TOKEN_OVERLAP = 2


@dataclass
class EvidenceItem:
    title: str
    link: str
    snippet: str
    source: str


@dataclass
class EvidenceResult:
    status: str
    query: str
    search_url: str
    items: list[EvidenceItem]
    note: str


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.removeprefix("www.").lower()


def _clean_snippet(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]


def _compact_query(headline: str, content: str) -> str:
    headline_words = headline.strip().split()[:12]
    query = " ".join(headline_words)

    if len(query) < 20 and content.strip():
        query = " ".join(
            [query, " ".join(content.strip().split()[:8])]
        ).strip()

    return query


def _trusted_site_filter() -> str:
    return " OR ".join(
        f"site:{domain}" for domain in TRUSTED_EVIDENCE_DOMAINS
    )


def _manual_search_url(query: str) -> str:
    return "https://www.google.com/search?q=" + quote_plus(
        f"{query} ({_trusted_site_filter()})"
    )


def _tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(value)
        if len(token) > 2
    }


def _is_relevant(query: str, item: EvidenceItem) -> bool:
    query_tokens = _tokens(query)

    if not query_tokens:
        return False

    item_tokens = _tokens(f"{item.title} {item.snippet}")
    return len(query_tokens & item_tokens) >= MIN_TOKEN_OVERLAP


def _dedupe_and_filter(query: str, items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen: set[str] = set()
    filtered: list[EvidenceItem] = []
    trusted_domains = tuple(TRUSTED_EVIDENCE_DOMAINS)

    for item in items:
        domain = _extract_domain(item.link)

        if not domain.endswith(trusted_domains):
            continue

        if not _is_relevant(query, item):
            continue

        if item.link in seen:
            continue

        seen.add(item.link)

        filtered.append(
            EvidenceItem(
                title=item.title,
                link=item.link,
                snippet=item.snippet,
                source=domain or item.source,
            )
        )

    return filtered[:EVIDENCE_SEARCH_RESULTS]


def _search_google(query: str) -> list[EvidenceItem]:
    params = urlencode(
        {
            "key": GOOGLE_SEARCH_API_KEY,
            "cx": GOOGLE_SEARCH_CX,
            "q": f"{query} ({_trusted_site_filter()})",
            "num": max(1, min(EVIDENCE_SEARCH_RESULTS, 10)),
        }
    )

    try:
        with urlopen(
            f"https://www.googleapis.com/customsearch/v1?{params}",
            timeout=EVIDENCE_SEARCH_TIMEOUT,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))

    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")

        try:
            payload = json.loads(detail)
            message = payload.get("error", {}).get("message", detail)
        except json.JSONDecodeError:
            message = detail

        raise RuntimeError(f"Google Custom Search failed: {message}") from exc

    items: list[EvidenceItem] = []

    for item in payload.get("items", []):
        link = str(item.get("link", ""))

        items.append(
            EvidenceItem(
                title=str(item.get("title", "")),
                link=link,
                snippet=str(item.get("snippet", "")),
                source=_extract_domain(link),
            )
        )

    return items


def _search_serpapi(query: str) -> tuple[list[EvidenceItem], str]:
    if requests is None:
        return [], "Install requests to use SerpAPI fallback."

    if not SERPAPI_API_KEY:
        return [], "SERPAPI_API_KEY is missing."

    search_query = f"{query} ({_trusted_site_filter()})"

    params = {
        "engine": "google",
        "q": search_query,
        "api_key": SERPAPI_API_KEY,
        "num": max(1, min(EVIDENCE_SEARCH_RESULTS, 10)),
    }

    try:
        response = requests.get(
            "https://serpapi.com/search.json",
            params=params,
            timeout=EVIDENCE_SEARCH_TIMEOUT,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout:
        return [], "SerpAPI search timed out."

    except requests.exceptions.ConnectionError:
        return [], "Could not connect to SerpAPI."

    except Exception as exc:
        return [], f"SerpAPI error: {str(exc)[:200]}"

    try:
        payload = response.json()
    except Exception:
        return [], "SerpAPI returned invalid JSON."

    if "error" in payload:
        return [], f"SerpAPI error: {payload.get('error')}"

    items: list[EvidenceItem] = []

    for item in payload.get("organic_results", []):
        link = str(item.get("link", ""))

        if not link:
            continue

        items.append(
            EvidenceItem(
                title=str(item.get("title", "")),
                link=link,
                snippet=_clean_snippet(str(item.get("snippet", ""))),
                source=_extract_domain(link),
            )
        )

    return items, ""


def search_evidence(headline: str, content: str = "") -> EvidenceResult:
    query = _compact_query(headline, content)
    search_url = _manual_search_url(query)

    if not query:
        return EvidenceResult(
            status="no_query",
            query=query,
            search_url=search_url,
            items=[],
            note="Headline or content is needed for source search.",
        )

    provider = "none"
    error = ""
    raw_items: list[EvidenceItem] = []

    if SERPAPI_API_KEY:
        provider = "serpapi"
        raw_items, serpapi_error = _search_serpapi(query)
        if serpapi_error:
            error = f"SerpAPI error: {serpapi_error}"
            if GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX:
                try:
                    provider = "google"
                    raw_items = _search_google(query)
                    error = ""
                except Exception as exc:
                    error = f"{error}; Google fallback error: {str(exc)[:220]}"

    elif GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX:
        try:
            provider = "google"
            raw_items = _search_google(query)
        except Exception as exc:
            error = f"Google error: {str(exc)[:220]}"

    else:
        error = "Set SERPAPI_API_KEY, or set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX."

    items = _dedupe_and_filter(query, raw_items)

    print("Evidence provider:", provider)
    print("Evidence query:", query)
    print("Evidence raw items:", len(raw_items))
    print("Evidence filtered items:", len(items))

    if items:
        return EvidenceResult(
            status="found",
            query=query,
            search_url=search_url,
            items=items,
            note=f"Found {len(items)} trusted-source match(es) using {provider}.",
        )

    note = "No matching trusted-source article was found."

    if error:
        note = f"{note} Search note: {error}"

    return EvidenceResult(
        status="no_results",
        query=query,
        search_url=search_url,
        items=[],
        note=note,
    )
