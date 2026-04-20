from __future__ import annotations

from dataclasses import dataclass
import json
import re
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen

from app.backend.config import (
    EVIDENCE_SEARCH_PROVIDER,
    EVIDENCE_SEARCH_RESULTS,
    EVIDENCE_SEARCH_TIMEOUT,
    GOOGLE_SEARCH_API_KEY,
    GOOGLE_SEARCH_CX,
    TAVILY_API_KEY,
    TRUSTED_EVIDENCE_DOMAINS,
)


MIN_TOKEN_OVERLAP = 2
MIN_RELEVANCE_SCORE = 0.12
TOKEN_PATTERN = re.compile(r"[\u0980-\u09ffA-Za-z0-9]+")
STOPWORDS = {
    "এবং",
    "এই",
    "একটি",
    "করে",
    "করা",
    "করতে",
    "কিন্তু",
    "জন্য",
    "তার",
    "তারা",
    "তিনি",
    "দিয়ে",
    "নিয়ে",
    "প্রতি",
    "বলা",
    "হবে",
    "হয়",
    "হয়েছে",
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
}


@dataclass
class EvidenceItem:
    title: str
    link: str
    snippet: str
    source: str


@dataclass
class EvidenceResult:
    status: str
    verdict_hint: str
    query: str
    search_url: str
    items: list[EvidenceItem]
    note: str


def _compact_text(value: str, max_words: int) -> str:
    return " ".join(value.split()[:max_words])


def build_evidence_query(category: str, headline: str, content: str) -> str:
    headline = headline.strip()
    if headline:
        return headline
    return " ".join(part for part in [category.strip(), _compact_text(content, 8)] if part).strip()


def _trusted_site_filter() -> str:
    return " OR ".join(f"site:{domain}" for domain in TRUSTED_EVIDENCE_DOMAINS)


def _google_search_url(query: str) -> str:
    return "https://www.google.com/search?" + urlencode(
        {"q": f"{query} ({_trusted_site_filter()})"}
    )


def _source_from_link(link: str) -> str:
    return link.split("/")[2].removeprefix("www.") if "://" in link else ""


def _tokens(value: str) -> set[str]:
    tokens = {token.lower() for token in TOKEN_PATTERN.findall(value)}
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _relevance_score(query_tokens: set[str], item: EvidenceItem) -> float:
    if not query_tokens:
        return 0.0
    item_tokens = _tokens(f"{item.title} {item.snippet}")
    overlap = query_tokens & item_tokens
    if len(overlap) < MIN_TOKEN_OVERLAP:
        return 0.0
    return len(overlap) / len(query_tokens)


def _filter_relevant_items(headline: str, content: str, items: list[EvidenceItem]) -> list[EvidenceItem]:
    query_tokens = _tokens(f"{headline} {_compact_text(content, 35)}")
    return [
        item
        for item in items
        if _relevance_score(query_tokens, item) >= MIN_RELEVANCE_SCORE
    ]


def _fetch_google_results(query: str) -> list[EvidenceItem]:
    params = urlencode(
        {
            "key": GOOGLE_SEARCH_API_KEY,
            "cx": GOOGLE_SEARCH_CX,
            "q": query,
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
        raise RuntimeError(f"Google Custom Search error {exc.code}: {message}") from exc

    items: list[EvidenceItem] = []
    for item in payload.get("items", []):
        link = str(item.get("link", ""))
        items.append(
            EvidenceItem(
                title=str(item.get("title", "")),
                link=link,
                snippet=str(item.get("snippet", "")),
                source=_source_from_link(link),
            )
        )
    return items


def _fetch_tavily_results(query: str) -> list[EvidenceItem]:
    payload = {
        "query": query,
        "topic": "news",
        "search_depth": "basic",
        "max_results": max(1, min(EVIDENCE_SEARCH_RESULTS, 20)),
        "include_answer": False,
        "include_raw_content": False,
        "include_domains": TRUSTED_EVIDENCE_DOMAINS,
    }
    request = Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=EVIDENCE_SEARCH_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(detail)
            message = payload.get("detail") or payload.get("message") or detail
        except json.JSONDecodeError:
            message = detail
        raise RuntimeError(f"Tavily search error {exc.code}: {message}") from exc

    items: list[EvidenceItem] = []
    for item in payload.get("results", []):
        link = str(item.get("url", ""))
        items.append(
            EvidenceItem(
                title=str(item.get("title", "")),
                link=link,
                snippet=str(item.get("content", "")),
                source=_source_from_link(link),
            )
        )
    return items


def verdict_from_evidence(model_label: str, item_count: int) -> tuple[str, str]:
    if item_count >= 2 and model_label == "real":
        return "likely_real", "Model predicts real and multiple relevant trusted-source matches were found."
    if item_count >= 2 and model_label == "fake":
        return "conflicting_evidence", "Model predicts fake, but relevant trusted-source matches exist; review sources."
    if item_count == 1:
        return "limited_evidence", "Only one relevant trusted-source match was found."
    if model_label == "fake":
        return "likely_fake_low_evidence", "Model predicts fake and no relevant trusted-source match was found."
    return "uncertain", "Model predicts real, but relevant trusted-source evidence was not found."


def check_evidence(
    *,
    category: str,
    headline: str,
    content: str,
    model_label: str,
) -> EvidenceResult:
    query = build_evidence_query(category, headline, content)
    search_url = _google_search_url(query)

    use_tavily = EVIDENCE_SEARCH_PROVIDER in {"auto", "tavily"} and bool(TAVILY_API_KEY)
    use_google = (
        EVIDENCE_SEARCH_PROVIDER in {"auto", "google"}
        and bool(GOOGLE_SEARCH_API_KEY)
        and bool(GOOGLE_SEARCH_CX)
    )

    if not use_tavily and not use_google:
        return EvidenceResult(
            status="not_configured",
            verdict_hint="model_only",
            query=query,
            search_url=search_url,
            items=[],
            note="Evidence search needs TAVILY_API_KEY, or GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX.",
        )

    try:
        raw_items = _fetch_tavily_results(query) if use_tavily else _fetch_google_results(query)
        items = _filter_relevant_items(headline, content, raw_items)
    except Exception as exc:
        if use_tavily and use_google:
            try:
                raw_items = _fetch_google_results(query)
                items = _filter_relevant_items(headline, content, raw_items)
            except Exception as google_exc:
                return EvidenceResult(
                    status="search_failed",
                    verdict_hint="model_only",
                    query=query,
                    search_url=search_url,
                    items=[],
                    note=f"Evidence search failed: {exc}; Google fallback failed: {google_exc}",
                )
            verdict_hint, note = verdict_from_evidence(model_label, len(items))
            return EvidenceResult(
                status="searched",
                verdict_hint=verdict_hint,
                query=query,
                search_url=search_url,
                items=items,
                note=note,
            )
        return EvidenceResult(
            status="search_failed",
            verdict_hint="model_only",
            query=query,
            search_url=search_url,
            items=[],
            note=f"Evidence search failed: {exc}",
        )

    verdict_hint, note = verdict_from_evidence(model_label, len(items))
    return EvidenceResult(
        status="searched",
        verdict_hint=verdict_hint,
        query=query,
        search_url=search_url,
        items=items,
        note=note,
    )
