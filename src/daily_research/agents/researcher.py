"""Researcher agent — gathers information from the web on a given topic."""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field

from daily_research.agents.base import chat, get_client
from daily_research.config import Settings

log = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Container for raw research output."""

    query: str
    search_results: list[dict] = field(default_factory=list)
    synthesis: str = ""


# ── internal helpers ────────────────────────────────────────────────


def _tavily_search(query: str, api_key: str, max_results: int = 10) -> list[dict]:
    """Run a Tavily search and return simplified results."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    resp = client.search(query, max_results=max_results)
    return [
        {"title": r.get("title", ""), "href": r["url"], "body": r.get("content", "")}
        for r in resp.get("results", [])
    ]


def _ddgs_search(query: str, max_results: int = 10) -> list[dict]:
    """Run a DuckDuckGo search via the ddgs package and return simplified results."""
    from ddgs import DDGS

    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=max_results))
    return [{"title": r["title"], "href": r["href"], "body": r["body"]} for r in raw]


def _web_search(query: str, settings: Settings, max_results: int = 10) -> list[dict]:
    """Run a web search using the best available backend.

    Prefers Tavily when TAVILY_API_KEY is set; falls back to DuckDuckGo.
    """
    if settings.tavily_api_key:
        return _tavily_search(query, settings.tavily_api_key, max_results)
    return _ddgs_search(query, max_results)


def _format_search_context(results: list[dict]) -> str:
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['title']}\n    {r['href']}\n    {r['body']}")
    return "\n\n".join(parts)


# ── public API ──────────────────────────────────────────────────────


RESEARCHER_SYSTEM = textwrap.dedent("""\
    You are a meticulous research analyst.  You will receive:
      1. A **research task description** written by the user.
      2. **Web search results** gathered for that task.

    Your job:
      • Synthesise the search results into a coherent body of knowledge.
      • Identify the most important facts, data, and insights.
      • Note any conflicting information or gaps.
      • Cite sources by number [1], [2], … when referencing search results.
      • Output well-structured prose with markdown headings.
      • Do NOT fabricate information — only use what is in the search results.
""")


def run_research(task_description: str, settings: Settings) -> ResearchResult:
    """Execute the full research pipeline for a single task.

    1. Ask the LLM to generate targeted search queries from the task.
    2. Run web searches.
    3. Synthesise findings with the LLM.
    """
    client = get_client(settings)
    model = settings.openai_model

    # Step 1: generate search queries
    queries_raw = chat(
        client,
        model,
        system=(
            "You are a search-query generator.  Given a research task, output 3-5 "
            "diverse, specific search queries (one per line, no numbering) that would "
            "best help answer the task."
        ),
        user=task_description,
        temperature=0.3,
        max_tokens=300,
    )
    queries = [q.strip() for q in queries_raw.strip().splitlines() if q.strip()]

    # Step 2: aggregate web search results
    all_results: list[dict] = []
    seen_hrefs: set[str] = set()
    for q in queries:
        try:
            for r in _web_search(q, settings, max_results=6):
                if r["href"] not in seen_hrefs:
                    seen_hrefs.add(r["href"])
                    all_results.append(r)
        except Exception:
            log.warning("Search failed for query %r", q, exc_info=True)
            continue  # search failures are non-fatal

    # Step 3: synthesise
    context = _format_search_context(all_results)
    synthesis = chat(
        client,
        model,
        system=RESEARCHER_SYSTEM,
        user=f"## Research Task\n\n{task_description}\n\n## Search Results\n\n{context}",
        temperature=0.3,
        max_tokens=4096,
    )

    return ResearchResult(
        query=task_description,
        search_results=all_results,
        synthesis=synthesis,
    )
