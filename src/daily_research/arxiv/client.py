"""arXiv API client — query, fetch metadata, and manage paper discovery."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import arxiv

from daily_research.graph.models import PaperNode, TaskConfig

log = logging.getLogger(__name__)


def _extract_arxiv_id(entry_id: str) -> str:
    """Extract the short arXiv ID from a full entry URL.

    'http://arxiv.org/abs/2301.12345v2' → '2301.12345'
    """
    m = re.search(r"(\d{4}\.\d{4,5})", entry_id)
    return m.group(1) if m else entry_id.rsplit("/", 1)[-1].split("v")[0]


def build_query(config: TaskConfig) -> str:
    """Build an arXiv API query string from a TaskConfig.

    Combines category filters (cat:cs.AI OR cat:cs.LG) with keyword
    filters using AND.  Adds date-range and exclude-keyword clauses
    when present.  Falls back to the task title if both categories
    and keywords are empty.
    """
    parts: list[str] = []

    if config.arxiv_categories:
        cat_expr = " OR ".join(f"cat:{c.strip()}" for c in config.arxiv_categories)
        parts.append(f"({cat_expr})")

    if config.keywords:
        kw_terms: list[str] = []
        for kw in config.keywords:
            kw = kw.strip()
            if " " in kw:
                kw_terms.append(f'"{kw}"')
            else:
                kw_terms.append(kw)
        kw_expr = " OR ".join(kw_terms)
        parts.append(f"({kw_expr})")

    if not parts:
        # Fallback: use the task title as a free-text query
        parts.append(config.title)

    query = " AND ".join(parts)

    # Exclude-keyword filter (ANDNOT each)
    if config.exclude_keywords:
        for ekw in config.exclude_keywords:
            ekw = ekw.strip()
            if " " in ekw:
                query += f' ANDNOT "{ekw}"'
            else:
                query += f" ANDNOT {ekw}"

    # Date range via arXiv submittedDate filter
    if config.date_from or config.date_to:
        d_from = config.date_from.replace("-", "") if config.date_from else "*"
        d_to = config.date_to.replace("-", "") if config.date_to else "*"
        # Append full-day timestamps
        if d_from != "*":
            d_from += "0000"
        if d_to != "*":
            d_to += "2359"
        query = f"({query}) AND submittedDate:[{d_from} TO {d_to}]"

    return query


def _result_to_paper(result: arxiv.Result) -> PaperNode:
    """Convert an arxiv.Result to a PaperNode."""
    paper_id = _extract_arxiv_id(result.entry_id)
    return PaperNode(
        paper_id=paper_id,
        title=result.title.strip().replace("\n", " "),
        authors=[a.name for a in result.authors],
        abstract=result.summary.strip().replace("\n", " "),
        year=result.published.year if result.published else None,
        categories=list(result.categories),
        pdf_url=result.pdf_url or "",
        submitted_date=(
            result.published.isoformat() if result.published else None
        ),
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )


def search_arxiv(
    query: str,
    *,
    max_results: int = 50,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
) -> list[PaperNode]:
    """Search arXiv and return a list of PaperNode objects.

    Uses the official arXiv API via the ``arxiv`` Python package.
    """
    log.info("arXiv query: %s  (max %d)", query, max_results)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=3.0,  # respect arXiv rate limits
        num_retries=3,
    )

    papers: list[PaperNode] = []
    try:
        for result in client.results(search):
            papers.append(_result_to_paper(result))
    except Exception:
        log.exception("arXiv search failed for query: %s", query)

    log.info("Fetched %d papers from arXiv", len(papers))
    return papers


def search_by_ids(arxiv_ids: list[str]) -> list[PaperNode]:
    """Fetch metadata for specific arXiv IDs."""
    if not arxiv_ids:
        return []

    log.info("Fetching %d specific arXiv papers", len(arxiv_ids))
    search = arxiv.Search(id_list=arxiv_ids)
    client = arxiv.Client(delay_seconds=3.0, num_retries=3)

    papers: list[PaperNode] = []
    try:
        for result in client.results(search):
            papers.append(_result_to_paper(result))
    except Exception:
        log.exception("arXiv ID fetch failed")

    return papers


def discover_new_papers(
    config: TaskConfig,
    *,
    known_ids: set[str] | None = None,
    max_results: int = 50,
) -> list[PaperNode]:
    """Search arXiv for papers matching the task, filtering out already-known IDs."""
    query = build_query(config)

    # Map task-level sort_by to arxiv.SortCriterion
    _SORT_MAP = {
        "submitteddate": arxiv.SortCriterion.SubmittedDate,
        "relevance": arxiv.SortCriterion.Relevance,
        "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
    }
    sort_criterion = _SORT_MAP.get(
        (config.sort_by or "submittedDate").lower(),
        arxiv.SortCriterion.SubmittedDate,
    )

    papers = search_arxiv(query, max_results=max_results, sort_by=sort_criterion)

    if known_ids:
        papers = [p for p in papers if p.paper_id not in known_ids]
        log.info("After dedup: %d new papers", len(papers))

    return papers
