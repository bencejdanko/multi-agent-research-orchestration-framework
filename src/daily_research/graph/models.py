"""Data models for the research knowledge graph."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Enums ───────────────────────────────────────────────────────────


class EdgeType(str, Enum):
    """Relationship types between papers in the research graph."""

    CITES = "CITES"
    EXTENDS = "EXTENDS"
    USES_DATASET = "USES_DATASET"
    COMPARES_TO = "COMPARES_TO"
    SAME_METHOD_FAMILY = "SAME_METHOD_FAMILY"
    SIMILAR_EMBEDDING = "SIMILAR_EMBEDDING"


# ── Core graph nodes ────────────────────────────────────────────────


@dataclass
class PaperNode:
    """A paper node in the research graph."""

    paper_id: str  # arXiv ID, e.g. "2301.12345"
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    venue: str | None = None
    categories: list[str] = field(default_factory=list)
    pdf_url: str = ""
    submitted_date: str | None = None

    # Populated by structured extractor
    method_type: str | None = None
    method_name: str | None = None
    tasks: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    novelty_claim: str | None = None
    limitations: str | None = None
    key_findings: list[str] = field(default_factory=list)
    referenced_arxiv_ids: list[str] = field(default_factory=list)

    # State
    full_text_extracted: bool = False
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict suitable for SQLite storage."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": json.dumps(self.authors),
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "categories": json.dumps(self.categories),
            "pdf_url": self.pdf_url,
            "submitted_date": self.submitted_date,
            "method_type": self.method_type,
            "method_name": self.method_name,
            "tasks": json.dumps(self.tasks),
            "datasets": json.dumps(self.datasets),
            "metrics": json.dumps(self.metrics),
            "novelty_claim": self.novelty_claim,
            "limitations": self.limitations,
            "key_findings": json.dumps(self.key_findings),
            "referenced_arxiv_ids": json.dumps(self.referenced_arxiv_ids),
            "full_text_extracted": int(self.full_text_extracted),
            "ingested_at": self.ingested_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> PaperNode:
        """Reconstruct from a SQLite row dict."""
        return cls(
            paper_id=row["paper_id"],
            title=row["title"],
            authors=json.loads(row.get("authors") or "[]"),
            abstract=row.get("abstract") or "",
            year=row.get("year"),
            venue=row.get("venue"),
            categories=json.loads(row.get("categories") or "[]"),
            pdf_url=row.get("pdf_url") or "",
            submitted_date=row.get("submitted_date"),
            method_type=row.get("method_type"),
            method_name=row.get("method_name"),
            tasks=json.loads(row.get("tasks") or "[]"),
            datasets=json.loads(row.get("datasets") or "[]"),
            metrics=json.loads(row.get("metrics") or "{}"),
            novelty_claim=row.get("novelty_claim"),
            limitations=row.get("limitations"),
            key_findings=json.loads(row.get("key_findings") or "[]"),
            referenced_arxiv_ids=json.loads(
                row.get("referenced_arxiv_ids") or "[]"
            ),
            full_text_extracted=bool(row.get("full_text_extracted")),
            ingested_at=row.get("ingested_at") or "",
        )


@dataclass
class GraphEdge:
    """A directed edge between two papers."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Chunk:
    """A section-aware text chunk from a paper, ready for embedding."""

    chunk_id: str
    paper_id: str
    section: str
    content: str
    token_count: int = 0
    embedding: bytes | None = None


# ── Extraction result ───────────────────────────────────────────────


@dataclass
class ExtractionResult:
    """Structured extraction output from the extractor agent."""

    method_type: str = ""
    method_name: str = ""
    tasks: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    novelty_claim: str = ""
    limitations: str = ""
    key_findings: list[str] = field(default_factory=list)
    referenced_arxiv_ids: list[str] = field(default_factory=list)


# ── Task configuration ──────────────────────────────────────────────


@dataclass
class TaskConfig:
    """Parsed research task with optional arXiv configuration."""

    title: str
    objective: str = ""
    raw_text: str = ""
    arxiv_categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    max_papers: int = 50
    citation_depth: int = 1
    questions: list[str] = field(default_factory=list)
    date_from: str | None = None  # YYYY-MM-DD
    date_to: str | None = None  # YYYY-MM-DD
    sort_by: str = "submittedDate"  # submittedDate | relevance | lastUpdatedDate

    @property
    def has_arxiv_config(self) -> bool:
        """True if this task has arXiv-specific search configuration."""
        return bool(self.arxiv_categories or self.keywords or self.date_from)


# ── Pipeline result ─────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Output of a single arXiv pipeline run."""

    new_papers: list[PaperNode] = field(default_factory=list)
    expanded_papers: list[PaperNode] = field(default_factory=list)
    digest: str = ""
    trends: dict[str, Any] = field(default_factory=dict)
    report_path: str | None = None
