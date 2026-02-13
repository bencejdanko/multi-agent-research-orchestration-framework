"""Trend detection, novelty scoring, and research landscape analysis.

Computes:
A. Embedding drift — track topic centroid shift over time
B. Method frequency — count method types per time window
C. Citation acceleration — rapid citation graph expansion
D. Novelty scoring — new methods, datasets, or SOTA improvements
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from daily_research.graph.models import EdgeType, PaperNode
from daily_research.graph.store import GraphStore
from daily_research.vectorstore.store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class TrendReport:
    """Aggregated trend and novelty analysis output."""

    # Embedding drift
    centroid_drift: float = 0.0
    drift_direction: str = ""

    # Method trends
    method_frequencies: dict[str, int] = field(default_factory=dict)
    emerging_methods: list[str] = field(default_factory=list)
    declining_methods: list[str] = field(default_factory=list)

    # Dataset trends
    dataset_frequencies: dict[str, int] = field(default_factory=dict)
    new_datasets: list[str] = field(default_factory=list)

    # Citation acceleration
    accelerating_papers: list[dict] = field(default_factory=list)

    # Novelty scoreboard
    novelty_scores: list[dict] = field(default_factory=list)

    # Summary
    trend_summary: str = ""
    trend_shift_detected: bool = False

    def to_dict(self) -> dict:
        return {
            "centroid_drift": self.centroid_drift,
            "drift_direction": self.drift_direction,
            "method_frequencies": self.method_frequencies,
            "emerging_methods": self.emerging_methods,
            "declining_methods": self.declining_methods,
            "dataset_frequencies": self.dataset_frequencies,
            "new_datasets": self.new_datasets,
            "accelerating_papers": self.accelerating_papers,
            "novelty_scores": self.novelty_scores,
            "trend_summary": self.trend_summary,
            "trend_shift_detected": self.trend_shift_detected,
        }


# ── A. Embedding Drift ─────────────────────────────────────────────


def compute_embedding_drift(
    vectors: VectorStore,
    graph: GraphStore,
    *,
    recent_days: int = 7,
) -> tuple[float, str]:
    """Compare the centroid of recent papers vs. older papers.

    Returns (drift_magnitude, drift_description).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=recent_days)).isoformat()

    # Get all papers with embeddings
    all_papers = graph.list_papers(limit=10000)
    recent_ids = [p.paper_id for p in all_papers if (p.ingested_at or "") >= cutoff]
    older_ids = [p.paper_id for p in all_papers if (p.ingested_at or "") < cutoff]

    if not recent_ids or not older_ids:
        return 0.0, "Not enough data for drift analysis"

    recent_centroid = vectors.compute_topic_centroid(recent_ids)
    older_centroid = vectors.compute_topic_centroid(older_ids)

    if recent_centroid is None or older_centroid is None:
        return 0.0, "Embeddings not available"

    # Cosine distance
    cos_sim = float(
        np.dot(recent_centroid, older_centroid)
        / (np.linalg.norm(recent_centroid) * np.linalg.norm(older_centroid) + 1e-9)
    )
    drift = 1.0 - cos_sim

    if drift > 0.15:
        desc = f"Significant topic drift detected (cosine distance: {drift:.3f})"
    elif drift > 0.05:
        desc = f"Moderate topic drift (cosine distance: {drift:.3f})"
    else:
        desc = f"Stable topic focus (cosine distance: {drift:.3f})"

    return drift, desc


# ── B. Method Frequency Analysis ───────────────────────────────────


def analyze_method_trends(
    graph: GraphStore,
    *,
    recent_days: int = 7,
) -> tuple[dict[str, int], list[str], list[str]]:
    """Compare method frequencies in recent vs. older papers.

    Returns (current_frequencies, emerging_methods, declining_methods).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=recent_days)).isoformat()
    all_papers = graph.list_papers(limit=10000)

    recent = [p for p in all_papers if (p.ingested_at or "") >= cutoff]
    older = [p for p in all_papers if (p.ingested_at or "") < cutoff]

    recent_methods = Counter(
        p.method_type for p in recent if p.method_type
    )
    older_methods = Counter(
        p.method_type for p in older if p.method_type
    )

    # Methods that appear in recent but not in older
    emerging = [m for m in recent_methods if m not in older_methods]

    # Methods that were common but have declined
    declining: list[str] = []
    for method, old_count in older_methods.items():
        recent_count = recent_methods.get(method, 0)
        if old_count >= 3 and recent_count == 0:
            declining.append(method)

    current_freq = dict(graph.get_method_frequencies())

    return current_freq, emerging, declining


# ── C. Citation Acceleration ────────────────────────────────────────


def find_accelerating_papers(
    graph: GraphStore,
    *,
    min_citations: int = 3,
) -> list[dict]:
    """Identify papers with rapidly growing citation counts in the graph.

    Looks for papers that have accumulated more than *min_citations*
    incoming CITES edges.
    """
    papers = graph.list_papers(limit=10000)
    results: list[dict] = []

    for paper in papers:
        incoming = graph.get_edges(paper.paper_id, edge_type=EdgeType.CITES, direction="incoming")
        if len(incoming) >= min_citations:
            results.append(
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "citation_count": len(incoming),
                    "submitted_date": paper.submitted_date,
                }
            )

    results.sort(key=lambda x: x["citation_count"], reverse=True)
    return results[:20]


# ── D. Novelty Scoring ─────────────────────────────────────────────


def score_novelty(
    paper: PaperNode,
    graph: GraphStore,
) -> dict:
    """Compute a novelty score for a paper based on graph context.

    Scoring heuristic:
    - +3 if method_name not seen in any prior paper
    - +2 for each previously unseen dataset
    - +2 if novelty_claim contains unique terms
    - +1 if paper has EXTENDS edge (builds on prior work — incremental)
    - Score capped at 10
    """
    score = 0
    reasons: list[str] = []

    # Check if method is novel
    if paper.method_name:
        existing = graph.get_papers_by_method(paper.method_type or "")
        existing_names = {
            p.method_name for p in existing
            if p.paper_id != paper.paper_id and p.method_name
        }
        if paper.method_name not in existing_names:
            score += 3
            reasons.append(f"Novel method: {paper.method_name}")

    # Check for new datasets
    known_datasets = set(graph.get_dataset_frequencies().keys())
    for ds in paper.datasets:
        if ds not in known_datasets:
            score += 2
            reasons.append(f"New dataset: {ds}")

    # Novelty claim present
    if paper.novelty_claim:
        score += 1
        reasons.append("Has explicit novelty claim")

    # Extends relationship (incremental innovation)
    extends_edges = graph.get_edges(
        paper.paper_id, edge_type=EdgeType.EXTENDS, direction="outgoing"
    )
    if extends_edges:
        score += 1
        reasons.append("Extends prior work")

    score = min(score, 10)

    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "novelty_score": score,
        "reasons": reasons,
    }


# ── Full Trend Report ──────────────────────────────────────────────


def generate_trend_report(
    graph: GraphStore,
    vectors: VectorStore,
    *,
    new_papers: list[PaperNode] | None = None,
    recent_days: int = 7,
) -> TrendReport:
    """Run all trend and novelty analyses and return a TrendReport."""
    report = TrendReport()

    # A. Embedding drift
    try:
        report.centroid_drift, report.drift_direction = compute_embedding_drift(
            vectors, graph, recent_days=recent_days
        )
    except Exception:
        log.exception("Embedding drift analysis failed")

    # B. Method trends
    try:
        freq, emerging, declining = analyze_method_trends(
            graph, recent_days=recent_days
        )
        report.method_frequencies = freq
        report.emerging_methods = emerging
        report.declining_methods = declining
    except Exception:
        log.exception("Method trend analysis failed")

    # C. Dataset frequencies
    try:
        report.dataset_frequencies = graph.get_dataset_frequencies()
        if new_papers:
            all_known = set(report.dataset_frequencies.keys())
            new_ds: list[str] = []
            for p in new_papers:
                for ds in p.datasets:
                    if ds not in all_known:
                        new_ds.append(ds)
            report.new_datasets = list(set(new_ds))
    except Exception:
        log.exception("Dataset analysis failed")

    # D. Citation acceleration
    try:
        report.accelerating_papers = find_accelerating_papers(graph)
    except Exception:
        log.exception("Citation acceleration analysis failed")

    # E. Novelty scores for new papers
    if new_papers:
        try:
            report.novelty_scores = [
                score_novelty(p, graph) for p in new_papers
            ]
            report.novelty_scores.sort(
                key=lambda x: x["novelty_score"], reverse=True
            )
        except Exception:
            log.exception("Novelty scoring failed")

    # Overall assessment
    report.trend_shift_detected = (
        report.centroid_drift > 0.1
        or len(report.emerging_methods) >= 2
        or any(s["novelty_score"] >= 7 for s in report.novelty_scores)
    )

    # Build summary text
    parts: list[str] = []
    if report.drift_direction:
        parts.append(report.drift_direction)
    if report.emerging_methods:
        parts.append(f"Emerging methods: {', '.join(report.emerging_methods)}")
    if report.new_datasets:
        parts.append(f"New datasets: {', '.join(report.new_datasets[:5])}")
    if report.novelty_scores:
        top = report.novelty_scores[0]
        parts.append(
            f"Highest novelty: {top['title']} (score {top['novelty_score']}/10)"
        )
    stats = graph.get_stats()
    parts.append(
        f"Graph: {stats['papers']} papers, {stats['edges']} edges, "
        f"{stats['chunks']} chunks"
    )

    report.trend_summary = " | ".join(parts) if parts else "No trends detected."

    return report
