"""Planner agent — decides what to fetch, expand, and analyse.

Separates planning from execution to prevent runaway tool loops.
The planner reasons about the current graph state and produces a
structured expansion plan with bounded scope.
"""

from __future__ import annotations

import json
import logging
import textwrap

from daily_research.agents.base import get_client
from daily_research.config import Settings
from daily_research.graph.models import PaperNode, TaskConfig
from daily_research.graph.store import GraphStore

log = logging.getLogger(__name__)

PLANNER_SYSTEM = textwrap.dedent("""\
    You are a research planning agent.  You receive:
    1. A research task description.
    2. A list of newly ingested papers (title + abstract).
    3. Summary statistics of the current research graph.

    Your job is to produce a JSON plan with:
    - "expand_citations": list of arXiv IDs whose references should be
      fetched and ingested (max {max_expansions}).  Prioritise papers that
      are highly cited, foundational, or central to the task.
    - "method_clusters_to_update": list of method_type strings that have
      gained enough new papers to warrant re-analysis.
    - "trend_shift_detected": boolean — true if the new papers suggest a
      meaningful shift in research direction compared to the existing graph.
    - "trend_description": short description of the shift (if detected).
    - "priority_papers": list of arXiv IDs from the new papers that are
      most important to read in detail.

    Constraints:
    - expand_citations must have at most {max_expansions} entries.
    - Be selective — only expand papers that will materially improve
      understanding of the research landscape.
    - If the graph is small (<20 papers), be more aggressive with expansion.

    Respond with a single JSON object.  No extra text.
""")


def _summarise_papers(papers: list[PaperNode], max_papers: int = 30) -> str:
    """Build a concise text summary of papers for the planner."""
    lines: list[str] = []
    for p in papers[:max_papers]:
        cats = ", ".join(p.categories[:3]) if p.categories else "?"
        lines.append(f"- [{p.paper_id}] {p.title} ({cats})")
        if p.abstract:
            # First sentence of abstract
            first_sent = p.abstract.split(". ")[0] + "."
            lines.append(f"  {first_sent[:200]}")
    return "\n".join(lines)


def _summarise_graph(graph: GraphStore) -> str:
    """Build a concise graph statistics summary."""
    stats = graph.get_stats()
    methods = graph.get_method_frequencies()
    top_methods = dict(list(methods.items())[:10])

    lines = [
        f"Papers: {stats['papers']}",
        f"Edges: {stats['edges']}",
        f"Chunks: {stats['chunks']}",
        "Top methods: " + json.dumps(top_methods),
    ]
    return "\n".join(lines)


def plan_expansions(
    new_papers: list[PaperNode],
    task_config: TaskConfig,
    graph: GraphStore,
    settings: Settings,
    *,
    max_expansions: int = 10,
) -> dict:
    """Ask the planner LLM to produce an expansion plan.

    Returns a dict with keys:
    - expand_citations: list[str] — arXiv IDs to expand
    - method_clusters_to_update: list[str]
    - trend_shift_detected: bool
    - trend_description: str
    - priority_papers: list[str]
    """
    if not new_papers:
        return {
            "expand_citations": [],
            "method_clusters_to_update": [],
            "trend_shift_detected": False,
            "trend_description": "",
            "priority_papers": [],
        }

    client = get_client(settings)
    model = settings.openai_model

    system = PLANNER_SYSTEM.format(max_expansions=max_expansions)

    user_msg = (
        f"## Research Task\n\n{task_config.raw_text}\n\n"
        f"## Newly Ingested Papers\n\n{_summarise_papers(new_papers)}\n\n"
        f"## Current Graph State\n\n{_summarise_graph(graph)}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        plan = json.loads(raw)
    except Exception:
        log.exception("Planner agent failed")
        return {
            "expand_citations": [],
            "method_clusters_to_update": [],
            "trend_shift_detected": False,
            "trend_description": "",
            "priority_papers": [],
        }

    # Enforce limits
    expand = plan.get("expand_citations", [])[:max_expansions]
    plan["expand_citations"] = expand

    log.info(
        "Planner: %d expansions, trend_shift=%s, %d priority papers",
        len(expand),
        plan.get("trend_shift_detected", False),
        len(plan.get("priority_papers", [])),
    )

    return plan


def should_expand_citation(
    paper_id: str,
    graph: GraphStore,
    *,
    max_depth: int = 2,
) -> bool:
    """Check if a paper's citations should be expanded based on graph depth."""
    # Count how many hops from a "root" paper this one is
    depth = 0
    current = paper_id
    visited: set[str] = set()

    while depth < max_depth:
        visited.add(current)
        # Check if this paper was cited by someone (i.e., it's an expansion)
        incoming = graph.get_edges(current, edge_type=None, direction="incoming")
        cite_sources = [
            e.source_id for e in incoming
            if e.source_id not in visited
        ]
        if not cite_sources:
            break  # This is a root paper
        current = cite_sources[0]
        depth += 1

    return depth < max_depth
