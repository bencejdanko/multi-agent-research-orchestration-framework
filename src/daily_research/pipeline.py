"""Pipeline — orchestrates the full arXiv research intelligence pipeline.

Implements the deterministic-ingestion / probabilistic-reasoning
architecture:

    Scheduler → Query → arXiv API → PDF → Parse → Chunk → Embed →
    Extract → Graph Update → Citation Expansion → Trend Analysis →
    Digest Generation
"""

from __future__ import annotations

import logging
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from daily_research.agents.base import chat, get_client
from daily_research.agents.extractor import extract_from_abstract, extract_structured
from daily_research.agents.planner import plan_expansions, should_expand_citation
from daily_research.analysis.trends import TrendReport, generate_trend_report
from daily_research.arxiv.chunker import chunk_paper
from daily_research.arxiv.client import discover_new_papers, search_by_ids
from daily_research.arxiv.pdf import (
    download_and_extract,
    find_arxiv_ids_in_text,
)
from daily_research.config import Settings
from daily_research.graph.models import (
    EdgeType,
    GraphEdge,
    PaperNode,
    PipelineResult,
    TaskConfig,
)
from daily_research.graph.store import GraphStore
from daily_research.vectorstore.store import VectorStore

log = logging.getLogger(__name__)
console = Console()


# ── Task parsing ────────────────────────────────────────────────────


def parse_task(task_text: str, fallback_title: str = "Research Task") -> TaskConfig:
    """Parse a task markdown file into a TaskConfig.

    Extracts optional arXiv configuration from an ``## arXiv`` section
    and questions from ``## Specific Questions``.
    """
    # Title from first heading
    title_match = re.search(r"^#\s+(.+)", task_text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else fallback_title

    # Objective from ## Objective section
    obj_match = re.search(
        r"##\s+Objective\s*\n(.*?)(?=\n##|\Z)", task_text, re.DOTALL
    )
    objective = obj_match.group(1).strip() if obj_match else ""

    # arXiv section
    arxiv_match = re.search(
        r"##\s+arXiv\s*\n(.*?)(?=\n##|\Z)", task_text, re.DOTALL | re.IGNORECASE
    )
    categories: list[str] = []
    keywords: list[str] = []
    exclude_keywords: list[str] = []
    max_papers = 50
    citation_depth = 1
    date_from: str | None = None
    date_to: str | None = None
    sort_by = "submittedDate"

    if arxiv_match:
        arxiv_block = arxiv_match.group(1)
        lines = arxiv_block.strip().splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip blank lines and comment-only lines
            if not line or line.startswith("#"):
                i += 1
                continue

            # Helper: strip inline YAML comments and list-item markers
            def _strip_yaml_value(v: str) -> str:
                """Remove inline ``# comments`` and surrounding whitespace,
                as well as leading list-item markers (``- ``)."""
                v = v.strip()
                if v.startswith("- "):
                    v = v[2:].strip()
                # strip inline comments (but keep '#' inside quotes)
                v = re.sub(r"\s+#\s.*$", "", v).strip()
                return v

            # Helper: collect YAML-style list items (lines starting with '- ')
            def _collect_list_items(start: int) -> tuple[list[str], int]:
                """Read consecutive ``- value`` lines from *start* onward."""
                items: list[str] = []
                j = start
                while j < len(lines):
                    raw = lines[j]
                    stripped = raw.strip()
                    if stripped.startswith("- "):
                        val = _strip_yaml_value(stripped)
                        if val:
                            items.append(val)
                        j += 1
                    elif stripped == "" or stripped.startswith("#"):
                        j += 1  # skip blanks / comments inside the list
                    else:
                        break  # next key encountered
                return items, j

            # Helper: parse a key that might have inline values or a YAML list
            def _parse_key_list(
                after_colon: str, next_idx: int
            ) -> tuple[list[str], int]:
                """Return parsed list values and the index to continue from."""
                after_colon = after_colon.strip()
                if after_colon:
                    # Inline: ``key: a, b, c``
                    after_colon = re.sub(r"\s+#\s.*$", "", after_colon)
                    vals = [v.strip() for v in after_colon.split(",") if v.strip()]
                    return vals, next_idx
                else:
                    # Multi-line list
                    return _collect_list_items(next_idx)

            # ── Dispatch on key ──────────────────────────────────────
            lower = line.lower()
            if lower.startswith("categories:"):
                after = line.split(":", 1)[1]
                categories, i = _parse_key_list(after, i + 1)
                # strip inline comments from individual category tokens
                categories = [re.sub(r"\s+#\s.*$", "", c).strip() for c in categories]
            elif lower.startswith("keywords:"):
                after = line.split(":", 1)[1]
                keywords, i = _parse_key_list(after, i + 1)
            elif lower.startswith("exclude_keywords:"):
                after = line.split(":", 1)[1]
                exclude_keywords, i = _parse_key_list(after, i + 1)
            elif lower.startswith("max_papers:"):
                try:
                    max_papers = int(
                        re.sub(r"\s+#\s.*$", "", line.split(":", 1)[1]).strip()
                    )
                except ValueError:
                    pass
                i += 1
            elif lower.startswith("citation_depth:"):
                try:
                    citation_depth = int(
                        re.sub(r"\s+#\s.*$", "", line.split(":", 1)[1]).strip()
                    )
                except ValueError:
                    pass
                i += 1
            elif lower.startswith("sort_by:"):
                sort_by = re.sub(r"\s+#\s.*$", "", line.split(":", 1)[1]).strip()
                i += 1
            elif lower.startswith("date_range:"):
                # Sub-keys: from / to on subsequent lines
                j = i + 1
                while j < len(lines):
                    sub = lines[j].strip()
                    if sub.lower().startswith("from:"):
                        date_from = re.sub(
                            r"\s+#\s.*$", "", sub.split(":", 1)[1]
                        ).strip()
                    elif sub.lower().startswith("to:"):
                        date_to = re.sub(
                            r"\s+#\s.*$", "", sub.split(":", 1)[1]
                        ).strip()
                    elif sub == "" or sub.startswith("#"):
                        j += 1
                        continue
                    else:
                        break
                    j += 1
                i = j
            else:
                i += 1
                continue

    # Questions
    q_match = re.search(
        r"##\s+Specific\s+Questions?\s*\n(.*?)(?=\n##|\Z)",
        task_text,
        re.DOTALL | re.IGNORECASE,
    )
    questions: list[str] = []
    if q_match:
        for line in q_match.group(1).strip().splitlines():
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                q = line[2:].strip()
                if q:
                    questions.append(q)

    return TaskConfig(
        title=title,
        objective=objective,
        raw_text=task_text,
        arxiv_categories=categories,
        keywords=keywords,
        exclude_keywords=exclude_keywords,
        max_papers=max_papers,
        citation_depth=citation_depth,
        questions=questions,
        date_from=date_from,
        date_to=date_to,
        sort_by=sort_by,
    )


# ── Digest generation ──────────────────────────────────────────────

_DIGEST_SYSTEM = textwrap.dedent("""\
    You are a research digest generator.  Given a set of newly discovered
    papers, structured extraction data, and trend analysis, produce a
    polished markdown research digest.

    Structure:
    # Research Digest: <topic>
    *Generated on <date>*

    ## Overview
    (2-3 paragraph summary of the day's findings)

    ## Key Papers
    (For each important paper, include title, arXiv ID, method, key
    findings, and novelty assessment.  Use ### sub-headings.)

    ## Trends & Shifts
    (Method trends, emerging topics, dataset shifts)

    ## Research Graph Update
    (Brief stats: new papers added, edges created, graph size)

    ## Novelty Highlights
    (Papers with high novelty scores, new methods, new datasets)

    ## References
    (arXiv links for all mentioned papers)

    Guidelines:
    • Be precise and cite paper IDs.
    • Use tables for comparative data.
    • Highlight genuinely novel contributions vs. incremental work.
""")

_REVIEW_DIGEST_SYSTEM = textwrap.dedent("""\
    You are a research digest generator.  No new papers were discovered
    today, so you will produce a comprehensive review digest based on
    the existing research graph — the full corpus of previously
    collected papers.

    Structure:
    # Research Review: <topic>
    *Generated on <date> — No new papers today; reviewing existing corpus*

    ## Landscape Overview
    (2-3 paragraph high-level summary of the research landscape based on
    all papers in the graph.  Cover dominant themes, major approaches,
    and the state of the field.)

    ## Key Papers
    (Highlight the most impactful or representative papers from the
    corpus.  Include title, arXiv ID, method, key findings, and why they
    matter.  Use ### sub-headings.)

    ## Method Taxonomy
    (Group and compare the methods found across the corpus.  Use a table
    if appropriate.)

    ## Dataset & Benchmark Summary
    (List datasets / benchmarks used, their frequency, and which methods
    were evaluated on them.)

    ## Trends & Gaps
    (Method trends, popular/emerging topics, under-explored areas, and
    open research questions.)

    ## Research Graph Stats
    (Current graph size: papers, edges, chunks.)

    ## References
    (arXiv links for all mentioned papers.)

    Guidelines:
    • Be precise and cite paper IDs.
    • Use tables for comparative data.
    • Focus on synthesising and connecting ideas across papers.
    • Point out gaps or opportunities for future work.
""")


def _generate_digest(
    task_config: TaskConfig,
    new_papers: list[PaperNode],
    trend_report: TrendReport,
    graph: GraphStore,
    settings: Settings,
) -> str:
    """Generate a markdown research digest using the LLM."""
    client = get_client(settings)
    model = settings.openai_model
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build paper summaries
    paper_sections: list[str] = []
    for p in new_papers[:30]:
        section = (
            f"### [{p.paper_id}] {p.title}\n"
            f"- **Authors:** {', '.join(p.authors[:5])}\n"
            f"- **Categories:** {', '.join(p.categories)}\n"
        )
        if p.method_name:
            section += f"- **Method:** {p.method_name} ({p.method_type})\n"
        if p.novelty_claim:
            section += f"- **Novelty:** {p.novelty_claim}\n"
        if p.key_findings:
            section += "- **Key Findings:**\n"
            for f in p.key_findings:
                section += f"  - {f}\n"
        if p.datasets:
            section += f"- **Datasets:** {', '.join(p.datasets)}\n"
        if p.limitations:
            section += f"- **Limitations:** {p.limitations}\n"
        paper_sections.append(section)

    # Build trend section
    trend_text = (
        f"## Trend Analysis\n\n"
        f"- Topic drift: {trend_report.drift_direction}\n"
        f"- Method frequencies: {trend_report.method_frequencies}\n"
    )
    if trend_report.emerging_methods:
        trend_text += f"- Emerging: {', '.join(trend_report.emerging_methods)}\n"
    if trend_report.new_datasets:
        trend_text += f"- New datasets: {', '.join(trend_report.new_datasets)}\n"
    if trend_report.novelty_scores:
        trend_text += "- Top novelty scores:\n"
        for ns in trend_report.novelty_scores[:5]:
            trend_text += f"  - {ns['title']}: {ns['novelty_score']}/10 — {', '.join(ns['reasons'])}\n"

    stats = graph.get_stats()
    graph_text = (
        f"## Graph Stats\n"
        f"- Papers: {stats['papers']}\n"
        f"- Edges: {stats['edges']}\n"
        f"- Chunks: {stats['chunks']}\n"
    )

    user_msg = (
        f"Topic: {task_config.title}\n"
        f"Date: {today}\n"
        f"New papers: {len(new_papers)}\n\n"
        f"## Paper Details\n\n{''.join(paper_sections)}\n\n"
        f"{trend_text}\n\n"
        f"{graph_text}"
    )

    return chat(
        client,
        model,
        system=_DIGEST_SYSTEM,
        user=user_msg,
        temperature=0.3,
        max_tokens=4096,
    )


def _generate_review_digest(
    task_config: TaskConfig,
    existing_papers: list[PaperNode],
    trend_report: TrendReport,
    graph: GraphStore,
    settings: Settings,
) -> str:
    """Generate a review digest from existing graph data (no new papers)."""
    client = get_client(settings)
    model = settings.openai_model
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build paper summaries (up to 30 most relevant)
    paper_sections: list[str] = []
    for p in existing_papers[:30]:
        section = (
            f"### [{p.paper_id}] {p.title}\n"
            f"- **Authors:** {', '.join(p.authors[:5])}\n"
            f"- **Categories:** {', '.join(p.categories)}\n"
        )
        if p.method_name:
            section += f"- **Method:** {p.method_name} ({p.method_type})\n"
        if p.novelty_claim:
            section += f"- **Novelty:** {p.novelty_claim}\n"
        if p.key_findings:
            section += "- **Key Findings:**\n"
            for f in p.key_findings:
                section += f"  - {f}\n"
        if p.datasets:
            section += f"- **Datasets:** {', '.join(p.datasets)}\n"
        if p.limitations:
            section += f"- **Limitations:** {p.limitations}\n"
        paper_sections.append(section)

    # Build trend section
    trend_text = (
        f"## Trend Analysis\n\n"
        f"- Topic drift: {trend_report.drift_direction}\n"
        f"- Method frequencies: {trend_report.method_frequencies}\n"
    )
    if trend_report.emerging_methods:
        trend_text += f"- Emerging: {', '.join(trend_report.emerging_methods)}\n"
    if trend_report.declining_methods:
        trend_text += f"- Declining: {', '.join(trend_report.declining_methods)}\n"
    if trend_report.dataset_frequencies:
        trend_text += f"- Dataset frequencies: {trend_report.dataset_frequencies}\n"
    if trend_report.accelerating_papers:
        trend_text += "- Most cited papers:\n"
        for ap in trend_report.accelerating_papers[:5]:
            trend_text += f"  - {ap['title']} ({ap['citation_count']} citations)\n"

    stats = graph.get_stats()
    graph_text = (
        f"## Graph Stats\n"
        f"- Papers: {stats['papers']}\n"
        f"- Edges: {stats['edges']}\n"
        f"- Chunks: {stats['chunks']}\n"
    )

    user_msg = (
        f"Topic: {task_config.title}\n"
        f"Objective: {task_config.objective}\n"
        f"Date: {today}\n"
        f"No new papers found today. Generate a review of the existing "
        f"research corpus ({len(existing_papers)} papers).\n\n"
        f"## Existing Papers in Graph\n\n{''.join(paper_sections)}\n\n"
        f"{trend_text}\n\n"
        f"{graph_text}"
    )

    return chat(
        client,
        model,
        system=_REVIEW_DIGEST_SYSTEM,
        user=user_msg,
        temperature=0.3,
        max_tokens=4096,
    )


# ── Main pipeline ──────────────────────────────────────────────────


def run_arxiv_pipeline(
    task_text: str,
    settings: Settings,
    *,
    skip_pdf: bool = False,
    skip_expansion: bool = False,
) -> PipelineResult:
    """Run the full arXiv research intelligence pipeline.

    Stages:
    1. Parse task configuration
    2. Discover new papers from arXiv
    3. Ingest metadata into graph
    4. Download PDFs + extract text + chunk (optional)
    5. Structured LLM extraction
    6. Generate embeddings
    7. Build graph edges (citations, similarity)
    8. Planner-controlled citation expansion
    9. Trend & novelty analysis
    10. Generate research digest
    """
    task_config = parse_task(task_text)
    graph = GraphStore(settings.data_dir / "research.db")
    vectors = VectorStore(graph, settings)
    pdf_cache = settings.data_dir / "pdf_cache"
    pdf_cache.mkdir(parents=True, exist_ok=True)

    result = PipelineResult()

    # ── Stage 1: Collect known IDs ──────────────────────────────────
    known_ids = {p.paper_id for p in graph.list_papers(limit=100_000)}
    console.print(f"  [dim]Graph has {len(known_ids)} known papers[/]")

    # ── Stage 2: Discover new papers ────────────────────────────────
    console.print("  [cyan]Searching arXiv…[/]")
    new_papers = discover_new_papers(
        task_config,
        known_ids=known_ids,
        max_results=task_config.max_papers,
    )
    console.print(f"  [green]✓[/] Found {len(new_papers)} new papers")

    if not new_papers:
        # Generate a review digest from existing graph data
        console.print("  [cyan]No new papers — generating review from existing graph…[/]")
        existing_papers = graph.list_papers(limit=200)
        trend_report = generate_trend_report(graph, vectors)
        result.trends = trend_report.to_dict()

        if existing_papers:
            try:
                digest = _generate_review_digest(
                    task_config, existing_papers, trend_report, graph, settings
                )
                result.digest = digest
            except Exception:
                log.warning("Review digest generation failed", exc_info=True)
                result.digest = _fallback_review_digest(
                    task_config, existing_papers, trend_report, graph
                )
        else:
            result.digest = (
                f"No new papers found for: {task_config.title}\n\n"
                f"The research graph is empty — run again when new papers "
                f"are available.\n\n{trend_report.trend_summary}"
            )

        # Update checkpoint
        graph.set_state(
            f"last_run:{task_config.title}",
            datetime.now(timezone.utc).isoformat(),
        )
        graph.close()
        return result

    # ── Stage 3: Ingest metadata ────────────────────────────────────
    for paper in new_papers:
        graph.upsert_paper(paper)
    result.new_papers = new_papers
    console.print(f"  [green]✓[/] Ingested {len(new_papers)} paper nodes")

    # ── Stage 4: PDF + text extraction ──────────────────────────────
    if not skip_pdf:
        console.print("  [cyan]Downloading and parsing PDFs…[/]")
        extracted_count = 0
        for paper in new_papers:
            if not paper.pdf_url:
                continue
            try:
                full_text, sections = download_and_extract(
                    paper.pdf_url, cache_dir=pdf_cache
                )
                if full_text:
                    # Chunk the paper
                    chunks = chunk_paper(sections, paper.paper_id)
                    for chunk in chunks:
                        graph.save_chunk(chunk)

                    # Find arXiv IDs in references
                    ref_ids = find_arxiv_ids_in_text(full_text)
                    paper.referenced_arxiv_ids = ref_ids

                    paper.full_text_extracted = True
                    graph.upsert_paper(paper)
                    extracted_count += 1
            except Exception:
                log.warning("PDF extraction failed for %s", paper.paper_id, exc_info=True)

        console.print(f"  [green]✓[/] Extracted text from {extracted_count} PDFs")

    # ── Stage 5: Structured extraction ──────────────────────────────
    console.print("  [cyan]Running structured extraction…[/]")
    for paper in new_papers:
        try:
            if paper.full_text_extracted:
                # Use full text from chunks
                chunks = graph.get_chunks(paper.paper_id)
                full_text = "\n\n".join(c.content for c in chunks)
                extraction = extract_structured(full_text, settings)
            else:
                # Fall back to abstract-only extraction
                extraction = extract_from_abstract(
                    paper.abstract, paper.title, settings
                )

            # Update paper with extraction results
            paper.method_type = extraction.method_type or paper.method_type
            paper.method_name = extraction.method_name or paper.method_name
            paper.tasks = extraction.tasks or paper.tasks
            paper.datasets = extraction.datasets or paper.datasets
            paper.metrics = extraction.metrics or paper.metrics
            paper.novelty_claim = extraction.novelty_claim or paper.novelty_claim
            paper.limitations = extraction.limitations or paper.limitations
            paper.key_findings = extraction.key_findings or paper.key_findings

            # Merge referenced arXiv IDs
            all_refs = set(paper.referenced_arxiv_ids) | set(extraction.referenced_arxiv_ids)
            paper.referenced_arxiv_ids = sorted(all_refs)

            graph.upsert_paper(paper)
        except Exception:
            log.warning("Extraction failed for %s", paper.paper_id, exc_info=True)

    console.print(f"  [green]✓[/] Structured extraction complete")

    # ── Stage 6: Generate embeddings ────────────────────────────────
    console.print("  [cyan]Generating embeddings…[/]")
    try:
        embedded = vectors.embed_new_chunks()
        console.print(f"  [green]✓[/] Embedded {embedded} chunks")

        # Also embed paper abstracts
        for paper in new_papers:
            if paper.abstract:
                vectors.embed_paper_abstract(paper.paper_id, paper.abstract)
    except Exception:
        log.warning("Embedding generation failed", exc_info=True)
        console.print("  [yellow]⚠[/] Embedding generation failed (continuing)")

    # ── Stage 7: Build edges ────────────────────────────────────────
    console.print("  [cyan]Building graph edges…[/]")
    edge_count = 0
    for paper in new_papers:
        # Citation edges
        for ref_id in paper.referenced_arxiv_ids:
            if graph.paper_exists(ref_id):
                graph.add_edge(
                    GraphEdge(
                        source_id=paper.paper_id,
                        target_id=ref_id,
                        edge_type=EdgeType.CITES,
                    )
                )
                edge_count += 1

        # Same-method-family edges
        if paper.method_type:
            same_method = graph.get_papers_by_method(paper.method_type)
            for other in same_method:
                if other.paper_id != paper.paper_id:
                    graph.add_edge(
                        GraphEdge(
                            source_id=paper.paper_id,
                            target_id=other.paper_id,
                            edge_type=EdgeType.SAME_METHOD_FAMILY,
                        )
                    )
                    edge_count += 1

        # Similarity edges (top-3 most similar papers)
        try:
            similar = vectors.find_similar_papers(paper.paper_id, top_k=3)
            for other_id, sim_score in similar:
                if sim_score > 0.85:  # High similarity threshold
                    graph.add_edge(
                        GraphEdge(
                            source_id=paper.paper_id,
                            target_id=other_id,
                            edge_type=EdgeType.SIMILAR_EMBEDDING,
                            metadata={"similarity": round(sim_score, 4)},
                        )
                    )
                    edge_count += 1
        except Exception:
            log.debug("Similarity edge failed for %s", paper.paper_id, exc_info=True)

    console.print(f"  [green]✓[/] Created {edge_count} edges")

    # ── Stage 8: Citation expansion (planner-controlled) ────────────
    expanded_papers: list[PaperNode] = []
    if not skip_expansion and task_config.citation_depth > 0:
        console.print("  [cyan]Running planner for citation expansion…[/]")
        try:
            plan = plan_expansions(
                new_papers,
                task_config,
                graph,
                settings,
                max_expansions=settings.max_citation_expansions,
            )

            expand_ids = [
                eid for eid in plan.get("expand_citations", [])
                if not graph.paper_exists(eid)
                and should_expand_citation(eid, graph, max_depth=task_config.citation_depth)
            ]

            if expand_ids:
                console.print(f"  [cyan]Expanding {len(expand_ids)} citations…[/]")
                fetched = search_by_ids(expand_ids)
                for paper in fetched:
                    graph.upsert_paper(paper)
                    expanded_papers.append(paper)

                    # Quick abstract-only extraction for expanded papers
                    try:
                        extraction = extract_from_abstract(
                            paper.abstract, paper.title, settings
                        )
                        paper.method_type = extraction.method_type
                        paper.method_name = extraction.method_name
                        paper.tasks = extraction.tasks
                        paper.datasets = extraction.datasets
                        paper.novelty_claim = extraction.novelty_claim
                        paper.key_findings = extraction.key_findings
                        graph.upsert_paper(paper)
                    except Exception:
                        log.debug("Extraction for expanded paper %s failed", paper.paper_id)

                console.print(f"  [green]✓[/] Expanded {len(expanded_papers)} citation papers")

            if plan.get("trend_shift_detected"):
                console.print(
                    f"  [yellow]⚡[/] Trend shift: {plan.get('trend_description', '?')}"
                )

        except Exception:
            log.warning("Citation expansion failed", exc_info=True)
            console.print("  [yellow]⚠[/] Citation expansion failed (continuing)")

    result.expanded_papers = expanded_papers

    # ── Stage 9: Trend analysis ─────────────────────────────────────
    console.print("  [cyan]Analysing trends…[/]")
    try:
        trend_report = generate_trend_report(
            graph, vectors, new_papers=new_papers
        )
        result.trends = trend_report.to_dict()
        console.print(f"  [green]✓[/] {trend_report.trend_summary}")
    except Exception:
        log.warning("Trend analysis failed", exc_info=True)
        trend_report = TrendReport()
        result.trends = trend_report.to_dict()

    # ── Stage 10: Generate digest ───────────────────────────────────
    console.print("  [cyan]Generating research digest…[/]")
    try:
        digest = _generate_digest(
            task_config, new_papers, trend_report, graph, settings
        )
        result.digest = digest
    except Exception:
        log.warning("Digest generation failed", exc_info=True)
        result.digest = _fallback_digest(task_config, new_papers, trend_report, graph)

    # Update checkpoint
    graph.set_state(
        f"last_run:{task_config.title}",
        datetime.now(timezone.utc).isoformat(),
    )

    graph.close()
    return result


def _fallback_digest(
    task_config: TaskConfig,
    new_papers: list[PaperNode],
    trend_report: TrendReport,
    graph: GraphStore,
) -> str:
    """Generate a basic digest without LLM (fallback)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stats = graph.get_stats()

    lines = [
        f"# Research Digest: {task_config.title}",
        f"*Generated on {today}*\n",
        f"## Overview",
        f"Found {len(new_papers)} new papers.\n",
        f"## Papers",
    ]
    for p in new_papers[:20]:
        lines.append(f"### [{p.paper_id}] {p.title}")
        lines.append(f"- Authors: {', '.join(p.authors[:5])}")
        lines.append(f"- Abstract: {p.abstract[:300]}…\n")

    lines.append(f"\n## Graph Stats")
    lines.append(f"- Papers: {stats['papers']}, Edges: {stats['edges']}, Chunks: {stats['chunks']}")
    lines.append(f"\n## Trend Summary")
    lines.append(trend_report.trend_summary)

    return "\n".join(lines)


def _fallback_review_digest(
    task_config: TaskConfig,
    existing_papers: list[PaperNode],
    trend_report: TrendReport,
    graph: GraphStore,
) -> str:
    """Generate a basic review digest without LLM (fallback)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stats = graph.get_stats()

    lines = [
        f"# Research Review: {task_config.title}",
        f"*Generated on {today} — No new papers today; reviewing existing corpus*\n",
        f"## Overview",
        f"Review of {len(existing_papers)} papers in the research graph.\n",
        f"## Papers",
    ]
    for p in existing_papers[:20]:
        lines.append(f"### [{p.paper_id}] {p.title}")
        lines.append(f"- Authors: {', '.join(p.authors[:5])}")
        if p.method_name:
            lines.append(f"- Method: {p.method_name} ({p.method_type})")
        if p.key_findings:
            lines.append("- Key Findings:")
            for finding in p.key_findings:
                lines.append(f"  - {finding}")
        lines.append(f"- Abstract: {p.abstract[:300]}…\n")

    if trend_report.method_frequencies:
        lines.append("\n## Method Frequencies")
        for method, count in list(trend_report.method_frequencies.items())[:10]:
            lines.append(f"- {method}: {count}")

    lines.append(f"\n## Graph Stats")
    lines.append(f"- Papers: {stats['papers']}, Edges: {stats['edges']}, Chunks: {stats['chunks']}")
    lines.append(f"\n## Trend Summary")
    lines.append(trend_report.trend_summary)

    return "\n".join(lines)