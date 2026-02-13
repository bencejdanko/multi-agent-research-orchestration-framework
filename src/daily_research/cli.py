"""CLI — manage research tasks, graph, and trigger runs."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from daily_research.config import get_settings
from daily_research.runner import discover_tasks, run_all_tasks, run_single_task

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


# ── Root group ──────────────────────────────────────────────────────


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """daily-research — automated agentic research system."""
    _setup_logging(verbose)


# ── tasks ───────────────────────────────────────────────────────────


@cli.group()
def tasks() -> None:
    """Manage research task files."""


@tasks.command("list")
def tasks_list() -> None:
    """List all pending research tasks."""
    settings = get_settings()
    found = discover_tasks(settings)
    if not found:
        console.print("[yellow]No tasks found.[/] Add .md files to", settings.tasks_dir)
        return
    table = Table(title="Research Tasks")
    table.add_column("#", style="dim")
    table.add_column("File")
    table.add_column("Size", justify="right")
    for i, p in enumerate(found, 1):
        table.add_row(str(i), p.name, f"{p.stat().st_size:,} B")
    console.print(table)


@tasks.command("add")
@click.argument("name")
@click.option("--from-file", type=click.Path(exists=True), help="Copy an existing .md file as a task.")
@click.option("--edit", "-e", is_flag=True, help="Open $EDITOR after creating the task stub.")
def tasks_add(name: str, from_file: str | None, edit: bool) -> None:
    """Create a new research task called NAME (.md extension added automatically)."""
    settings = get_settings()
    settings.ensure_dirs()
    slug = name.replace(" ", "-").lower()
    dest = settings.tasks_dir / f"{slug}.md"

    if dest.exists():
        console.print(f"[red]Task already exists:[/] {dest}")
        raise SystemExit(1)

    if from_file:
        shutil.copy2(from_file, dest)
        console.print(f"[green]Copied[/] {from_file} → {dest}")
    else:
        dest.write_text(
            f"# {name}\n\n"
            "<!-- Describe the research task in detail below. -->\n\n"
            "## Objective\n\n\n"
            "## Specific Questions\n\n- \n",
            encoding="utf-8",
        )
        console.print(f"[green]Created[/] {dest}")

    if edit:
        click.edit(filename=str(dest))


@tasks.command("remove")
@click.argument("name")
def tasks_remove(name: str) -> None:
    """Remove a research task by name (without .md extension)."""
    settings = get_settings()
    slug = name.replace(" ", "-").lower()
    dest = settings.tasks_dir / f"{slug}.md"
    if not dest.exists():
        console.print(f"[red]Not found:[/] {dest}")
        raise SystemExit(1)
    dest.unlink()
    console.print(f"[green]Removed[/] {dest.name}")


@tasks.command("show")
@click.argument("name")
def tasks_show(name: str) -> None:
    """Print the contents of a task file."""
    settings = get_settings()
    slug = name.replace(" ", "-").lower()
    dest = settings.tasks_dir / f"{slug}.md"
    if not dest.exists():
        console.print(f"[red]Not found:[/] {dest}")
        raise SystemExit(1)
    console.print(dest.read_text(encoding="utf-8"))


# ── run ─────────────────────────────────────────────────────────────


@cli.command()
@click.argument("name", required=False)
@click.option("--no-discord", is_flag=True, help="Skip sending to Discord.")
@click.option("--web-only", is_flag=True, help="Force web-search mode (skip arXiv pipeline).")
def run(name: str | None, no_discord: bool, web_only: bool) -> None:
    """Run research for a single task (by NAME) or all tasks if omitted."""
    settings = get_settings()

    if name:
        slug = name.replace(" ", "-").lower()
        task_path = settings.tasks_dir / f"{slug}.md"
        if not task_path.exists():
            console.print(f"[red]Task not found:[/] {task_path}")
            raise SystemExit(1)
        run_single_task(task_path, settings, skip_discord=no_discord, web_only=web_only)
    else:
        run_all_tasks(settings, skip_discord=no_discord, web_only=web_only)


# ── reports ─────────────────────────────────────────────────────────


@cli.group()
def reports() -> None:
    """Manage generated reports."""


@reports.command("list")
def reports_list() -> None:
    """List all generated reports."""
    settings = get_settings()
    settings.ensure_dirs()
    found = sorted(settings.reports_dir.glob("*.md"), reverse=True)
    if not found:
        console.print("[yellow]No reports yet.[/]")
        return
    table = Table(title="Reports")
    table.add_column("#", style="dim")
    table.add_column("File")
    table.add_column("Size", justify="right")
    for i, p in enumerate(found, 1):
        table.add_row(str(i), p.name, f"{p.stat().st_size:,} B")
    console.print(table)


@reports.command("show")
@click.argument("filename")
def reports_show(filename: str) -> None:
    """Print a report by filename."""
    settings = get_settings()
    path = settings.reports_dir / filename
    if not path.exists():
        console.print(f"[red]Not found:[/] {path}")
        raise SystemExit(1)
    console.print(path.read_text(encoding="utf-8"))


@reports.command("clean")
@click.confirmation_option(prompt="Delete ALL reports?")
def reports_clean() -> None:
    """Delete all generated reports."""
    settings = get_settings()
    settings.ensure_dirs()
    count = 0
    for p in settings.reports_dir.glob("*.md"):
        p.unlink()
        count += 1
    console.print(f"[green]Deleted {count} report(s).[/]")


# ── config ──────────────────────────────────────────────────────────


@cli.command("config")
def show_config() -> None:
    """Show current configuration (masks secrets)."""
    settings = get_settings()
    table = Table(title="Configuration")
    table.add_column("Key")
    table.add_column("Value")

    def _mask(s: str) -> str:
        if not s:
            return "[dim]<not set>[/]"
        return s[:8] + "…" if len(s) > 12 else s

    table.add_row("OPENAI_MODEL", settings.openai_model)
    table.add_row("OPENAI_API_KEY", _mask(settings.openai_api_key))
    table.add_row("OPENAI_BASE_URL", str(settings.openai_base_url or "[dim]default[/]"))
    table.add_row("EMBEDDING_MODEL", settings.embedding_model)
    table.add_row("DISCORD_WEBHOOK_URL", _mask(settings.discord_webhook_url))
    table.add_row("TASKS_DIR", str(settings.tasks_dir))
    table.add_row("REPORTS_DIR", str(settings.reports_dir))
    table.add_row("DATA_DIR", str(settings.data_dir))
    table.add_row("MAX_PAPERS_PER_RUN", str(settings.max_papers_per_run))
    table.add_row("CITATION_DEPTH", str(settings.citation_depth))
    table.add_row("MAX_CITATION_EXPANSIONS", str(settings.max_citation_expansions))
    console.print(table)


# ── cron helper ─────────────────────────────────────────────────────


@cli.command("cron-install")
@click.option(
    "--schedule",
    default="0 8 * * *",
    show_default=True,
    help="Cron schedule expression.",
)
def cron_install(schedule: str) -> None:
    """Print a crontab line you can install to automate runs."""
    import sys

    python = sys.executable
    console.print("\nAdd this line to your crontab ([bold]crontab -e[/]):\n")
    console.print(
        f"  {schedule}  cd {Path.cwd()} && {python} -m daily_research.cli run 2>&1 "
        f">> {Path.cwd() / 'cron.log'}"
    )
    console.print()


# ── graph ───────────────────────────────────────────────────────────


@cli.group()
def graph() -> None:
    """Inspect and manage the research knowledge graph."""


@graph.command("stats")
def graph_stats() -> None:
    """Show research graph statistics."""
    from daily_research.graph.store import GraphStore

    settings = get_settings()
    settings.ensure_dirs()
    db_path = settings.data_dir / "research.db"

    if not db_path.exists():
        console.print("[yellow]No research graph found.[/] Run an arXiv task first.")
        return

    store = GraphStore(db_path)
    stats = store.get_stats()

    table = Table(title="Research Graph")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Papers", str(stats["papers"]))
    table.add_row("Edges", str(stats["edges"]))
    table.add_row("Chunks", str(stats["chunks"]))
    console.print(table)

    methods = store.get_method_frequencies()
    if methods:
        mt = Table(title="Method Frequencies")
        mt.add_column("Method")
        mt.add_column("Count", justify="right")
        for method, count in list(methods.items())[:15]:
            mt.add_row(method, str(count))
        console.print(mt)

    datasets = store.get_dataset_frequencies()
    if datasets:
        dt = Table(title="Dataset Frequencies")
        dt.add_column("Dataset")
        dt.add_column("Count", justify="right")
        for ds, count in list(datasets.items())[:15]:
            dt.add_row(ds, str(count))
        console.print(dt)

    store.close()


@graph.command("papers")
@click.option("--recent", type=int, default=None, help="Show papers from last N days.")
@click.option("--limit", type=int, default=20, help="Max papers to show.")
def graph_papers(recent: int | None, limit: int) -> None:
    """List papers in the research graph."""
    from daily_research.graph.store import GraphStore

    settings = get_settings()
    db_path = settings.data_dir / "research.db"

    if not db_path.exists():
        console.print("[yellow]No research graph found.[/]")
        return

    store = GraphStore(db_path)

    if recent:
        papers = store.get_recent_papers(days=recent)[:limit]
    else:
        papers = store.list_papers(limit=limit)

    if not papers:
        console.print("[yellow]No papers in graph.[/]")
        store.close()
        return

    table = Table(title=f"Papers ({len(papers)} shown)")
    table.add_column("arXiv ID", style="cyan")
    table.add_column("Title", max_width=50)
    table.add_column("Method", max_width=20)
    table.add_column("Year")
    table.add_column("Categories", max_width=15)
    for p in papers:
        table.add_row(
            p.paper_id,
            p.title[:50],
            p.method_type or "-",
            str(p.year or "-"),
            ", ".join(p.categories[:2]),
        )
    console.print(table)
    store.close()


@graph.command("trends")
@click.option("--days", type=int, default=7, help="Lookback window in days.")
def graph_trends(days: int) -> None:
    """Show trend analysis from the research graph."""
    from daily_research.analysis.trends import generate_trend_report
    from daily_research.graph.store import GraphStore
    from daily_research.vectorstore.store import VectorStore

    settings = get_settings()
    db_path = settings.data_dir / "research.db"

    if not db_path.exists():
        console.print("[yellow]No research graph found.[/]")
        return

    store = GraphStore(db_path)
    vectors = VectorStore(store, settings)

    report = generate_trend_report(store, vectors, recent_days=days)

    console.print(f"\n[bold]Trend Report[/] (last {days} days)\n")
    console.print(f"  Topic drift: {report.drift_direction}")

    if report.emerging_methods:
        console.print(f"  Emerging methods: {', '.join(report.emerging_methods)}")
    if report.declining_methods:
        console.print(f"  Declining methods: {', '.join(report.declining_methods)}")
    if report.new_datasets:
        console.print(f"  New datasets: {', '.join(report.new_datasets)}")

    if report.novelty_scores:
        console.print("\n  [bold]Top Novelty Scores:[/]")
        for ns in report.novelty_scores[:5]:
            console.print(
                f"    [{ns['novelty_score']}/10] {ns['title']}"
            )
            for reason in ns["reasons"]:
                console.print(f"      - {reason}")

    if report.accelerating_papers:
        console.print("\n  [bold]Most Cited (in graph):[/]")
        for ap in report.accelerating_papers[:5]:
            console.print(
                f"    [{ap['citation_count']} citations] {ap['title']}"
            )

    console.print()
    store.close()


@graph.command("export")
@click.argument("output", default="graph-export.json")
def graph_export(output: str) -> None:
    """Export the research graph to a JSON file."""
    import json

    from daily_research.graph.store import GraphStore

    settings = get_settings()
    db_path = settings.data_dir / "research.db"

    if not db_path.exists():
        console.print("[yellow]No research graph found.[/]")
        return

    store = GraphStore(db_path)
    papers = store.list_papers(limit=100_000)

    export_data = {
        "papers": [p.to_dict() for p in papers],
        "stats": store.get_stats(),
        "method_frequencies": store.get_method_frequencies(),
        "dataset_frequencies": store.get_dataset_frequencies(),
    }

    out_path = Path(output)
    out_path.write_text(json.dumps(export_data, indent=2, default=str), encoding="utf-8")
    console.print(f"[green]Exported {len(papers)} papers →[/] {out_path}")
    store.close()


# ── arxiv ───────────────────────────────────────────────────────────


@cli.group()
def arxiv() -> None:
    """arXiv paper search and ingestion."""


@arxiv.command("search")
@click.argument("query")
@click.option("--max-results", "-n", type=int, default=10, help="Max results.")
def arxiv_search(query: str, max_results: int) -> None:
    """Search arXiv for papers matching QUERY."""
    from daily_research.arxiv.client import search_arxiv

    papers = search_arxiv(query, max_results=max_results)

    if not papers:
        console.print("[yellow]No results found.[/]")
        return

    table = Table(title=f"arXiv Results ({len(papers)})")
    table.add_column("ID", style="cyan")
    table.add_column("Title", max_width=60)
    table.add_column("Year")
    table.add_column("Categories", max_width=20)
    for p in papers:
        table.add_row(
            p.paper_id,
            p.title[:60],
            str(p.year or "?"),
            ", ".join(p.categories[:3]),
        )
    console.print(table)


@arxiv.command("ingest")
@click.argument("arxiv_id")
def arxiv_ingest(arxiv_id: str) -> None:
    """Manually ingest a specific arXiv paper by ID."""
    from daily_research.agents.extractor import extract_from_abstract
    from daily_research.arxiv.client import search_by_ids
    from daily_research.graph.store import GraphStore

    settings = get_settings()
    settings.ensure_dirs()

    papers = search_by_ids([arxiv_id])
    if not papers:
        console.print(f"[red]Paper not found:[/] {arxiv_id}")
        return

    paper = papers[0]
    store = GraphStore(settings.data_dir / "research.db")
    store.upsert_paper(paper)

    console.print(f"[green]Ingested:[/] {paper.title}")

    # Quick extraction from abstract
    try:
        extraction = extract_from_abstract(paper.abstract, paper.title, settings)
        paper.method_type = extraction.method_type
        paper.method_name = extraction.method_name
        paper.tasks = extraction.tasks
        paper.datasets = extraction.datasets
        paper.novelty_claim = extraction.novelty_claim
        paper.key_findings = extraction.key_findings
        store.upsert_paper(paper)
        console.print(f"  Method: {extraction.method_type} / {extraction.method_name}")
        console.print(f"  Novelty: {extraction.novelty_claim}")
    except Exception:
        console.print("  [yellow]⚠[/] Extraction failed")

    store.close()


# ── entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
