"""Runner — orchestrates the full research pipeline for one or many tasks.

Supports two modes:
- **Web search** (legacy): DuckDuckGo/Tavily → LLM synthesis → report
- **arXiv pipeline** (new): arXiv API → PDF parsing → structured extraction →
  research graph → citation expansion → trend analysis → digest

Mode is selected automatically based on whether the task file contains
an ``## arXiv`` configuration section.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from daily_research.agents.reporter import generate_report
from daily_research.agents.researcher import run_research
from daily_research.agents.summarizer import summarise_for_discord
from daily_research.config import Settings, get_settings
from daily_research.discord import send_to_discord
from daily_research.pdf_export import markdown_to_pdf
from daily_research.pipeline import parse_task, run_arxiv_pipeline
from daily_research.s3 import upload_to_s3

log = logging.getLogger(__name__)
console = Console()


# ── Task loading ────────────────────────────────────────────────────


def _title_from_md(text: str, fallback: str) -> str:
    """Extract a title from the first markdown heading, or use the filename."""
    m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def discover_tasks(settings: Settings) -> list[Path]:
    """Return all .md files in the tasks directory, sorted by name."""
    settings.ensure_dirs()
    tasks = sorted(settings.tasks_dir.glob("*.md"))
    return tasks


# ── Single-task pipeline ────────────────────────────────────────────


def run_single_task(
    task_path: Path,
    settings: Settings | None = None,
    *,
    skip_discord: bool = False,
    web_only: bool = False,
) -> Path:
    """Run the full pipeline for a single task file.

    If the task contains an ``## arXiv`` section (and *web_only* is False),
    the arXiv research-graph pipeline is used.  Otherwise the legacy
    web-search pipeline runs.

    Returns the path to the generated report.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()

    task_text = task_path.read_text(encoding="utf-8")
    title = _title_from_md(task_text, task_path.stem)
    task_config = parse_task(task_text, fallback_title=task_path.stem)

    console.print(Panel(f"[bold cyan]Researching:[/] {title}"))

    # ── Route: arXiv pipeline ───────────────────────────────────────
    if task_config.has_arxiv_config and not web_only:
        console.print("  [dim]Mode: arXiv research graph pipeline[/]")
        pipeline_result = run_arxiv_pipeline(task_text, settings)
        report_md = pipeline_result.digest

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^\w\-]", "_", title.lower())[:60]
        report_path = settings.reports_dir / f"{timestamp}_{slug}.md"
        report_path.write_text(report_md, encoding="utf-8")
        console.print(f"  [green]✓[/] Digest saved → {report_path.name}")

        n_new = len(pipeline_result.new_papers)
        n_exp = len(pipeline_result.expanded_papers)
        console.print(
            f"  [green]✓[/] {n_new} new papers, {n_exp} expanded citations"
        )

    # ── Route: legacy web-search pipeline ───────────────────────────
    else:
        console.print("  [dim]Mode: web search pipeline[/]")

        # 1. Research
        with console.status("[yellow]Gathering search results & synthesising…"):
            research = run_research(task_text, settings)
        console.print(f"  [green]✓[/] Collected {len(research.search_results)} sources")

        # 2. Report
        with console.status("[yellow]Writing report…"):
            report_md = generate_report(research, settings)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^\w\-]", "_", title.lower())[:60]
        report_path = settings.reports_dir / f"{timestamp}_{slug}.md"
        report_path.write_text(report_md, encoding="utf-8")
        console.print(f"  [green]✓[/] Report saved → {report_path.name}")

    # ── PDF generation & S3 upload ──────────────────────────────────
    pdf_url: str | None = None
    try:
        with console.status("[yellow]Generating PDF…"):
            pdf_path = markdown_to_pdf(report_path)
        console.print(f"  [green]✓[/] PDF generated → {pdf_path.name}")

        with console.status("[yellow]Uploading PDF to S3…"):
            pdf_url = upload_to_s3(pdf_path, settings=settings)
        if pdf_url:
            console.print(f"  [green]✓[/] PDF uploaded → {pdf_url}")
        else:
            console.print("  [yellow]⚠[/] S3 upload skipped (not configured or failed)")
    except Exception as exc:
        log.warning("PDF generation/upload failed: %s", exc)
        console.print(f"  [yellow]⚠[/] PDF step failed: {exc}")

    # ── Discord delivery ────────────────────────────────────────────
    if not skip_discord:
        with console.status("[yellow]Summarising for Discord…"):
            summary = summarise_for_discord(report_md, title, settings)
        sent = send_to_discord(
            summary,
            task_title=title,
            report_path=report_path,
            pdf_url=pdf_url,
            settings=settings,
        )
        if sent:
            console.print("  [green]✓[/] Discord notification sent")
        else:
            console.print("  [yellow]⚠[/] Discord delivery skipped or failed")
    else:
        console.print("  [dim]⏭  Discord skipped (--no-discord)[/]")

    return report_path


# ── Batch run ───────────────────────────────────────────────────────


def run_all_tasks(
    settings: Settings | None = None,
    *,
    skip_discord: bool = False,
    web_only: bool = False,
) -> list[Path]:
    """Discover and run every task in the tasks directory."""
    settings = settings or get_settings()
    tasks = discover_tasks(settings)

    if not tasks:
        console.print("[yellow]No task files found in[/]", settings.tasks_dir)
        return []

    console.print(f"[bold]Found {len(tasks)} task(s)[/]\n")
    reports: list[Path] = []
    for task_path in tasks:
        try:
            report = run_single_task(
                task_path, settings, skip_discord=skip_discord, web_only=web_only
            )
            reports.append(report)
        except Exception as exc:
            console.print(f"  [red]✗[/] Failed: {exc}")
            log.exception("Task %s failed", task_path.name)

    console.print(f"\n[bold green]Done — {len(reports)}/{len(tasks)} reports generated.[/]")
    return reports
