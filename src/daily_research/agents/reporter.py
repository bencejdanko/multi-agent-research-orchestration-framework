"""Reporter agent — turns raw research into a polished markdown report."""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone

from daily_research.agents.base import chat, get_client
from daily_research.agents.researcher import ResearchResult
from daily_research.config import Settings

REPORTER_SYSTEM = textwrap.dedent("""\
    You are a professional technical writer.  You will receive a research
    synthesis and its source list.  Produce a **polished markdown report**
    with the following structure:

    # <Title>
    *Generated on <date>*

    ## Executive Summary
    (2-3 paragraph overview)

    ## Key Findings
    (detailed sections with ### sub-headings)

    ## Sources
    (numbered list with clickable links)

    Guidelines:
    • Write clearly and concisely.
    • Use bullet points and tables where appropriate.
    • Cite sources inline as [1], [2], etc.
    • Do not invent facts beyond what the synthesis provides.
""")


def _build_source_list(results: list[dict]) -> str:
    return "\n".join(
        f"[{i}] [{r['title']}]({r['href']})" for i, r in enumerate(results, 1)
    )


def generate_report(
    research: ResearchResult,
    settings: Settings,
) -> str:
    """Generate a full markdown report from research results."""
    client = get_client(settings)
    model = settings.openai_model
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sources = _build_source_list(research.search_results)
    user_msg = (
        f"Date: {today}\n\n"
        f"## Research Synthesis\n\n{research.synthesis}\n\n"
        f"## Source List\n\n{sources}"
    )
    return chat(
        client,
        model,
        system=REPORTER_SYSTEM,
        user=user_msg,
        temperature=0.25,
        max_tokens=4096,
    )
