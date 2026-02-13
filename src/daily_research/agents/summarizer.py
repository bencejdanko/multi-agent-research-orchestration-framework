"""Summariser agent — creates a concise Discord-friendly summary."""

from __future__ import annotations

import textwrap

from daily_research.agents.base import chat, get_client
from daily_research.config import Settings

SUMMARISER_SYSTEM = textwrap.dedent("""\
    You are a concise summariser.  Given a full research report in markdown,
    produce a **short summary** suitable for posting in a Discord channel.

    Rules:
    • Maximum ~1800 characters (Discord limit is 2000; leave room for a header).
    • Use **bold** and *italic* sparingly for emphasis.
    • Bullet points are fine.
    • End with a one-line note like "Full report attached." or similar.
    • Do not include the source list — just mention "see full report for sources".
    • Be informative but punchy.
""")


def summarise_for_discord(report_md: str, task_title: str, settings: Settings) -> str:
    """Return a Discord-ready summary of the report."""
    client = get_client(settings)
    model = settings.openai_model

    summary = chat(
        client,
        model,
        system=SUMMARISER_SYSTEM,
        user=f"# Task: {task_title}\n\n{report_md}",
        temperature=0.3,
        max_tokens=600,
    )
    return summary
