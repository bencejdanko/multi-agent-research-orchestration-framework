"""Discord webhook integration — send summaries and optional file attachments."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from daily_research.config import Settings

log = logging.getLogger(__name__)


def send_to_discord(
    summary: str,
    *,
    task_title: str,
    report_path: Path | None = None,
    pdf_url: str | None = None,
    settings: Settings,
) -> bool:
    """Post a summary (and optional report file) to the configured Discord webhook.

    If *pdf_url* is provided it is appended to the message so readers can
    download the full PDF report.

    Returns True on success, False otherwise.
    """
    url = settings.discord_webhook_url
    if not url:
        log.warning("DISCORD_WEBHOOK_URL is not set — skipping Discord delivery.")
        return False

    header = f"📋 **Research Report: {task_title}**\n\n"
    content = header + summary

    if pdf_url:
        content += f"\n\n📄 **[Download PDF Report]({pdf_url})**"

    # Discord message limit is 2000 chars
    if len(content) > 2000:
        content = content[:1997] + "…"

    try:
        # If we have a report file, send as multipart with the file attached
        if report_path and report_path.exists():
            with report_path.open("rb") as f:
                resp = httpx.post(
                    url,
                    data={"content": content},
                    files={"file": (report_path.name, f, "text/markdown")},
                    timeout=30,
                )
        else:
            resp = httpx.post(url, json={"content": content}, timeout=30)

        resp.raise_for_status()
        log.info("Discord message sent successfully.")
        return True

    except httpx.HTTPStatusError as exc:
        log.error("Discord API error %s: %s", exc.response.status_code, exc.response.text)
    except httpx.RequestError as exc:
        log.error("Discord request failed: %s", exc)

    return False
