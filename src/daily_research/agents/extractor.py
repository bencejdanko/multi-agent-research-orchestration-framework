"""Extractor agent — structured information extraction from papers.

Uses schema-constrained JSON output to extract methods, datasets,
novelty claims, metrics, and key findings from paper text.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap

from daily_research.agents.base import get_client
from daily_research.config import Settings
from daily_research.graph.models import ExtractionResult

log = logging.getLogger(__name__)

# ── JSON repair helpers ────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL
)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    m = _JSON_FENCE_RE.search(text)
    return m.group(1) if m else text


def _repair_json(raw: str) -> str:
    """Best-effort repair of truncated / malformed JSON from LLM output.

    Handles:
    - Markdown code fences around JSON
    - Unterminated strings (missing closing quote)
    - Missing closing brackets / braces
    - Trailing commas before } or ]
    """
    raw = _strip_markdown_fences(raw).strip()

    # Remove trailing commas before closing brackets
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    # Try parsing as-is first
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # Attempt to close unterminated strings and brackets
    # Strategy: append closing characters until it parses or we give up
    open_braces = raw.count("{") - raw.count("}")
    open_brackets = raw.count("[") - raw.count("]")
    open_quotes = raw.count('"') % 2  # odd means unterminated string

    suffix = ""
    if open_quotes:
        suffix += '"'
    # close any remaining arrays then objects
    suffix += "]" * max(open_brackets, 0)
    suffix += "}" * max(open_braces, 0)

    if suffix:
        repaired = raw + suffix
        # Clean trailing commas that may have appeared before our closers
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            json.loads(repaired)
            log.info("Repaired truncated JSON (appended %r)", suffix)
            return repaired
        except json.JSONDecodeError:
            pass

    return raw  # return as-is; caller will handle the error

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "method_type": {
            "type": "string",
            "description": "High-level method category (e.g., 'Transformer', 'Diffusion Model', 'Reinforcement Learning', 'Mixture of Experts')",
        },
        "method_name": {
            "type": "string",
            "description": "Specific method/model name proposed in the paper",
        },
        "tasks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "ML/AI tasks addressed (e.g., 'text classification', 'image generation')",
        },
        "datasets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Datasets used in experiments (e.g., 'ImageNet', 'GSM8K')",
        },
        "metrics": {
            "type": "object",
            "description": "Reported metrics as {metric_name: {value, dataset}} objects",
        },
        "novelty_claim": {
            "type": "string",
            "description": "The paper's primary novelty claim in one sentence",
        },
        "limitations": {
            "type": "string",
            "description": "Key limitations acknowledged or apparent",
        },
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-5 most important findings or contributions",
        },
        "referenced_arxiv_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "arXiv IDs (e.g., '2301.12345') found in the references",
        },
    },
    "required": [
        "method_type",
        "method_name",
        "tasks",
        "datasets",
        "novelty_claim",
        "key_findings",
    ],
}

EXTRACTOR_SYSTEM = textwrap.dedent("""\
    You are a precise academic paper analysis engine.  Given the text of a
    research paper (possibly truncated), extract structured information.

    Rules:
    • Extract ONLY information explicitly stated in the paper.
    • Do NOT fabricate metrics, datasets, or claims.
    • For method_type, use a broad category (e.g., "Transformer",
      "Graph Neural Network", "Diffusion Model").
    • For method_name, use the specific name the authors give their approach.
    • For metrics, include the metric name, value, and which dataset it was
      measured on.  If exact values aren't available, omit.
    • For referenced_arxiv_ids, extract only IDs that match the pattern
      YYMM.NNNNN (e.g., "2301.12345").
    • key_findings should be 3-5 bullet points summarising the most
      important contributions.

    Respond with a single JSON object matching the specified schema.
    Do NOT include any text outside the JSON.
""")


def extract_structured(
    paper_text: str,
    settings: Settings,
    *,
    max_input_chars: int = 60_000,
    max_retries: int = 2,
) -> ExtractionResult:
    """Run structured extraction on paper text via LLM.

    Truncates input to *max_input_chars* to stay within context limits.
    Retries up to *max_retries* times on JSON parse failures.
    Returns an ExtractionResult dataclass.
    """
    client = get_client(settings)
    model = settings.openai_model

    # Truncate if needed (keep beginning + end for best coverage)
    if len(paper_text) > max_input_chars:
        half = max_input_chars // 2
        paper_text = (
            paper_text[:half]
            + "\n\n[…text truncated…]\n\n"
            + paper_text[-half:]
        )

    last_error: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTOR_SYSTEM},
                    {"role": "user", "content": paper_text},
                ],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"

            # Check for truncated response (finish_reason != 'stop')
            finish_reason = resp.choices[0].finish_reason
            if finish_reason == "length":
                log.warning(
                    "LLM output truncated (finish_reason=length), "
                    "attempting JSON repair (attempt %d/%d)",
                    attempt + 1, 1 + max_retries,
                )

            # Repair and parse JSON
            repaired = _repair_json(raw)
            data = json.loads(repaired)

            # Handle LLM returning a list instead of a dict
            if isinstance(data, list):
                log.warning(
                    "LLM returned a JSON array instead of object; "
                    "using first element"
                )
                data = data[0] if data else {}

            if not isinstance(data, dict):
                log.warning(
                    "LLM returned unexpected JSON type %s; skipping",
                    type(data).__name__,
                )
                data = {}

            return ExtractionResult(
                method_type=data.get("method_type", ""),
                method_name=data.get("method_name", ""),
                tasks=data.get("tasks", []),
                datasets=data.get("datasets", []),
                metrics=data.get("metrics", {}),
                novelty_claim=data.get("novelty_claim", ""),
                limitations=data.get("limitations", ""),
                key_findings=data.get("key_findings", []),
                referenced_arxiv_ids=data.get("referenced_arxiv_ids", []),
            )

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            last_error = exc
            log.warning(
                "Structured extraction attempt %d/%d failed: %s",
                attempt + 1, 1 + max_retries, exc,
            )
            continue
        except Exception:
            log.exception("Structured extraction failed (non-retryable)")
            return ExtractionResult()

    log.error(
        "Structured extraction failed after %d attempts: %s",
        1 + max_retries, last_error,
    )
    return ExtractionResult()


def extract_from_abstract(
    abstract: str, title: str, settings: Settings
) -> ExtractionResult:
    """Quick extraction using only title + abstract (no PDF required).

    Cheaper and faster than full-text extraction, suitable for
    initial triage before deciding to download the PDF.
    """
    prompt = f"# {title}\n\n## Abstract\n\n{abstract}"
    return extract_structured(prompt, settings, max_input_chars=8000)
