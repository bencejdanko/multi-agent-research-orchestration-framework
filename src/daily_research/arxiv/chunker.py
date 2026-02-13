"""Section-aware chunking for academic papers."""

from __future__ import annotations

import hashlib
import logging
import re

from daily_research.graph.models import Chunk

log = logging.getLogger(__name__)

# Approximate tokens per word (for English text)
_TOKENS_PER_WORD = 1.3
_DEFAULT_MAX_TOKENS = 1000
_DEFAULT_OVERLAP_TOKENS = 100


def _estimate_tokens(text: str) -> int:
    """Estimate token count from word count."""
    return int(len(text.split()) * _TOKENS_PER_WORD)


def _chunk_id(paper_id: str, section: str, index: int) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{paper_id}:{section}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text on blank lines, preserving non-empty paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_section(
    text: str,
    paper_id: str,
    section: str,
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk a single section into token-limited pieces.

    Strategy:
    1. Split by paragraph boundaries first.
    2. Merge paragraphs until max_tokens is reached.
    3. If a single paragraph exceeds max_tokens, split by sentences.

    Each chunk is tagged with paper_id and section name.
    """
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    chunk_idx = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        # If a single paragraph is too big, split by sentences
        if para_tokens > max_tokens:
            # Flush current buffer first
            if current_parts:
                content = "\n\n".join(current_parts)
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(paper_id, section, chunk_idx),
                        paper_id=paper_id,
                        section=section,
                        content=content,
                        token_count=_estimate_tokens(content),
                    )
                )
                chunk_idx += 1
                current_parts = []
                current_tokens = 0

            # Split paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_buffer: list[str] = []
            sent_tokens = 0
            for sent in sentences:
                st = _estimate_tokens(sent)
                if sent_tokens + st > max_tokens and sent_buffer:
                    content = " ".join(sent_buffer)
                    chunks.append(
                        Chunk(
                            chunk_id=_chunk_id(paper_id, section, chunk_idx),
                            paper_id=paper_id,
                            section=section,
                            content=content,
                            token_count=_estimate_tokens(content),
                        )
                    )
                    chunk_idx += 1
                    # Keep last sentence as overlap
                    if overlap_tokens > 0 and sent_buffer:
                        sent_buffer = sent_buffer[-1:]
                        sent_tokens = _estimate_tokens(sent_buffer[0])
                    else:
                        sent_buffer = []
                        sent_tokens = 0
                sent_buffer.append(sent)
                sent_tokens += st

            if sent_buffer:
                content = " ".join(sent_buffer)
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(paper_id, section, chunk_idx),
                        paper_id=paper_id,
                        section=section,
                        content=content,
                        token_count=_estimate_tokens(content),
                    )
                )
                chunk_idx += 1

            continue

        # Normal case: accumulate paragraphs
        if current_tokens + para_tokens > max_tokens and current_parts:
            content = "\n\n".join(current_parts)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(paper_id, section, chunk_idx),
                    paper_id=paper_id,
                    section=section,
                    content=content,
                    token_count=_estimate_tokens(content),
                )
            )
            chunk_idx += 1
            # Keep last paragraph as overlap context
            if overlap_tokens > 0 and current_parts:
                last = current_parts[-1]
                current_parts = [last]
                current_tokens = _estimate_tokens(last)
            else:
                current_parts = []
                current_tokens = 0

        current_parts.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_parts:
        content = "\n\n".join(current_parts)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(paper_id, section, chunk_idx),
                paper_id=paper_id,
                section=section,
                content=content,
                token_count=_estimate_tokens(content),
            )
        )

    return chunks


def chunk_paper(
    sections: dict[str, str],
    paper_id: str,
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
    skip_sections: set[str] | None = None,
) -> list[Chunk]:
    """Chunk an entire paper (dict of section→text) into Chunk objects.

    Optionally skip certain sections (e.g. references, appendix).
    """
    skip = skip_sections or {"references", "appendix"}
    all_chunks: list[Chunk] = []

    for section_name, section_text in sections.items():
        if section_name.lower() in skip:
            continue
        chunks = chunk_section(
            section_text,
            paper_id,
            section_name,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_chunks.extend(chunks)

    log.info(
        "Chunked paper %s: %d chunks from %d sections",
        paper_id,
        len(all_chunks),
        len(sections) - len(skip.intersection(sections)),
    )
    return all_chunks
