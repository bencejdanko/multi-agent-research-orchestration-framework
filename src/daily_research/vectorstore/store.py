"""Vector store — embedding generation and similarity search.

Uses OpenAI embeddings API and numpy-based cosine similarity,
with persistence via the SQLite graph store.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from openai import OpenAI

from daily_research.config import Settings
from daily_research.graph.models import Chunk
from daily_research.graph.store import GraphStore

log = logging.getLogger(__name__)

# Default embedding dimension for text-embedding-3-small
_EMBEDDING_DIM = 1536


def _get_embedding_client(settings: Settings) -> OpenAI:
    kwargs: dict = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


class VectorStore:
    """Embedding generation and nearest-neighbour search over paper chunks.

    Embeddings are generated via the OpenAI API and persisted in the
    graph store's SQLite database.  Similarity search uses in-memory
    numpy cosine similarity.
    """

    def __init__(self, graph: GraphStore, settings: Settings) -> None:
        self.graph = graph
        self.settings = settings
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = _get_embedding_client(self.settings)
        return self._client

    # ── Embedding generation ────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Call the OpenAI embeddings API for a batch of texts."""
        if not texts:
            return []

        model = self.settings.embedding_model
        # API accepts up to 2048 inputs; batch if needed
        batch_size = 128
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Truncate very long texts to avoid token limits
            batch = [t[:8000] for t in batch]
            try:
                resp = self.client.embeddings.create(model=model, input=batch)
                for item in resp.data:
                    all_embeddings.append(
                        np.array(item.embedding, dtype=np.float32)
                    )
            except Exception:
                log.exception("Embedding API call failed for batch %d", i)
                # Fill with zeros as fallback
                for _ in batch:
                    all_embeddings.append(np.zeros(_EMBEDDING_DIM, dtype=np.float32))

        return all_embeddings

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        results = self._embed_texts([text])
        return results[0] if results else np.zeros(_EMBEDDING_DIM, dtype=np.float32)

    # ── Chunk embedding ─────────────────────────────────────────────

    def embed_new_chunks(self, batch_limit: int = 500) -> int:
        """Generate embeddings for all un-embedded chunks in the store.

        Returns the number of newly embedded chunks.
        """
        chunks = self.graph.get_chunks_without_embeddings(limit=batch_limit)
        if not chunks:
            return 0

        log.info("Embedding %d chunks…", len(chunks))
        texts = [c.content for c in chunks]
        embeddings = self._embed_texts(texts)

        for chunk, emb in zip(chunks, embeddings):
            self.graph.update_chunk_embedding(
                chunk.chunk_id, emb.tobytes()
            )

        log.info("Embedded %d chunks", len(chunks))
        return len(chunks)

    def embed_paper_abstract(self, paper_id: str, abstract: str) -> None:
        """Embed and store a paper's abstract embedding."""
        emb = self.embed_text(abstract)
        self.graph.update_paper_embedding(paper_id, emb.tobytes())

    # ── Similarity search ───────────────────────────────────────────

    def search_similar_chunks(
        self,
        query: str,
        *,
        top_k: int = 10,
        paper_filter: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Find the most similar chunks to a query string.

        Returns list of (chunk, cosine_similarity) sorted descending.
        """
        query_emb = self.embed_text(query)
        return self._search_by_embedding(
            query_emb, top_k=top_k, paper_filter=paper_filter
        )

    def _search_by_embedding(
        self,
        query_emb: np.ndarray,
        *,
        top_k: int = 10,
        paper_filter: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search using a pre-computed embedding vector."""
        all_embedded = self.graph.get_all_chunk_embeddings()
        if not all_embedded:
            return []

        # Filter by paper if requested
        if paper_filter:
            all_embedded = [
                (cid, pid, emb) for cid, pid, emb in all_embedded if pid == paper_filter
            ]

        # Build matrix and compute cosine similarity
        chunk_ids: list[str] = []
        paper_ids: list[str] = []
        emb_list: list[np.ndarray] = []

        for chunk_id, paper_id, emb_bytes in all_embedded:
            arr = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            if arr.shape[0] == 0:
                continue
            chunk_ids.append(chunk_id)
            paper_ids.append(paper_id)
            emb_list.append(arr)

        if not emb_list:
            return []

        matrix = np.stack(emb_list)
        # Normalise
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        matrix = matrix / norms

        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm

        scores = matrix @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[Chunk, float]] = []
        for idx in top_indices:
            cid = chunk_ids[idx]
            chunks = self.graph.get_chunks(paper_ids[idx])
            chunk = next((c for c in chunks if c.chunk_id == cid), None)
            if chunk:
                results.append((chunk, float(scores[idx])))

        return results

    def find_similar_papers(
        self, paper_id: str, *, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Find papers most similar to a given paper by abstract embedding.

        Returns list of (paper_id, similarity) sorted descending.
        """
        paper = self.graph.get_paper(paper_id)
        if not paper:
            return []

        # Use the paper's stored abstract embedding
        from daily_research.graph.store import GraphStore

        cur = self.graph._conn.execute(
            "SELECT paper_id, embedding FROM papers WHERE embedding IS NOT NULL AND paper_id != ?",
            (paper_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return []

        # Get query paper's embedding
        cur2 = self.graph._conn.execute(
            "SELECT embedding FROM papers WHERE paper_id = ?", (paper_id,)
        )
        row = cur2.fetchone()
        if not row or not row["embedding"]:
            # Generate it
            self.embed_paper_abstract(paper_id, paper.abstract)
            cur2 = self.graph._conn.execute(
                "SELECT embedding FROM papers WHERE paper_id = ?", (paper_id,)
            )
            row = cur2.fetchone()
            if not row or not row["embedding"]:
                return []

        query_emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        if query_emb.shape[0] == 0:
            return []

        paper_ids: list[str] = []
        emb_list: list[np.ndarray] = []
        for r in rows:
            if r["embedding"]:
                arr = np.frombuffer(r["embedding"], dtype=np.float32).copy()
                if arr.shape[0] > 0:
                    paper_ids.append(r["paper_id"])
                    emb_list.append(arr)

        if not emb_list:
            return []

        matrix = np.stack(emb_list)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        matrix = matrix / norms

        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm

        scores = matrix @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(paper_ids[i], float(scores[i])) for i in top_indices]

    # ── Cluster centroid ────────────────────────────────────────────

    def compute_topic_centroid(
        self, paper_ids: list[str] | None = None
    ) -> np.ndarray | None:
        """Compute the average embedding across papers (or a subset).

        Useful for tracking topic drift over time.
        """
        if paper_ids:
            placeholders = ",".join("?" for _ in paper_ids)
            cur = self.graph._conn.execute(
                f"SELECT embedding FROM papers WHERE paper_id IN ({placeholders}) AND embedding IS NOT NULL",
                paper_ids,
            )
        else:
            cur = self.graph._conn.execute(
                "SELECT embedding FROM papers WHERE embedding IS NOT NULL"
            )

        embs: list[np.ndarray] = []
        for row in cur.fetchall():
            if row["embedding"]:
                arr = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                if arr.shape[0] > 0:
                    embs.append(arr)

        if not embs:
            return None

        return np.mean(np.stack(embs), axis=0)
