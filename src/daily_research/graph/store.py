"""SQLite-backed persistent research knowledge graph store."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from daily_research.graph.models import Chunk, EdgeType, GraphEdge, PaperNode

log = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS papers (
    paper_id            TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    authors             TEXT DEFAULT '[]',
    abstract            TEXT DEFAULT '',
    year                INTEGER,
    venue               TEXT,
    categories          TEXT DEFAULT '[]',
    pdf_url             TEXT DEFAULT '',
    submitted_date      TEXT,
    method_type         TEXT,
    method_name         TEXT,
    tasks               TEXT DEFAULT '[]',
    datasets            TEXT DEFAULT '[]',
    metrics             TEXT DEFAULT '{}',
    novelty_claim       TEXT,
    limitations         TEXT,
    key_findings        TEXT DEFAULT '[]',
    referenced_arxiv_ids TEXT DEFAULT '[]',
    full_text_extracted INTEGER DEFAULT 0,
    ingested_at         TEXT,
    embedding           BLOB
);

CREATE TABLE IF NOT EXISTS edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    metadata    TEXT DEFAULT '{}',
    created_at  TEXT,
    UNIQUE(source_id, target_id, edge_type)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    paper_id    TEXT NOT NULL,
    section     TEXT DEFAULT '',
    content     TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    embedding   BLOB,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

CREATE TABLE IF NOT EXISTS run_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_papers_year      ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_ingested   ON papers(ingested_at);
CREATE INDEX IF NOT EXISTS idx_edges_source       ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target       ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type         ON edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_chunks_paper       ON chunks(paper_id);
"""


class GraphStore:
    """Persistent research graph backed by SQLite.

    Stores paper nodes, edges, text chunks, and run-state checkpoints.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Papers ──────────────────────────────────────────────────────

    def upsert_paper(self, paper: PaperNode) -> None:
        """Insert or update a paper node."""
        d = paper.to_dict()
        cols = ", ".join(d.keys())
        placeholders = ", ".join(f":{k}" for k in d.keys())
        update_clause = ", ".join(
            f"{k}=excluded.{k}" for k in d.keys() if k != "paper_id"
        )
        sql = (
            f"INSERT INTO papers ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(paper_id) DO UPDATE SET {update_clause}"
        )
        self._conn.execute(sql, d)
        self._conn.commit()

    def get_paper(self, paper_id: str) -> PaperNode | None:
        """Fetch a single paper by ID."""
        cur = self._conn.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        )
        row = cur.fetchone()
        return PaperNode.from_row(dict(row)) if row else None

    def paper_exists(self, paper_id: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)
        )
        return cur.fetchone() is not None

    def list_papers(
        self,
        *,
        since: str | None = None,
        categories: list[str] | None = None,
        limit: int = 200,
    ) -> list[PaperNode]:
        """List papers, optionally filtered by ingestion date or category."""
        sql = "SELECT * FROM papers WHERE 1=1"
        params: list[str] = []
        if since:
            sql += " AND ingested_at >= ?"
            params.append(since)
        sql += " ORDER BY ingested_at DESC LIMIT ?"
        params.append(str(limit))
        cur = self._conn.execute(sql, params)
        rows = cur.fetchall()
        papers = [PaperNode.from_row(dict(r)) for r in rows]
        if categories:
            cat_set = set(categories)
            papers = [
                p for p in papers if cat_set.intersection(p.categories)
            ]
        return papers

    def count_papers(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM papers")
        return cur.fetchone()[0]

    def update_paper_embedding(self, paper_id: str, embedding: bytes) -> None:
        self._conn.execute(
            "UPDATE papers SET embedding = ? WHERE paper_id = ?",
            (embedding, paper_id),
        )
        self._conn.commit()

    def get_papers_without_extraction(self, limit: int = 50) -> list[PaperNode]:
        """Papers that have been ingested but not yet fully extracted."""
        cur = self._conn.execute(
            "SELECT * FROM papers WHERE full_text_extracted = 0 "
            "ORDER BY ingested_at DESC LIMIT ?",
            (limit,),
        )
        return [PaperNode.from_row(dict(r)) for r in cur.fetchall()]

    # ── Edges ───────────────────────────────────────────────────────

    def add_edge(self, edge: GraphEdge) -> None:
        """Insert an edge (ignores duplicates)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                edge.source_id,
                edge.target_id,
                edge.edge_type.value,
                json.dumps(edge.metadata),
                edge.created_at,
            ),
        )
        self._conn.commit()

    def get_edges(
        self,
        paper_id: str,
        *,
        edge_type: EdgeType | None = None,
        direction: str = "outgoing",
    ) -> list[GraphEdge]:
        """Get edges for a paper.  direction: 'outgoing', 'incoming', or 'both'."""
        results: list[GraphEdge] = []
        clauses: list[str] = []
        params: list[str] = []

        if direction in ("outgoing", "both"):
            sql = "SELECT * FROM edges WHERE source_id = ?"
            p: list[str] = [paper_id]
            if edge_type:
                sql += " AND edge_type = ?"
                p.append(edge_type.value)
            cur = self._conn.execute(sql, p)
            for row in cur.fetchall():
                results.append(
                    GraphEdge(
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        edge_type=EdgeType(row["edge_type"]),
                        metadata=json.loads(row["metadata"] or "{}"),
                        created_at=row["created_at"] or "",
                    )
                )

        if direction in ("incoming", "both"):
            sql = "SELECT * FROM edges WHERE target_id = ?"
            p = [paper_id]
            if edge_type:
                sql += " AND edge_type = ?"
                p.append(edge_type.value)
            cur = self._conn.execute(sql, p)
            for row in cur.fetchall():
                results.append(
                    GraphEdge(
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        edge_type=EdgeType(row["edge_type"]),
                        metadata=json.loads(row["metadata"] or "{}"),
                        created_at=row["created_at"] or "",
                    )
                )

        return results

    def count_edges(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM edges")
        return cur.fetchone()[0]

    # ── Chunks ──────────────────────────────────────────────────────

    def save_chunk(self, chunk: Chunk) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, paper_id, section, content, token_count, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk.chunk_id,
                chunk.paper_id,
                chunk.section,
                chunk.content,
                chunk.token_count,
                chunk.embedding,
            ),
        )
        self._conn.commit()

    def get_chunks(self, paper_id: str) -> list[Chunk]:
        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE paper_id = ? ORDER BY chunk_id", (paper_id,)
        )
        return [
            Chunk(
                chunk_id=row["chunk_id"],
                paper_id=row["paper_id"],
                section=row["section"],
                content=row["content"],
                token_count=row["token_count"],
                embedding=row["embedding"],
            )
            for row in cur.fetchall()
        ]

    def get_chunks_without_embeddings(self, limit: int = 500) -> list[Chunk]:
        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE embedding IS NULL LIMIT ?", (limit,)
        )
        return [
            Chunk(
                chunk_id=row["chunk_id"],
                paper_id=row["paper_id"],
                section=row["section"],
                content=row["content"],
                token_count=row["token_count"],
                embedding=None,
            )
            for row in cur.fetchall()
        ]

    def update_chunk_embedding(self, chunk_id: str, embedding: bytes) -> None:
        self._conn.execute(
            "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
            (embedding, chunk_id),
        )
        self._conn.commit()

    def get_all_chunk_embeddings(self) -> list[tuple[str, str, bytes]]:
        """Return (chunk_id, paper_id, embedding) for all embedded chunks."""
        cur = self._conn.execute(
            "SELECT chunk_id, paper_id, embedding FROM chunks WHERE embedding IS NOT NULL"
        )
        return [(r["chunk_id"], r["paper_id"], r["embedding"]) for r in cur.fetchall()]

    def count_chunks(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM chunks")
        return cur.fetchone()[0]

    # ── Run state / checkpoints ─────────────────────────────────────

    def set_state(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_state (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_state(self, key: str) -> str | None:
        cur = self._conn.execute(
            "SELECT value FROM run_state WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return row["value"] if row else None

    # ── Statistics ──────────────────────────────────────────────────

    def get_stats(self) -> dict[str, int]:
        return {
            "papers": self.count_papers(),
            "edges": self.count_edges(),
            "chunks": self.count_chunks(),
        }

    def get_method_frequencies(self) -> dict[str, int]:
        """Count occurrences of each method_type across papers."""
        cur = self._conn.execute(
            "SELECT method_type, COUNT(*) as cnt FROM papers "
            "WHERE method_type IS NOT NULL GROUP BY method_type ORDER BY cnt DESC"
        )
        return {row["method_type"]: row["cnt"] for row in cur.fetchall()}

    def get_dataset_frequencies(self) -> dict[str, int]:
        """Count how often each dataset appears across papers."""
        cur = self._conn.execute(
            "SELECT datasets FROM papers WHERE datasets != '[]'"
        )
        freq: dict[str, int] = {}
        for row in cur.fetchall():
            for ds in json.loads(row["datasets"]):
                freq[ds] = freq.get(ds, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: -x[1]))

    def get_papers_by_method(self, method_type: str) -> list[PaperNode]:
        cur = self._conn.execute(
            "SELECT * FROM papers WHERE method_type = ? ORDER BY submitted_date DESC",
            (method_type,),
        )
        return [PaperNode.from_row(dict(r)) for r in cur.fetchall()]

    def get_recent_papers(self, days: int = 7) -> list[PaperNode]:
        """Papers ingested in the last N days."""
        from datetime import datetime, timedelta, timezone

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()
        return self.list_papers(since=cutoff)

    def get_citation_graph(self, paper_id: str, depth: int = 1) -> dict:
        """Build a citation sub-graph starting from paper_id."""
        visited: set[str] = set()
        nodes: list[dict] = []
        edge_list: list[dict] = []

        def _traverse(pid: str, current_depth: int) -> None:
            if pid in visited or current_depth > depth:
                return
            visited.add(pid)
            paper = self.get_paper(pid)
            if paper:
                nodes.append({"paper_id": pid, "title": paper.title})
            for edge in self.get_edges(pid, edge_type=EdgeType.CITES):
                edge_list.append(
                    {"source": edge.source_id, "target": edge.target_id}
                )
                _traverse(edge.target_id, current_depth + 1)

        _traverse(paper_id, 0)
        return {"nodes": nodes, "edges": edge_list}
