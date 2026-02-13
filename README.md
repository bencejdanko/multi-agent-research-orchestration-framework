# daily-research

Automated agentic research intelligence system — discovers papers on arXiv,
builds a self-growing knowledge graph, tracks trends and novelty, generates
polished markdown digests, and pushes summaries to Discord.

## Architecture

```
tasks/*.md
   ↓
Task Parser  ─────────────────────────────────────────────┐
   ↓ (arXiv mode)                            (web mode)   │
Topic Query Generator                      Researcher     │
   ↓                                       (DuckDuckGo)   │
arXiv API + Metadata Ingestion                 ↓           │
   ↓                                       Reporter       │
PDF Acquisition (PyMuPDF)                      ↓           │
   ↓                                       reports/*.md   │
Section-Aware Chunking                         │           │
   ↓                                           │           │
Embedding + Vector Store                       │           │
   ↓                                           │           │
Structured Extraction (LLM)                    │           │
   ↓                                           │           │
Research Graph Update ←────────────────────────┘           │
   ↓                                                       │
Planner Agent (controlled expansion)                       │
   ↓                                                       │
Citation Expansion (recursive, depth-limited)              │
   ↓                                                       │
Trend + Novelty Analysis                                   │
   ↓                                                       │
Daily Digest Generator ──→ reports/*.md                    │
   ↓                                                       │
Summariser Agent ──→ Discord webhook  ←────────────────────┘
```

### Pipeline Modes

The system automatically selects the pipeline mode based on task configuration:

| Mode | Trigger | Pipeline |
|------|---------|----------|
| **arXiv** | Task has `## arXiv` section | arXiv API → PDF → graph → trends → digest |
| **Web** | No arXiv config | DuckDuckGo/Tavily → synthesis → report |

### Agent Pipeline (arXiv mode)

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **Query Generator** | Builds arXiv search queries from categories + keywords |
| 2 | **arXiv Client** | Fetches paper metadata via arXiv API |
| 3 | **PDF Parser** | Downloads PDFs, extracts text with PyMuPDF |
| 4 | **Chunker** | Section-aware splitting (800–1200 tokens) |
| 5 | **Embedding Engine** | OpenAI embeddings → numpy cosine similarity |
| 6 | **Extractor Agent** | Schema-constrained JSON extraction (methods, datasets, novelty) |
| 7 | **Graph Builder** | Upserts nodes + edges (CITES, EXTENDS, SIMILAR, etc.) |
| 8 | **Planner Agent** | Decides citation expansions (depth ≤ 2, bounded) |
| 9 | **Trend Analyser** | Embedding drift, method frequency, novelty scoring |
| 10 | **Digest Generator** | Produces structured markdown research digest |

### Research Graph

```
Paper Node ──CITES──→ Paper Node
    │                     │
    ├──EXTENDS──→         │
    ├──SAME_METHOD_FAMILY─┤
    ├──USES_DATASET──→ Dataset
    ├──COMPARES_TO──→     │
    └──SIMILAR_EMBEDDING──┘
```

**Paper Node** fields: `paper_id`, `title`, `authors`, `abstract`, `year`,
`categories`, `method_type`, `method_name`, `tasks`, `datasets`, `metrics`,
`novelty_claim`, `limitations`, `key_findings`, `referenced_arxiv_ids`

## Quick Start

```bash
# 1. Create & activate the venv
source .venv/bin/activate

# 2. Install the project (includes arXiv + PyMuPDF + numpy)
uv pip install -e .

# 3. Configure secrets
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and optionally DISCORD_WEBHOOK_URL

# 4. Create a task with arXiv configuration
research tasks add "Mixture of Experts"
# Then edit the task file — see Task File Format below

# 5. Run research (auto-selects arXiv or web pipeline)
research run mixture-of-experts

# Or run ALL tasks
research run

# 6. Inspect the research graph
research graph stats
research graph papers --recent 7
research graph trends --days 14
```

## CLI Reference

```
research                             # top-level help
research tasks list                  # list task files
research tasks add NAME              # create a new task stub
research tasks add NAME --from-file path/to/file.md
research tasks show NAME             # print task contents
research tasks remove NAME           # delete a task

research run [NAME]                  # run one or all tasks
research run --no-discord            # skip Discord delivery
research run --web-only              # force web-search mode

research reports list                # list generated reports
research reports show FILENAME       # print a report
research reports clean               # delete all reports

research graph stats                 # show graph statistics
research graph papers [--recent N]   # list papers in graph
research graph trends [--days N]     # show trend analysis
research graph export [FILE]         # export graph to JSON

research arxiv search QUERY          # search arXiv directly
research arxiv ingest ARXIV_ID       # manually ingest a paper

research config                      # show current configuration
research cron-install                # print a crontab line
```

## Task File Format

Tasks are plain markdown files in `tasks/`. Add an `## arXiv` section to
enable the research graph pipeline:

```markdown
# Mixture of Experts Architectures

## Objective

Survey recent advances in Mixture-of-Experts (MoE) architectures
for large language models.

## arXiv
categories: cs.AI, cs.LG, cs.CL
keywords: mixture of experts, sparse models, expert routing
max_papers: 50
citation_depth: 2

## Specific Questions

- What routing strategies are most effective?
- How does MoE compare to dense scaling?
- What are the training stability challenges?
```

### arXiv Section Fields

| Field | Default | Description |
|-------|---------|-------------|
| `categories` | — | arXiv category filters (comma-separated) |
| `keywords` | — | Search keywords (comma-separated) |
| `max_papers` | `50` | Max papers to fetch per run |
| `citation_depth` | `1` | Max citation expansion depth (0–2) |

If the `## arXiv` section is omitted, the task runs through the legacy
web-search pipeline (DuckDuckGo / Tavily).

## Configuration

All settings come from environment variables (or `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI (or compatible) API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for all agents |
| `OPENAI_BASE_URL` | *(default)* | Override for compatible APIs |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for embeddings |
| `DISCORD_WEBHOOK_URL` | — | Discord channel webhook URL |
| `TASKS_DIR` | `tasks` | Directory for task files |
| `REPORTS_DIR` | `reports` | Directory for generated reports |
| `DATA_DIR` | `data` | Directory for SQLite DB + PDF cache |
| `MAX_PAPERS_PER_RUN` | `50` | Global paper limit per pipeline run |
| `CITATION_DEPTH` | `1` | Default citation expansion depth |
| `MAX_CITATION_EXPANSIONS` | `10` | Max papers to expand per run |
| `TAVILY_API_KEY` | — | Tavily API key (optional, for web mode) |

## Design Principles

1. **Deterministic ingestion, probabilistic reasoning** — parsing and
   storage are deterministic; the LLM is only used for extraction,
   classification, and novelty reasoning.

2. **Separate planning from execution** — the Planner agent decides what
   to fetch/expand; the executor performs bounded operations. This prevents
   runaway tool loops.

3. **Structured over freeform** — all extraction produces schema-constrained
   JSON, never raw summaries.

4. **Controlled graph expansion** — citation depth is bounded (≤ 2),
   daily expansion is capped, and deduplication prevents re-ingestion.

## Data Storage

| Component | Backend | Location |
|-----------|---------|----------|
| Research graph | SQLite (WAL mode) | `data/research.db` |
| Embeddings | SQLite (BLOB columns) | `data/research.db` |
| PDF cache | File system | `data/pdf_cache/` |
| Reports | Markdown files | `reports/` |

## Cron Scheduling

```bash
research cron-install --schedule "0 8 * * *"
crontab -e  # paste the printed line
```

## License

MIT
