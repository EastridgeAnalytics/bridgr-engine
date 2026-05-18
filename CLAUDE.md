# Bridgr Engine

LadybugDB fork + Python graph database package. **This repo is MIT / public.** Never put proprietary code (agent logic, vertical prompts, credit metering, customer-specific details) here.

## Before Starting Work

Read the architecture docs from the strategy repo first:
```
Read C:\Users\eastr\Projects\Bridgr_20260512\README.md
Read C:\Users\eastr\Projects\Bridgr_20260512\OPEN_ITEMS.md
Read C:\Users\eastr\Projects\Bridgr_20260512\technical-specs\REQUIREMENTS.md
```

Then read this repo's docs:
```
Read C:\Users\eastr\Projects\bridgr-engine\AGENTS.md
Read C:\Users\eastr\Projects\bridgr-engine\bindings\python\ARGUS_INTEGRATION.md
```

## Build & Test (Python Package)

```bash
cd bindings/python
pip install -e ".[dev]"    # install bridgr + dev deps
pytest                      # run all tests (127 pass, 6 skip)
pytest -v                   # verbose output
```

The 6 skipped tests require the LadybugDB `algo` extension. Cypher-based algorithm tests always run.

## Build (C++ Engine)

See `AGENTS.md` for full build commands. Quick start:

```bash
make release               # Release build
make test                  # Run C++ tests
make pytest                # Run Python tests
```

Windows (PowerShell):
```powershell
cmake -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release .
cmake --build build/release --config Release
```

## What Lives Here

**Python package (`bindings/python/src/bridgr/`):**
- `database.py` — Core Database class (Cypher CRUD, transactions, Arrow/Pandas export)
- `argus.py` — BridgrStore (drop-in CaseGraph replacement for Argus)
- `algorithms.py` — GraphAlgorithms (WCC, PageRank, Louvain, SCC, K-Core + Cypher-based)
- `vector.py` — VectorIndex (HNSW: cosine, L2, dotproduct, hybrid search)
- `audit.py` — AuditedDatabase (append-only mutation log)
- `export.py` — DataExporter (Parquet/CSV import/export)
- `migrate.py` — Filesystem → .lbug migration tool
- `mcp_server.py` — MCP server (24 tools for AI agent access)
- `exceptions.py` — Error hierarchy

**C++ engine:** LadybugDB fork with CSR storage, Cypher, factorized joins, WCOJ, built-in extensions (algo, vector, fts, json).

## What Does NOT Live Here

- Product assembly, CLI, unified session, configuration → `bridgr` (private, product repo)
- Agent runtime, credit metering, API key management → `bridgr-agent` (private, intelligence library)
- Schema inference, import orchestration → `bridgr-agent`
- Vertical prompts (legal, fraud, retail) → `bridgr-agent`
- Memory persistence (save_memory, recall_memories) → `bridgr-agent`
- Customer names, pricing, deal details → `Bridgr_20260512` (strategy repo)
- Entity resolution (ER_Agentic) → separate repo

## MCP Server

```bash
python -m bridgr.mcp_server --db /path/to/database.lbug
```

24 tools shipped (v0.1 + v0.2 complete):

**v0.1 (core CRUD):** `query`, `read_node`, `write_node`, `delete_node`, `create_edge`, `search`, `traverse_graph`, `list_node_types`, `get_edges`, `create_node_table`, `create_edge_table`, `list_schema`.

**v0.2 (engine API):** `begin_transaction`, `commit_transaction`, `rollback_transaction`, `drop_table`, `alter_table`, `run_algorithm`, `bulk_import`, `create_vector_index`, `vector_search`, `hybrid_search`, `get_audit_log`, `export_data`.

Structured error codes: `SCHEMA_CONFLICT`, `NOT_FOUND`, `DUPLICATE`, `TRANSACTION_ERROR`, `VALIDATION_ERROR`, `CONFIRMATION_REQUIRED`.

## Key Rules

- **This is a public MIT repo.** Every commit is visible. No secrets, no customer data, no proprietary logic.
- **Cypher is the query language.** DataFusion/SQL is deferred. Don't build SQL support without explicit decision.
- **The engine is the authority.** When the agent calls MCP tools, the engine validates and executes. Errors flow back through tool responses.
- **Arrow-native.** Data flows through PyArrow. Query results available as dicts, Arrow tables, or Pandas DataFrames.

## Related Repos

| Repo | Path | Purpose |
|------|------|---------|
| bridgr (private) | `C:\Users\eastr\Projects\bridgr` | **The product.** Assembles engine + agent into unified SDK, CLI, and API. |
| Bridgr_20260512 (strategy) | `C:\Users\eastr\Projects\Bridgr_20260512` | Architecture docs, decisions, roadmap |
| bridgr-agent (private) | `C:\Users\eastr\Projects\bridgr-agent` | Intelligence library (schema inference, prompts, memory) |
| Argus | `C:\Users\eastr\Projects\Argus_CaseMap_Claw` | First customer of the Bridgr product |
| bridgr-server (private) | `C:\Users\eastr\Projects\bridgr-server` | Rust server wrapper (Tier 3): gRPC, Arrow Flight, REST, MCP-over-HTTP, auth, ethical walls |
| ER_Agentic | `C:\Users\eastr\Projects\ER_Agentic` | Entity resolution module |

Do not mention timelines and delivery dates in the conversation. Prioritization is okay, but not amounts of time to complete items.
