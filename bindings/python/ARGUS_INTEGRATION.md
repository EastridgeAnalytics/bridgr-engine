# Argus Integration Guide — Swapping CaseGraph for BridgrStore

This guide documents the exact code changes needed in the Argus repo to replace the filesystem-based CaseGraph with BridgrStore (backed by LadybugDB). Covers stories A-6 through A-11.

## Prerequisites

```bash
pip install bridgr   # or: pip install -e /path/to/bridgr-engine/bindings/python
```

## Step 1: Add storage backend toggle (A-7)

In `api/nullclaw/config.py` (or wherever settings live):

```python
import os
STORAGE_BACKEND = os.getenv("ARGUS_STORAGE_BACKEND", "filesystem")  # or "bridgr"
```

## Step 2: Create a factory function (A-7)

In `api/nullclaw/node_store.py`, add at the bottom:

```python
def get_store(case_dir: Path):
    """Return the appropriate storage backend based on config."""
    from .config import STORAGE_BACKEND

    if STORAGE_BACKEND == "bridgr":
        from bridgr.argus import BridgrStore
        store = BridgrStore(case_dir)
        store.load()
        return store
    else:
        store = CaseGraph(case_dir)
        store.load()
        return store
```

## Step 3: Swap in route files (A-7)

In each route file that uses CaseGraph, change the graph cache:

### `routes_data.py`

```python
# BEFORE:
from .node_store import CaseGraph
_graph_cache: dict[str, CaseGraph] = {}

def _get_graph() -> CaseGraph:
    cp = _case_path()
    key = str(cp)
    if key not in _graph_cache:
        g = CaseGraph(cp)
        g.load()
        _graph_cache[key] = g
    return _graph_cache[key]

# AFTER:
from .node_store import get_store
_graph_cache: dict[str, Any] = {}

def _get_graph():
    cp = _case_path()
    key = str(cp)
    if key not in _graph_cache:
        _graph_cache[key] = get_store(cp)
    return _graph_cache[key]
```

No other changes needed in routes_data.py — BridgrStore has the same public API as CaseGraph:
- `read_node(id)` ✓
- `write_node(id, type, frontmatter, body)` ✓
- `delete_node(id, force=, cascade=)` ✓
- `list_nodes(type)` ✓
- `node_exists(id)` ✓
- `unique_id(slug)` ✓
- `get_by_short_name(name)` ✓
- `all_short_names()` ✓
- `get_issue_tree()` ✓
- `get_descendants(id)` ✓
- `get_references_to(id)` ✓
- `rename_entity(id, new_name)` ✓
- `get_facts_about(entity_id)` ✓
- `get_facts_citing(source_id)` ✓
- `get_facts_bearing_on(issue_id)` ✓
- `next_fact_number()` ✓
- `set_counter(value)` ✓

### Same pattern for:
- `routes_case.py` — case open/create
- `routes_processing.py` — extraction writes
- `routes_chat.py` — context enrichment

## Step 4: Swap in MCP server (A-8)

In `mcp_server.py`, the tools call `graph.read_node()`, `graph.write_node()`, etc.
Since BridgrStore has the same API, no tool implementation changes needed — only the store initialization.

## Step 5: Update extraction pipeline (A-9)

In `extraction_pipeline.py`, the batch mode change:

```python
# BEFORE:
graph._batch_mode = True
# ... create nodes ...
graph._batch_mode = False
graph._rebuild_index()

# AFTER:
graph.begin_batch()
# ... create nodes (same write_node calls) ...
graph.end_batch()
# No need for _rebuild_index — edges are synced automatically
```

## Step 6: Case creation (A-7)

In `routes_case.py`, when creating a new case:

```python
# BEFORE: mkdir for each node type
for d in ["entities", "facts", "sources", "issues", ...]:
    (case_dir / d).mkdir(exist_ok=True)

# AFTER (with bridgr backend):
# BridgrStore.load() creates the .lbug file and schema automatically.
# The node type directories are still created for source-docs/ compatibility.
store = get_store(case_dir)
```

## Step 7: Migration of existing cases

For existing cases with .md files:

```python
from bridgr.migrate import migrate_case
migrate_case(Path("~/.config/Argus/cases/MyCaseName"))
```

This creates `bridgr.lbug` alongside the existing .md files. Original files are untouched.

## Step 8: Rollback plan

To switch back to filesystem storage:
```bash
export ARGUS_STORAGE_BACKEND=filesystem
```

The .md files are never deleted, so CaseGraph continues to work. The bridgr.lbug file is ignored when STORAGE_BACKEND=filesystem.

## Step 9: Run existing tests against BridgrStore (A-6)

```python
# In conftest.py or test fixtures:
import os
os.environ["ARGUS_STORAGE_BACKEND"] = "bridgr"

# All existing CaseGraph tests should pass with BridgrStore
# because the public API is identical.
```

## Step 10: Performance benchmarks (A-11)

Run the Phase 0 validation suite to compare:
```bash
cd bridgr-engine
python validation/phase0_validate.py
```

Key metrics from validation:
- Single node lookup: 1.6ms (CaseGraph: <1ms dict lookup)
- Facts for entity: 2.9ms (CaseGraph: <5ms reverse index)
- 3-hop traversal: 3.0ms (CaseGraph: impossible)
- Canvas data: 3.2ms (CaseGraph: ~100ms)
- Batch 500 nodes: 197ms (CaseGraph: ~2s file writes)

## Files changed summary

| File | Change | Effort |
|------|--------|--------|
| `config.py` | Add STORAGE_BACKEND env var | 2 lines |
| `node_store.py` | Add get_store() factory | 10 lines |
| `routes_data.py` | Change _get_graph() | 5 lines |
| `routes_case.py` | Change case init | 3 lines |
| `routes_processing.py` | Change store import | 2 lines |
| `routes_chat.py` | Change store import | 2 lines |
| `mcp_server.py` | Change store import | 2 lines |
| `extraction_pipeline.py` | begin_batch/end_batch | 4 lines |
| `conftest.py` | Add backend toggle for tests | 5 lines |

**Total: ~35 lines changed across 9 files.** No API contract changes. Frontend unchanged.
