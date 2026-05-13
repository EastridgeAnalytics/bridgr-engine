"""Tests for MCP v0.2 tools (the 9 new tools added to mcp_server.py).

Each tool has at least one happy-path and one error-path test.
Extension-dependent tests (algo, vector) use skipif markers.
"""

import csv
import os
import tempfile

import pytest

import bridgr
from bridgr.audit import AuditedDatabase
from bridgr.exceptions import SchemaError
from bridgr.mcp_server import _dispatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    d = bridgr.open(":memory:")
    d.create_node_table("Person", {"id": "STRING PRIMARY KEY", "name": "STRING", "age": "INT64"})
    d.create_node_table("Org", {"id": "STRING PRIMARY KEY", "title": "STRING"})
    d.create_edge_table("WORKS_AT", "Person", "Org")
    d.create_node("Person", {"id": "p1", "name": "Alice", "age": 30})
    d.create_node("Person", {"id": "p2", "name": "Bob", "age": 25})
    d.create_node("Org", {"id": "o1", "title": "Acme"})
    d.create_edge("WORKS_AT", "p1", "o1", from_label="Person", to_label="Org")
    d.create_edge("WORKS_AT", "p2", "o1", from_label="Person", to_label="Org")
    yield d
    d.close()


@pytest.fixture
def audited_db():
    d = AuditedDatabase(":memory:", actor="test-actor")
    d.create_node_table("Item", {"id": "STRING PRIMARY KEY", "val": "STRING"})
    d.create_node("Item", {"id": "i1", "val": "one"})
    d.create_node("Item", {"id": "i2", "val": "two"})
    yield d
    d.close()


def _algo_extension_available():
    try:
        d = bridgr.open(":memory:")
        d.execute("LOAD EXTENSION algo")
        d.close()
        return True
    except Exception:
        return False


algo_available = pytest.mark.skipif(
    not _algo_extension_available(),
    reason="algo extension not available",
)


def _vector_extension_available():
    try:
        d = bridgr.open(":memory:")
        d.execute("LOAD EXTENSION vector")
        d.close()
        return True
    except Exception:
        return False


vector_available = pytest.mark.skipif(
    not _vector_extension_available(),
    reason="vector extension not available",
)


# ===========================================================================
# alter_table
# ===========================================================================

class TestAlterTable:
    def test_add_column(self, db):
        result = _dispatch(db, "alter_table", {
            "label": "Person",
            "operation": "add_column",
            "column_name": "email",
            "column_type": "STRING",
        })
        assert result["success"] is True
        assert result["operation"] == "add_column"

        schema = _dispatch(db, "list_schema", {})
        person_cols = None
        for t in schema["node_tables"]:
            if t["label"] == "Person":
                person_cols = [c.get("name", c.get("property name", "")) for c in t["columns"]]
                break
        assert person_cols is not None
        assert "email" in person_cols

    def test_drop_column(self, db):
        result = _dispatch(db, "alter_table", {
            "label": "Person",
            "operation": "drop_column",
            "column_name": "age",
        })
        assert result["success"] is True

    def test_rename_column(self, db):
        result = _dispatch(db, "alter_table", {
            "label": "Person",
            "operation": "rename_column",
            "old_name": "name",
            "new_name": "full_name",
        })
        assert result["success"] is True

    def test_alter_nonexistent_table(self, db):
        result = _dispatch(db, "alter_table", {
            "label": "Ghost",
            "operation": "add_column",
            "column_name": "x",
            "column_type": "STRING",
        })
        assert result["success"] is False
        assert "error" in result

    def test_unknown_operation(self, db):
        result = _dispatch(db, "alter_table", {
            "label": "Person",
            "operation": "explode",
        })
        assert result["success"] is False
        assert "Unknown operation" in result["error"]


# ===========================================================================
# drop_table (already has a Tool def, but we test the v0.2 spec behavior)
# ===========================================================================

class TestDropTable:
    def test_drop_existing(self):
        db = bridgr.open(":memory:")
        db.create_node_table("Temp", {"id": "STRING PRIMARY KEY"})
        result = _dispatch(db, "drop_table", {"label": "Temp", "confirm": True})
        assert result["success"] is True
        db.close()

    def test_drop_nonexistent(self, db):
        with pytest.raises(SchemaError):
            _dispatch(db, "drop_table", {"label": "Phantom", "confirm": True})

    def test_drop_without_confirm(self, db):
        result = _dispatch(db, "drop_table", {"label": "Person"})
        assert result["success"] is False
        assert "confirm" in result.get("error", "").lower() or "CONFIRMATION" in result.get("code", "")


# ===========================================================================
# run_algorithm — Cypher-based (always work)
# ===========================================================================

class TestRunAlgorithmCypher:
    def test_degree_centrality(self, db):
        result = _dispatch(db, "run_algorithm", {
            "algorithm": "degree_centrality",
            "node_label": "Person",
            "edge_label": "WORKS_AT",
        })
        assert "results" in result
        assert result["count"] > 0
        for r in result["results"]:
            assert "node_id" in r
            assert "total_degree" in r

    def test_shortest_path(self, db):
        db.create_node("Person", {"id": "p3", "name": "Charlie", "age": 35})
        db.create_edge_table("KNOWS", "Person", "Person")
        db.create_edge("KNOWS", "p1", "p2", from_label="Person", to_label="Person")
        db.create_edge("KNOWS", "p2", "p3", from_label="Person", to_label="Person")

        result = _dispatch(db, "run_algorithm", {
            "algorithm": "shortest_path",
            "node_label": "Person",
            "edge_label": "KNOWS",
            "source_id": "p1",
            "target_id": "p3",
        })
        assert result["algorithm"] == "shortest_path"
        assert result["count"] == 1

    def test_shortest_path_no_path(self, db):
        db.create_node("Person", {"id": "p3", "name": "Charlie", "age": 35})
        db.create_edge_table("KNOWS", "Person", "Person")
        result = _dispatch(db, "run_algorithm", {
            "algorithm": "shortest_path",
            "node_label": "Person",
            "edge_label": "KNOWS",
            "source_id": "p1",
            "target_id": "p3",
        })
        assert result["count"] == 0

    def test_node_similarity(self, db):
        result = _dispatch(db, "run_algorithm", {
            "algorithm": "node_similarity",
            "node_label": "Person",
            "edge_label": "WORKS_AT",
            "source_id": "p1",
            "target_id": "p2",
        })
        assert result["algorithm"] == "node_similarity"
        assert "score" in result["results"][0]

    def test_unknown_algorithm(self, db):
        result = _dispatch(db, "run_algorithm", {
            "algorithm": "magic",
            "node_label": "Person",
            "edge_label": "WORKS_AT",
        })
        assert "error" in result


# ===========================================================================
# run_algorithm — extension-based (require algo extension)
# ===========================================================================

@algo_available
class TestRunAlgorithmExtension:
    def test_pagerank(self, db):
        result = _dispatch(db, "run_algorithm", {
            "algorithm": "pagerank",
            "node_label": "Person",
            "edge_label": "WORKS_AT",
        })
        assert result["count"] > 0
        assert "score" in result["results"][0] or "rank" in str(result["results"][0])

    def test_wcc(self, db):
        result = _dispatch(db, "run_algorithm", {
            "algorithm": "wcc",
            "node_label": "Person",
            "edge_label": "WORKS_AT",
        })
        assert result["count"] > 0


# ===========================================================================
# bulk_import
# ===========================================================================

class TestBulkImport:
    def test_import_csv(self, db):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "age"])
            writer.writerow(["p10", "Zara", "40"])
            writer.writerow(["p11", "Yuki", "28"])
            path = f.name

        try:
            db.create_node_table("Import", {"id": "STRING PRIMARY KEY", "name": "STRING", "age": "INT64"})
            result = _dispatch(db, "bulk_import", {
                "label": "Import",
                "path": path,
                "format": "csv",
            })
            assert result.get("imported", 0) >= 2
            assert result["label"] == "Import"
        finally:
            os.unlink(path)

    def test_import_nonexistent_table(self, db):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["id", "val"])
            writer.writerow(["x1", "test"])
            path = f.name

        try:
            result = _dispatch(db, "bulk_import", {
                "label": "Nope",
                "path": path,
            })
            assert result.get("success") is False or "error" in result
        finally:
            os.unlink(path)


# ===========================================================================
# create_vector_index
# ===========================================================================

@vector_available
class TestCreateVectorIndex:
    def test_create_index(self):
        db = bridgr.open(":memory:")
        db.execute("CREATE NODE TABLE Doc(id STRING PRIMARY KEY, emb FLOAT[3])")
        result = _dispatch(db, "create_vector_index", {
            "table": "Doc",
            "property": "emb",
            "metric": "cosine",
        })
        assert result["success"] is True
        assert result["index_name"] == "Doc_emb_idx"
        db.close()

    def test_create_on_nonexistent(self):
        db = bridgr.open(":memory:")
        result = _dispatch(db, "create_vector_index", {
            "table": "Ghost",
            "property": "emb",
        })
        assert result["success"] is False
        db.close()


# ===========================================================================
# vector_search
# ===========================================================================

@vector_available
class TestVectorSearch:
    def test_search(self):
        db = bridgr.open(":memory:")
        db.execute("CREATE NODE TABLE Doc(id STRING PRIMARY KEY, emb FLOAT[3])")
        db.execute("CREATE (:Doc {id: 'd1', emb: [1.0, 0.0, 0.0]})")
        db.execute("CREATE (:Doc {id: 'd2', emb: [0.0, 1.0, 0.0]})")

        _dispatch(db, "create_vector_index", {
            "table": "Doc",
            "property": "emb",
            "index_name": "doc_idx",
        })
        result = _dispatch(db, "vector_search", {
            "table": "Doc",
            "index_name": "doc_idx",
            "query_vector": [1.0, 0.0, 0.0],
            "k": 2,
        })
        assert result["count"] >= 1
        db.close()

    def test_search_nonexistent_index(self):
        db = bridgr.open(":memory:")
        result = _dispatch(db, "vector_search", {
            "table": "X",
            "index_name": "nope",
            "query_vector": [0.0],
        })
        assert "error" in result
        db.close()


# ===========================================================================
# hybrid_search
# ===========================================================================

@vector_available
class TestHybridSearch:
    def test_vector_only(self):
        db = bridgr.open(":memory:")
        db.execute("CREATE NODE TABLE Doc(id STRING PRIMARY KEY, emb FLOAT[3])")
        db.execute("CREATE (:Doc {id: 'd1', emb: [1.0, 0.0, 0.0]})")

        _dispatch(db, "create_vector_index", {
            "table": "Doc",
            "property": "emb",
            "index_name": "doc_idx",
        })
        result = _dispatch(db, "hybrid_search", {
            "table": "Doc",
            "index": "doc_idx",
            "query_vector": [1.0, 0.0, 0.0],
            "k": 1,
        })
        assert "results" in result
        db.close()

    def test_nonexistent_index(self):
        db = bridgr.open(":memory:")
        result = _dispatch(db, "hybrid_search", {
            "table": "X",
            "index": "nope",
            "query_vector": [0.0],
        })
        assert "error" in result
        db.close()


# ===========================================================================
# get_audit_log
# ===========================================================================

class TestGetAuditLog:
    def test_with_audited_db(self, audited_db):
        result = _dispatch(audited_db, "get_audit_log", {"limit": 50})
        assert result["count"] >= 2
        assert len(result["entries"]) >= 2

    def test_filter_by_operation(self, audited_db):
        result = _dispatch(audited_db, "get_audit_log", {
            "operation": "create",
            "limit": 50,
        })
        assert result["count"] >= 2

    def test_target_id_history(self, audited_db):
        result = _dispatch(audited_db, "get_audit_log", {"target_id": "i1"})
        assert result["count"] >= 1

    def test_plain_db_no_audit_table(self, db):
        result = _dispatch(db, "get_audit_log", {})
        assert result["entries"] == []
        assert result["count"] == 0


# ===========================================================================
# export_data
# ===========================================================================

class TestExportData:
    def test_export_csv(self, db):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            result = _dispatch(db, "export_data", {
                "label": "Person",
                "path": path,
                "format": "csv",
            })
            assert result.get("exported", 0) >= 2
            assert result["format"] == "csv"
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_export_parquet(self, db):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            result = _dispatch(db, "export_data", {
                "label": "Person",
                "path": path,
                "format": "parquet",
            })
            assert result.get("exported", 0) >= 2
            assert result["format"] == "parquet"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_export_nonexistent_table(self, db):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            result = _dispatch(db, "export_data", {
                "label": "Ghost",
                "path": path,
                "format": "csv",
            })
            assert result.get("success") is False or "error" in result
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ===========================================================================
# Tool count sanity check
# ===========================================================================

# ===========================================================================
# Transaction tools
# ===========================================================================

class TestTransactionTools:
    def test_begin_commit(self, db):
        result = _dispatch(db, "begin_transaction", {})
        assert result["success"] is True

        _dispatch(db, "write_node", {
            "label": "Person",
            "properties": {"id": "p99", "name": "TxTest", "age": 1},
        })

        result = _dispatch(db, "commit_transaction", {})
        assert result["success"] is True

        node = _dispatch(db, "read_node", {"node_id": "p99", "label": "Person"})
        assert node["found"] is True

    def test_begin_rollback(self, db):
        _dispatch(db, "begin_transaction", {})

        _dispatch(db, "write_node", {
            "label": "Person",
            "properties": {"id": "p98", "name": "Rolled", "age": 2},
        })

        result = _dispatch(db, "rollback_transaction", {})
        assert result["success"] is True

        node = _dispatch(db, "read_node", {"node_id": "p98", "label": "Person"})
        assert node["found"] is False

    def test_commit_without_begin(self, db):
        from bridgr.exceptions import TransactionError
        with pytest.raises(TransactionError):
            _dispatch(db, "commit_transaction", {})


# ===========================================================================
# Tool count sanity check
# ===========================================================================

def test_tool_count():
    from bridgr.mcp_server import TOOLS
    tool_names = [t.name for t in TOOLS]
    assert len(tool_names) == 24, f"Expected 24 tools, got {len(tool_names)}: {tool_names}"

    v02_tools = {"alter_table", "run_algorithm", "bulk_import",
                 "create_vector_index", "vector_search", "hybrid_search",
                 "get_audit_log", "export_data"}
    for name in v02_tools:
        assert name in tool_names, f"Missing v0.2 tool: {name}"

    tx_tools = {"begin_transaction", "commit_transaction", "rollback_transaction"}
    for name in tx_tools:
        assert name in tool_names, f"Missing transaction tool: {name}"


def test_tool_names_unique():
    from bridgr.mcp_server import TOOLS
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names)), f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"
