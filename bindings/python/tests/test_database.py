"""Tests for bridgr.Database — covers R1, R2 (Cypher), R9, R10, R11."""

import shutil
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import bridgr
from bridgr import Database
from bridgr.exceptions import (
    NodeNotFoundError,
    DuplicateNodeError,
    TransactionError,
    SchemaError,
)

TEST_DIR = Path(__file__).parent / "test_dbs"


@pytest.fixture(autouse=True)
def clean_test_dir():
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR))
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR), ignore_errors=True)


@pytest.fixture
def db():
    d = bridgr.open(":memory:")
    d.create_node_table("Entity", {
        "id": "STRING PRIMARY KEY",
        "name": "STRING",
        "entity_type": "STRING",
        "confidence": "DOUBLE",
        "status": "STRING",
        "aliases": "STRING[]",
    })
    d.create_node_table("Fact", {
        "id": "STRING PRIMARY KEY",
        "fact_number": "INT64",
        "summary": "STRING",
        "confidence": "DOUBLE",
        "polarity": "STRING",
    })
    d.create_edge_table("INVOLVES", "Fact", "Entity", {"role": "STRING"})
    d.create_edge_table("CONNECTED_TO", "Entity", "Entity", {
        "relationship_type": "STRING",
        "context": "STRING",
    })
    yield d
    d.close()


# ------------------------------------------------------------------
# R1: Embedded Engine — open/close/persist
# ------------------------------------------------------------------

class TestEmbeddedEngine:
    def test_open_memory(self):
        d = bridgr.open(":memory:")
        assert d is not None
        d.close()

    def test_open_disk(self):
        path = str(TEST_DIR / "test.lbug")
        d = bridgr.open(path)
        d.create_node_table("Test", {"id": "STRING PRIMARY KEY", "val": "STRING"})
        d.create_node("Test", {"id": "t1", "val": "hello"})
        d.close()

        d2 = bridgr.open(path)
        node = d2.get_node("t1", "Test")
        assert node is not None
        assert node["val"] == "hello"
        d2.close()

    def test_context_manager(self):
        with bridgr.open(":memory:") as d:
            d.create_node_table("T", {"id": "STRING PRIMARY KEY"})
            d.create_node("T", {"id": "x"})
            assert d.get_node("x", "T") is not None

    def test_version(self):
        assert bridgr.__version__ == "0.1.0"


# ------------------------------------------------------------------
# R9: Node CRUD
# ------------------------------------------------------------------

class TestNodeCRUD:
    def test_create_and_get(self, db):
        db.create_node("Entity", {
            "id": "e1", "name": "John Smith", "entity_type": "person",
            "confidence": 0.95, "status": "confirmed", "aliases": ["J. Smith"],
        })
        node = db.get_node("e1")
        assert node is not None
        assert node["name"] == "John Smith"
        assert node["entity_type"] == "person"
        assert node["confidence"] == 0.95
        assert node["_label"] == "Entity"

    def test_get_node_not_found(self, db):
        assert db.get_node("nonexistent") is None

    def test_get_node_by_label(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Test", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        node = db.get_node("e1", label="Entity")
        assert node is not None
        assert node["name"] == "Test"

    def test_get_nodes_by_type(self, db):
        db.create_node("Entity", {"id": "e1", "name": "A", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        db.create_node("Entity", {"id": "e2", "name": "B", "entity_type": "org",
                                   "confidence": 0.8, "status": "ok", "aliases": []})
        nodes = db.get_nodes_by_type("Entity")
        assert len(nodes) == 2
        names = {n["name"] for n in nodes}
        assert names == {"A", "B"}

    def test_update_node(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Old", "entity_type": "person",
                                   "confidence": 0.5, "status": "unconfirmed", "aliases": []})
        db.update_node("e1", {"name": "New Name", "confidence": 0.99})
        node = db.get_node("e1")
        assert node["name"] == "New Name"
        assert node["confidence"] == 0.99
        assert node["entity_type"] == "person"  # preserved

    def test_update_nonexistent_raises(self, db):
        with pytest.raises(NodeNotFoundError):
            db.update_node("ghost", {"name": "Nope"})

    def test_delete_node(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Doomed", "entity_type": "person",
                                   "confidence": 0.5, "status": "ok", "aliases": []})
        db.delete_node("e1")
        assert db.get_node("e1") is None

    def test_delete_nonexistent_raises(self, db):
        with pytest.raises(NodeNotFoundError):
            db.delete_node("ghost")

    def test_create_duplicate_raises(self, db):
        db.create_node("Entity", {"id": "e1", "name": "First", "entity_type": "person",
                                   "confidence": 0.5, "status": "ok", "aliases": []})
        with pytest.raises(DuplicateNodeError):
            db.create_node("Entity", {"id": "e1", "name": "Second", "entity_type": "person",
                                       "confidence": 0.5, "status": "ok", "aliases": []})

    def test_array_properties(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Multi", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok",
                                   "aliases": ["Alias1", "Alias2", "Alias3"]})
        node = db.get_node("e1")
        assert node["aliases"] == ["Alias1", "Alias2", "Alias3"]


# ------------------------------------------------------------------
# R9: Edge CRUD
# ------------------------------------------------------------------

class TestEdgeCRUD:
    def _seed(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Alice", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        db.create_node("Entity", {"id": "e2", "name": "Bob", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        db.create_node("Fact", {"id": "f1", "fact_number": 1, "summary": "Test fact",
                                 "confidence": 0.8, "polarity": "supports"})

    def test_create_and_get_edges(self, db):
        self._seed(db)
        db.create_edge("CONNECTED_TO", "e1", "e2",
                        {"relationship_type": "knows", "context": "test"},
                        from_label="Entity", to_label="Entity")
        edges = db.get_edges("e1", label="Entity")
        assert len(edges) >= 1
        conn_edges = [e for e in edges if e["type"] == "CONNECTED_TO"]
        assert len(conn_edges) == 1
        assert conn_edges[0]["to_id"] == "e2"

    def test_create_edge_with_properties(self, db):
        self._seed(db)
        db.create_edge("INVOLVES", "f1", "e1",
                        {"role": "subject"},
                        from_label="Fact", to_label="Entity")
        edges = db.get_edges("f1", label="Fact")
        involves = [e for e in edges if e["type"] == "INVOLVES"]
        assert len(involves) == 1

    def test_delete_edge(self, db):
        self._seed(db)
        db.create_edge("CONNECTED_TO", "e1", "e2",
                        {"relationship_type": "knows", "context": "test"},
                        from_label="Entity", to_label="Entity")
        db.delete_edge("CONNECTED_TO", "e1", "e2")
        edges = db.get_edges("e1", label="Entity")
        conn_edges = [e for e in edges if e["type"] == "CONNECTED_TO"]
        assert len(conn_edges) == 0

    def test_get_edges_both_directions(self, db):
        self._seed(db)
        db.create_edge("CONNECTED_TO", "e1", "e2",
                        {"relationship_type": "knows", "context": "test"},
                        from_label="Entity", to_label="Entity")
        outgoing = db.get_edges("e1", label="Entity")
        incoming = db.get_edges("e2", label="Entity")
        out_conn = [e for e in outgoing if e["type"] == "CONNECTED_TO" and e["direction"] == "outgoing"]
        in_conn = [e for e in incoming if e["type"] == "CONNECTED_TO" and e["direction"] == "incoming"]
        assert len(out_conn) == 1
        assert len(in_conn) == 1


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

class TestSearch:
    def test_basic_search(self, db):
        db.create_node("Entity", {"id": "e1", "name": "John Smith", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        db.create_node("Entity", {"id": "e2", "name": "Jane Doe", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        results = db.search("Smith")
        assert len(results) == 1
        assert results[0]["name"] == "John Smith"

    def test_case_insensitive_search(self, db):
        db.create_node("Entity", {"id": "e1", "name": "John Smith", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        results = db.search("john smith")
        assert len(results) == 1

    def test_search_across_types(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Smith Corp", "entity_type": "org",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        db.create_node("Fact", {"id": "f1", "fact_number": 1, "summary": "Smith testified",
                                 "confidence": 0.8, "polarity": "supports"})
        results = db.search("Smith")
        assert len(results) == 2
        labels = {r["_label"] for r in results}
        assert labels == {"Entity", "Fact"}

    def test_search_no_results(self, db):
        results = db.search("nonexistent_term_xyz")
        assert len(results) == 0

    def test_search_in_aliases(self, db):
        db.create_node("Entity", {"id": "e1", "name": "International Business Machines",
                                   "entity_type": "org", "confidence": 0.9, "status": "ok",
                                   "aliases": ["IBM", "Big Blue"]})
        results = db.search("IBM")
        assert len(results) == 1


# ------------------------------------------------------------------
# Canvas data
# ------------------------------------------------------------------

class TestCanvasData:
    def test_canvas_returns_nodes_and_edges(self, db):
        db.create_node("Entity", {"id": "e1", "name": "A", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        db.create_node("Entity", {"id": "e2", "name": "B", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        db.create_edge("CONNECTED_TO", "e1", "e2",
                        {"relationship_type": "knows", "context": "test"},
                        from_label="Entity", to_label="Entity")
        canvas = db.get_canvas_data()
        assert len(canvas["nodes"]) == 2
        assert len(canvas["edges"]) == 1


# ------------------------------------------------------------------
# R10: Transactions / Batch mode
# ------------------------------------------------------------------

class TestTransactions:
    def test_commit(self, db):
        db.begin_transaction()
        db.create_node("Entity", {"id": "e1", "name": "Tx Test", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        db.commit()
        assert db.get_node("e1") is not None

    def test_rollback(self, db):
        db.create_node("Entity", {"id": "e_before", "name": "Before", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        db.begin_transaction()
        db.create_node("Entity", {"id": "e_rolled", "name": "Rolled", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        db.rollback()
        assert db.get_node("e_before") is not None
        assert db.get_node("e_rolled") is None

    def test_double_begin_raises(self, db):
        db.begin_transaction()
        with pytest.raises(TransactionError):
            db.begin_transaction()
        db.rollback()

    def test_commit_without_begin_raises(self, db):
        with pytest.raises(TransactionError):
            db.commit()

    def test_batch_mode_aliases(self, db):
        db.begin_batch()
        db.create_node("Entity", {"id": "e1", "name": "Batch", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        db.end_batch()
        assert db.get_node("e1") is not None


# ------------------------------------------------------------------
# R11: Project isolation
# ------------------------------------------------------------------

class TestProjectIsolation:
    def test_separate_databases(self):
        path_a = str(TEST_DIR / "case_a.lbug")
        path_b = str(TEST_DIR / "case_b.lbug")

        db_a = bridgr.open(path_a)
        db_a.create_node_table("Entity", {"id": "STRING PRIMARY KEY", "name": "STRING"})
        db_a.create_node("Entity", {"id": "shared_id", "name": "Case A Entity"})

        db_b = bridgr.open(path_b)
        db_b.create_node_table("Entity", {"id": "STRING PRIMARY KEY", "name": "STRING"})
        db_b.create_node("Entity", {"id": "shared_id", "name": "Case B Entity"})

        node_a = db_a.get_node("shared_id", "Entity")
        node_b = db_b.get_node("shared_id", "Entity")

        assert node_a["name"] == "Case A Entity"
        assert node_b["name"] == "Case B Entity"

        db_a.close()
        db_b.close()


# ------------------------------------------------------------------
# Raw Cypher execution + output formats
# ------------------------------------------------------------------

class TestRawExecution:
    def test_execute_cypher(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Test", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        result = db.execute("MATCH (e:Entity) RETURN e.name")
        assert result.has_next()
        assert result.get_next()[0] == "Test"

    def test_query_returns_dicts(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Test", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        rows = db.query("MATCH (e:Entity) RETURN e.name AS name, e.confidence AS conf")
        assert len(rows) == 1
        assert rows[0]["name"] == "Test"
        assert rows[0]["conf"] == 0.9

    def test_query_arrow(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Test", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        table = db.query_arrow("MATCH (e:Entity) RETURN e.name, e.confidence")
        assert table.num_rows == 1

    def test_query_df(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Test", "entity_type": "person",
                                   "confidence": 0.9, "status": "ok", "aliases": []})
        df = db.query_df("MATCH (e:Entity) RETURN e.name AS name, e.confidence AS conf")
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Test"


# ------------------------------------------------------------------
# Graph queries (path traversal, shortest path)
# ------------------------------------------------------------------

class TestGraphQueries:
    def _build_graph(self, db):
        for i in range(1, 6):
            db.create_node("Entity", {
                "id": f"e{i}", "name": f"Entity {i}", "entity_type": "person",
                "confidence": 0.9, "status": "ok", "aliases": [],
            })
        db.create_edge("CONNECTED_TO", "e1", "e2",
                        {"relationship_type": "knows", "context": ""},
                        from_label="Entity", to_label="Entity")
        db.create_edge("CONNECTED_TO", "e2", "e3",
                        {"relationship_type": "knows", "context": ""},
                        from_label="Entity", to_label="Entity")
        db.create_edge("CONNECTED_TO", "e3", "e4",
                        {"relationship_type": "knows", "context": ""},
                        from_label="Entity", to_label="Entity")
        db.create_edge("CONNECTED_TO", "e4", "e5",
                        {"relationship_type": "knows", "context": ""},
                        from_label="Entity", to_label="Entity")

    def test_variable_length_path(self, db):
        self._build_graph(db)
        rows = db.query(
            "MATCH (a:Entity {id: 'e1'})-[:CONNECTED_TO*1..3]->(b:Entity) "
            "RETURN DISTINCT b.name AS name"
        )
        names = {r["name"] for r in rows}
        assert "Entity 2" in names
        assert "Entity 3" in names
        assert "Entity 4" in names

    def test_shortest_path(self, db):
        self._build_graph(db)
        rows = db.query(
            "MATCH p = (a:Entity {id: 'e1'})-[:CONNECTED_TO* SHORTEST 1..10]->(b:Entity {id: 'e5'}) "
            "RETURN length(p) AS path_length"
        )
        assert len(rows) == 1
        assert rows[0]["path_length"] == 4
