"""Tests for Bridgr MCP server — stories R4-1, R4-2, R4-3."""

import shutil
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import bridgr
from bridgr.mcp_server import _dispatch, create_server, TOOLS

TEST_DIR = Path(__file__).parent / "test_mcp_dbs"


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
    })
    d.create_node_table("Fact", {
        "id": "STRING PRIMARY KEY",
        "summary": "STRING",
        "confidence": "DOUBLE",
    })
    d.create_edge_table("INVOLVES", "Fact", "Entity", {"role": "STRING"})
    d.create_edge_table("CONNECTED_TO", "Entity", "Entity", {"relationship_type": "STRING"})
    yield d
    d.close()


def _seed(db):
    db.create_node("Entity", {"id": "e1", "name": "Smith", "entity_type": "person", "confidence": 0.9})
    db.create_node("Entity", {"id": "e2", "name": "Jones", "entity_type": "person", "confidence": 0.8})
    db.create_node("Entity", {"id": "e3", "name": "Acme Corp", "entity_type": "org", "confidence": 0.95})
    db.create_node("Fact", {"id": "f1", "summary": "Smith met Jones", "confidence": 0.85})
    db.create_edge("INVOLVES", "f1", "e1", {"role": "subject"}, from_label="Fact", to_label="Entity")
    db.create_edge("INVOLVES", "f1", "e2", {"role": "object"}, from_label="Fact", to_label="Entity")
    db.create_edge("CONNECTED_TO", "e1", "e3", {"relationship_type": "works_at"}, from_label="Entity", to_label="Entity")
    db.create_edge("CONNECTED_TO", "e2", "e3", {"relationship_type": "works_at"}, from_label="Entity", to_label="Entity")


class TestToolDefinitions:
    def test_tools_list_not_empty(self):
        assert len(TOOLS) >= 8

    def test_all_tools_have_schema(self):
        for tool in TOOLS:
            assert tool.name
            assert tool.description
            assert tool.inputSchema


class TestQueryTool:
    def test_basic_query(self, db):
        _seed(db)
        result = _dispatch(db, "query", {
            "cypher": "MATCH (e:Entity) RETURN e.name ORDER BY e.name",
        })
        assert result["count"] == 3
        names = [r["e.name"] for r in result["rows"]]
        assert "Smith" in names

    def test_parameterized_query(self, db):
        _seed(db)
        result = _dispatch(db, "query", {
            "cypher": "MATCH (e:Entity {id: $id}) RETURN e.name",
            "params": {"id": "e1"},
        })
        assert result["count"] == 1
        assert result["rows"][0]["e.name"] == "Smith"

    def test_traversal_query(self, db):
        _seed(db)
        result = _dispatch(db, "query", {
            "cypher": "MATCH (e:Entity {id: 'e1'})-[:CONNECTED_TO]->(org:Entity) RETURN org.name",
        })
        assert result["count"] == 1
        assert result["rows"][0]["org.name"] == "Acme Corp"


class TestReadNodeTool:
    def test_read_existing(self, db):
        _seed(db)
        result = _dispatch(db, "read_node", {"node_id": "e1"})
        assert result["found"] is True
        assert result["node"]["name"] == "Smith"

    def test_read_with_label(self, db):
        _seed(db)
        result = _dispatch(db, "read_node", {"node_id": "e1", "label": "Entity"})
        assert result["found"] is True

    def test_read_not_found(self, db):
        result = _dispatch(db, "read_node", {"node_id": "ghost"})
        assert result["found"] is False


class TestWriteNodeTool:
    def test_create_node(self, db):
        result = _dispatch(db, "write_node", {
            "label": "Entity",
            "properties": {"id": "new1", "name": "New Entity", "entity_type": "person", "confidence": 0.7},
        })
        assert result["action"] == "created"
        node = db.get_node("new1", "Entity")
        assert node["name"] == "New Entity"

    def test_update_node(self, db):
        _seed(db)
        result = _dispatch(db, "write_node", {
            "label": "Entity",
            "properties": {"id": "e1", "name": "Updated Smith"},
        })
        assert result["action"] == "updated"
        node = db.get_node("e1", "Entity")
        assert node["name"] == "Updated Smith"

    def test_write_missing_id(self, db):
        result = _dispatch(db, "write_node", {
            "label": "Entity",
            "properties": {"name": "No ID"},
        })
        assert "error" in result


class TestDeleteNodeTool:
    def test_delete_existing(self, db):
        _seed(db)
        result = _dispatch(db, "delete_node", {"node_id": "e1"})
        assert result["deleted"] is True
        assert db.get_node("e1") is None

    def test_delete_not_found(self, db):
        result = _dispatch(db, "delete_node", {"node_id": "ghost"})
        assert result["deleted"] is False


class TestCreateEdgeTool:
    def test_create_edge(self, db):
        _seed(db)
        result = _dispatch(db, "create_edge", {
            "edge_type": "CONNECTED_TO",
            "from_id": "e1",
            "to_id": "e2",
            "from_label": "Entity",
            "to_label": "Entity",
            "properties": {"relationship_type": "knows"},
        })
        assert result["created"] is True


class TestSearchTool:
    def test_search(self, db):
        _seed(db)
        result = _dispatch(db, "search", {"query": "Smith"})
        assert result["count"] >= 1

    def test_search_with_labels(self, db):
        _seed(db)
        result = _dispatch(db, "search", {"query": "Smith", "labels": ["Entity"]})
        assert result["count"] >= 1

    def test_search_no_results(self, db):
        _seed(db)
        result = _dispatch(db, "search", {"query": "nonexistent_xyz"})
        assert result["count"] == 0


class TestTraverseGraphTool:
    def test_traverse(self, db):
        _seed(db)
        result = _dispatch(db, "traverse_graph", {
            "start_node_id": "e1",
            "start_label": "Entity",
            "max_depth": 2,
        })
        assert result["node_count"] >= 1
        reached_ids = {n["id"] for n in result["reachable_nodes"]}
        assert "e3" in reached_ids  # Smith -> Acme Corp via CONNECTED_TO

    def test_traverse_with_edge_filter(self, db):
        _seed(db)
        result = _dispatch(db, "traverse_graph", {
            "start_node_id": "e1",
            "start_label": "Entity",
            "edge_types": ["CONNECTED_TO"],
            "max_depth": 1,
        })
        assert result["node_count"] >= 1


class TestListNodeTypesTool:
    def test_list_types(self, db):
        _seed(db)
        result = _dispatch(db, "list_node_types", {})
        labels = {t["label"] for t in result["types"]}
        assert "Entity" in labels
        assert "Fact" in labels
        entity_type = next(t for t in result["types"] if t["label"] == "Entity")
        assert entity_type["count"] == 3


class TestGetEdgesTool:
    def test_get_edges(self, db):
        _seed(db)
        result = _dispatch(db, "get_edges", {"node_id": "e1", "label": "Entity"})
        assert result["count"] >= 1
        edge_types = {e["type"] for e in result["edges"]}
        assert "CONNECTED_TO" in edge_types


class TestServerCreation:
    def test_create_server(self):
        db_path = str(TEST_DIR / "server_test.lbug")
        server = create_server(db_path)
        assert server is not None
