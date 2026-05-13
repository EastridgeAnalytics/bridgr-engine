"""Tests for Bridgr Audit Trail — stories R12-1, R12-2."""

import json
import shutil
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bridgr.audit import AuditedDatabase, AuditLog

TEST_DIR = Path(__file__).parent / "test_audit_dbs"


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
    d = AuditedDatabase(":memory:", actor="test_user")
    d.create_node_table("Entity", {
        "id": "STRING PRIMARY KEY",
        "name": "STRING",
        "confidence": "DOUBLE",
    })
    d.create_edge_table("KNOWS", "Entity", "Entity")
    yield d
    d.close()


class TestAuditCreate:
    def test_create_logs_entry(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Smith", "confidence": 0.9})
        history = db.audit_log.get_history("e1")
        assert len(history) == 1
        assert history[0]["operation"] == "create"
        assert history[0]["target_id"] == "e1"
        assert history[0]["actor"] == "test_user"

    def test_create_logs_properties(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Smith", "confidence": 0.9})
        history = db.audit_log.get_history("e1")
        changes = history[0]["changes"]
        assert changes["properties"]["name"] == "Smith"

    def test_create_has_timestamp(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Smith", "confidence": 0.9})
        history = db.audit_log.get_history("e1")
        assert history[0]["ts"].startswith("2026-")


class TestAuditUpdate:
    def test_update_logs_entry(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Old", "confidence": 0.5})
        db.update_node("e1", {"name": "New", "confidence": 0.99})
        history = db.audit_log.get_history("e1")
        assert len(history) == 2
        update_entry = history[0]  # most recent first
        assert update_entry["operation"] == "update"

    def test_update_logs_before_after(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Old", "confidence": 0.5})
        db.update_node("e1", {"name": "New"})
        history = db.audit_log.get_history("e1")
        update_entry = history[0]
        assert update_entry["changes"]["before"]["name"] == "Old"
        assert update_entry["changes"]["after"]["name"] == "New"


class TestAuditDelete:
    def test_delete_logs_entry(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Doomed", "confidence": 0.5})
        db.delete_node("e1")
        history = db.audit_log.get_history("e1")
        assert len(history) == 2
        delete_entry = history[0]
        assert delete_entry["operation"] == "delete"


class TestAuditEdge:
    def test_create_edge_logs(self, db):
        db.create_node("Entity", {"id": "e1", "name": "A", "confidence": 0.9})
        db.create_node("Entity", {"id": "e2", "name": "B", "confidence": 0.9})
        db.create_edge("KNOWS", "e1", "e2", from_label="Entity", to_label="Entity")
        history = db.audit_log.get_history("e1->e2")
        assert len(history) == 1
        assert history[0]["operation"] == "create_edge"
        assert history[0]["target_type"] == "KNOWS"


class TestAuditQuery:
    def test_query_by_operation(self, db):
        db.create_node("Entity", {"id": "e1", "name": "A", "confidence": 0.9})
        db.create_node("Entity", {"id": "e2", "name": "B", "confidence": 0.9})
        db.update_node("e1", {"name": "Updated"})
        creates = db.audit_log.query(operation="create")
        updates = db.audit_log.query(operation="update")
        assert len(creates) == 2
        assert len(updates) == 1

    def test_query_by_actor(self, db):
        db.create_node("Entity", {"id": "e1", "name": "A", "confidence": 0.9})
        results = db.audit_log.query(actor="test_user")
        assert len(results) >= 1

    def test_query_by_target_type(self, db):
        db.create_node("Entity", {"id": "e1", "name": "A", "confidence": 0.9})
        results = db.audit_log.query(target_type="Entity")
        assert len(results) >= 1

    def test_count(self, db):
        assert db.audit_log.count() == 0
        db.create_node("Entity", {"id": "e1", "name": "A", "confidence": 0.9})
        assert db.audit_log.count() == 1


class TestAuditExport:
    def test_export_jsonl(self, db):
        db.create_node("Entity", {"id": "e1", "name": "A", "confidence": 0.9})
        db.create_node("Entity", {"id": "e2", "name": "B", "confidence": 0.9})
        db.update_node("e1", {"name": "Updated"})

        export_path = str(TEST_DIR / "audit.jsonl")
        count = db.audit_log.export_jsonl(export_path)
        assert count == 3

        with open(export_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            entry = json.loads(line)
            assert "operation" in entry
            assert "ts" in entry


class TestAuditedDatabaseCompatibility:
    def test_works_as_regular_database(self, db):
        """AuditedDatabase should work exactly like Database for all operations."""
        db.create_node("Entity", {"id": "e1", "name": "Test", "confidence": 0.9})
        node = db.get_node("e1", "Entity")
        assert node["name"] == "Test"

        db.update_node("e1", {"name": "Updated"})
        node = db.get_node("e1", "Entity")
        assert node["name"] == "Updated"

        nodes = db.get_nodes_by_type("Entity")
        assert len(nodes) == 1

        results = db.search("Updated")
        assert len(results) == 1

    def test_actor_tracked(self, db):
        db.create_node("Entity", {"id": "e1", "name": "Test", "confidence": 0.9})
        history = db.audit_log.get_history("e1")
        assert history[0]["actor"] == "test_user"

    def test_different_actors(self):
        db = AuditedDatabase(":memory:", actor="alice")
        db.create_node_table("T", {"id": "STRING PRIMARY KEY", "v": "STRING"})
        db.create_node("T", {"id": "t1", "v": "by_alice"})
        db.actor = "bob"
        db.update_node("t1", {"v": "by_bob"})

        history = db.audit_log.get_history("t1")
        actors = [h["actor"] for h in history]
        assert "alice" in actors
        assert "bob" in actors
        db.close()
