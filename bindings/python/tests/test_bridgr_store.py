"""Tests for BridgrStore — Argus CaseGraph replacement (stories A-1 through A-4)."""

import shutil
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bridgr.argus import BridgrStore, ReferentialIntegrityError

TEST_DIR = Path(__file__).parent / "test_cases"


@pytest.fixture(autouse=True)
def clean_test_dir():
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR))
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR), ignore_errors=True)


@pytest.fixture
def store():
    s = BridgrStore(TEST_DIR / "test_case")
    s.load()
    yield s
    s.close()


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TestInit:
    def test_load_creates_db(self):
        s = BridgrStore(TEST_DIR / "new_case")
        s.load()
        assert (TEST_DIR / "new_case" / "bridgr.lbug").exists()
        s.close()

    def test_load_idempotent(self):
        s = BridgrStore(TEST_DIR / "idem_case")
        s.load()
        s.close()
        s2 = BridgrStore(TEST_DIR / "idem_case")
        s2.load()
        s2.close()

    def test_ensure_fresh_noop(self, store):
        store.ensure_fresh()


# ------------------------------------------------------------------
# Node CRUD
# ------------------------------------------------------------------

class TestNodeCRUD:
    def test_write_and_read(self, store):
        store.write_node("entity_smith", "entity", {
            "id": "entity_smith",
            "name": "John Smith",
            "type": "entity",
            "short_name": "smith",
            "entity_type": "person",
            "confidence": 0.95,
        }, "Some body text about Smith.")

        node = store.read_node("entity_smith")
        assert node is not None
        assert node["type"] == "entity"
        assert node["frontmatter"]["name"] == "John Smith"
        assert node["frontmatter"]["confidence"] == 0.95
        assert node["body"] == "Some body text about Smith."

    def test_read_nonexistent(self, store):
        assert store.read_node("ghost") is None

    def test_write_update(self, store):
        store.write_node("e1", "entity", {"id": "e1", "name": "Old"})
        store.write_node("e1", "entity", {"id": "e1", "name": "New"})
        node = store.read_node("e1")
        assert node["frontmatter"]["name"] == "New"

    def test_delete_node(self, store):
        store.write_node("e1", "entity", {"id": "e1", "name": "Doomed"})
        assert store.delete_node("e1", force=True)
        assert store.read_node("e1") is None

    def test_delete_nonexistent(self, store):
        assert not store.delete_node("ghost")

    def test_delete_with_referential_integrity(self, store):
        store.write_node("e1", "entity", {"id": "e1", "name": "Target"})
        store.write_node("f1", "fact", {
            "id": "f1", "summary": "Test", "involves": ["e1"],
        })
        with pytest.raises(ReferentialIntegrityError):
            store.delete_node("e1")

    def test_delete_force(self, store):
        store.write_node("e1", "entity", {"id": "e1", "name": "Target"})
        store.write_node("f1", "fact", {
            "id": "f1", "summary": "Test", "involves": ["e1"],
        })
        assert store.delete_node("e1", force=True)

    def test_list_nodes(self, store):
        store.write_node("e1", "entity", {"id": "e1", "name": "Alice"})
        store.write_node("e2", "entity", {"id": "e2", "name": "Bob"})
        store.write_node("f1", "fact", {"id": "f1", "summary": "Fact 1"})
        entities = store.list_nodes("entity")
        assert len(entities) == 2
        names = {e["name"] for e in entities}
        assert names == {"Alice", "Bob"}

    def test_node_exists(self, store):
        assert not store.node_exists("e1")
        store.write_node("e1", "entity", {"id": "e1", "name": "Test"})
        assert store.node_exists("e1")

    def test_unique_id(self, store):
        assert store.unique_id("entity_smith") == "entity_smith"
        store.write_node("entity_smith", "entity", {"id": "entity_smith", "name": "Smith"})
        assert store.unique_id("entity_smith") == "entity_smith-2"
        store.write_node("entity_smith-2", "entity", {"id": "entity_smith-2", "name": "Smith 2"})
        assert store.unique_id("entity_smith") == "entity_smith-3"


# ------------------------------------------------------------------
# Short names
# ------------------------------------------------------------------

class TestShortNames:
    def test_get_by_short_name(self, store):
        store.write_node("e1", "entity", {
            "id": "e1", "name": "John Smith", "short_name": "smith",
        })
        node = store.get_by_short_name("smith")
        assert node is not None
        assert node["frontmatter"]["name"] == "John Smith"

    def test_get_by_short_name_case_insensitive(self, store):
        store.write_node("e1", "entity", {
            "id": "e1", "name": "John Smith", "short_name": "Smith",
        })
        assert store.get_by_short_name("smith") is not None
        assert store.get_by_short_name("SMITH") is not None

    def test_all_short_names(self, store):
        store.write_node("e1", "entity", {"id": "e1", "name": "A", "short_name": "a"})
        store.write_node("e2", "entity", {"id": "e2", "name": "B", "short_name": "b"})
        mapping = store.all_short_names()
        assert mapping == {"a": "e1", "b": "e2"}


# ------------------------------------------------------------------
# Issue tree
# ------------------------------------------------------------------

class TestIssueTree:
    def test_flat_issues(self, store):
        store.write_node("i1", "issue", {"id": "i1", "title": "Issue 1"})
        store.write_node("i2", "issue", {"id": "i2", "title": "Issue 2"})
        tree = store.get_issue_tree()
        assert len(tree) == 2

    def test_nested_issues(self, store):
        store.write_node("i1", "issue", {"id": "i1", "title": "Parent"})
        store.write_node("i2", "issue", {"id": "i2", "title": "Child", "parent_id": "i1"})
        tree = store.get_issue_tree()
        assert len(tree) == 1
        assert tree[0]["title"] == "Parent"
        assert len(tree[0]["children"]) == 1
        assert tree[0]["children"][0]["title"] == "Child"


# ------------------------------------------------------------------
# Traversals
# ------------------------------------------------------------------

class TestTraversals:
    def _seed(self, store):
        store.write_node("e1", "entity", {"id": "e1", "name": "Smith"})
        store.write_node("e2", "entity", {"id": "e2", "name": "Jones"})
        store.write_node("s1", "source", {"id": "s1", "filename": "doc.pdf"})
        store.write_node("i1", "issue", {"id": "i1", "title": "Main Issue"})
        store.write_node("f1", "fact", {
            "id": "f1", "summary": "Fact one",
            "involves": ["e1", "e2"],
            "sources": ["s1"],
            "bears_on": ["i1"],
        })
        store.write_node("f2", "fact", {
            "id": "f2", "summary": "Fact two",
            "involves": ["e1"],
            "sources": ["s1"],
        })

    def test_get_facts_about(self, store):
        self._seed(store)
        facts = store.get_facts_about("e1")
        assert len(facts) == 2
        summaries = {f["summary"] for f in facts}
        assert summaries == {"Fact one", "Fact two"}

    def test_get_facts_about_single(self, store):
        self._seed(store)
        facts = store.get_facts_about("e2")
        assert len(facts) == 1
        assert facts[0]["summary"] == "Fact one"

    def test_get_facts_citing(self, store):
        self._seed(store)
        facts = store.get_facts_citing("s1")
        assert len(facts) == 2

    def test_get_facts_bearing_on(self, store):
        self._seed(store)
        facts = store.get_facts_bearing_on("i1")
        assert len(facts) == 1
        assert facts[0]["summary"] == "Fact one"

    def test_get_references_to(self, store):
        self._seed(store)
        refs = store.get_references_to("e1")
        assert len(refs) == 2
        ref_sources = {r["node_id"] for r in refs}
        assert ref_sources == {"f1", "f2"}


# ------------------------------------------------------------------
# Counters
# ------------------------------------------------------------------

class TestCounters:
    def test_next_fact_number(self, store):
        assert store.next_fact_number() == 1
        assert store.next_fact_number() == 2
        assert store.next_fact_number() == 3

    def test_set_counter(self, store):
        store.set_counter(100)
        assert store.next_fact_number() == 101


# ------------------------------------------------------------------
# Batch mode
# ------------------------------------------------------------------

class TestBatchMode:
    def test_batch_write(self, store):
        store.begin_batch()
        for i in range(50):
            store.write_node(f"e{i}", "entity", {"id": f"e{i}", "name": f"Entity {i}"})
        store.end_batch()
        entities = store.list_nodes("entity")
        assert len(entities) == 50


# ------------------------------------------------------------------
# Rename entity
# ------------------------------------------------------------------

class TestRename:
    def test_rename_entity(self, store):
        store.write_node("e1", "entity", {"id": "e1", "name": "Old Name", "short_name": "old"})
        store.write_node("f1", "fact", {"id": "f1", "summary": "Test", "involves": ["e1"]})
        updated = store.rename_entity("e1", "New Name")
        node = store.read_node("e1")
        assert node["frontmatter"]["name"] == "New Name"
        assert "f1" in updated


# ------------------------------------------------------------------
# Persistence across close/reopen
# ------------------------------------------------------------------

class TestPersistence:
    def test_data_persists(self):
        case_dir = TEST_DIR / "persist_test"
        s1 = BridgrStore(case_dir)
        s1.load()
        s1.write_node("e1", "entity", {"id": "e1", "name": "Persist Me"})
        s1.close()

        s2 = BridgrStore(case_dir)
        s2.load()
        node = s2.read_node("e1")
        assert node is not None
        assert node["frontmatter"]["name"] == "Persist Me"
        s2.close()
