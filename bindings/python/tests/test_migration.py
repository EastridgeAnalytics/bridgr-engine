"""Tests for filesystem-to-Bridgr migration (story A-5)."""

import json
import shutil
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bridgr.argus import BridgrStore
from bridgr.migrate import migrate_case, parse_md_file

TEST_DIR = Path(__file__).parent / "test_migration"


@pytest.fixture(autouse=True)
def clean_test_dir():
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR))
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR), ignore_errors=True)


def _create_md(case_dir: Path, node_type_dir: str, filename: str, frontmatter: dict, body: str = ""):
    """Helper to create a .md file in the case directory."""
    import yaml
    d = case_dir / node_type_dir
    d.mkdir(parents=True, exist_ok=True)
    fm_str = yaml.dump(frontmatter, default_flow_style=False)
    (d / filename).write_text(f"---\n{fm_str}---\n\n{body}", encoding="utf-8")


def _build_test_case(case_dir: Path):
    """Build a synthetic Hubley-like case for testing."""
    _create_md(case_dir, "entities", "entity_smith.md", {
        "id": "entity_smith",
        "type": "person",
        "name": "John Smith",
        "short_name": "smith",
        "aliases": ["J. Smith", "Smith, John"],
        "role": "witness",
        "confidence": 0.95,
    }, "Key witness in the investigation.")

    _create_md(case_dir, "entities", "entity_acme.md", {
        "id": "entity_acme",
        "type": "organization",
        "name": "Acme Corp",
        "short_name": "acme",
        "aliases": ["Acme Corporation", "ACME"],
        "confidence": 0.9,
    })

    _create_md(case_dir, "entities", "entity_doe.md", {
        "id": "entity_doe",
        "type": "person",
        "name": "Jane Doe",
        "short_name": "doe",
        "confidence": 0.85,
    })

    _create_md(case_dir, "facts", "fact-001.md", {
        "id": "fact-001",
        "type": "fact",
        "fact_number": 1,
        "summary": "Smith emailed Doe about merger terms on March 15",
        "confidence": 0.92,
        "status": "undisputed",
        "involves": ["entity_smith", "entity_doe"],
        "sources": [{"source_id": "source_email_archive", "pincite": "p. 42"}],
        "bears_on": [{"issue": "issue_insider", "direction": "supports", "strength": "strong"}],
        "polarity": "affirmed",
        "claim_type": "communication",
    }, "Email dated 2024-03-15 from Smith to Doe discussing merger terms.")

    _create_md(case_dir, "facts", "fact-002.md", {
        "id": "fact-002",
        "type": "fact",
        "fact_number": 2,
        "summary": "Acme Corp announced merger publicly on April 1",
        "confidence": 0.99,
        "status": "undisputed",
        "involves": ["entity_acme"],
        "sources": [{"source_id": "source_press_release"}],
    })

    _create_md(case_dir, "facts", "fact-003.md", {
        "id": "fact-003",
        "type": "fact",
        "fact_number": 3,
        "summary": "Doe purchased Acme stock on March 20",
        "confidence": 0.88,
        "involves": ["entity_doe", "entity_acme"],
        "sources": [{"source_id": "source_trading_records", "pincite": "row 147"}],
        "bears_on": [{"issue": "issue_insider", "direction": "supports"}],
    })

    _create_md(case_dir, "sources", "source_email_archive.md", {
        "id": "source_email_archive",
        "type": "source",
        "filename": "smith_email_archive.pst",
        "doc_type": "correspondence",
    })

    _create_md(case_dir, "sources", "source_press_release.md", {
        "id": "source_press_release",
        "type": "source",
        "filename": "acme_merger_announcement.pdf",
        "doc_type": "exhibit",
    })

    _create_md(case_dir, "sources", "source_trading_records.md", {
        "id": "source_trading_records",
        "type": "source",
        "filename": "doe_brokerage_statements.xlsx",
        "doc_type": "discovery",
    })

    _create_md(case_dir, "issues", "issue_insider.md", {
        "id": "issue_insider",
        "type": "issue",
        "title": "Insider Trading Allegation",
        "description": "Did Doe trade on material non-public information?",
    })

    _create_md(case_dir, "issues", "issue_communication.md", {
        "id": "issue_communication",
        "type": "issue",
        "title": "Pre-Announcement Communication",
        "parent_id": "issue_insider",
    })

    counters = {"fact_number": 3}
    (case_dir / "counters.json").write_text(json.dumps(counters), encoding="utf-8")


# ------------------------------------------------------------------
# Parse tests
# ------------------------------------------------------------------

class TestParseMd:
    def test_parse_md_file(self):
        case_dir = TEST_DIR / "parse_test"
        _create_md(case_dir, "entities", "test.md",
                   {"id": "test", "name": "Test Entity"},
                   "Body text here.")
        fm, body = parse_md_file(case_dir / "entities" / "test.md")
        assert fm["id"] == "test"
        assert fm["name"] == "Test Entity"
        assert body == "Body text here."

    def test_parse_no_frontmatter(self):
        d = TEST_DIR / "no_fm"
        d.mkdir(parents=True)
        (d / "test.md").write_text("Just a plain file.", encoding="utf-8")
        fm, body = parse_md_file(d / "test.md")
        assert fm == {}
        assert "plain file" in body


# ------------------------------------------------------------------
# Migration tests
# ------------------------------------------------------------------

class TestMigration:
    def test_full_migration(self):
        case_dir = TEST_DIR / "hubley_case"
        _build_test_case(case_dir)

        stats = migrate_case(case_dir, verbose=False)

        assert stats["nodes_imported"] == 11  # 3 entities + 3 facts + 3 sources + 2 issues
        assert len(stats["errors"]) == 0

    def test_migration_preserves_data(self):
        case_dir = TEST_DIR / "preserve_case"
        _build_test_case(case_dir)
        migrate_case(case_dir, verbose=False)

        store = BridgrStore(case_dir)
        store.load()

        smith = store.read_node("entity_smith")
        assert smith is not None
        assert smith["frontmatter"]["name"] == "John Smith"
        assert smith["frontmatter"]["aliases"] == ["J. Smith", "Smith, John"]
        assert smith["body"] == "Key witness in the investigation."

        fact1 = store.read_node("fact-001")
        assert fact1 is not None
        assert "Smith emailed Doe" in fact1["frontmatter"]["summary"]

        store.close()

    def test_migration_creates_edges(self):
        case_dir = TEST_DIR / "edges_case"
        _build_test_case(case_dir)
        migrate_case(case_dir, verbose=False)

        store = BridgrStore(case_dir)
        store.load()

        facts_about_smith = store.get_facts_about("entity_smith")
        assert len(facts_about_smith) >= 1
        summaries = {f["summary"] for f in facts_about_smith}
        assert any("Smith emailed Doe" in s for s in summaries)

        store.close()

    def test_migration_preserves_counter(self):
        case_dir = TEST_DIR / "counter_case"
        _build_test_case(case_dir)
        migrate_case(case_dir, verbose=False)

        store = BridgrStore(case_dir)
        store.load()
        next_num = store.next_fact_number()
        assert next_num == 4  # counter was at 3, next should be 4
        store.close()

    def test_migration_idempotent(self):
        case_dir = TEST_DIR / "idem_case"
        _build_test_case(case_dir)
        migrate_case(case_dir, verbose=False)
        migrate_case(case_dir, verbose=False)  # should not crash or duplicate

        store = BridgrStore(case_dir)
        store.load()
        entities = store.list_nodes("entity")
        assert len(entities) == 3  # still 3, not 6
        store.close()

    def test_migration_empty_case(self):
        case_dir = TEST_DIR / "empty_case"
        case_dir.mkdir(parents=True)
        stats = migrate_case(case_dir, verbose=False)
        assert stats["nodes_imported"] == 0

    def test_original_files_untouched(self):
        case_dir = TEST_DIR / "untouched_case"
        _build_test_case(case_dir)

        original_files = set()
        for d in ["entities", "facts", "sources", "issues"]:
            dir_path = case_dir / d
            if dir_path.exists():
                for f in dir_path.glob("*.md"):
                    original_files.add(str(f))
                    assert f.exists()

        migrate_case(case_dir, verbose=False)

        for f_path in original_files:
            assert Path(f_path).exists(), f"Original file was deleted: {f_path}"
