"""Tests for Bridgr Export/Import — story R8-1."""

import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import bridgr
from bridgr.export import DataExporter

TEST_DIR = Path(__file__).parent / "test_export"


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
        "confidence": "DOUBLE",
    })
    d.create_node("Entity", {"id": "e1", "name": "Smith", "confidence": 0.9})
    d.create_node("Entity", {"id": "e2", "name": "Jones", "confidence": 0.8})
    d.create_node("Entity", {"id": "e3", "name": "Acme Corp", "confidence": 0.95})
    yield d
    d.close()


class TestParquetExport:
    def test_export_to_parquet(self, db):
        exporter = DataExporter(db)
        path = str(TEST_DIR / "entities.parquet")
        count = exporter.to_parquet("Entity", path)
        assert count == 3
        assert Path(path).exists()

    def test_parquet_readable_by_pyarrow(self, db):
        exporter = DataExporter(db)
        path = str(TEST_DIR / "entities.parquet")
        exporter.to_parquet("Entity", path)
        table = pq.read_table(path)
        assert table.num_rows == 3
        col_names = table.column_names
        assert any("id" in c for c in col_names)

    def test_parquet_readable_by_pandas(self, db):
        exporter = DataExporter(db)
        path = str(TEST_DIR / "entities.parquet")
        exporter.to_parquet("Entity", path)
        import pandas as pd
        df = pd.read_parquet(path)
        assert len(df) == 3


class TestCSVExport:
    def test_export_to_csv(self, db):
        exporter = DataExporter(db)
        path = str(TEST_DIR / "entities.csv")
        count = exporter.to_csv("Entity", path)
        assert count == 3
        assert Path(path).exists()

    def test_csv_has_header(self, db):
        exporter = DataExporter(db)
        path = str(TEST_DIR / "entities.csv")
        exporter.to_csv("Entity", path)
        with open(path, encoding="utf-8") as f:
            header = f.readline().strip()
        assert "id" in header.lower() or "n.id" in header.lower()


class TestQueryExport:
    def test_query_to_parquet(self, db):
        exporter = DataExporter(db)
        path = str(TEST_DIR / "high_conf.parquet")
        count = exporter.query_to_parquet(
            "MATCH (e:Entity) WHERE e.confidence > 0.85 RETURN e.id, e.name, e.confidence",
            path,
        )
        assert count == 2  # Smith (0.9) and Acme (0.95)

    def test_query_to_csv(self, db):
        exporter = DataExporter(db)
        path = str(TEST_DIR / "all.csv")
        exporter.query_to_csv(
            "MATCH (e:Entity) RETURN e.id, e.name",
            path,
        )
        assert Path(path).exists()


class TestCSVImport:
    def test_roundtrip_csv(self, db):
        exporter = DataExporter(db)
        csv_path = str(TEST_DIR / "entities_rt.csv")
        exporter.to_csv("Entity", csv_path)

        db2 = bridgr.open(":memory:")
        db2.create_node_table("Entity", {
            "id": "STRING PRIMARY KEY",
            "name": "STRING",
            "confidence": "DOUBLE",
        })
        exporter2 = DataExporter(db2)
        exporter2.from_csv("Entity", csv_path)

        nodes = db2.get_nodes_by_type("Entity")
        assert len(nodes) == 3
        names = {n["name"] for n in nodes}
        assert "Smith" in names
        db2.close()
