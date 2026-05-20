"""Tests for Delta Lake writeback — to_delta_lake() and query_to_delta_lake()."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

deltalake = pytest.importorskip("deltalake")

import bridgr
from bridgr.export import DataExporter, to_delta_lake, query_to_delta_lake


@pytest.fixture
def db():
    """Create an in-memory database with Entity nodes for export testing."""
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


@pytest.fixture
def delta_dir():
    """Provide a temporary directory for Delta Lake output, cleaned up after test."""
    tmpdir = tempfile.mkdtemp(prefix="bridgr_delta_export_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestToDeltaLakeFunction:
    """Tests for the standalone to_delta_lake() function."""

    def test_basic_export(self, db, delta_dir):
        """Export nodes to Delta and verify row count."""
        delta_path = os.path.join(delta_dir, "entities")
        result = to_delta_lake(db, "Entity", delta_path)

        assert result["rows_written"] == 3
        assert result["delta_path"] == delta_path
        assert "columns" in result

    def test_delta_table_readable(self, db, delta_dir):
        """Exported Delta table is readable by deltalake."""
        delta_path = os.path.join(delta_dir, "entities_read")
        to_delta_lake(db, "Entity", delta_path)

        dt = deltalake.DeltaTable(delta_path)
        table = dt.to_pyarrow_table()
        assert table.num_rows == 3

    def test_column_names_clean(self, db, delta_dir):
        """Column names should not have 'n.' prefix from Cypher RETURN n.*."""
        delta_path = os.path.join(delta_dir, "entities_cols")
        result = to_delta_lake(db, "Entity", delta_path)

        # Columns should be clean (no "n." prefix)
        for col in result["columns"]:
            assert not col.startswith("n."), f"Column '{col}' has 'n.' prefix"

        # Verify by reading back
        dt = deltalake.DeltaTable(delta_path)
        table = dt.to_pyarrow_table()
        for col in table.column_names:
            assert not col.startswith("n.")

    def test_data_roundtrip(self, db, delta_dir):
        """Values written to Delta match the source data."""
        delta_path = os.path.join(delta_dir, "entities_rt")
        to_delta_lake(db, "Entity", delta_path)

        dt = deltalake.DeltaTable(delta_path)
        table = dt.to_pyarrow_table()
        df = table.to_pandas()

        names = set(df["name"].tolist())
        assert names == {"Smith", "Jones", "Acme Corp"}

    def test_overwrite_mode(self, db, delta_dir):
        """Overwrite mode replaces existing data."""
        delta_path = os.path.join(delta_dir, "entities_ow")
        to_delta_lake(db, "Entity", delta_path, mode="overwrite")
        to_delta_lake(db, "Entity", delta_path, mode="overwrite")

        dt = deltalake.DeltaTable(delta_path)
        table = dt.to_pyarrow_table()
        assert table.num_rows == 3  # not 6

    def test_append_mode(self, db, delta_dir):
        """Append mode adds to existing data."""
        delta_path = os.path.join(delta_dir, "entities_app")
        to_delta_lake(db, "Entity", delta_path, mode="overwrite")
        to_delta_lake(db, "Entity", delta_path, mode="append")

        dt = deltalake.DeltaTable(delta_path)
        table = dt.to_pyarrow_table()
        assert table.num_rows == 6  # 3 + 3


class TestQueryToDeltaLakeFunction:
    """Tests for the standalone query_to_delta_lake() function."""

    def test_basic_query_export(self, db, delta_dir):
        """Export query results to Delta and verify."""
        delta_path = os.path.join(delta_dir, "query_basic")
        result = query_to_delta_lake(
            db,
            "MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.confidence AS confidence",
            delta_path,
        )

        assert result["rows_written"] == 3
        assert result["delta_path"] == delta_path
        assert "columns" in result
        assert len(result["columns"]) == 3

    def test_filtered_query(self, db, delta_dir):
        """Export filtered query results."""
        delta_path = os.path.join(delta_dir, "query_filtered")
        result = query_to_delta_lake(
            db,
            "MATCH (e:Entity) WHERE e.confidence > 0.85 RETURN e.id AS id, e.name AS name",
            delta_path,
        )

        assert result["rows_written"] == 2  # Smith (0.9) and Acme (0.95)

        dt = deltalake.DeltaTable(delta_path)
        table = dt.to_pyarrow_table()
        df = table.to_pandas()
        names = set(df["name"].tolist())
        assert names == {"Smith", "Acme Corp"}

    def test_query_result_columns(self, db, delta_dir):
        """Columns in result match the query's RETURN clause."""
        delta_path = os.path.join(delta_dir, "query_cols")
        result = query_to_delta_lake(
            db,
            "MATCH (e:Entity) RETURN e.name AS entity_name, e.confidence AS score",
            delta_path,
        )
        assert "entity_name" in result["columns"]
        assert "score" in result["columns"]


class TestDataExporterDeltaMethods:
    """Tests for the DataExporter class methods (dict return values)."""

    def test_exporter_to_delta_returns_dict(self, db, delta_dir):
        """DataExporter.to_delta_lake() returns a dict with metadata."""
        exporter = DataExporter(db)
        delta_path = os.path.join(delta_dir, "exporter_nodes")
        result = exporter.to_delta_lake("Entity", delta_path)

        assert isinstance(result, dict)
        assert result["rows_written"] == 3
        assert result["delta_path"] == delta_path
        assert isinstance(result["columns"], list)

    def test_exporter_query_to_delta_returns_dict(self, db, delta_dir):
        """DataExporter.query_to_delta_lake() returns a dict with metadata."""
        exporter = DataExporter(db)
        delta_path = os.path.join(delta_dir, "exporter_query")
        result = exporter.query_to_delta_lake(
            "MATCH (e:Entity) RETURN e.id AS id, e.name AS name",
            delta_path,
        )

        assert isinstance(result, dict)
        assert result["rows_written"] == 3
        assert "columns" in result
        assert result["delta_path"] == delta_path
