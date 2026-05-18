"""Tests for Database.from_delta_lake() — Delta Lake import support."""

import os
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

deltalake = pytest.importorskip("deltalake")

from bridgr.database import Database


@pytest.fixture
def delta_table_path():
    """Create a synthetic Delta table and return its path."""
    tmpdir = tempfile.mkdtemp(prefix="bridgr_delta_")
    table = pa.table({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "name": ["Alice Smith", "Bob Jones", "Carol White", "Dave Brown", "Eve Black"],
        "email": ["alice@example.com", "bob@example.com", "carol@example.com", "dave@example.com", "eve@example.com"],
        "age": [30, 45, 28, 52, 35],
        "balance": [1000.50, 2500.00, 750.25, 3200.00, 1800.75],
        "active": [True, True, False, True, False],
    })
    path = os.path.join(tmpdir, "customers")
    deltalake.write_deltalake(path, table)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def empty_delta_table_path():
    """Create an empty Delta table."""
    tmpdir = tempfile.mkdtemp(prefix="bridgr_delta_empty_")
    table = pa.table({"id": pa.array([], type=pa.string()), "value": pa.array([], type=pa.float64())})
    path = os.path.join(tmpdir, "empty")
    deltalake.write_deltalake(path, table)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestFromDeltaLake:
    def test_basic_import(self, delta_table_path):
        """Import a Delta table and verify node count."""
        db = Database.from_delta_lake(delta_table_path, node_label="Customer")
        rows = db.query("MATCH (n:Customer) RETURN count(n) AS cnt")
        assert rows[0]["cnt"] == 5
        db.close()

    def test_properties_preserved(self, delta_table_path):
        """Verify node properties match source data."""
        db = Database.from_delta_lake(
            delta_table_path, node_label="Customer", primary_key="customer_id"
        )
        rows = db.query(
            "MATCH (n:Customer {customer_id: 'C001'}) RETURN n.name AS name, n.age AS age, n.balance AS balance"
        )
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice Smith"
        assert rows[0]["age"] == 30
        assert abs(rows[0]["balance"] - 1000.50) < 0.01
        db.close()

    def test_primary_key_auto_detection(self, delta_table_path):
        """Auto-detects customer_id as PK (ends with _id)."""
        db = Database.from_delta_lake(delta_table_path, node_label="Customer")
        # Should be queryable by PK
        rows = db.query("MATCH (n:Customer {customer_id: 'C003'}) RETURN n.name AS name")
        assert rows[0]["name"] == "Carol White"
        db.close()

    def test_explicit_primary_key(self, delta_table_path):
        """Explicit primary key overrides auto-detection."""
        db = Database.from_delta_lake(
            delta_table_path, node_label="Customer", primary_key="email"
        )
        rows = db.query(
            "MATCH (n:Customer {email: 'bob@example.com'}) RETURN n.name AS name"
        )
        assert rows[0]["name"] == "Bob Jones"
        db.close()

    def test_empty_table(self, empty_delta_table_path):
        """Empty Delta table creates schema but no nodes."""
        db = Database.from_delta_lake(empty_delta_table_path, node_label="Empty")
        rows = db.query("MATCH (n:Empty) RETURN count(n) AS cnt")
        assert rows[0]["cnt"] == 0
        db.close()

    def test_in_memory_database(self, delta_table_path):
        """Default db_path creates in-memory database."""
        db = Database.from_delta_lake(delta_table_path)
        rows = db.query("MATCH (n:Entity) RETURN count(n) AS cnt")
        assert rows[0]["cnt"] == 5
        db.close()

    def test_invalid_primary_key_raises(self, delta_table_path):
        """Specifying a non-existent PK column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            Database.from_delta_lake(delta_table_path, primary_key="nonexistent")

    def test_boolean_type_preserved(self, delta_table_path):
        """Boolean columns are properly typed."""
        db = Database.from_delta_lake(
            delta_table_path, node_label="Customer", primary_key="customer_id"
        )
        rows = db.query("MATCH (n:Customer {customer_id: 'C001'}) RETURN n.active AS active")
        assert rows[0]["active"] is True
        db.close()

    def test_round_trip_query(self, delta_table_path):
        """Import, query with filter, verify results."""
        db = Database.from_delta_lake(
            delta_table_path, node_label="Customer", primary_key="customer_id"
        )
        rows = db.query("MATCH (n:Customer) WHERE n.age > 40 RETURN n.name AS name ORDER BY n.name")
        names = [r["name"] for r in rows]
        assert names == ["Bob Jones", "Dave Brown"]
        db.close()


class TestFromDeltaLakeTypes:
    """Test Arrow type → Cypher type mapping."""

    def test_int_types(self):
        tmpdir = tempfile.mkdtemp(prefix="bridgr_delta_int_")
        try:
            table = pa.table({
                "id": pa.array(["A", "B"]),
                "int32_col": pa.array([1, 2], type=pa.int32()),
                "int64_col": pa.array([100, 200], type=pa.int64()),
            })
            path = os.path.join(tmpdir, "ints")
            deltalake.write_deltalake(path, table)
            db = Database.from_delta_lake(path, node_label="Typed", primary_key="id")
            rows = db.query("MATCH (n:Typed {id: 'A'}) RETURN n.int32_col AS i32, n.int64_col AS i64")
            assert rows[0]["i32"] == 1
            assert rows[0]["i64"] == 100
            db.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_date_timestamp(self):
        import datetime

        tmpdir = tempfile.mkdtemp(prefix="bridgr_delta_date_")
        try:
            table = pa.table({
                "id": pa.array(["X"]),
                "date_col": pa.array([datetime.date(2026, 5, 17)]),
                "ts_col": pa.array([datetime.datetime(2026, 5, 17, 10, 30, 0)]),
            })
            path = os.path.join(tmpdir, "dates")
            deltalake.write_deltalake(path, table)
            db = Database.from_delta_lake(path, node_label="Dated", primary_key="id")
            rows = db.query("MATCH (n:Dated {id: 'X'}) RETURN n.date_col AS d, n.ts_col AS ts")
            assert rows[0]["d"] is not None
            assert rows[0]["ts"] is not None
            db.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_nested_fallback_to_string(self):
        """Nested/struct types fall back to STRING."""
        tmpdir = tempfile.mkdtemp(prefix="bridgr_delta_nested_")
        try:
            table = pa.table({
                "id": pa.array(["R1", "R2"]),
                "tags": pa.array([["a", "b"], ["c"]], type=pa.list_(pa.string())),
            })
            path = os.path.join(tmpdir, "nested")
            deltalake.write_deltalake(path, table)
            db = Database.from_delta_lake(path, node_label="Tagged", primary_key="id")
            rows = db.query("MATCH (n:Tagged) RETURN count(n) AS cnt")
            assert rows[0]["cnt"] == 2
            db.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
