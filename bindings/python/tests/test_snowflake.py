"""Tests for Snowflake connector — all tests use mocked Snowflake connections."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bridgr.database import Database


TEST_DIR = Path(__file__).parent / "test_snowflake_dbs"


@pytest.fixture(autouse=True)
def clean_test_dir():
    """Create and clean up test database directory."""
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR))
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR), ignore_errors=True)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_connection(tables: dict[str, pa.Table]) -> MagicMock:
    """Build a mock Snowflake connection.

    Args:
        tables: Mapping of table_name -> PyArrow table to return from fetch_arrow_all().
    """
    conn = MagicMock()
    cursor = MagicMock()

    # Track SQL statements executed
    executed_sqls: list[str] = []

    def mock_execute(sql):
        executed_sqls.append(sql)
        # Set the "current table" for the next fetch_arrow_all call
        for table_name, arrow_table in tables.items():
            if table_name in sql:
                cursor._current_table = arrow_table
                return
        cursor._current_table = None

    cursor.execute = mock_execute
    cursor.executed_sqls = executed_sqls

    def mock_fetch_arrow_all():
        return getattr(cursor, "_current_table", None)

    cursor.fetch_arrow_all = mock_fetch_arrow_all
    conn.cursor.return_value = cursor
    conn.close = MagicMock()
    return conn


def _customers_table() -> pa.Table:
    """Standard 3-row customer table for tests."""
    return pa.table({
        "customer_id": pa.array(["C001", "C002", "C003"]),
        "name": pa.array(["Alice", "Bob", "Carol"]),
        "email": pa.array(["alice@ex.com", "bob@ex.com", "carol@ex.com"]),
        "balance": pa.array([1000.0, 2500.0, 750.0]),
    })


def _transactions_table() -> pa.Table:
    """Standard 4-row transactions table for tests."""
    return pa.table({
        "txn_id": pa.array(["T001", "T002", "T003", "T004"]),
        "customer_id": pa.array(["C001", "C001", "C002", "C003"]),
        "amount": pa.array([100.0, 200.0, 150.0, 300.0]),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSnowflakeConnector:
    """Tests for bridgr.connectors.snowflake.from_snowflake()."""

    def test_single_table_import(self):
        """Import 1 table, verify nodes exist in the database."""
        from bridgr.connectors.snowflake import from_snowflake

        db = Database(str(TEST_DIR / "single.lbug"))
        mock_conn = _make_mock_connection({"CUSTOMERS": _customers_table()})

        stats = from_snowflake(
            db,
            tables=["CUSTOMERS"],
            connection=mock_conn,
        )

        assert stats["CUSTOMERS"]["rows"] == 3
        assert stats["CUSTOMERS"]["columns"] == 4
        assert stats["CUSTOMERS"]["label"] == "Customers"
        assert stats["CUSTOMERS"]["primary_key"] == "customer_id"
        rows = db.query("MATCH (n:Customers) RETURN count(n) AS cnt")
        assert rows[0]["cnt"] == 3
        db.close()

    def test_multi_table_import(self):
        """Import 2 tables, verify both node labels exist."""
        from bridgr.connectors.snowflake import from_snowflake

        db = Database(str(TEST_DIR / "multi.lbug"))
        mock_conn = _make_mock_connection({
            "CUSTOMERS": _customers_table(),
            "TRANSACTIONS": _transactions_table(),
        })

        stats = from_snowflake(
            db,
            tables=["CUSTOMERS", "TRANSACTIONS"],
            primary_key_map={"TRANSACTIONS": "txn_id"},
            connection=mock_conn,
        )

        assert stats["CUSTOMERS"]["rows"] == 3
        assert stats["TRANSACTIONS"]["rows"] == 4
        cust_count = db.query("MATCH (n:Customers) RETURN count(n) AS cnt")
        assert cust_count[0]["cnt"] == 3
        txn_count = db.query("MATCH (n:Transactions) RETURN count(n) AS cnt")
        assert txn_count[0]["cnt"] == 4
        db.close()

    def test_custom_label_map(self):
        """node_label_map overrides the default PascalCase label."""
        from bridgr.connectors.snowflake import from_snowflake

        db = Database(str(TEST_DIR / "label.lbug"))
        mock_conn = _make_mock_connection({"CUSTOMERS": _customers_table()})

        stats = from_snowflake(
            db,
            tables=["CUSTOMERS"],
            node_label_map={"CUSTOMERS": "Client"},
            connection=mock_conn,
        )

        assert stats["CUSTOMERS"]["label"] == "Client"
        rows = db.query("MATCH (n:Client) RETURN count(n) AS cnt")
        assert rows[0]["cnt"] == 3
        db.close()

    def test_custom_primary_key(self):
        """primary_key_map overrides auto-detection."""
        from bridgr.connectors.snowflake import from_snowflake

        db = Database(str(TEST_DIR / "pk.lbug"))
        mock_conn = _make_mock_connection({"CUSTOMERS": _customers_table()})

        stats = from_snowflake(
            db,
            tables=["CUSTOMERS"],
            primary_key_map={"CUSTOMERS": "email"},
            connection=mock_conn,
        )

        # Should be queryable by email as PK
        rows = db.query("MATCH (n:Customers {email: 'alice@ex.com'}) RETURN n.name AS name")
        assert rows[0]["name"] == "Alice"
        db.close()

    def test_sql_filter(self):
        """WHERE clause appended via sql_filter_map."""
        from bridgr.connectors.snowflake import from_snowflake

        db = Database(str(TEST_DIR / "filter.lbug"))

        # The mock returns all rows regardless, but we verify the SQL was built correctly
        mock_conn = _make_mock_connection({"CUSTOMERS": _customers_table()})

        from_snowflake(
            db,
            tables=["CUSTOMERS"],
            sql_filter_map={"CUSTOMERS": "balance > 500"},
            connection=mock_conn,
        )

        cursor = mock_conn.cursor()
        # Check that the SQL included the WHERE clause
        assert any("WHERE balance > 500" in sql for sql in cursor.executed_sqls)
        db.close()

    def test_empty_table_skipped(self):
        """0 rows handled gracefully — table marked as skipped."""
        from bridgr.connectors.snowflake import from_snowflake

        db = Database(str(TEST_DIR / "empty.lbug"))
        empty = pa.table({
            "id": pa.array([], type=pa.string()),
            "value": pa.array([], type=pa.float64()),
        })
        mock_conn = _make_mock_connection({"EMPTY_TABLE": empty})

        stats = from_snowflake(
            db,
            tables=["EMPTY_TABLE"],
            connection=mock_conn,
        )

        assert stats["EMPTY_TABLE"]["rows"] == 0
        db.close()

    def test_synthetic_pk_when_none_detected(self):
        """_row_id added when no PK column is detectable."""
        from bridgr.connectors.snowflake import from_snowflake

        db = Database(str(TEST_DIR / "synth_pk.lbug"))
        # Table with ONLY float/bool columns — no string/int, no _id/key/pk patterns
        # This forces _find_primary_key to return None, triggering synthetic _row_id
        table = pa.table({
            "temperature": pa.array([98.6, 99.1, 97.8]),
            "pressure": pa.array([1.0, 2.0, 3.0]),
            "valid": pa.array([True, False, True]),
        })
        mock_conn = _make_mock_connection({"READINGS": table})

        stats = from_snowflake(
            db,
            tables=["READINGS"],
            connection=mock_conn,
        )

        assert stats["READINGS"]["rows"] == 3
        assert stats["READINGS"]["primary_key"] == "_row_id"
        rows = db.query("MATCH (n:Readings) RETURN n._row_id AS rid")
        row_ids = [r["rid"] for r in rows]
        assert "0" in row_ids
        assert "1" in row_ids
        assert "2" in row_ids
        db.close()

    def test_uppercase_column_pk_matching(self):
        """CUSTOMER_ID matches as PK (Snowflake uppercase convention)."""
        from bridgr.connectors.snowflake import from_snowflake

        db = Database(str(TEST_DIR / "upper.lbug"))
        # Snowflake returns uppercase column names by default
        table = pa.table({
            "CUSTOMER_ID": pa.array(["C001", "C002"]),
            "NAME": pa.array(["Alice", "Bob"]),
        })
        mock_conn = _make_mock_connection({"CUSTOMERS": table})

        stats = from_snowflake(
            db,
            tables=["CUSTOMERS"],
            connection=mock_conn,
        )

        assert stats["CUSTOMERS"]["rows"] == 2
        rows = db.query("MATCH (n:Customers {CUSTOMER_ID: 'C001'}) RETURN n.NAME AS name")
        assert rows[0]["name"] == "Alice"
        db.close()

    def test_connection_reuse(self):
        """Pre-built connection is not closed by from_snowflake."""
        from bridgr.connectors.snowflake import from_snowflake

        db = Database(str(TEST_DIR / "reuse.lbug"))
        mock_conn = _make_mock_connection({"CUSTOMERS": _customers_table()})

        from_snowflake(
            db,
            tables=["CUSTOMERS"],
            connection=mock_conn,
        )

        # Connection should NOT be closed when passed in
        mock_conn.close.assert_not_called()
        db.close()

    def test_password_from_env(self, monkeypatch):
        """SNOWFLAKE_PASSWORD env var is used when password= not provided."""
        from bridgr.connectors.snowflake import from_snowflake

        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "s3cr3t")

        db = Database(str(TEST_DIR / "env_pw.lbug"))

        # Create a mock snowflake.connector module with connect() that returns
        # a working mock connection
        mock_sf_connector = MagicMock()
        mock_conn = _make_mock_connection({"CUSTOMERS": _customers_table()})
        mock_sf_connector.connect.return_value = mock_conn

        # Patch at the module level — after the password check passes, the code
        # does `import snowflake.connector` which consults sys.modules
        mock_sf_pkg = MagicMock()
        mock_sf_pkg.connector = mock_sf_connector

        with patch.dict(sys.modules, {
            "snowflake": mock_sf_pkg,
            "snowflake.connector": mock_sf_connector,
        }):
            from_snowflake(
                db,
                account="test_acct",
                user="test_user",
                warehouse="WH",
                database="DB",
                schema="PUBLIC",
                tables=["CUSTOMERS"],
            )

            mock_sf_connector.connect.assert_called_once_with(
                account="test_acct",
                user="test_user",
                password="s3cr3t",
                warehouse="WH",
                database="DB",
                schema="PUBLIC",
            )
        db.close()

    def test_no_password_raises(self, monkeypatch):
        """ValueError raised when no password provided and env var unset."""
        from bridgr.connectors.snowflake import from_snowflake

        monkeypatch.delenv("SNOWFLAKE_PASSWORD", raising=False)

        db = Database(str(TEST_DIR / "no_pw.lbug"))

        with pytest.raises(ValueError, match="Snowflake password required"):
            from_snowflake(
                db,
                account="test_acct",
                user="test_user",
                tables=["CUSTOMERS"],
            )
        db.close()

    def test_import_error_without_package(self, monkeypatch):
        """ImportError raised with pip hint when snowflake-connector is missing."""
        from bridgr.connectors.snowflake import from_snowflake

        db = Database(str(TEST_DIR / "no_pkg.lbug"))

        monkeypatch.delenv("SNOWFLAKE_PASSWORD", raising=False)

        # Temporarily remove snowflake from sys.modules to simulate missing package
        saved = sys.modules.get("snowflake")
        saved_connector = sys.modules.get("snowflake.connector")
        sys.modules["snowflake"] = None  # type: ignore[assignment]
        sys.modules["snowflake.connector"] = None  # type: ignore[assignment]

        try:
            # No connection= passed, so it tries to import snowflake.connector
            with pytest.raises(ImportError, match="pip install bridgr\\[snowflake\\]"):
                from_snowflake(
                    db,
                    account="test_acct",
                    user="test_user",
                    password="secret",
                    tables=["CUSTOMERS"],
                )
        finally:
            # Restore
            if saved is not None:
                sys.modules["snowflake"] = saved
            else:
                sys.modules.pop("snowflake", None)
            if saved_connector is not None:
                sys.modules["snowflake.connector"] = saved_connector
            else:
                sys.modules.pop("snowflake.connector", None)
        db.close()

    def test_temp_files_cleaned_up(self):
        """No leftover parquet files after import completes."""
        from bridgr.connectors.snowflake import from_snowflake

        db = Database(str(TEST_DIR / "cleanup.lbug"))
        mock_conn = _make_mock_connection({"CUSTOMERS": _customers_table()})

        # Track temp directory before
        temp_dir = tempfile.gettempdir()
        parquets_before = set(
            f for f in os.listdir(temp_dir) if f.endswith(".parquet")
        )

        from_snowflake(
            db,
            tables=["CUSTOMERS"],
            connection=mock_conn,
        )

        # Check no new parquet files remain
        parquets_after = set(
            f for f in os.listdir(temp_dir) if f.endswith(".parquet")
        )
        new_parquets = parquets_after - parquets_before
        assert len(new_parquets) == 0, f"Leftover temp files: {new_parquets}"
        db.close()

    def test_default_label_removes_underscores(self):
        """CUSTOMER_RECORDS -> CustomerRecords (PascalCase from underscored table name)."""
        from bridgr.connectors.snowflake import _table_name_to_label

        assert _table_name_to_label("CUSTOMER_RECORDS") == "CustomerRecords"
        assert _table_name_to_label("purchase_orders") == "PurchaseOrders"
        assert _table_name_to_label("TRANSACTIONS") == "Transactions"
        assert _table_name_to_label("order_line_items") == "OrderLineItems"
