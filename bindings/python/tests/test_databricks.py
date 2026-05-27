"""Tests for Databricks connector (engine, MIT) — read + auth only, all mocked."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from bridgr.database import Database


TEST_DIR = Path(__file__).parent / "test_databricks_dbs"


@pytest.fixture(autouse=True)
def clean_test_dir():
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR), ignore_errors=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR), ignore_errors=True)


def _make_mock_connection(tables: dict[str, pa.Table]) -> MagicMock:
    conn = MagicMock()
    cursor = MagicMock()
    executed_sqls: list[str] = []

    def mock_execute(sql):
        executed_sqls.append(sql)
        for table_name, arrow_table in tables.items():
            if table_name in sql:
                cursor._current_table = arrow_table
                return
        cursor._current_table = None

    cursor.execute = mock_execute
    cursor.executed_sqls = executed_sqls
    cursor.fetchall_arrow = lambda: getattr(cursor, "_current_table", None)
    conn.cursor.return_value = cursor
    conn.close = MagicMock()
    return conn


def _customers_table() -> pa.Table:
    return pa.table({
        "customer_id": pa.array(["C001", "C002", "C003"]),
        "name": pa.array(["Alice", "Bob", "Carol"]),
        "balance": pa.array([1000.0, 2500.0, 750.0]),
    })


def _transactions_table() -> pa.Table:
    return pa.table({
        "txn_id": pa.array(["T001", "T002", "T003"]),
        "customer_id": pa.array(["C001", "C001", "C002"]),
        "amount": pa.array([100.0, 200.0, 150.0]),
    })


# ===========================================================================
# from_databricks() tests (12)
# ===========================================================================


class TestFromDatabricks:

    def test_single_table(self):
        from bridgr.connectors.databricks import from_databricks

        db = Database(str(TEST_DIR / "single.lbug"))
        mock_conn = _make_mock_connection({"customers": _customers_table()})

        stats = from_databricks(db, tables=["customers"], connection=mock_conn)

        assert stats["customers"]["rows"] == 3
        assert stats["customers"]["label"] == "Customers"
        assert stats["customers"]["primary_key"] == "customer_id"
        db.close()

    def test_multi_table(self):
        from bridgr.connectors.databricks import from_databricks

        db = Database(str(TEST_DIR / "multi.lbug"))
        mock_conn = _make_mock_connection({
            "customers": _customers_table(),
            "transactions": _transactions_table(),
        })

        stats = from_databricks(
            db, tables=["customers", "transactions"],
            primary_key_map={"transactions": "txn_id"},
            connection=mock_conn,
        )

        assert stats["customers"]["rows"] == 3
        assert stats["transactions"]["rows"] == 3
        db.close()

    def test_label_map(self):
        from bridgr.connectors.databricks import from_databricks

        db = Database(str(TEST_DIR / "label.lbug"))
        mock_conn = _make_mock_connection({"customers": _customers_table()})

        stats = from_databricks(
            db, tables=["customers"], node_label_map={"customers": "Person"},
            connection=mock_conn,
        )

        assert stats["customers"]["label"] == "Person"
        db.close()

    def test_pk_map(self):
        from bridgr.connectors.databricks import from_databricks

        db = Database(str(TEST_DIR / "pk.lbug"))
        mock_conn = _make_mock_connection({"customers": _customers_table()})

        from_databricks(
            db, tables=["customers"], primary_key_map={"customers": "name"},
            connection=mock_conn,
        )

        rows = db.query("MATCH (n:Customers {name: 'Alice'}) RETURN n.balance AS bal")
        assert rows[0]["bal"] == 1000.0
        db.close()

    def test_filter_map(self):
        from bridgr.connectors.databricks import from_databricks

        db = Database(str(TEST_DIR / "filter.lbug"))
        mock_conn = _make_mock_connection({"customers": _customers_table()})

        from_databricks(
            db, tables=["customers"], sql_filter_map={"customers": "balance > 1000"},
            connection=mock_conn,
        )

        cursor = mock_conn.cursor()
        assert any("WHERE balance > 1000" in sql for sql in cursor.executed_sqls)
        db.close()

    def test_empty_table(self):
        from bridgr.connectors.databricks import from_databricks

        db = Database(str(TEST_DIR / "empty.lbug"))
        empty = pa.table({"id": pa.array([], type=pa.string())})
        mock_conn = _make_mock_connection({"empty": empty})

        stats = from_databricks(db, tables=["empty"], connection=mock_conn)

        assert stats["empty"]["rows"] == 0
        db.close()

    def test_casing_preserved(self):
        from bridgr.connectors.databricks import from_databricks

        db = Database(str(TEST_DIR / "case.lbug"))
        table = pa.table({
            "customerId": pa.array(["C001"]),
            "Full_Name": pa.array(["Alice"]),
        })
        mock_conn = _make_mock_connection({"data": table})

        from_databricks(
            db, tables=["data"], primary_key_map={"data": "customerId"},
            connection=mock_conn,
        )

        rows = db.query("MATCH (n:Data {customerId: 'C001'}) RETURN n.Full_Name AS name")
        assert rows[0]["name"] == "Alice"
        db.close()

    def test_connection_reuse(self):
        from bridgr.connectors.databricks import from_databricks

        db = Database(str(TEST_DIR / "reuse.lbug"))
        mock_conn = _make_mock_connection({"customers": _customers_table()})

        from_databricks(db, tables=["customers"], connection=mock_conn)

        mock_conn.close.assert_not_called()
        db.close()

    def test_synthetic_pk(self):
        from bridgr.connectors.databricks import from_databricks

        db = Database(str(TEST_DIR / "synth.lbug"))
        table = pa.table({"temp": pa.array([1.0, 2.0])})
        mock_conn = _make_mock_connection({"readings": table})

        stats = from_databricks(db, tables=["readings"], connection=mock_conn)

        assert stats["readings"]["primary_key"] == "_row_id"
        db.close()

    def test_temp_file_cleanup(self):
        from bridgr.connectors.databricks import from_databricks

        db = Database(str(TEST_DIR / "tmp.lbug"))
        mock_conn = _make_mock_connection({"customers": _customers_table()})

        temp_dir = tempfile.gettempdir()
        before = set(f for f in os.listdir(temp_dir) if f.endswith(".parquet"))

        from_databricks(db, tables=["customers"], connection=mock_conn)

        after = set(f for f in os.listdir(temp_dir) if f.endswith(".parquet"))
        assert len(after - before) == 0
        db.close()

    def test_fq_table_with_catalog(self):
        from bridgr.connectors.databricks import from_databricks

        db = Database(str(TEST_DIR / "fq.lbug"))
        mock_conn = _make_mock_connection({"customers": _customers_table()})

        from_databricks(
            db, tables=["customers"], catalog="main", schema="prod",
            connection=mock_conn,
        )

        cursor = mock_conn.cursor()
        assert any("main.prod.customers" in sql for sql in cursor.executed_sqls)
        db.close()

    def test_label_transform(self):
        from bridgr.connectors.databricks import _table_name_to_label

        assert _table_name_to_label("customer_records") == "CustomerRecords"
        assert _table_name_to_label("orders") == "Orders"
        assert _table_name_to_label("line_item_details") == "LineItemDetails"


# ===========================================================================
# Auth tests (5)
# ===========================================================================


class TestDatabricksAuth:

    def test_oauth_m2m(self):
        from bridgr.connectors.databricks import _databricks_connect

        mock_dbsql = MagicMock()
        mock_core = MagicMock()
        mock_core.Config.return_value = "cfg_obj"
        mock_core.oauth_service_principal.return_value = "provider"
        mock_db = MagicMock()
        mock_db.sql = mock_dbsql

        with patch.dict(sys.modules, {
            "databricks": mock_db,
            "databricks.sql": mock_dbsql,
            "databricks.sdk": MagicMock(),
            "databricks.sdk.core": mock_core,
        }):
            _databricks_connect(
                server_hostname="h", http_path="/p",
                client_id="c", client_secret="s",
            )

        mock_core.Config.assert_called_once_with(
            host="https://h", client_id="c", client_secret="s",
        )
        mock_core.oauth_service_principal.assert_called_once_with("cfg_obj")
        kw = mock_dbsql.connect.call_args[1]
        assert kw.get("credentials_provider") == "provider"

    def test_pat(self):
        from bridgr.connectors.databricks import _databricks_connect

        mock_dbsql = MagicMock()
        mock_db = MagicMock()
        mock_db.sql = mock_dbsql

        with patch.dict(sys.modules, {
            "databricks": mock_db,
            "databricks.sql": mock_dbsql,
        }):
            _databricks_connect(
                server_hostname="h", http_path="/p", access_token="tok",
            )

        kw = mock_dbsql.connect.call_args[1]
        assert kw["access_token"] == "tok"

    def test_env_fallback(self, monkeypatch):
        from bridgr.connectors.databricks import _databricks_connect

        monkeypatch.setenv("DATABRICKS_TOKEN", "envtok")
        mock_dbsql = MagicMock()
        mock_db = MagicMock()
        mock_db.sql = mock_dbsql

        with patch.dict(sys.modules, {
            "databricks": mock_db,
            "databricks.sql": mock_dbsql,
        }):
            _databricks_connect(server_hostname="h", http_path="/p")

        kw = mock_dbsql.connect.call_args[1]
        assert kw["access_token"] == "envtok"

    def test_no_creds_raises(self, monkeypatch):
        from bridgr.connectors.databricks import _databricks_connect

        monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
        mock_dbsql = MagicMock()

        with patch.dict(sys.modules, {
            "databricks": MagicMock(),
            "databricks.sql": mock_dbsql,
        }):
            with pytest.raises(ValueError, match="credentials required"):
                _databricks_connect(server_hostname="h", http_path="/p")

    def test_missing_package(self):
        from bridgr.connectors.databricks import _databricks_connect

        saved = sys.modules.get("databricks")
        saved_sql = sys.modules.get("databricks.sql")
        sys.modules["databricks"] = None  # type: ignore[assignment]
        sys.modules["databricks.sql"] = None  # type: ignore[assignment]

        try:
            with pytest.raises(ImportError, match="pip install"):
                _databricks_connect(
                    server_hostname="h", http_path="/p", access_token="t",
                )
        finally:
            if saved is not None:
                sys.modules["databricks"] = saved
            else:
                sys.modules.pop("databricks", None)
            if saved_sql is not None:
                sys.modules["databricks.sql"] = saved_sql
            else:
                sys.modules.pop("databricks.sql", None)
