"""Snowflake connector — fetch tables as Arrow, bulk-import into Bridgr.

Usage:
    from bridgr.connectors.snowflake import from_snowflake

    stats = from_snowflake(
        db,
        account="xy12345.us-east-1",
        user="analyst",
        private_key_file="~/.ssh/snowflake_key.p8",
        warehouse="COMPUTE_WH",
        database="PROD",
        schema="PUBLIC",
        tables=["CUSTOMERS", "TRANSACTIONS"],
    )
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from bridgr.database import Database, _arrow_schema_to_cypher, _find_primary_key
from bridgr.export import _cypher_path


def _table_name_to_label(table_name: str) -> str:
    """Convert a SQL table name to a PascalCase node label.

    CUSTOMER_RECORDS -> CustomerRecords
    purchase_orders -> PurchaseOrders
    TRANSACTIONS -> Transactions
    """
    parts = table_name.split("_")
    return "".join(part.capitalize() for part in parts if part)


def _snowflake_connect(
    *,
    account: str,
    user: str,
    private_key_file: str | None = None,
    private_key_passphrase: str | None = None,
    password: str | None = None,
    warehouse: str | None = None,
    database: str | None = None,
    schema: str | None = None,
    role: str | None = None,
) -> Any:
    """Create a Snowflake connection with key pair auth (preferred) or password.

    Returns:
        snowflake.connector.Connection
    """
    if private_key_file is None:
        resolved_password = password or os.environ.get("SNOWFLAKE_PASSWORD")
        if not resolved_password:
            raise ValueError(
                "Snowflake credentials required. Pass private_key_file= (recommended) "
                "or password= or set SNOWFLAKE_PASSWORD env var."
            )

    try:
        import snowflake.connector
    except ImportError as e:
        raise ImportError(
            "snowflake-connector-python package required. "
            "Install with: pip install bridgr[snowflake]"
        ) from e

    connect_kwargs: dict[str, Any] = {
        "account": account,
        "user": user,
    }
    if warehouse is not None:
        connect_kwargs["warehouse"] = warehouse
    if database is not None:
        connect_kwargs["database"] = database
    if schema is not None:
        connect_kwargs["schema"] = schema
    if role is not None:
        connect_kwargs["role"] = role

    if private_key_file is not None:
        from cryptography.hazmat.primitives import serialization

        key_path = Path(private_key_file)
        if not key_path.exists():
            raise FileNotFoundError(f"Private key file not found: {private_key_file}")

        key_data = key_path.read_bytes()
        passphrase = private_key_passphrase.encode() if private_key_passphrase else None
        private_key = serialization.load_pem_private_key(key_data, password=passphrase)
        pkb = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        connect_kwargs["private_key"] = pkb
    else:
        connect_kwargs["password"] = resolved_password

    return snowflake.connector.connect(**connect_kwargs)


def from_snowflake(
    db: Database,
    *,
    account: str | None = None,
    user: str | None = None,
    password: str | None = None,
    private_key_file: str | None = None,
    private_key_passphrase: str | None = None,
    warehouse: str | None = None,
    database: str | None = None,
    schema: str | None = None,
    role: str | None = None,
    tables: list[str],
    node_label_map: dict[str, str] | None = None,
    primary_key_map: dict[str, str] | None = None,
    sql_filter_map: dict[str, str] | None = None,
    connection: Any | None = None,
) -> dict[str, Any]:
    """Import Snowflake tables as nodes into a Bridgr database.

    For each table: fetches as Arrow via cursor.fetch_arrow_all(), writes a
    temporary Parquet file, then bulk-imports via COPY FROM into Bridgr.

    Args:
        db: Target Bridgr Database instance.
        account: Snowflake account identifier (e.g., "xy12345.us-east-1").
        user: Snowflake username.
        password: Snowflake password. Falls back to SNOWFLAKE_PASSWORD env var.
        private_key_file: Path to PEM-encoded private key file (preferred over password).
        private_key_passphrase: Passphrase for encrypted private key.
        warehouse: Snowflake warehouse name.
        database: Snowflake database name.
        schema: Snowflake schema name.
        role: Snowflake role for scoped permissions.
        tables: List of table names to import.
        node_label_map: Override node labels. {table_name: label}.
        primary_key_map: Override primary keys. {table_name: column_name}.
        sql_filter_map: WHERE clauses to append. {table_name: "status = 'ACTIVE'"}.
        connection: Pre-built Snowflake connection (reused, not closed).

    Returns:
        Dict with per-table stats: {table: {rows, columns, label, primary_key}}.

    Raises:
        ImportError: If snowflake-connector-python is not installed.
        ValueError: If no credentials are provided.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    owns_connection = connection is None

    if owns_connection:
        connection = _snowflake_connect(
            account=account,
            user=user,
            private_key_file=private_key_file,
            private_key_passphrase=private_key_passphrase,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema,
            role=role,
        )

    node_label_map = node_label_map or {}
    primary_key_map = primary_key_map or {}
    sql_filter_map = sql_filter_map or {}

    stats: dict[str, Any] = {}

    tmp_files: list[str] = []

    try:
        cursor = connection.cursor()
        for table in tables:
            label = node_label_map.get(table, _table_name_to_label(table))
            sql = f"SELECT * FROM {table}"
            where_clause = sql_filter_map.get(table)
            if where_clause:
                sql += f" WHERE {where_clause}"

            cursor.execute(sql)
            arrow_table: pa.Table = cursor.fetch_arrow_all()

            if arrow_table is None or arrow_table.num_rows == 0:
                stats[table] = {"rows": 0, "columns": 0, "label": label, "primary_key": ""}
                continue

            pk_col = primary_key_map.get(table)
            if pk_col is None:
                pk_col = _find_primary_key(arrow_table.schema, label_hint=label)

            if pk_col and pk_col not in arrow_table.column_names:
                lower_map = {c.lower(): c for c in arrow_table.column_names}
                if pk_col.lower() in lower_map:
                    pk_col = lower_map[pk_col.lower()]
                else:
                    pk_col = None

            if pk_col is None:
                pk_col = "_row_id"
                ids = pa.array([str(i) for i in range(arrow_table.num_rows)])
                arrow_table = arrow_table.append_column("_row_id", ids)

            props = _arrow_schema_to_cypher(arrow_table.schema, pk_col)
            db.create_node_table(label, props)

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                tmp_path = f.name
            tmp_files.append(tmp_path)

            pq.write_table(arrow_table, tmp_path)
            cp = _cypher_path(tmp_path)
            db.execute(f'COPY {label} FROM "{cp}"')

            stats[table] = {
                "rows": arrow_table.num_rows,
                "columns": arrow_table.num_columns,
                "label": label,
                "primary_key": pk_col,
            }

        cursor.close()
    finally:
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass

        if owns_connection and connection is not None:
            connection.close()

    return stats
