"""Databricks connector — fetch Unity Catalog tables via SQL Warehouse into Bridgr.

Usage:
    from bridgr.connectors.databricks import from_databricks

    stats = from_databricks(
        db,
        server_hostname="adb-1234567890.12.azuredatabricks.net",
        http_path="/sql/1.0/warehouses/abc123",
        access_token="dapi...",
        catalog="main",
        tables=["customers", "transactions"],
    )
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from bridgr.database import Database, _arrow_schema_to_cypher, _find_primary_key
from bridgr.export import _cypher_path


def _table_name_to_label(table_name: str) -> str:
    """Convert a SQL table name to a PascalCase node label."""
    parts = table_name.split("_")
    return "".join(part.capitalize() for part in parts if part)


def _databricks_connect(
    *,
    server_hostname: str,
    http_path: str,
    access_token: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    catalog: str | None = None,
    schema: str | None = None,
) -> Any:
    """Create a Databricks SQL connection.

    Auth priority: OAuth M2M (client_id + client_secret) > PAT (access_token or
    DATABRICKS_TOKEN env var).

    Returns:
        databricks.sql.Connection
    """
    try:
        from databricks import sql as databricks_sql
    except ImportError as e:
        raise ImportError(
            "databricks-sql-connector package required. "
            "Install with: pip install bridgr[databricks]"
        ) from e

    connect_kwargs: dict[str, Any] = {
        "server_hostname": server_hostname,
        "http_path": http_path,
    }
    if catalog is not None:
        connect_kwargs["catalog"] = catalog
    if schema is not None:
        connect_kwargs["schema"] = schema

    if client_id is not None and client_secret is not None:
        from databricks.sdk.core import Config, oauth_service_principal

        cfg = Config(
            host=f"https://{server_hostname}",
            client_id=client_id,
            client_secret=client_secret,
        )
        connect_kwargs["credentials_provider"] = oauth_service_principal(cfg)
    else:
        token = access_token or os.environ.get("DATABRICKS_TOKEN")
        if not token:
            raise ValueError(
                "Databricks credentials required. Pass client_id + client_secret "
                "(OAuth M2M, recommended) or access_token or set DATABRICKS_TOKEN env var."
            )
        connect_kwargs["access_token"] = token

    return databricks_sql.connect(**connect_kwargs)


def from_databricks(
    db: Database,
    *,
    server_hostname: str | None = None,
    http_path: str | None = None,
    access_token: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    catalog: str | None = None,
    schema: str = "default",
    tables: list[str],
    node_label_map: dict[str, str] | None = None,
    primary_key_map: dict[str, str] | None = None,
    sql_filter_map: dict[str, str] | None = None,
    connection: Any | None = None,
) -> dict[str, Any]:
    """Import Databricks tables as nodes into a Bridgr database.

    Column casing is preserved as-is (unlike Snowflake's uppercase convention).

    Args:
        db: Target Bridgr Database instance.
        server_hostname: Databricks workspace hostname.
        http_path: SQL Warehouse HTTP path.
        access_token: PAT token. Falls back to DATABRICKS_TOKEN env var.
        client_id: OAuth M2M client ID (preferred over PAT).
        client_secret: OAuth M2M client secret.
        catalog: Unity Catalog name.
        schema: Schema name (default: "default").
        tables: List of table names to import.
        node_label_map: Override node labels. {table_name: label}.
        primary_key_map: Override primary keys. {table_name: column_name}.
        sql_filter_map: WHERE clauses to append. {table_name: "status = 'ACTIVE'"}.
        connection: Pre-built Databricks SQL connection (reused, not closed).

    Returns:
        Dict with per-table stats: {table: {rows, columns, label, primary_key}}.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    owns_connection = connection is None

    if owns_connection:
        connection = _databricks_connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
            client_id=client_id,
            client_secret=client_secret,
            catalog=catalog,
            schema=schema,
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

            fq_table = table
            if "." not in table and catalog:
                fq_table = f"{catalog}.{schema}.{table}"

            sql = f"SELECT * FROM {fq_table}"
            where_clause = sql_filter_map.get(table)
            if where_clause:
                sql += f" WHERE {where_clause}"

            cursor.execute(sql)
            arrow_table: pa.Table = cursor.fetchall_arrow()

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
