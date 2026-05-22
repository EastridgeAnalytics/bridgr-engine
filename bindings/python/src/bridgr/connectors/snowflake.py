"""Snowflake connector — fetch tables as Arrow, bulk-import into Bridgr.

Usage:
    from bridgr.connectors.snowflake import from_snowflake

    stats = from_snowflake(
        db,
        account="xy12345.us-east-1",
        user="analyst",
        warehouse="COMPUTE_WH",
        database="PROD",
        schema="PUBLIC",
        tables=["CUSTOMERS", "TRANSACTIONS"],
    )
"""

from __future__ import annotations

import os
import tempfile
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


def from_snowflake(
    db: Database,
    *,
    account: str | None = None,
    user: str | None = None,
    password: str | None = None,
    warehouse: str | None = None,
    database: str | None = None,
    schema: str | None = None,
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
        warehouse: Snowflake warehouse name.
        database: Snowflake database name.
        schema: Snowflake schema name.
        tables: List of table names to import.
        node_label_map: Override node labels. {table_name: label}.
        primary_key_map: Override primary keys. {table_name: column_name}.
        sql_filter_map: WHERE clauses to append. {table_name: "status = 'ACTIVE'"}.
        connection: Pre-built Snowflake connection (reused, not closed).

    Returns:
        Dict with: tables_imported, total_rows, per_table stats.

    Raises:
        ImportError: If snowflake-connector-python is not installed.
        ValueError: If no password is provided and SNOWFLAKE_PASSWORD is unset.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Resolve password
    resolved_password = password or os.environ.get("SNOWFLAKE_PASSWORD")
    owns_connection = connection is None

    if owns_connection:
        if not resolved_password:
            raise ValueError(
                "Snowflake password required. Pass password= or set SNOWFLAKE_PASSWORD env var."
            )

        try:
            import snowflake.connector
        except ImportError as e:
            raise ImportError(
                "snowflake-connector-python package required. "
                "Install with: pip install bridgr[snowflake]"
            ) from e

        connection = snowflake.connector.connect(
            account=account,
            user=user,
            password=resolved_password,
            warehouse=warehouse,
            database=database,
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
            sql = f"SELECT * FROM {table}"
            where_clause = sql_filter_map.get(table)
            if where_clause:
                sql += f" WHERE {where_clause}"

            cursor.execute(sql)
            arrow_table: pa.Table = cursor.fetch_arrow_all()

            if arrow_table is None or arrow_table.num_rows == 0:
                stats[table] = {"rows": 0, "columns": 0, "label": label, "primary_key": ""}
                continue

            # Determine primary key
            pk_col = primary_key_map.get(table)
            if pk_col is None:
                # Try case-insensitive match for uppercase Snowflake columns
                pk_col = _find_primary_key(arrow_table.schema, label_hint=label)

            if pk_col and pk_col not in arrow_table.column_names:
                # Try case-insensitive fallback for Snowflake uppercase
                lower_map = {c.lower(): c for c in arrow_table.column_names}
                if pk_col.lower() in lower_map:
                    pk_col = lower_map[pk_col.lower()]
                else:
                    pk_col = None

            # Synthetic PK if nothing detected
            if pk_col is None:
                pk_col = "_row_id"
                ids = pa.array([str(i) for i in range(arrow_table.num_rows)])
                arrow_table = arrow_table.append_column("_row_id", ids)

            # Map schema and create node table
            props = _arrow_schema_to_cypher(arrow_table.schema, pk_col)
            db.create_node_table(label, props)

            # Write temp Parquet, COPY FROM
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
        # Clean up temp files
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass

        # Close connection only if we created it
        if owns_connection and connection is not None:
            connection.close()

    return stats
