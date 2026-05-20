"""Bridgr Export/Import — Parquet, CSV, and Delta Lake data interchange.

Wraps LadybugDB's COPY TO/FROM for bulk data export and import.
Supports Parquet (columnar, efficient), CSV (universal), and Delta Lake
(versioned lakehouse format).

Usage:
    db = bridgr.open("case.lbug")
    from bridgr.export import DataExporter, to_delta_lake, query_to_delta_lake

    # Class-based API
    exporter = DataExporter(db)
    exporter.to_parquet("Entity", "entities.parquet")
    exporter.to_csv("Fact", "facts.csv")
    exporter.from_csv("Entity", "new_entities.csv")

    # Delta Lake standalone functions
    result = to_delta_lake(db, "Entity", "/data/entities_delta")
    result = query_to_delta_lake(db, "MATCH (e:Entity) RETURN e.*", "/data/query_delta")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bridgr.database import Database

__all__ = ["DataExporter", "to_delta_lake", "query_to_delta_lake"]


def _cypher_path(path: str) -> str:
    """Convert a filesystem path to a Cypher-safe string (forward slashes, double-quoted)."""
    return str(Path(path).resolve()).replace("\\", "/")


class DataExporter:
    """Bulk data export and import for a Bridgr database."""

    def __init__(self, db: Database):
        self._db = db

    def to_parquet(self, label: str, path: str) -> int:
        """Export all nodes of a type to a Parquet file.

        Returns the number of rows exported.
        """
        cp = _cypher_path(path)
        self._db.execute(
            f'COPY (MATCH (n:{label}) RETURN n.*) TO "{cp}"'
        )
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return table.num_rows

    def to_csv(self, label: str, path: str, *, header: bool = True) -> int:
        """Export all nodes of a type to a CSV file.

        Returns the number of rows exported.
        """
        cp = _cypher_path(path)
        header_str = "true" if header else "false"
        self._db.execute(
            f'COPY (MATCH (n:{label}) RETURN n.*) TO "{cp}" (header={header_str})'
        )
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                count += 1
        if header:
            count -= 1
        return count

    def from_csv(self, label: str, path: str) -> None:
        """Import nodes from a CSV file into a node table.

        The CSV must have a header row matching the table's column names.
        """
        cp = _cypher_path(path)
        self._db.execute(f'COPY {label} FROM "{cp}"')

    def from_parquet(self, label: str, path: str) -> None:
        """Import nodes from a Parquet file into a node table."""
        cp = _cypher_path(path)
        self._db.execute(f'COPY {label} FROM "{cp}"')

    def query_to_parquet(self, cypher: str, path: str, params: dict[str, Any] | None = None) -> int:
        """Export the result of any Cypher query to Parquet."""
        cp = _cypher_path(path)
        self._db.execute(f'COPY ({cypher}) TO "{cp}"')
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return table.num_rows

    def query_to_csv(self, cypher: str, path: str) -> None:
        """Export the result of any Cypher query to CSV."""
        cp = _cypher_path(path)
        self._db.execute(f'COPY ({cypher}) TO "{cp}" (header=true)')

    def to_delta_lake(
        self,
        label: str,
        path: str,
        *,
        mode: str = "overwrite",
        partition_by: list[str] | None = None,
    ) -> dict[str, Any]:
        """Export all nodes of a type to a Delta Lake table.

        Args:
            label: Node type to export (e.g., "Customer").
            path: Delta Lake table path (local, s3://, gs://, az://).
            mode: Write mode — "overwrite" or "append".
            partition_by: Optional list of columns to partition by.

        Returns:
            Dict with: rows_written, delta_path, columns.

        Requires:
            pip install bridgr[deltalake]
        """
        try:
            import deltalake
        except ImportError as e:
            raise ImportError(
                "deltalake package required. Install with: pip install bridgr[deltalake]"
            ) from e

        import tempfile
        import os
        import pyarrow.parquet as pq

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp_path = f.name
        try:
            cp = _cypher_path(tmp_path)
            self._db.execute(f'COPY (MATCH (n:{label}) RETURN n.*) TO "{cp}"')
            table = pq.read_table(tmp_path)
            # Strip "n." prefix from column names (Kuzu's RETURN n.* produces "n.col")
            clean_names = [
                c.replace("n.", "", 1) if c.startswith("n.") else c
                for c in table.column_names
            ]
            table = table.rename_columns(clean_names)
            deltalake.write_deltalake(
                path, table, mode=mode, partition_by=partition_by,
            )
            return {
                "rows_written": table.num_rows,
                "delta_path": str(path),
                "columns": clean_names,
            }
        finally:
            os.unlink(tmp_path)

    def query_to_delta_lake(
        self,
        cypher: str,
        path: str,
        *,
        mode: str = "overwrite",
        partition_by: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Export the result of any Cypher query to a Delta Lake table.

        Args:
            cypher: Cypher query whose results to export.
            path: Delta Lake table path (local, s3://, gs://, az://).
            mode: Write mode — "overwrite" or "append".
            partition_by: Optional list of columns to partition by.
            params: Optional Cypher query parameters.

        Returns:
            Dict with: rows_written, columns, delta_path.

        Requires:
            pip install bridgr[deltalake]
        """
        try:
            import deltalake
        except ImportError as e:
            raise ImportError(
                "deltalake package required. Install with: pip install bridgr[deltalake]"
            ) from e

        import tempfile
        import os
        import pyarrow.parquet as pq

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp_path = f.name
        try:
            cp = _cypher_path(tmp_path)
            self._db.execute(f'COPY ({cypher}) TO "{cp}"')
            table = pq.read_table(tmp_path)
            deltalake.write_deltalake(
                path, table, mode=mode, partition_by=partition_by,
            )
            return {
                "rows_written": table.num_rows,
                "columns": table.column_names,
                "delta_path": str(path),
            }
        finally:
            os.unlink(tmp_path)


# ------------------------------------------------------------------
# Standalone functions (convenience API matching spec D3)
# ------------------------------------------------------------------


def to_delta_lake(
    db: Database,
    node_label: str,
    delta_path: str,
    mode: str = "overwrite",
) -> dict[str, Any]:
    """Export all nodes of a type to a Delta Lake table.

    Convenience function that wraps DataExporter.to_delta_lake().

    Args:
        db: Bridgr Database instance.
        node_label: Node type to export (e.g., "Customer").
        delta_path: Delta Lake table path (local, s3://, gs://, az://).
        mode: Write mode — "overwrite" or "append".

    Returns:
        Dict with: rows_written, delta_path, columns.

    Requires:
        pip install bridgr[deltalake]
    """
    exporter = DataExporter(db)
    return exporter.to_delta_lake(node_label, delta_path, mode=mode)


def query_to_delta_lake(
    db: Database,
    cypher: str,
    delta_path: str,
    mode: str = "overwrite",
) -> dict[str, Any]:
    """Export Cypher query results to a Delta Lake table.

    Convenience function that wraps DataExporter.query_to_delta_lake().

    Args:
        db: Bridgr Database instance.
        cypher: Cypher query whose results to export.
        delta_path: Delta Lake table path (local, s3://, gs://, az://).
        mode: Write mode — "overwrite" or "append".

    Returns:
        Dict with: rows_written, columns, delta_path.

    Requires:
        pip install bridgr[deltalake]
    """
    exporter = DataExporter(db)
    return exporter.query_to_delta_lake(cypher, delta_path, mode=mode)
