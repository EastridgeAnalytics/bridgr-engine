"""Bridgr Export/Import — Parquet and CSV data interchange.

Wraps LadybugDB's COPY TO/FROM for bulk data export and import.
Supports Parquet (columnar, efficient) and CSV (universal).

Usage:
    db = bridgr.open("case.lbug")
    from bridgr.export import DataExporter

    exporter = DataExporter(db)
    exporter.to_parquet("Entity", "entities.parquet")
    exporter.to_csv("Fact", "facts.csv")

    exporter.from_csv("Entity", "new_entities.csv")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bridgr.database import Database


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
