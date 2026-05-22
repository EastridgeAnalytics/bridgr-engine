"""Bridgr — MIT-licensed embedded graph database engine.

The open-source foundation: database CRUD, Cypher queries, 7 core algorithms,
vector search, Delta Lake import/export, and schema utilities.

Proprietary features (Leiden, temporal analytics, Snowflake connector,
fraud scoring, streaming, ER integration, audit trail, alerts, triggers)
are available in the ``bridgr_platform`` package.
"""

from bridgr.database import Database
from bridgr.algorithms import GraphAlgorithms
from bridgr.vector import VectorIndex
from bridgr.export import DataExporter, to_delta_lake, query_to_delta_lake
from bridgr.schema_utils import arrow_schema_to_cypher, find_primary_key, cypher_path
from bridgr.exceptions import (
    BridgrError,
    NodeNotFoundError,
    EdgeNotFoundError,
    DuplicateNodeError,
    TransactionError,
    SchemaError,
)

__version__ = "0.1.0"
__all__ = [
    # Core database
    "Database",
    "open",
    # Algorithms (MIT: WCC, Louvain, PageRank, SCC, K-Core, degree, shortest path)
    "GraphAlgorithms",
    # Vector search
    "VectorIndex",
    # Export/Import
    "DataExporter",
    "to_delta_lake",
    "query_to_delta_lake",
    # Schema utilities
    "arrow_schema_to_cypher",
    "find_primary_key",
    "cypher_path",
    # Exceptions
    "BridgrError",
    "NodeNotFoundError",
    "EdgeNotFoundError",
    "DuplicateNodeError",
    "TransactionError",
    "SchemaError",
]


def open(path: str) -> Database:
    """Open or create a Bridgr database at the given path.

    Args:
        path: Filesystem path for the database. Use ":memory:" for in-memory mode.

    Returns:
        A Database instance ready for queries.
    """
    return Database(path)
