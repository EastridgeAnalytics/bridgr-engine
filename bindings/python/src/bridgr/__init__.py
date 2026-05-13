"""Bridgr — AI-native embedded graph database for enterprise."""

from bridgr.database import Database
from bridgr.argus import BridgrStore
from bridgr.migrate import migrate_case
from bridgr.algorithms import GraphAlgorithms
from bridgr.vector import VectorIndex
from bridgr.audit import AuditedDatabase, AuditLog
from bridgr.export import DataExporter
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
    "Database",
    "BridgrStore",
    "GraphAlgorithms",
    "VectorIndex",
    "AuditedDatabase",
    "AuditLog",
    "DataExporter",
    "migrate_case",
    "open",
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
