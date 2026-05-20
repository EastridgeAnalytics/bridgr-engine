"""Bridgr — AI-native embedded graph database for enterprise."""

from bridgr.database import Database
from bridgr.argus import BridgrStore
from bridgr.migrate import migrate_case
from bridgr.algorithms import GraphAlgorithms
from bridgr.vector import VectorIndex
from bridgr.audit import AuditedDatabase, AuditLog
from bridgr.export import DataExporter, to_delta_lake, query_to_delta_lake
from bridgr.similarity_graph import SimilarityGraphBuilder
from bridgr.streaming import IncrementalWriter
from bridgr.scoring import FraudScorer
from bridgr.triggers import AlgorithmTrigger
from bridgr.alerts import Alert, AlertEngine, AlertRule

try:
    from bridgr.kafka_consumer import BridgrKafkaConsumer
except ImportError:
    BridgrKafkaConsumer = None  # type: ignore[assignment,misc]
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
    "Alert",
    "AlertEngine",
    "AlertRule",
    "AlgorithmTrigger",
    "BridgrKafkaConsumer",
    "AuditLog",
    "AuditedDatabase",
    "BridgrError",
    "BridgrStore",
    "DataExporter",
    "Database",
    "DuplicateNodeError",
    "EdgeNotFoundError",
    "FraudScorer",
    "GraphAlgorithms",
    "IncrementalWriter",
    "NodeNotFoundError",
    "SchemaError",
    "SimilarityGraphBuilder",
    "TransactionError",
    "VectorIndex",
    "migrate_case",
    "open",
    "query_to_delta_lake",
    "to_delta_lake",
]


def open(path: str) -> Database:
    """Open or create a Bridgr database at the given path.

    Args:
        path: Filesystem path for the database. Use ":memory:" for in-memory mode.

    Returns:
        A Database instance ready for queries.
    """
    return Database(path)
