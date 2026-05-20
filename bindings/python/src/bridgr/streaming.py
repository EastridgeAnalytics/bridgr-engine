"""Incremental graph writes — upsert nodes/edges without full reimport.

Enables near real-time graph updates from streaming sources (Kafka, API).

Usage:
    from bridgr.streaming import IncrementalWriter

    writer = IncrementalWriter(db)
    result = writer.upsert_node("Customer", "C001", {"name": "John", "risk_score": 0.8})
    # result: {"action": "created", "id": "C001"}

    writer.upsert_edge("SIMILAR_TO", "C001", "C002", {"weight": 3})
    # result: {"action": "created"}

    stats = writer.batch_upsert_nodes("Customer", [
        {"id": "C001", "name": "John"},
        {"id": "C002", "name": "Jane"},
    ])
    # stats: {"created": 1, "updated": 1, "elapsed_ms": 2.3}
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from bridgr.database import Database

__all__ = ["IncrementalWriter"]


class IncrementalWriter:
    """Incremental graph write operations for streaming workloads.

    Uses MERGE semantics: create if not exists, update properties if exists.
    Integrates with AlgorithmTrigger to re-run analytics after writes.
    """

    def __init__(self, db: Database, *, trigger: Any | None = None):
        """Create an IncrementalWriter.

        Args:
            db: Database instance.
            trigger: Optional AlgorithmTrigger to notify on writes.
        """
        self._db = db
        self._trigger = trigger
        self._write_count: dict[str, int] = {}

    @property
    def write_counts(self) -> dict[str, int]:
        """Per-label write counts since this writer was created."""
        return dict(self._write_count)

    # ------------------------------------------------------------------
    # Single-record operations
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        label: str,
        pk_value: str | dict[str, Any],
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert or update a node. Returns {action: 'created'|'updated', id: str}.

        Uses MERGE semantics: create if not exists, update properties if exists.
        Target: <5ms single operation.

        Supports two calling conventions:
            writer.upsert_node("Customer", "C001", {"name": "John"})
            writer.upsert_node("Customer", {"id": "C001", "name": "John"})

        Args:
            label: Node table label.
            pk_value: Primary key value (str), or a dict of all properties
                including the PK (for backward compatibility with batch callers).
            properties: Additional properties to set. Ignored when pk_value is a dict.

        Returns:
            Dict with 'action' ('created' or 'updated') and 'id'.
        """
        if isinstance(pk_value, dict):
            # Legacy calling convention: upsert_node("Label", {full_props_dict})
            all_props = dict(pk_value)
            pk_col = self._find_pk(label)
            actual_pk = all_props.get(pk_col)
            if actual_pk is None:
                raise ValueError(f"Missing primary key '{pk_col}' in properties")
        else:
            pk_col = self._find_pk(label)
            actual_pk = pk_value
            all_props = {pk_col: pk_value}
            if properties:
                all_props.update(properties)

        # Check if node already exists
        existing = self._db.query(
            f"MATCH (n:{label} {{{pk_col}: $pk_val}}) RETURN count(n) AS cnt",
            {"pk_val": actual_pk},
        )
        already_exists = existing and existing[0].get("cnt", 0) > 0

        # Build MERGE + SET
        set_parts: list[str] = []
        params: dict[str, Any] = {"pk_val": actual_pk}
        for k, v in all_props.items():
            if k == pk_col:
                continue
            param_name = f"p_{k}"
            set_parts.append(f"n.{k} = ${param_name}")
            params[param_name] = v
        params["p_updated_at"] = datetime.now(timezone.utc).isoformat()
        set_parts.append("n.updated_at = $p_updated_at")

        set_clause = ", ".join(set_parts) if set_parts else ""
        cypher = f"MERGE (n:{label} {{{pk_col}: $pk_val}})"
        if set_clause:
            cypher += f" SET {set_clause}"

        self._db.execute(cypher, params)
        self._write_count[label] = self._write_count.get(label, 0) + 1
        if self._trigger:
            self._trigger.notify_write(1)

        action = "updated" if already_exists else "created"
        return {"action": action, "id": str(actual_pk)}

    def upsert_edge(
        self,
        edge_label: str,
        from_id: str,
        to_id: str,
        properties: dict[str, Any] | None = None,
        *,
        from_label: str | None = None,
        to_label: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update an edge. Returns {action: 'created'|'updated'}.

        Uses MERGE semantics. Target: <5ms single operation.

        Args:
            edge_label: Relationship type label.
            from_id: Source node primary key value.
            to_id: Target node primary key value.
            properties: Optional edge properties.
            from_label: Source node table label. Auto-detected if not provided.
            to_label: Target node table label. Auto-detected if not provided.

        Returns:
            Dict with 'action' ('created' or 'updated').
        """
        # Resolve node labels from edge table metadata if not provided
        if from_label is None or to_label is None:
            resolved_from, resolved_to = self._resolve_edge_endpoints(edge_label)
            from_label = from_label or resolved_from
            to_label = to_label or resolved_to

        from_pk = self._find_pk(from_label)
        to_pk = self._find_pk(to_label)

        # Check if edge already exists
        existing = self._db.query(
            f"MATCH (a:{from_label} {{{from_pk}: $from_id}})"
            f"-[r:{edge_label}]->"
            f"(b:{to_label} {{{to_pk}: $to_id}}) RETURN count(r) AS cnt",
            {"from_id": from_id, "to_id": to_id},
        )
        already_exists = existing and existing[0].get("cnt", 0) > 0

        params: dict[str, Any] = {"from_id": from_id, "to_id": to_id}
        set_parts: list[str] = []
        if properties:
            for k, v in properties.items():
                param_name = f"e_{k}"
                set_parts.append(f"r.{k} = ${param_name}")
                params[param_name] = v
        params["e_updated_at"] = datetime.now(timezone.utc).isoformat()
        set_parts.append("r.updated_at = $e_updated_at")

        set_clause = " SET " + ", ".join(set_parts)

        self._db.execute(
            f"MATCH (a:{from_label} {{{from_pk}: $from_id}}), "
            f"(b:{to_label} {{{to_pk}: $to_id}}) "
            f"MERGE (a)-[r:{edge_label}]->(b){set_clause}",
            params,
        )
        self._write_count[edge_label] = self._write_count.get(edge_label, 0) + 1
        if self._trigger:
            self._trigger.notify_write(1)

        action = "updated" if already_exists else "created"
        return {"action": action}

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_upsert_nodes(
        self,
        label: str,
        nodes: list[dict[str, Any]],
        pk_column: str = "id",
        *,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """Batch upsert nodes. Returns {created: int, updated: int, elapsed_ms: float}.

        Target: <500ms for batch of 100.

        Args:
            label: Node table label.
            nodes: List of property dicts, each containing the PK column.
            pk_column: Name of the primary key column in the dicts.
            batch_size: Internal processing batch size.

        Returns:
            Dict with 'created', 'updated', 'errors', and 'elapsed_ms'.
        """
        start = time.perf_counter()
        created = 0
        updated = 0
        errors: list[str] = []

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            for record in batch:
                try:
                    result = self.upsert_node(label, record)
                    if result["action"] == "created":
                        created += 1
                    else:
                        updated += 1
                except Exception as e:
                    errors.append(str(e))

        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "created": created,
            "updated": updated,
            "errors": errors,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    def batch_upsert_edges(
        self,
        edge_label: str,
        edges: list[dict[str, Any]],
        from_col: str = "from_id",
        to_col: str = "to_id",
        *,
        from_label: str | None = None,
        to_label: str | None = None,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """Batch upsert edges. Returns {created: int, updated: int, elapsed_ms: float}.

        Each edge dict must contain the from/to ID columns. Additional keys
        become edge properties.

        Args:
            edge_label: Relationship type label.
            edges: List of dicts, each with from/to IDs and optional properties.
            from_col: Key name for the source node ID in each dict.
            to_col: Key name for the target node ID in each dict.
            from_label: Source node table label. Auto-detected if not provided.
            to_label: Target node table label. Auto-detected if not provided.
            batch_size: Internal processing batch size.

        Returns:
            Dict with 'created', 'updated', 'errors', and 'elapsed_ms'.
        """
        start = time.perf_counter()
        created = 0
        updated = 0
        errors: list[str] = []

        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]
            for edge_dict in batch:
                try:
                    fid = edge_dict[from_col]
                    tid = edge_dict[to_col]
                    props = {
                        k: v
                        for k, v in edge_dict.items()
                        if k not in (from_col, to_col)
                    }
                    result = self.upsert_edge(
                        edge_label,
                        fid,
                        tid,
                        props or None,
                        from_label=from_label,
                        to_label=to_label,
                    )
                    if result["action"] == "created":
                        created += 1
                    else:
                        updated += 1
                except Exception as e:
                    errors.append(str(e))

        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "created": created,
            "updated": updated,
            "errors": errors,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_pk(self, label: str) -> str:
        """Discover the primary key column for a node table."""
        try:
            rows = self._db.query(f"CALL table_info('{label}') RETURN *")
            for r in rows:
                if r.get("isPrimaryKey") or r.get("is_primary_key"):
                    return str(r.get("name", "id"))
            if rows:
                return str(rows[0].get("name", "id"))
        except Exception:
            pass
        return "id"

    def _resolve_edge_endpoints(
        self, edge_label: str
    ) -> tuple[str, str]:
        """Resolve FROM and TO node labels for an edge table."""
        try:
            rows = self._db.query(
                f"CALL show_connection('{edge_label}') RETURN *"
            )
            if rows:
                src = rows[0].get("source table name", "")
                dst = rows[0].get("destination table name", "")
                if src and dst:
                    return src, dst
        except Exception:
            pass
        raise ValueError(
            f"Cannot resolve endpoint labels for edge '{edge_label}'. "
            f"Pass from_label and to_label explicitly."
        )
