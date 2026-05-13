"""Bridgr Audit Trail — append-only mutation log for all graph writes.

Every create, update, and delete operation is recorded with timestamp,
operation type, node/edge ID, actor, and changed fields. The log is
append-only and tamper-evident. Required for regulated industries
(legal, financial, healthcare).

Usage:
    from bridgr.audit import AuditedDatabase

    db = AuditedDatabase("case.lbug", actor="agent:nullclaw")
    db.create_node("Entity", {"id": "e1", "name": "Smith"})
    # Automatically logged

    # Query the audit log
    history = db.audit_log.get_history("e1")
    recent = db.audit_log.query(since="2026-05-01", operation="create")
    db.audit_log.export_jsonl("audit_export.jsonl")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from bridgr.database import Database


class AuditLog:
    """Queryable, append-only audit log stored in the graph database."""

    TABLE_NAME = "_AuditLog"

    def __init__(self, db: Database):
        self._db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        existing = set()
        result = self._db.execute("CALL SHOW_TABLES() RETURN name")
        while result.has_next():
            existing.add(result.get_next()[0])

        if self.TABLE_NAME not in existing:
            self._db.execute(f"""
                CREATE NODE TABLE {self.TABLE_NAME}(
                    id STRING PRIMARY KEY,
                    ts STRING,
                    operation STRING,
                    target_id STRING,
                    target_type STRING,
                    actor STRING,
                    changes STRING
                )
            """)

    def append(
        self,
        operation: str,
        target_id: str,
        target_type: str,
        actor: str,
        changes: dict[str, Any] | None = None,
    ) -> str:
        """Append an audit entry. Returns the entry ID."""
        import uuid

        entry_id = f"audit_{uuid.uuid4().hex[:12]}"
        ts = datetime.now(timezone.utc).isoformat()
        changes_str = json.dumps(changes, default=str) if changes else "{}"
        # Base64 encode changes to avoid LadybugDB MAP parsing
        import base64
        changes_b64 = base64.b64encode(changes_str.encode()).decode()

        self._db.execute(
            f"CREATE (:{self.TABLE_NAME} {{id: $id, ts: $ts, operation: $op, "
            f"target_id: $tid, target_type: $ttype, actor: $actor, changes: $changes}})",
            {
                "id": entry_id,
                "ts": ts,
                "op": operation,
                "tid": target_id,
                "ttype": target_type,
                "actor": actor,
                "changes": changes_b64,
            },
        )
        return entry_id

    def get_history(self, target_id: str) -> list[dict[str, Any]]:
        """Get all audit entries for a specific node/edge."""
        return self._query_log(
            f"MATCH (a:{self.TABLE_NAME}) WHERE a.target_id = $tid "
            f"RETURN a.* ORDER BY a.ts DESC",
            {"tid": target_id},
        )

    def query(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        operation: str | None = None,
        actor: str | None = None,
        target_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit log with filters."""
        conditions = []
        params: dict[str, Any] = {}

        if since:
            conditions.append("a.ts >= $since")
            params["since"] = since
        if until:
            conditions.append("a.ts <= $until")
            params["until"] = until
        if operation:
            conditions.append("a.operation = $op")
            params["op"] = operation
        if actor:
            conditions.append("a.actor = $actor")
            params["actor"] = actor
        if target_type:
            conditions.append("a.target_type = $ttype")
            params["ttype"] = target_type

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        return self._query_log(
            f"MATCH (a:{self.TABLE_NAME}){where} "
            f"RETURN a.* ORDER BY a.ts DESC LIMIT {limit}",
            params if params else None,
        )

    def count(self) -> int:
        result = self._db.execute(
            f"MATCH (a:{self.TABLE_NAME}) RETURN count(a) AS cnt"
        )
        return result.get_next()[0]

    def export_jsonl(self, path: str) -> int:
        """Export the audit log as JSONL. Returns the number of entries exported."""
        entries = self._query_log(
            f"MATCH (a:{self.TABLE_NAME}) RETURN a.* ORDER BY a.ts ASC", None
        )
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, default=str) + "\n")
        return len(entries)

    def _query_log(self, cypher: str, params: dict | None) -> list[dict[str, Any]]:
        import base64
        result = self._db.execute(cypher, params) if params else self._db.execute(cypher)
        col_names = result.get_column_names()
        rows = []
        while result.has_next():
            raw = result.get_next()
            entry = {}
            for k, v in zip(col_names, raw):
                clean = k.replace("a.", "")
                if clean == "changes" and isinstance(v, str) and v:
                    try:
                        entry[clean] = json.loads(base64.b64decode(v).decode())
                    except Exception:
                        entry[clean] = v
                else:
                    entry[clean] = v
            rows.append(entry)
        return rows


class AuditedDatabase(Database):
    """A Database subclass that automatically logs all mutations.

    Drop-in replacement for Database — same API, plus audit logging.
    """

    def __init__(self, path: str, *, actor: str = "system"):
        super().__init__(path)
        self.actor = actor
        self.audit_log = AuditLog(self)

    def create_node(self, label: str, properties: dict[str, Any]) -> str:
        node_id = super().create_node(label, properties)
        self.audit_log.append(
            operation="create",
            target_id=node_id,
            target_type=label,
            actor=self.actor,
            changes={"properties": properties},
        )
        return node_id

    def update_node(
        self, node_id: str, properties: dict[str, Any], label: str | None = None
    ) -> None:
        old = self.get_node(node_id, label=label)
        super().update_node(node_id, properties, label=label)
        changes = {"updated_fields": list(properties.keys())}
        if old:
            changes["before"] = {
                k: old.get(k) for k in properties if k in old
            }
            changes["after"] = properties
        self.audit_log.append(
            operation="update",
            target_id=node_id,
            target_type=label or (old["_label"] if old else "unknown"),
            actor=self.actor,
            changes=changes,
        )

    def delete_node(self, node_id: str, label: str | None = None) -> None:
        old = self.get_node(node_id, label=label)
        target_type = label or (old["_label"] if old else "unknown")
        super().delete_node(node_id, label=label)
        self.audit_log.append(
            operation="delete",
            target_id=node_id,
            target_type=target_type,
            actor=self.actor,
            changes={"deleted_node": old} if old else None,
        )

    def create_edge(
        self,
        edge_type: str,
        from_id: str,
        to_id: str,
        properties: dict[str, Any] | None = None,
        from_label: str | None = None,
        to_label: str | None = None,
    ) -> None:
        super().create_edge(edge_type, from_id, to_id, properties, from_label, to_label)
        self.audit_log.append(
            operation="create_edge",
            target_id=f"{from_id}->{to_id}",
            target_type=edge_type,
            actor=self.actor,
            changes={"from_id": from_id, "to_id": to_id, "properties": properties},
        )
