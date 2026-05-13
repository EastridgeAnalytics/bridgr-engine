"""Bridgr Database — thin wrapper around LadybugDB with a clean Python API."""

from __future__ import annotations

import re
from typing import Any

import real_ladybug as _lb

from bridgr.exceptions import (
    BridgrError,
    DuplicateNodeError,
    EdgeNotFoundError,
    NodeNotFoundError,
    SchemaError,
    TransactionError,
)

_CYPHER_TYPE_MAP = {
    str: "STRING",
    int: "INT64",
    float: "DOUBLE",
    bool: "BOOLEAN",
    list: "STRING[]",
}


class Database:
    """An embedded Bridgr graph database.

    Wraps LadybugDB (real_ladybug) with a Bridgr-branded Python API.
    Provides node/edge CRUD, transactions, search, and schema management.
    """

    def __init__(self, path: str):
        self._path = path
        self._db = _lb.Database(path)
        self._conn = _lb.Connection(self._db)
        self._in_transaction = False
        self._schema_cache: dict[str, dict[str, str]] = {}

    @property
    def path(self) -> str:
        return self._path

    def close(self) -> None:
        del self._conn
        del self._db
        self._conn = None  # type: ignore[assignment]
        self._db = None  # type: ignore[assignment]

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Raw query execution
    # ------------------------------------------------------------------

    def execute(self, cypher: str, params: dict[str, Any] | None = None) -> _lb.QueryResult:
        """Execute a raw Cypher query."""
        if params:
            return self._conn.execute(cypher, params)
        return self._conn.execute(cypher)

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as a list of dicts."""
        result = self.execute(cypher, params)
        rows = []
        while result.has_next():
            row = result.get_next()
            col_names = result.get_column_names()
            rows.append(dict(zip(col_names, row)))
        return rows

    def query_arrow(self, cypher: str, params: dict[str, Any] | None = None):
        """Execute a Cypher query and return an Arrow table."""
        result = self.execute(cypher, params)
        return result.get_as_arrow()

    def query_df(self, cypher: str, params: dict[str, Any] | None = None):
        """Execute a Cypher query and return a Pandas DataFrame."""
        result = self.execute(cypher, params)
        return result.get_as_df()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_node_table(self, label: str, properties: dict[str, str]) -> None:
        """Create a node table with the given label and property types.

        Args:
            label: Node label (e.g., "Entity", "Fact").
            properties: Mapping of property name to Cypher type string.
                        The first property should be the primary key.

        Example:
            db.create_node_table("Entity", {
                "id": "STRING PRIMARY KEY",
                "name": "STRING",
                "confidence": "DOUBLE",
            })
        """
        cols = ", ".join(f"{name} {dtype}" for name, dtype in properties.items())
        try:
            self._conn.execute(f"CREATE NODE TABLE {label}({cols})")
        except RuntimeError as e:
            raise SchemaError(str(e)) from e
        self._schema_cache[label] = properties

    def create_edge_table(
        self, label: str, from_label: str, to_label: str, properties: dict[str, str] | None = None
    ) -> None:
        """Create an edge (relationship) table.

        Args:
            label: Edge label (e.g., "INVOLVES").
            from_label: Source node table label.
            to_label: Target node table label.
            properties: Optional edge properties.
        """
        props = ""
        if properties:
            props = ", " + ", ".join(f"{name} {dtype}" for name, dtype in properties.items())
        try:
            self._conn.execute(
                f"CREATE REL TABLE {label}(FROM {from_label} TO {to_label}{props})"
            )
        except RuntimeError as e:
            raise SchemaError(str(e)) from e

    def drop_table(self, label: str) -> None:
        try:
            self._conn.execute(f"DROP TABLE {label}")
        except RuntimeError as e:
            raise SchemaError(str(e)) from e
        self._schema_cache.pop(label, None)

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def create_node(self, label: str, properties: dict[str, Any]) -> str:
        """Create a node and return its id.

        Raises DuplicateNodeError if a node with the same primary key exists.
        """
        prop_str = ", ".join(f"{k}: ${k}" for k in properties)
        try:
            self._conn.execute(
                f"CREATE (:{label} {{{prop_str}}})", properties
            )
        except RuntimeError as e:
            msg = str(e)
            if "primary key" in msg.lower() or "duplicate" in msg.lower() or "already exists" in msg.lower():
                pk_val = properties.get("id", str(properties))
                raise DuplicateNodeError(str(pk_val)) from e
            raise BridgrError(msg) from e
        return str(properties.get("id", ""))

    def get_node(self, node_id: str, label: str | None = None) -> dict[str, Any] | None:
        """Get a node by its primary key ID.

        If label is provided, searches only that table. Otherwise searches all node tables.
        Returns None if not found.
        """
        if label:
            labels = [label]
        else:
            labels = self._get_node_labels()

        for lbl in labels:
            result = self._conn.execute(
                f"MATCH (n:{lbl} {{id: $id}}) RETURN n.*", {"id": node_id}
            )
            if result.has_next():
                row = result.get_next()
                col_names = result.get_column_names()
                node = dict(zip(col_names, row))
                cleaned = {}
                for k, v in node.items():
                    clean_key = k.replace("n.", "")
                    cleaned[clean_key] = v
                cleaned["_label"] = lbl
                return cleaned
        return None

    def get_nodes_by_type(self, label: str) -> list[dict[str, Any]]:
        """Get all nodes of a given type/label."""
        result = self._conn.execute(f"MATCH (n:{label}) RETURN n.*")
        col_names = result.get_column_names()
        rows = []
        while result.has_next():
            row = result.get_next()
            node = {}
            for k, v in zip(col_names, row):
                node[k.replace("n.", "")] = v
            node["_label"] = label
            rows.append(node)
        return rows

    def update_node(self, node_id: str, properties: dict[str, Any], label: str | None = None) -> None:
        """Update a node's properties (merge semantics — unmentioned properties preserved).

        Raises NodeNotFoundError if the node doesn't exist.
        """
        if not label:
            existing = self.get_node(node_id)
            if existing is None:
                raise NodeNotFoundError(node_id)
            label = existing["_label"]

        set_parts = []
        params: dict[str, Any] = {"id": node_id}
        for k, v in properties.items():
            if k in ("id", "_label"):
                continue
            set_parts.append(f"n.{k} = ${k}")
            params[k] = v

        if not set_parts:
            return

        set_clause = ", ".join(set_parts)
        self._conn.execute(
            f"MATCH (n:{label} {{id: $id}}) SET {set_clause}", params
        )

    def delete_node(self, node_id: str, label: str | None = None) -> None:
        """Delete a node and all its connected edges.

        Raises NodeNotFoundError if the node doesn't exist.
        """
        if not label:
            existing = self.get_node(node_id)
            if existing is None:
                raise NodeNotFoundError(node_id)
            label = existing["_label"]

        self._conn.execute(
            f"MATCH (n:{label} {{id: $id}}) DETACH DELETE n", {"id": node_id}
        )

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def create_edge(
        self,
        edge_type: str,
        from_id: str,
        to_id: str,
        properties: dict[str, Any] | None = None,
        from_label: str | None = None,
        to_label: str | None = None,
    ) -> None:
        """Create an edge between two nodes.

        If from_label/to_label are not provided, searches all node tables.
        """
        from_lbl = from_label
        to_lbl = to_label
        if not from_lbl:
            node = self.get_node(from_id)
            if node is None:
                raise NodeNotFoundError(from_id)
            from_lbl = node["_label"]
        if not to_lbl:
            node = self.get_node(to_id)
            if node is None:
                raise NodeNotFoundError(to_id)
            to_lbl = node["_label"]

        props = ""
        params: dict[str, Any] = {"from_id": from_id, "to_id": to_id}
        if properties:
            prop_parts = []
            for k, v in properties.items():
                prop_parts.append(f"{k}: $prop_{k}")
                params[f"prop_{k}"] = v
            props = " {" + ", ".join(prop_parts) + "}"

        self._conn.execute(
            f"MATCH (a:{from_lbl} {{id: $from_id}}), (b:{to_lbl} {{id: $to_id}}) "
            f"CREATE (a)-[:{edge_type}{props}]->(b)",
            params,
        )

    def get_edges(self, node_id: str, label: str | None = None) -> list[dict[str, Any]]:
        """Get all edges connected to a node (both incoming and outgoing)."""
        if not label:
            node = self.get_node(node_id)
            if node is None:
                return []
            label = node["_label"]

        edges = []

        out_result = self._conn.execute(
            f"MATCH (n:{label} {{id: $nid}})-[r]->(m) "
            f"RETURN r, label(r) AS type, n.id AS from_id, m.id AS to_id",
            {"nid": node_id},
        )
        while out_result.has_next():
            row = out_result.get_next()
            rel_dict = row[0]
            props = {k: v for k, v in rel_dict.items() if not k.startswith("_")}
            edges.append({
                "type": row[1],
                "from_id": row[2],
                "to_id": row[3],
                "props": props,
                "direction": "outgoing",
            })

        in_result = self._conn.execute(
            f"MATCH (n:{label} {{id: $nid}})<-[r]-(m) "
            f"RETURN r, label(r) AS type, m.id AS from_id, n.id AS to_id",
            {"nid": node_id},
        )
        while in_result.has_next():
            row = in_result.get_next()
            rel_dict = row[0]
            props = {k: v for k, v in rel_dict.items() if not k.startswith("_")}
            edges.append({
                "type": row[1],
                "from_id": row[2],
                "to_id": row[3],
                "props": props,
                "direction": "incoming",
            })

        return edges

    def delete_edge(self, edge_type: str, from_id: str, to_id: str) -> None:
        """Delete an edge by type and endpoint IDs."""
        labels = self._get_node_labels()
        for from_lbl in labels:
            for to_lbl in labels:
                try:
                    self._conn.execute(
                        f"MATCH (a:{from_lbl} {{id: $from_id}})-[r:{edge_type}]->(b:{to_lbl} {{id: $to_id}}) DELETE r",
                        {"from_id": from_id, "to_id": to_id},
                    )
                    return
                except RuntimeError:
                    continue

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_text: str, labels: list[str] | None = None) -> list[dict[str, Any]]:
        """Search across node properties for a keyword string.

        Returns matching nodes with their label and matching content.
        """
        if labels is None:
            labels = self._get_node_labels()

        results = []
        search_lower = query_text.lower()

        for lbl in labels:
            try:
                all_nodes = self._conn.execute(f"MATCH (n:{lbl}) RETURN n.*")
                col_names = all_nodes.get_column_names()
                while all_nodes.has_next():
                    row = all_nodes.get_next()
                    node = {}
                    matched = False
                    for k, v in zip(col_names, row):
                        clean_key = k.replace("n.", "")
                        node[clean_key] = v
                        if isinstance(v, str) and search_lower in v.lower():
                            matched = True
                        elif isinstance(v, list):
                            for item in v:
                                if isinstance(item, str) and search_lower in item.lower():
                                    matched = True
                                    break
                    if matched:
                        node["_label"] = lbl
                        results.append(node)
            except RuntimeError:
                continue
        return results

    # ------------------------------------------------------------------
    # Canvas / bulk retrieval
    # ------------------------------------------------------------------

    def get_canvas_data(self, labels: list[str] | None = None) -> dict[str, list]:
        """Get all nodes and edges for visualization (Evidence Board canvas)."""
        if labels is None:
            labels = self._get_node_labels()

        nodes = []
        for lbl in labels:
            try:
                result = self._conn.execute(f"MATCH (n:{lbl}) RETURN n.*")
                col_names = result.get_column_names()
                while result.has_next():
                    row = result.get_next()
                    node = {}
                    for k, v in zip(col_names, row):
                        node[k.replace("n.", "")] = v
                    node["_label"] = lbl
                    nodes.append(node)
            except RuntimeError:
                continue

        edges = []
        try:
            result = self._conn.execute(
                "MATCH (a)-[r]->(b) RETURN label(r) AS type, a.id AS from_id, b.id AS to_id"
            )
            while result.has_next():
                row = result.get_next()
                col_names = result.get_column_names()
                edges.append(dict(zip(col_names, row)))
        except RuntimeError:
            pass

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def begin_transaction(self) -> None:
        if self._in_transaction:
            raise TransactionError("Transaction already in progress")
        self._conn.execute("BEGIN TRANSACTION")
        self._in_transaction = True

    def commit(self) -> None:
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")
        self._conn.execute("COMMIT")
        self._in_transaction = False

    def rollback(self) -> None:
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")
        self._conn.execute("ROLLBACK")
        self._in_transaction = False

    # Batch mode aliases for BridgrStore compatibility
    def begin_batch(self) -> None:
        self.begin_transaction()

    def end_batch(self) -> None:
        self.commit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_node_labels(self) -> list[str]:
        """Get all node table labels in the database."""
        result = self._conn.execute("CALL SHOW_TABLES() RETURN name, type")
        labels = []
        while result.has_next():
            row = result.get_next()
            if row[1] == "NODE":
                labels.append(row[0])
        return labels

