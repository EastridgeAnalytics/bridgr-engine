"""Bridgr Vector Search — Python API for LadybugDB's HNSW vector index.

Wraps LadybugDB's built-in vector extension for similarity search
and hybrid vector+graph queries.

Usage:
    db = bridgr.open("my.lbug")
    # Create a node table with vector column
    db.execute("CREATE NODE TABLE Doc(id STRING PRIMARY KEY, title STRING, embedding FLOAT[384])")

    from bridgr.vector import VectorIndex
    vi = VectorIndex(db)
    vi.create_index("Doc", "doc_emb", "embedding", metric="cosine")
    results = vi.search("Doc", "doc_emb", query_vector, k=10)
"""

from __future__ import annotations

from typing import Any

from bridgr.database import Database


class VectorIndex:
    """Manages HNSW vector indices on a Bridgr database."""

    def __init__(self, db: Database):
        self._db = db

    def create_index(
        self,
        table_name: str,
        index_name: str,
        property_name: str,
        *,
        metric: str = "cosine",
        mu: int = 30,
        ml: int = 60,
        efc: int = 200,
    ) -> None:
        """Create an HNSW vector index on a node table property.

        Args:
            table_name: Node table containing the vector column.
            index_name: Name for the index.
            property_name: Vector property (must be FLOAT[N] or DOUBLE[N]).
            metric: Distance metric — 'cosine', 'l2', 'l2sq', or 'dotproduct'.
            mu: Max degree in upper HNSW layer.
            ml: Max degree in lower HNSW layer.
            efc: Candidate vertices during construction.
        """
        self._db.execute("LOAD EXTENSION vector")
        self._db.execute(
            f"CALL CREATE_VECTOR_INDEX('{table_name}', '{index_name}', "
            f"'{property_name}', mu := {mu}, ml := {ml}, "
            f"metric := '{metric}', efc := {efc})"
        )

    def drop_index(self, table_name: str, index_name: str) -> None:
        self._db.execute(f"CALL DROP_VECTOR_INDEX('{table_name}', '{index_name}')")

    def search(
        self,
        table_name: str,
        index_name: str,
        query_vector: list[float],
        k: int = 10,
        *,
        efs: int = 200,
    ) -> list[dict[str, Any]]:
        """Search for k nearest neighbors by vector similarity.

        Returns list of dicts with 'node' (the full node) and 'distance'.
        """
        result = self._db.execute(
            f"CALL QUERY_VECTOR_INDEX('{table_name}', '{index_name}', "
            f"$qvec, {k}, efs := {efs}) RETURN node, distance",
            {"qvec": query_vector},
        )
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({"node": row[0], "distance": row[1]})
        return rows

    def hybrid_search(
        self,
        table_name: str,
        index_name: str,
        query_vector: list[float],
        k: int = 10,
        *,
        traverse_edge: str | None = None,
        traverse_depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Hybrid vector+graph search: find by embedding, then traverse.

        Two-step approach (aligned with Rust server implementation):
        1. Pure ANN search to get matched nodes + distances.
        2. Per-node graph traversal to collect connected neighbors.

        Returns list of dicts with ``matched_node``, ``node_id``,
        ``distance``, and ``connected`` (list of neighbor dicts).
        """
        if not traverse_edge:
            return self.search(table_name, index_name, query_vector, k)

        # Step 1: Pure ANN search
        search_results = self.search(table_name, index_name, query_vector, k)

        # Step 2: For each matched node, traverse edges
        enriched: list[dict[str, Any]] = []
        for match in search_results:
            node = match["node"]
            node_id = (
                node.get("id") or node.get("_ID") or node.get("node_id")
                if isinstance(node, dict)
                else str(node)
            )

            connected: list[dict[str, Any]] = []
            if node_id:
                try:
                    traverse_rows = self._db.query(
                        f"MATCH (n:{table_name} {{id: $nid}})"
                        f"-[:{traverse_edge}*1..{traverse_depth}]-(c) "
                        f"RETURN DISTINCT c.*, label(c) AS _label",
                        {"nid": node_id},
                    )
                    for r in traverse_rows:
                        props = {}
                        label = r.get("_label", "")
                        for k, v in r.items():
                            if k == "_label":
                                continue
                            clean = k[2:] if k.startswith("c.") else k
                            if not clean.startswith("_"):
                                props[clean] = v
                        props["label"] = label
                        connected.append(props)
                except RuntimeError:
                    pass

            enriched.append({
                "matched_node": node if isinstance(node, dict) else {"id": node_id},
                "node_id": node_id,
                "distance": match["distance"],
                "connected": connected,
            })

        return enriched

    def list_indices(self) -> list[dict[str, Any]]:
        """List all vector indices in the database."""
        result = self._db.execute("CALL SHOW_INDEXES() RETURN *")
        indices = []
        while result.has_next():
            row = result.get_next()
            col_names = result.get_column_names()
            indices.append(dict(zip(col_names, row)))
        return indices
