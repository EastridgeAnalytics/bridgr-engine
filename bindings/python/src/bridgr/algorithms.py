"""Bridgr Graph Algorithms — Python API for graph analytics.

Wraps LadybugDB's built-in algo extension (WCC, PageRank, Louvain, SCC, K-Core)
and provides custom implementations for degree centrality, node similarity,
shortest path, and label propagation.

Usage:
    db = bridgr.open("my.lbug")
    from bridgr.algorithms import GraphAlgorithms
    algo = GraphAlgorithms(db)

    # Built-in algorithms (via LadybugDB algo extension)
    components = algo.weakly_connected_components("Entity", "CONNECTED_TO")
    scores = algo.pagerank("Entity", "CONNECTED_TO")
    communities = algo.louvain("Entity", "CONNECTED_TO")

    # Cypher-based algorithms
    path = algo.shortest_path("e1", "e5", "Entity", max_depth=10)
    centrality = algo.degree_centrality("Entity", "CONNECTED_TO")
"""

from __future__ import annotations

from typing import Any

from bridgr.database import Database


class GraphAlgorithms:
    """Graph analytics algorithms on a Bridgr database."""

    def __init__(self, db: Database):
        self._db = db
        self._algo_loaded = False

    def _ensure_algo(self) -> None:
        if not self._algo_loaded:
            try:
                self._db.execute("LOAD EXTENSION algo")
            except RuntimeError:
                pass
            self._algo_loaded = True

    # ------------------------------------------------------------------
    # Built-in algorithms (LadybugDB algo extension)
    # ------------------------------------------------------------------

    def weakly_connected_components(
        self, node_label: str, edge_label: str
    ) -> list[dict[str, Any]]:
        """Assign a component ID to each node. Nodes in the same connected
        component share the same component ID.

        Returns list of {node_id, component_id}.
        """
        self._ensure_algo()
        graph_name = f"_wcc_{node_label}_{edge_label}"
        self._project_graph(graph_name, node_label, edge_label)
        try:
            return self._db.query(
                f"CALL weakly_connected_component('{graph_name}') "
                f"RETURN node.id AS node_id, component_id "
                f"ORDER BY component_id, node_id"
            )
        finally:
            self._drop_graph(graph_name)

    def pagerank(
        self,
        node_label: str,
        edge_label: str,
        *,
        damping: float = 0.85,
        iterations: int = 20,
        tolerance: float = 1e-6,
    ) -> list[dict[str, Any]]:
        """Compute PageRank scores for all nodes.

        Returns list of {node_id, score} ordered by score descending.
        """
        self._ensure_algo()
        graph_name = f"_pr_{node_label}_{edge_label}"
        self._project_graph(graph_name, node_label, edge_label)
        try:
            return self._db.query(
                f"CALL pagerank('{graph_name}', "
                f"normalizeInitial := true, dampingFactor := {damping}, "
                f"maxIterations := {iterations}, delta := {tolerance}) "
                f"RETURN node.id AS node_id, rank AS score "
                f"ORDER BY score DESC"
            )
        finally:
            self._drop_graph(graph_name)

    def louvain(
        self, node_label: str, edge_label: str, *, max_iterations: int = 10
    ) -> list[dict[str, Any]]:
        """Detect communities using the Louvain algorithm.

        Returns list of {node_id, community_id}.
        """
        self._ensure_algo()
        graph_name = f"_louv_{node_label}_{edge_label}"
        self._project_graph(graph_name, node_label, edge_label)
        try:
            return self._db.query(
                f"CALL community_detection('{graph_name}', "
                f"maxIterations := {max_iterations}) "
                f"RETURN node.id AS node_id, community_id "
                f"ORDER BY community_id, node_id"
            )
        finally:
            self._drop_graph(graph_name)

    def strongly_connected_components(
        self, node_label: str, edge_label: str
    ) -> list[dict[str, Any]]:
        """Find strongly connected components in a directed graph.

        Returns list of {node_id, component_id}.
        """
        self._ensure_algo()
        graph_name = f"_scc_{node_label}_{edge_label}"
        self._project_graph(graph_name, node_label, edge_label)
        try:
            return self._db.query(
                f"CALL strongly_connected_component('{graph_name}') "
                f"RETURN node.id AS node_id, component_id "
                f"ORDER BY component_id, node_id"
            )
        finally:
            self._drop_graph(graph_name)

    def k_core(
        self, node_label: str, edge_label: str, *, k: int = 2
    ) -> list[dict[str, Any]]:
        """K-core decomposition — find the maximal subgraph where every node
        has degree >= k.

        Returns list of {node_id, core_number}.
        """
        self._ensure_algo()
        graph_name = f"_kcore_{node_label}_{edge_label}"
        self._project_graph(graph_name, node_label, edge_label)
        try:
            return self._db.query(
                f"CALL k_core('{graph_name}', k := {k}) "
                f"RETURN node.id AS node_id, core "
                f"ORDER BY core DESC, node_id"
            )
        finally:
            self._drop_graph(graph_name)

    # ------------------------------------------------------------------
    # Cypher-based algorithms
    # ------------------------------------------------------------------

    def shortest_path(
        self,
        from_id: str,
        to_id: str,
        node_label: str,
        *,
        edge_label: str | None = None,
        max_depth: int = 10,
    ) -> list[dict[str, Any]] | None:
        """Find the shortest path between two nodes.

        Returns list of {node_id, hop} representing the path, or None if no path exists.
        """
        edge = f":{edge_label}" if edge_label else ""
        rows = self._db.query(
            f"MATCH p = (a:{node_label} {{id: $from_id}})"
            f"-[{edge}* SHORTEST 1..{max_depth}]-"
            f"(b:{node_label} {{id: $to_id}}) "
            f"RETURN length(p) AS path_length, nodes(p) AS path_nodes",
            {"from_id": from_id, "to_id": to_id},
        )
        if not rows:
            return None
        result = rows[0]
        if result.get("path_nodes"):
            result["path_node_ids"] = [
                n.get("id", n.get("_ID", "")) if isinstance(n, dict) else str(n)
                for n in result["path_nodes"]
            ]
        return result

    def degree_centrality(
        self, node_label: str, edge_label: str
    ) -> list[dict[str, Any]]:
        """Compute degree centrality (in-degree, out-degree, total) for each node.

        Returns list of {node_id, in_degree, out_degree, total_degree}
        ordered by total_degree descending.
        """
        rows = self._db.query(
            f"MATCH (n:{node_label}) RETURN n.id AS node_id"
        )
        results = []
        for row in rows:
            nid = row["node_id"]
            out_result = self._db.query(
                f"MATCH (:{node_label} {{id: $nid}})-[:{edge_label}]->() RETURN count(*) AS cnt",
                {"nid": nid},
            )
            out_deg = out_result[0]["cnt"] if out_result else 0
            in_result = self._db.query(
                f"MATCH (:{node_label} {{id: $nid}})<-[:{edge_label}]-() RETURN count(*) AS cnt",
                {"nid": nid},
            )
            in_deg = in_result[0]["cnt"] if in_result else 0
            results.append({
                "node_id": nid,
                "in_degree": in_deg,
                "out_degree": out_deg,
                "total_degree": in_deg + out_deg,
            })
        results.sort(key=lambda x: x["total_degree"], reverse=True)
        return results

    def node_similarity(
        self,
        node_id_a: str,
        node_id_b: str,
        node_label: str,
        edge_label: str,
        *,
        metric: str = "jaccard",
    ) -> float:
        """Compute similarity between two nodes based on their shared neighbors.

        Supports 'jaccard' (|A∩B|/|A∪B|) and 'overlap' (|A∩B|/min(|A|,|B|)).
        """
        neighbors_a = set()
        neighbors_b = set()

        rows_a = self._db.query(
            f"MATCH (n:{node_label} {{id: $id}})-[:{edge_label}]-(m) RETURN m.id",
            {"id": node_id_a},
        )
        for r in rows_a:
            neighbors_a.add(r["m.id"])

        rows_b = self._db.query(
            f"MATCH (n:{node_label} {{id: $id}})-[:{edge_label}]-(m) RETURN m.id",
            {"id": node_id_b},
        )
        for r in rows_b:
            neighbors_b.add(r["m.id"])

        intersection = neighbors_a & neighbors_b

        if metric == "jaccard":
            union = neighbors_a | neighbors_b
            if not union:
                return 0.0
            return len(intersection) / len(union)
        elif metric == "overlap":
            min_size = min(len(neighbors_a), len(neighbors_b))
            if min_size == 0:
                return 0.0
            return len(intersection) / min_size
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'jaccard' or 'overlap'.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_graph(self, name: str, node_label: str, edge_label: str) -> None:
        try:
            self._db.execute(f"CALL DROP_PROJECTED_GRAPH('{name}')")
        except RuntimeError:
            pass
        self._db.execute(
            f"CALL PROJECT_GRAPH('{name}', ['{node_label}'], ['{edge_label}'])"
        )

    def _drop_graph(self, name: str) -> None:
        try:
            self._db.execute(f"CALL DROP_PROJECTED_GRAPH('{name}')")
        except RuntimeError:
            pass
