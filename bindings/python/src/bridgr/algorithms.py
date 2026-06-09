"""Bridgr Graph Algorithms — MIT-licensed core graph analytics.

Wraps LadybugDB's built-in algo extension (WCC, PageRank, Louvain, SCC, K-Core)
and provides custom implementations for degree centrality and shortest path.

Additional algorithms (Leiden, triangle count, centrality measures, link
prediction, FastRP embeddings, label propagation, node similarity) are
available in the proprietary ``bridgr_platform.algorithms`` extension.

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

from collections import deque
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
                self._db.execute("INSTALL algo")
            except RuntimeError:
                pass
            try:
                self._db.execute("LOAD EXTENSION algo")
            except RuntimeError:
                pass
            self._algo_loaded = True

    def _primary_key(self, node_label: str) -> str:
        """Return the declared primary-key property of a node table.

        Algorithm results bind the projected node as ``node``; the business key
        is read back via ``node.<pk>``. Hardcoding ``node.id`` broke on tables
        whose primary key isn't named ``id`` (e.g. H-E-B's ``record_id``). This
        mirrors the ``table_info`` lookup used elsewhere in the SDK and falls
        back to ``id`` when the PK can't be determined.
        """
        try:
            rows = self._db.query(f"CALL table_info('{node_label}') RETURN *")
            for r in rows:
                # `table_info` marks the PK in a column literally named
                # "primary key" (with a space); accept camel/snake spellings too
                # for forward-compat across engine versions.
                if r.get("primary key") or r.get("isPrimaryKey") or r.get("is_primary_key"):
                    return str(r.get("name", "id"))
        except RuntimeError:
            pass
        return "id"

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
                f"CALL WEAKLY_CONNECTED_COMPONENTS('{graph_name}') "
                f"RETURN node.{self._primary_key(node_label)} AS node_id, group_id AS component_id "
                f"ORDER BY group_id, node_id"
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
                f"CALL PAGE_RANK('{graph_name}') "
                f"RETURN node.{self._primary_key(node_label)} AS node_id, rank AS score "
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
                f"CALL LOUVAIN('{graph_name}') "
                f"RETURN node.{self._primary_key(node_label)} AS node_id, louvain_id AS community_id "
                f"ORDER BY louvain_id, node_id"
            )
        finally:
            self._drop_graph(graph_name)

    def leiden(
        self,
        node_label: str,
        edge_label: str,
        *,
        resolution: float = 1.0,
        max_iterations: int = 10,
        weight_property: str | None = None,
        objective: str = "modularity",
    ) -> list[dict[str, Any]]:
        """Detect communities using the Leiden algorithm.

        Like Louvain but guarantees internally-connected communities (repairs
        Louvain's disconnected-community defect). Leiden is an MIT engine
        algorithm (``CALL LEIDEN``, GVE-backed), so it lives on the MIT base
        alongside ``louvain()``. The proprietary ``bridgr_platform`` package adds
        an ER/Scan-integrated wrapper over this same call.

        Args:
            resolution: Modularity resolution (gamma); higher -> more, smaller
                communities.
            max_iterations: Max local-moving iterations per phase (wired to the
                engine's ``maxIterations``; previously accepted but dropped).
            weight_property: Numeric edge property to use as the edge weight
                (e.g. a Splink ``match_probability``). ``None`` (default) leaves
                every edge weight 1 (the historical unweighted behavior). Weights
                must be non-negative; the engine rejects negatives.
            objective: Only ``"modularity"`` is supported today; ``"cpm"``
                (resolution-limit-free, the fix for ER over-merge) ships in a
                later engine build and raises until then.

        Returns list of {node_id, community_id}.
        """
        if objective != "modularity":
            raise NotImplementedError(
                f"objective={objective!r} is not available yet; the CPM objective "
                "ships in a later engine build. Use objective='modularity'."
            )
        self._ensure_algo()
        graph_name = f"_leid_{node_label}_{edge_label}"
        args = [
            f"resolution := {float(resolution)}",
            f"maxIterations := {int(max_iterations)}",
        ]
        if weight_property:
            wp = weight_property.replace("'", "''")
            args.append(f"weight_property := '{wp}'")
        arg_str = ", ".join(args)
        self._project_graph(graph_name, node_label, edge_label)
        try:
            return self._db.query(
                f"CALL LEIDEN('{graph_name}', {arg_str}) "
                f"RETURN node.{self._primary_key(node_label)} AS node_id, community_id "
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
                f"CALL STRONGLY_CONNECTED_COMPONENTS('{graph_name}') "
                f"RETURN node.{self._primary_key(node_label)} AS node_id, group_id AS component_id "
                f"ORDER BY group_id, node_id"
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
                f"RETURN node.{self._primary_key(node_label)} AS node_id, core "
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_adjacency(
        self, node_label: str | None
    ) -> tuple[dict[str, set[str]], list[str], dict[str, str]]:
        """Build an undirected adjacency list from the database.

        Args:
            node_label: Optional filter. If None, uses all node types.

        Returns:
            Tuple of (adjacency dict, sorted node ID list, node-id-to-label map).
        """
        if node_label:
            rows = self._db.query(
                f"MATCH (n:{node_label}) RETURN n.id AS id"
            )
        else:
            # Fetch from all node tables
            labels = self._db._get_node_labels()
            rows = []
            for lbl in labels:
                rows.extend(self._db.query(
                    f"MATCH (n:{lbl}) RETURN n.id AS id"
                ))

        node_ids = [r["id"] for r in rows]
        node_labels_map: dict[str, str] = {}
        if node_label:
            for nid in node_ids:
                node_labels_map[nid] = node_label
        else:
            for lbl in self._db._get_node_labels():
                lbl_rows = self._db.query(
                    f"MATCH (n:{lbl}) RETURN n.id AS id"
                )
                for r in lbl_rows:
                    node_labels_map[r["id"]] = lbl

        # Build undirected adjacency from all edges
        adj: dict[str, set[str]] = {nid: set() for nid in node_ids}
        node_set = set(node_ids)
        try:
            edge_rows = self._db.query(
                "MATCH (a)-[]->(b) RETURN a.id AS src, b.id AS dst"
            )
            for er in edge_rows:
                src, dst = er["src"], er["dst"]
                if src in node_set and dst in node_set:
                    adj[src].add(dst)
                    adj[dst].add(src)
        except RuntimeError:
            pass

        node_ids.sort()
        return adj, node_ids, node_labels_map

    def _bfs_distances(
        self, source: str, adj: dict[str, set[str]]
    ) -> tuple[int, int]:
        """Run BFS from source, return (sum of distances, reachable count)."""
        dist: dict[str, int] = {source: 0}
        queue: deque[str] = deque([source])
        dist_sum = 0
        while queue:
            v = queue.popleft()
            for w in adj.get(v, set()):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    dist_sum += dist[w]
                    queue.append(w)
        return dist_sum, len(dist)

    def _project_graph(self, name: str, node_label: str, edge_label: str) -> None:
        try:
            self._db.execute(f"CALL DROP_PROJECTED_GRAPH('{name}')")
        except RuntimeError:
            pass
        # Auto-detect endpoint tables for the edge to handle bipartite graphs
        node_labels = self._get_edge_endpoints(edge_label) or [node_label]
        label_list = ", ".join(f"'{l}'" for l in node_labels)
        self._db.execute(
            f"CALL PROJECT_GRAPH('{name}', [{label_list}], ['{edge_label}'])"
        )

    def _get_edge_endpoints(self, edge_label: str) -> list[str] | None:
        try:
            rows = self._db.query(f"CALL show_connection('{edge_label}') RETURN *")
            labels = set()
            for r in rows:
                src = r.get("source table name")
                dst = r.get("destination table name")
                if src:
                    labels.add(src)
                if dst:
                    labels.add(dst)
            return list(labels) if labels else None
        except Exception:
            return None

    def _drop_graph(self, name: str) -> None:
        try:
            self._db.execute(f"CALL DROP_PROJECTED_GRAPH('{name}')")
        except RuntimeError:
            pass
