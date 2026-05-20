"""Bridgr Graph Algorithms — Python API for graph analytics.

Wraps LadybugDB's built-in algo extension (WCC, PageRank, Louvain, SCC, K-Core)
and provides custom implementations for degree centrality, node similarity,
shortest path, triangle count, closeness/betweenness centrality, link prediction,
and FastRP graph embeddings.

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

    # Phase 2 algorithms
    triangles = algo.triangle_count("Entity")
    closeness = algo.closeness_centrality("Entity")
    betweenness = algo.betweenness_centrality("Entity")
    prediction = algo.link_prediction("node1", "node2")
    embeddings = algo.fast_rp("Entity", dimension=64)
"""

from __future__ import annotations

import math
import random
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
                f"RETURN node.id AS node_id, group_id AS component_id "
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
                f"CALL LOUVAIN('{graph_name}') "
                f"RETURN node.id AS node_id, louvain_id AS community_id "
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
    ) -> list[dict[str, Any]]:
        """Detect communities using the Leiden algorithm.

        Guarantees internally connected communities, unlike Louvain.
        Returns list of {node_id, community_id}.
        """
        self._ensure_algo()
        graph_name = f"_leid_{node_label}_{edge_label}"
        try:
            self._project_graph(graph_name, node_label, edge_label)
            return self._db.query(
                f"CALL LEIDEN('{graph_name}', resolution := {resolution}) "
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
                f"CALL STRONGLY_CONNECTED_COMPONENTS('{graph_name}') "
                f"RETURN node.id AS node_id, group_id AS component_id "
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
    # Phase 2: Triangle Count / Clustering Coefficient
    # ------------------------------------------------------------------

    def triangle_count(self, node_label: str | None = None) -> dict[str, Any]:
        """Count triangles and compute clustering coefficients.

        For each node, counts the number of triangles it participates in and
        computes the local clustering coefficient (ratio of actual to possible
        triangles among its neighbors).

        Args:
            node_label: Optional node label filter. If None, uses all node types.

        Returns:
            Dictionary with ``total_triangles`` (int) and ``nodes`` (list of
            dicts with ``id``, ``triangles``, and ``clustering_coefficient``).
        """
        adj, node_ids, node_labels = self._build_adjacency(node_label)

        node_triangles: dict[str, int] = {nid: 0 for nid in node_ids}

        # Count triangles: for each edge (u,v) where u < v, count common
        # neighbors w > v.  Each triangle is counted exactly once.
        sorted_ids = sorted(node_ids)
        id_rank = {nid: i for i, nid in enumerate(sorted_ids)}
        total_triangles = 0

        for u in sorted_ids:
            neighbors_u = adj.get(u, set())
            for v in neighbors_u:
                if id_rank[v] <= id_rank[u]:
                    continue
                neighbors_v = adj.get(v, set())
                common = neighbors_u & neighbors_v
                for w in common:
                    if id_rank[w] <= id_rank[v]:
                        continue
                    total_triangles += 1
                    node_triangles[u] += 1
                    node_triangles[v] += 1
                    node_triangles[w] += 1

        nodes = []
        for nid in node_ids:
            tri = node_triangles[nid]
            deg = len(adj.get(nid, set()))
            # Clustering coefficient = 2 * triangles / (degree * (degree - 1))
            if deg >= 2:
                cc = (2.0 * tri) / (deg * (deg - 1))
            else:
                cc = 0.0
            nodes.append({
                "id": nid,
                "label": node_labels.get(nid, ""),
                "triangles": tri,
                "clustering_coefficient": cc,
            })

        return {"total_triangles": total_triangles, "nodes": nodes}

    # ------------------------------------------------------------------
    # Phase 2: Closeness Centrality
    # ------------------------------------------------------------------

    def closeness_centrality(
        self,
        node_label: str | None = None,
        sample_size: int = 0,
    ) -> list[dict[str, Any]]:
        """Compute closeness centrality for all nodes (or a sample).

        Closeness centrality measures how close a node is to all other reachable
        nodes: ``closeness(v) = (reachable - 1) / sum(shortest_path_distances)``.

        Args:
            node_label: Optional node label filter.
            sample_size: When > 0, only compute BFS from a random sample of
                nodes to approximate centrality for large graphs.

        Returns:
            List of dicts ``{id, label, closeness}`` sorted by closeness descending.
        """
        adj, node_ids, node_labels_map = self._build_adjacency(node_label)
        n = len(node_ids)
        if n == 0:
            return []

        targets = node_ids
        if sample_size > 0 and sample_size < n:
            targets = random.sample(node_ids, sample_size)

        results = []
        for source in targets:
            dist_sum, reachable = self._bfs_distances(source, adj)
            if reachable <= 1 or dist_sum == 0:
                closeness = 0.0
            else:
                closeness = (reachable - 1) / dist_sum
            results.append({
                "id": source,
                "label": node_labels_map.get(source, ""),
                "closeness": closeness,
            })

        results.sort(key=lambda x: x["closeness"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Phase 2: Betweenness Centrality (Brandes' Algorithm)
    # ------------------------------------------------------------------

    def betweenness_centrality(
        self,
        node_label: str | None = None,
        sample_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Compute betweenness centrality (approximated via sampling).

        Betweenness centrality measures the fraction of shortest paths between
        all pairs of nodes that pass through a given node.  Uses Brandes'
        algorithm with optional source sampling for scalability.

        Args:
            node_label: Optional node label filter.
            sample_size: Number of source nodes to sample.  Use 0 or a value
                >= the node count for exact computation.

        Returns:
            List of dicts ``{id, label, betweenness}`` sorted descending.
        """
        adj, node_ids, node_labels_map = self._build_adjacency(node_label)
        n = len(node_ids)
        if n <= 2:
            return [
                {"id": nid, "label": node_labels_map.get(nid, ""), "betweenness": 0.0}
                for nid in node_ids
            ]

        cb: dict[str, float] = {nid: 0.0 for nid in node_ids}

        # Determine sources (all or sampled)
        if sample_size <= 0 or sample_size >= n:
            sources = node_ids
        else:
            sources = random.sample(node_ids, min(sample_size, n))

        for s in sources:
            # Brandes' single-source shortest-path accumulation
            stack: list[str] = []
            pred: dict[str, list[str]] = {nid: [] for nid in node_ids}
            sigma: dict[str, int] = {nid: 0 for nid in node_ids}
            sigma[s] = 1
            dist: dict[str, int] = {nid: -1 for nid in node_ids}
            dist[s] = 0

            queue: deque[str] = deque([s])
            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in adj.get(v, set()):
                    # First visit?
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    # Shortest path through v?
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta: dict[str, float] = {nid: 0.0 for nid in node_ids}
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    cb[w] += delta[w]

        # Normalize: for undirected graphs divide by 2, then scale by
        # sampling fraction if we sampled.
        scale = 1.0
        if len(sources) < n:
            scale = n / len(sources)
        # Undirected normalization: each pair counted twice in Brandes
        for nid in node_ids:
            cb[nid] = (cb[nid] / 2.0) * scale

        results = [
            {"id": nid, "label": node_labels_map.get(nid, ""), "betweenness": cb[nid]}
            for nid in node_ids
        ]
        results.sort(key=lambda x: x["betweenness"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Phase 2: Link Prediction (Common Neighbors, Adamic-Adar, Jaccard)
    # ------------------------------------------------------------------

    def link_prediction(self, node_a_id: str, node_b_id: str) -> dict[str, Any]:
        """Predict the likelihood of a link between two specific nodes.

        Computes three link prediction scores based on neighborhood overlap:

        - **common_neighbors**: count of shared neighbors
        - **adamic_adar**: ``sum(1 / log(degree(z)))`` for each common neighbor z
        - **jaccard**: ``|intersection| / |union|`` of neighbor sets

        Args:
            node_a_id: ID of the first node.
            node_b_id: ID of the second node.

        Returns:
            Dict with ``common_neighbors``, ``adamic_adar``, ``jaccard``, and
            ``predicted`` (True if adamic_adar > 0).
        """
        adj, _, _ = self._build_adjacency(node_label=None)

        neighbors_a = adj.get(node_a_id, set())
        neighbors_b = adj.get(node_b_id, set())

        common = neighbors_a & neighbors_b
        union = neighbors_a | neighbors_b

        common_neighbors = len(common)
        jaccard = len(common) / len(union) if union else 0.0

        adamic_adar = 0.0
        for z in common:
            deg_z = len(adj.get(z, set()))
            if deg_z > 1:
                adamic_adar += 1.0 / math.log(deg_z)

        return {
            "common_neighbors": common_neighbors,
            "adamic_adar": adamic_adar,
            "jaccard": jaccard,
            "predicted": adamic_adar > 0.0,
        }

    def predict_links(
        self,
        node_label: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find the top-k most likely missing links among nodes of a type.

        Evaluates every non-connected pair of nodes with the given label and
        scores them using Adamic-Adar, common neighbors, and Jaccard.

        Args:
            node_label: Label of nodes to consider.
            top_k: Maximum number of predicted links to return.

        Returns:
            List of dicts ``{source_id, target_id, common_neighbors,
            adamic_adar, jaccard}`` sorted by adamic_adar descending.
        """
        adj, node_ids, _ = self._build_adjacency(node_label)

        candidates: list[dict[str, Any]] = []
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1:]:
                # Skip if already connected
                if b in adj.get(a, set()):
                    continue
                neighbors_a = adj.get(a, set())
                neighbors_b = adj.get(b, set())
                common = neighbors_a & neighbors_b
                if not common:
                    continue

                union = neighbors_a | neighbors_b
                jaccard = len(common) / len(union) if union else 0.0

                adamic_adar = 0.0
                for z in common:
                    deg_z = len(adj.get(z, set()))
                    if deg_z > 1:
                        adamic_adar += 1.0 / math.log(deg_z)

                candidates.append({
                    "source_id": a,
                    "target_id": b,
                    "common_neighbors": len(common),
                    "adamic_adar": adamic_adar,
                    "jaccard": jaccard,
                })

        candidates.sort(key=lambda x: x["adamic_adar"], reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Phase 2: FastRP (Fast Random Projection) Graph Embeddings
    # ------------------------------------------------------------------

    def fast_rp(
        self,
        node_label: str,
        dimension: int = 128,
        iterations: int = 3,
        normalization_strength: float = 0.0,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compute FastRP graph embeddings.

        FastRP (Fast Random Projection) generates low-dimensional node
        embeddings by iteratively averaging random projections of neighbor
        embeddings.  Uses numpy when available, falls back to pure Python.

        Args:
            node_label: Label of nodes to embed.
            dimension: Embedding dimensionality.
            iterations: Number of neighbor-averaging iterations.
            normalization_strength: L2 normalization strength.  When > 0,
                embeddings are L2-normalized after each iteration.
            seed: Optional random seed for reproducibility.

        Returns:
            List of dicts ``{id, label, embedding}`` where embedding is a
            list of floats with length ``dimension``.

        Reference:
            Chen et al., "Fast Graph Representation Learning with PyTorch
            Geometric", 2019.
        """
        adj, node_ids, node_labels_map = self._build_adjacency(node_label)
        n = len(node_ids)
        if n == 0:
            return []

        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        try:
            return self._fast_rp_numpy(
                adj, node_ids, node_labels_map, id_to_idx, n,
                dimension, iterations, normalization_strength, seed,
            )
        except ImportError:
            return self._fast_rp_pure(
                adj, node_ids, node_labels_map, id_to_idx, n,
                dimension, iterations, normalization_strength, seed,
            )

    def _fast_rp_numpy(
        self,
        adj: dict[str, set[str]],
        node_ids: list[str],
        node_labels_map: dict[str, str],
        id_to_idx: dict[str, int],
        n: int,
        dimension: int,
        iterations: int,
        normalization_strength: float,
        seed: int | None,
    ) -> list[dict[str, Any]]:
        """FastRP implementation using numpy for performance."""
        import numpy as np

        rng = np.random.default_rng(seed)

        # Step 1: Random initial projection — sparse ±1/sqrt(d) entries
        scale = 1.0 / math.sqrt(dimension)
        embeddings = rng.choice(
            [-scale, scale], size=(n, dimension)
        ).astype(np.float64)

        # Step 2: Iterative neighbor averaging
        for _ in range(iterations):
            new_embeddings = np.zeros_like(embeddings)
            for nid in node_ids:
                idx = id_to_idx[nid]
                neighbors = adj.get(nid, set())
                if not neighbors:
                    new_embeddings[idx] = embeddings[idx]
                    continue
                neighbor_sum = np.zeros(dimension, dtype=np.float64)
                for nbr in neighbors:
                    neighbor_sum += embeddings[id_to_idx[nbr]]
                new_embeddings[idx] = neighbor_sum / len(neighbors)
            embeddings = new_embeddings

            # Optional L2 normalization
            if normalization_strength > 0:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                embeddings = embeddings / norms

        return [
            {
                "id": nid,
                "label": node_labels_map.get(nid, ""),
                "embedding": embeddings[id_to_idx[nid]].tolist(),
            }
            for nid in node_ids
        ]

    def _fast_rp_pure(
        self,
        adj: dict[str, set[str]],
        node_ids: list[str],
        node_labels_map: dict[str, str],
        id_to_idx: dict[str, int],
        n: int,
        dimension: int,
        iterations: int,
        normalization_strength: float,
        seed: int | None,
    ) -> list[dict[str, Any]]:
        """Pure-Python FastRP fallback (no numpy)."""
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(dimension)

        # Step 1: Random initial projection
        embeddings: list[list[float]] = [
            [rng.choice([-scale, scale]) for _ in range(dimension)]
            for _ in range(n)
        ]

        # Step 2: Iterative neighbor averaging
        for _ in range(iterations):
            new_embeddings: list[list[float]] = [
                [0.0] * dimension for _ in range(n)
            ]
            for nid in node_ids:
                idx = id_to_idx[nid]
                neighbors = adj.get(nid, set())
                if not neighbors:
                    new_embeddings[idx] = list(embeddings[idx])
                    continue
                deg = len(neighbors)
                for nbr in neighbors:
                    nbr_emb = embeddings[id_to_idx[nbr]]
                    for d in range(dimension):
                        new_embeddings[idx][d] += nbr_emb[d]
                for d in range(dimension):
                    new_embeddings[idx][d] /= deg
            embeddings = new_embeddings

            # Optional L2 normalization
            if normalization_strength > 0:
                for idx in range(n):
                    norm = math.sqrt(
                        sum(x * x for x in embeddings[idx])
                    )
                    if norm > 1e-10:
                        embeddings[idx] = [x / norm for x in embeddings[idx]]

        return [
            {
                "id": nid,
                "label": node_labels_map.get(nid, ""),
                "embedding": embeddings[id_to_idx[nid]],
            }
            for nid in node_ids
        ]

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
