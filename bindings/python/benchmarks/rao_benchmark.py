"""Rao graph-benchmark: 9 queries on a social network graph.

Replicates the 9 queries from Rao's graph-benchmark
(github.com/prrao87/graph-benchmark) on a synthetic social network.

Default scale: 100K persons, 2.4M KNOWS edges. Configurable via CLI args.

The 9 queries cover:
  Q1: Single node lookup by ID
  Q2: 1-hop neighbors
  Q3: 2-hop traversal (friends-of-friends)
  Q4: 3-hop traversal
  Q5: Shortest path between two nodes
  Q6: PageRank (top 10)
  Q7: Community detection (WCC or Louvain)
  Q8: Triangle count
  Q9: Pattern match (triangles containing a specific node)

Usage:
    python -m benchmarks.rao_benchmark                  # full benchmark (100K nodes)
    python -m benchmarks.rao_benchmark --nodes 1000     # quick test
    python -m benchmarks.rao_benchmark --nodes 10000 --edges 100000
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from typing import Any

from bridgr.database import Database

from benchmarks.bench_utils import (
    AlgorithmResult,
    TimingResult,
    algo_extension_available,
    format_algorithm_table,
    format_timing_table,
    generate_social_network,
    load_algo_extension,
    time_callable,
    time_query,
)


def _pick_seed_nodes(db: Database, count: int = 5) -> list[int]:
    """Pick seed node IDs spread across the graph for diverse benchmarks."""
    rows = db.query("MATCH (p:Person) RETURN p.id AS id ORDER BY p.id")
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    step = max(1, len(ids) // (count + 1))
    return [ids[step * (i + 1)] for i in range(min(count, len(ids)))]


def run_benchmark(
    node_count: int = 100_000,
    edge_count: int = 2_400_000,
    *,
    warmup: int = 3,
    runs: int = 10,
    seed: int = 42,
) -> str:
    """Run the full 9-query Rao benchmark and return formatted results.

    Args:
        node_count: Number of Person nodes to generate.
        edge_count: Number of KNOWS edges to generate.
        warmup: Warmup iterations per query.
        runs: Timed iterations per query.
        seed: RNG seed for reproducibility.

    Returns:
        Formatted benchmark report as a string.
    """
    # --- Setup ---
    print(f"Setting up: generating {node_count:,} nodes, {edge_count:,} edges...")
    db = Database(":memory:")
    gen_stats = generate_social_network(db, node_count, edge_count, seed=seed)
    print(f"Graph loaded in {gen_stats['generation_time_s']:.1f}s "
          f"({gen_stats['nodes']:,} nodes, {gen_stats['edges']:,} edges)")

    has_algo = load_algo_extension(db)
    if has_algo:
        print("Algo extension loaded.")
    else:
        print("Algo extension not available. Q6/Q7 will use Cypher fallbacks.")

    # Pick seed nodes for parameterized queries
    seeds = _pick_seed_nodes(db, 5)
    if len(seeds) < 2:
        print("ERROR: Not enough nodes generated. Aborting.")
        db.close()
        return "Benchmark failed: insufficient nodes."

    node_a = seeds[0]
    node_b = seeds[-1]

    timing_results: list[TimingResult] = []
    algo_results: list[AlgorithmResult] = []

    # ------------------------------------------------------------------
    # Q1: Single node lookup by ID
    # ------------------------------------------------------------------
    print("  Q1: Single node lookup...")
    q1 = time_query(
        db,
        "MATCH (p:Person {id: $id}) RETURN p.id, p.name, p.age, p.city",
        warmup=warmup,
        runs=runs,
        params={"id": node_a},
    )
    timing_results.append(TimingResult(
        name="Q1: Single node lookup",
        median_ms=q1["median_ms"],
        p95_ms=q1["p95_ms"],
        p99_ms=q1["p99_ms"],
        runs=runs,
        warmup=warmup,
    ))

    # ------------------------------------------------------------------
    # Q2: 1-hop neighbors
    # ------------------------------------------------------------------
    print("  Q2: 1-hop neighbors...")
    q2 = time_query(
        db,
        "MATCH (p:Person {id: $id})-[:KNOWS]-(friend) RETURN friend.id, friend.name",
        warmup=warmup,
        runs=runs,
        params={"id": node_a},
    )
    timing_results.append(TimingResult(
        name="Q2: 1-hop neighbors",
        median_ms=q2["median_ms"],
        p95_ms=q2["p95_ms"],
        p99_ms=q2["p99_ms"],
        extra={"result_count": q2["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # Q3: 2-hop traversal (friends-of-friends)
    # ------------------------------------------------------------------
    print("  Q3: 2-hop traversal...")
    q3 = time_query(
        db,
        "MATCH (p:Person {id: $id})-[:KNOWS]-()-[:KNOWS]-(fof) "
        "WHERE fof.id <> $id "
        "RETURN DISTINCT fof.id LIMIT 100",
        warmup=warmup,
        runs=runs,
        params={"id": node_a},
    )
    timing_results.append(TimingResult(
        name="Q3: 2-hop (friends-of-friends)",
        median_ms=q3["median_ms"],
        p95_ms=q3["p95_ms"],
        p99_ms=q3["p99_ms"],
        extra={"result_count": q3["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # Q4: 3-hop traversal
    # ------------------------------------------------------------------
    print("  Q4: 3-hop traversal...")
    q4 = time_query(
        db,
        "MATCH (p:Person {id: $id})-[:KNOWS*1..3]-(distant) "
        "WHERE distant.id <> $id "
        "RETURN DISTINCT distant.id LIMIT 100",
        warmup=warmup,
        runs=runs,
        params={"id": node_a},
    )
    timing_results.append(TimingResult(
        name="Q4: 3-hop traversal",
        median_ms=q4["median_ms"],
        p95_ms=q4["p95_ms"],
        p99_ms=q4["p99_ms"],
        extra={"result_count": q4["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # Q5: Shortest path between two nodes
    # ------------------------------------------------------------------
    print("  Q5: Shortest path...")
    q5 = time_query(
        db,
        f"MATCH p = (a:Person {{id: $from_id}})"
        f"-[:KNOWS* SHORTEST 1..10]-"
        f"(b:Person {{id: $to_id}}) "
        f"RETURN length(p) AS path_length",
        warmup=warmup,
        runs=runs,
        params={"from_id": node_a, "to_id": node_b},
    )
    timing_results.append(TimingResult(
        name="Q5: Shortest path",
        median_ms=q5["median_ms"],
        p95_ms=q5["p95_ms"],
        p99_ms=q5["p99_ms"],
        extra={"result_count": q5["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # Q6: PageRank (top 10)
    # ------------------------------------------------------------------
    print("  Q6: PageRank...")
    if has_algo:
        # Use native algo extension
        def _run_pagerank() -> list[dict[str, Any]]:
            graph_name = "_bench_rao_pr"
            try:
                db.execute(f"CALL DROP_PROJECTED_GRAPH('{graph_name}')")
            except RuntimeError:
                pass
            db.execute(f"CALL PROJECT_GRAPH('{graph_name}', ['Person'], ['KNOWS'])")
            result = db.query(
                f"CALL PAGE_RANK('{graph_name}') "
                f"RETURN node.id AS node_id, rank AS score "
                f"ORDER BY score DESC LIMIT 10"
            )
            db.execute(f"CALL DROP_PROJECTED_GRAPH('{graph_name}')")
            return result

        q6_timing = time_callable(_run_pagerank, warmup=warmup, runs=runs)
        q6_method = "native (algo extension)"
    else:
        # Cypher-based degree approximation as PageRank proxy
        def _run_pagerank_cypher() -> list[dict[str, Any]]:
            return db.query(
                "MATCH (p:Person)-[:KNOWS]-(n) "
                "RETURN p.id AS node_id, count(n) AS degree "
                "ORDER BY degree DESC LIMIT 10"
            )

        q6_timing = time_callable(_run_pagerank_cypher, warmup=warmup, runs=runs)
        q6_method = "cypher degree proxy"

    timing_results.append(TimingResult(
        name=f"Q6: PageRank ({q6_method})",
        median_ms=q6_timing["median_ms"],
        p95_ms=q6_timing["p95_ms"],
        p99_ms=q6_timing["p99_ms"],
    ))

    # ------------------------------------------------------------------
    # Q7: Community detection (WCC or Louvain)
    # ------------------------------------------------------------------
    print("  Q7: Community detection...")
    if has_algo:
        def _run_wcc() -> list[dict[str, Any]]:
            graph_name = "_bench_rao_wcc"
            try:
                db.execute(f"CALL DROP_PROJECTED_GRAPH('{graph_name}')")
            except RuntimeError:
                pass
            db.execute(f"CALL PROJECT_GRAPH('{graph_name}', ['Person'], ['KNOWS'])")
            result = db.query(
                f"CALL WEAKLY_CONNECTED_COMPONENTS('{graph_name}') "
                f"RETURN group_id, count(*) AS size "
                f"ORDER BY size DESC LIMIT 10"
            )
            db.execute(f"CALL DROP_PROJECTED_GRAPH('{graph_name}')")
            return result

        q7_timing = time_callable(_run_wcc, warmup=warmup, runs=runs)
        q7_method = "WCC (native)"
    else:
        # Cypher-based connected component approximation via BFS label propagation
        def _run_community_cypher() -> list[dict[str, Any]]:
            return db.query(
                "MATCH (p:Person)-[:KNOWS]-(n) "
                "WITH p.city AS community, count(*) AS size "
                "RETURN community, size ORDER BY size DESC LIMIT 10"
            )

        q7_timing = time_callable(_run_community_cypher, warmup=warmup, runs=runs)
        q7_method = "city grouping proxy"

    timing_results.append(TimingResult(
        name=f"Q7: Community detect ({q7_method})",
        median_ms=q7_timing["median_ms"],
        p95_ms=q7_timing["p95_ms"],
        p99_ms=q7_timing["p99_ms"],
    ))

    # ------------------------------------------------------------------
    # Q8: Triangle count (global)
    # ------------------------------------------------------------------
    print("  Q8: Triangle count...")
    q8 = time_query(
        db,
        "MATCH (a:Person)-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a) "
        "WHERE a.id < b.id AND b.id < c.id "
        "RETURN count(*) AS triangle_count",
        warmup=warmup,
        runs=runs,
    )
    timing_results.append(TimingResult(
        name="Q8: Triangle count (global)",
        median_ms=q8["median_ms"],
        p95_ms=q8["p95_ms"],
        p99_ms=q8["p99_ms"],
        extra={"result_count": q8["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # Q9: Pattern match — triangles containing a specific node
    # ------------------------------------------------------------------
    print("  Q9: Pattern match (node triangles)...")
    q9 = time_query(
        db,
        "MATCH (a:Person {id: $id})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a) "
        "WHERE b.id < c.id "
        "RETURN b.id AS b, c.id AS c",
        warmup=warmup,
        runs=runs,
        params={"id": node_a},
    )
    timing_results.append(TimingResult(
        name="Q9: Pattern match (triangles)",
        median_ms=q9["median_ms"],
        p95_ms=q9["p95_ms"],
        p99_ms=q9["p99_ms"],
        extra={"result_count": q9["last_result_count"]},
    ))

    # --- Format output ---
    header = (
        f"Bridgr Rao Graph-Benchmark Results\n"
        f"===================================\n"
        f"Nodes: {gen_stats['nodes']:,} | Edges: {gen_stats['edges']:,}\n"
        f"Engine: LadybugDB (Kuzu fork) v0.1\n"
        f"Date: {date.today().isoformat()}\n"
        f"Graph generation: {gen_stats['generation_time_s']:.1f}s\n"
        f"Algo extension: {'yes' if has_algo else 'no'}\n"
        f"Warmup: {warmup} | Runs: {runs}\n"
    )

    table = format_timing_table(timing_results, "Query Latencies")

    db.close()

    report = f"{header}\n{table}\n"
    return report


def main() -> None:
    """CLI entry point for the Rao benchmark."""
    parser = argparse.ArgumentParser(
        description="Bridgr Rao Graph-Benchmark: 9 queries on a social network graph."
    )
    parser.add_argument(
        "--nodes", type=int, default=100_000,
        help="Number of Person nodes (default: 100000)",
    )
    parser.add_argument(
        "--edges", type=int, default=2_400_000,
        help="Number of KNOWS edges (default: 2400000)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup iterations per query (default: 3)",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Timed iterations per query (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    report = run_benchmark(
        node_count=args.nodes,
        edge_count=args.edges,
        warmup=args.warmup,
        runs=args.runs,
        seed=args.seed,
    )
    print(report)


if __name__ == "__main__":
    main()
