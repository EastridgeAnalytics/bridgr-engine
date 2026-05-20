"""Shared benchmark utilities for Bridgr engine performance testing.

Provides graph generation (social networks, power-law graphs), query timing
with warmup/runs and percentile stats, and result formatting as markdown tables.

All generators use COPY FROM via temp Parquet files for bulk-load speed.
"""

from __future__ import annotations

import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from bridgr.database import Database
from bridgr.export import _cypher_path

# Optional: numpy for robust percentile computation
try:
    import numpy as _np

    def _percentile(values: list[float], pct: float) -> float:
        return float(_np.percentile(values, pct))

except ImportError:
    _np = None  # type: ignore[assignment]

    def _percentile(values: list[float], pct: float) -> float:
        """Pure-Python linear-interpolation percentile (matches numpy default)."""
        if not values:
            return 0.0
        s = sorted(values)
        k = (pct / 100.0) * (len(s) - 1)
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        frac = k - lo
        return s[lo] * (1 - frac) + s[hi] * frac


def _median(values: list[float]) -> float:
    """Return the median of a list of floats."""
    return _percentile(values, 50.0)


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    """Latency measurements for a single benchmark query."""

    name: str
    median_ms: float
    p95_ms: float
    p99_ms: float
    runs: int = 10
    warmup: int = 3
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: median={self.median_ms:.2f}ms p95={self.p95_ms:.2f}ms p99={self.p99_ms:.2f}ms"


@dataclass
class AlgorithmResult:
    """Timing and summary for an algorithm benchmark."""

    name: str
    time_seconds: float
    graph_nodes: int
    graph_edges: int
    result_summary: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: {self.time_seconds:.3f}s on {self.graph_nodes}N/{self.graph_edges}E — {self.result_summary}"


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------


def generate_social_network(
    db: Database,
    node_count: int = 100_000,
    edge_count: int = 2_400_000,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a random social network and bulk-load via COPY FROM Parquet.

    Creates a Person node table with ``node_count`` nodes and a KNOWS edge
    table with ``edge_count`` undirected edges (stored as directed pairs).

    Args:
        db: Target database (should be empty or have no Person/KNOWS tables).
        node_count: Number of Person nodes.
        edge_count: Number of KNOWS edges.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with ``nodes``, ``edges``, ``generation_time_s``.
    """
    rng = random.Random(seed)
    t0 = time.monotonic()

    # --- Create schema ---
    db.create_node_table("Person", {
        "id": "INT64 PRIMARY KEY",
        "name": "STRING",
        "age": "INT64",
        "city": "STRING",
    })
    db.create_edge_table("KNOWS", "Person", "Person", properties={
        "since": "INT64",
    })

    tmpdir = tempfile.mkdtemp(prefix="bridgr_bench_social_")

    try:
        # --- Generate nodes ---
        cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
            "Jacksonville", "San Francisco", "Columbus", "Charlotte", "Indianapolis",
            "Seattle", "Denver", "Nashville", "Portland", "Memphis",
        ]
        ids = list(range(node_count))
        names = [f"Person_{i}" for i in ids]
        ages = [rng.randint(18, 80) for _ in ids]
        city_col = [rng.choice(cities) for _ in ids]

        node_table = pa.table({
            "id": pa.array(ids, type=pa.int64()),
            "name": pa.array(names, type=pa.string()),
            "age": pa.array(ages, type=pa.int64()),
            "city": pa.array(city_col, type=pa.string()),
        })
        node_path = os.path.join(tmpdir, "persons.parquet")
        pq.write_table(node_table, node_path)
        db.execute(f'COPY Person FROM "{_cypher_path(node_path)}"')

        # --- Generate edges (no self-loops, no exact duplicates) ---
        edge_set: set[tuple[int, int]] = set()
        while len(edge_set) < edge_count:
            batch_size = min(edge_count - len(edge_set), 500_000)
            for _ in range(batch_size):
                a = rng.randint(0, node_count - 1)
                b = rng.randint(0, node_count - 1)
                if a != b:
                    edge_set.add((min(a, b), max(a, b)))

        from_ids = []
        to_ids = []
        sinces = []
        for a, b in edge_set:
            from_ids.append(a)
            to_ids.append(b)
            sinces.append(rng.randint(2000, 2025))

        # Trim to exact count if over (set dedup may produce fewer, loop handles)
        actual_edges = len(from_ids)

        edge_table = pa.table({
            "from": pa.array(from_ids, type=pa.int64()),
            "to": pa.array(to_ids, type=pa.int64()),
            "since": pa.array(sinces, type=pa.int64()),
        })
        edge_path = os.path.join(tmpdir, "knows.parquet")
        pq.write_table(edge_table, edge_path)
        db.execute(f'COPY KNOWS FROM "{_cypher_path(edge_path)}"')

    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = time.monotonic() - t0
    return {
        "nodes": node_count,
        "edges": actual_edges,
        "generation_time_s": elapsed,
    }


def generate_power_law_graph(
    db: Database,
    scale: int = 16,
    *,
    edge_factor: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a Graph500-style power-law graph and bulk-load via Parquet.

    Uses the R-MAT (Recursive Matrix) generator:
      - Nodes: 2^scale
      - Edges: ~nodes * edge_factor
      - Parameters: a=0.57, b=0.19, c=0.19 (standard Graph500 Kronecker)

    The resulting graph has a power-law degree distribution typical of
    real-world social/web graphs.

    Args:
        db: Target database (should be empty or have no Node/EDGE tables).
        scale: Log2 of node count (16 = 65,536 nodes).
        edge_factor: Multiplier for edge count (edges = 2^scale * edge_factor).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with ``nodes``, ``edges``, ``generation_time_s``, ``scale``.
    """
    rng = random.Random(seed)
    t0 = time.monotonic()

    n_nodes = 1 << scale
    n_edges = n_nodes * edge_factor

    # R-MAT parameters (Graph500 standard)
    a, b, c = 0.57, 0.19, 0.19
    # d = 1 - a - b - c = 0.05 (implicit)

    # --- Create schema ---
    db.create_node_table("Node", {
        "id": "INT64 PRIMARY KEY",
    })
    db.create_edge_table("EDGE", "Node", "Node", properties={
        "weight": "DOUBLE",
    })

    tmpdir = tempfile.mkdtemp(prefix="bridgr_bench_powerlaw_")

    try:
        # --- Generate nodes ---
        node_table = pa.table({
            "id": pa.array(list(range(n_nodes)), type=pa.int64()),
        })
        node_path = os.path.join(tmpdir, "nodes.parquet")
        pq.write_table(node_table, node_path)
        db.execute(f'COPY Node FROM "{_cypher_path(node_path)}"')

        # --- R-MAT edge generation ---
        from_ids: list[int] = []
        to_ids: list[int] = []
        weights: list[float] = []
        edge_set: set[tuple[int, int]] = set()

        for _ in range(n_edges * 2):  # over-generate, then dedup
            u, v = 0, 0
            for depth in range(scale):
                r = rng.random()
                if r < a:
                    pass  # top-left quadrant
                elif r < a + b:
                    v += 1 << (scale - 1 - depth)
                elif r < a + b + c:
                    u += 1 << (scale - 1 - depth)
                else:
                    u += 1 << (scale - 1 - depth)
                    v += 1 << (scale - 1 - depth)

            if u != v and (u, v) not in edge_set:
                edge_set.add((u, v))
                from_ids.append(u)
                to_ids.append(v)
                weights.append(rng.random())

            if len(from_ids) >= n_edges:
                break

        # Trim to target
        actual_edges = min(len(from_ids), n_edges)
        from_ids = from_ids[:actual_edges]
        to_ids = to_ids[:actual_edges]
        weights = weights[:actual_edges]

        edge_table = pa.table({
            "from": pa.array(from_ids, type=pa.int64()),
            "to": pa.array(to_ids, type=pa.int64()),
            "weight": pa.array(weights, type=pa.float64()),
        })
        edge_path = os.path.join(tmpdir, "edges.parquet")
        pq.write_table(edge_table, edge_path)
        db.execute(f'COPY EDGE FROM "{_cypher_path(edge_path)}"')

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = time.monotonic() - t0
    return {
        "nodes": n_nodes,
        "edges": actual_edges,
        "generation_time_s": elapsed,
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# Query timing
# ---------------------------------------------------------------------------


def time_query(
    db: Database,
    cypher: str,
    *,
    warmup: int = 3,
    runs: int = 10,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Time a Cypher query with warmup runs, returning latency percentiles.

    Args:
        db: Database to query.
        cypher: Cypher query string.
        warmup: Number of warmup runs (discarded).
        runs: Number of timed runs.
        params: Optional query parameters.

    Returns:
        Dict with ``median_ms``, ``p95_ms``, ``p99_ms``, ``min_ms``, ``max_ms``,
        ``runs``, ``warmup``, ``last_result_count``.
    """
    # Warmup
    for _ in range(warmup):
        db.query(cypher, params)

    # Timed runs
    latencies: list[float] = []
    last_count = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        result = db.query(cypher, params)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        last_count = len(result)

    return {
        "median_ms": round(_median(latencies), 3),
        "p95_ms": round(_percentile(latencies, 95.0), 3),
        "p99_ms": round(_percentile(latencies, 99.0), 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "runs": runs,
        "warmup": warmup,
        "last_result_count": last_count,
    }


def time_callable(
    fn: Any,
    *,
    warmup: int = 3,
    runs: int = 10,
) -> dict[str, Any]:
    """Time an arbitrary callable with warmup, returning latency percentiles.

    Args:
        fn: Zero-argument callable to benchmark.
        warmup: Number of warmup invocations (discarded).
        runs: Number of timed invocations.

    Returns:
        Dict with ``median_ms``, ``p95_ms``, ``p99_ms``, ``min_ms``, ``max_ms``.
    """
    for _ in range(warmup):
        fn()

    latencies: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

    return {
        "median_ms": round(_median(latencies), 3),
        "p95_ms": round(_percentile(latencies, 95.0), 3),
        "p99_ms": round(_percentile(latencies, 99.0), 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "runs": runs,
        "warmup": warmup,
    }


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def format_timing_table(results: list[TimingResult], title: str) -> str:
    """Format a list of TimingResults as a markdown table.

    Args:
        results: List of TimingResult objects.
        title: Table title.

    Returns:
        Markdown-formatted table string.
    """
    lines = [
        title,
        "=" * len(title),
        "",
        "| Query                                | Median (ms) | P95 (ms)  | P99 (ms)  |",
        "|--------------------------------------|-------------|-----------|-----------|",
    ]
    for r in results:
        name = r.name.ljust(36)
        med = f"{r.median_ms:>11.3f}"
        p95 = f"{r.p95_ms:>9.3f}"
        p99 = f"{r.p99_ms:>9.3f}"
        lines.append(f"| {name} | {med} | {p95} | {p99} |")

    return "\n".join(lines)


def format_algorithm_table(results: list[AlgorithmResult], title: str) -> str:
    """Format a list of AlgorithmResults as a markdown table.

    Args:
        results: List of AlgorithmResult objects.
        title: Table title.

    Returns:
        Markdown-formatted table string.
    """
    lines = [
        title,
        "=" * len(title),
        "",
        "| Algorithm                | Nodes     | Edges      | Time (s)  | Result Summary                    |",
        "|--------------------------|-----------|------------|-----------|-----------------------------------|",
    ]
    for r in results:
        name = r.name.ljust(24)
        nodes = f"{r.graph_nodes:>9,}"
        edges = f"{r.graph_edges:>10,}"
        t = f"{r.time_seconds:>9.3f}"
        summary = r.result_summary[:35].ljust(35)
        lines.append(f"| {name} | {nodes} | {edges} | {t} | {summary} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Algo extension detection
# ---------------------------------------------------------------------------


def algo_extension_available() -> bool:
    """Return True if the LadybugDB algo extension can be loaded."""
    try:
        test_db = Database(":memory:")
        test_db.execute("INSTALL algo")
        test_db.execute("LOAD EXTENSION algo")
        test_db.close()
        return True
    except Exception:
        return False


def load_algo_extension(db: Database) -> bool:
    """Attempt to load the algo extension on an existing database.

    Returns True if successful, False otherwise.
    """
    try:
        db.execute("INSTALL algo")
    except RuntimeError:
        pass
    try:
        db.execute("LOAD EXTENSION algo")
        return True
    except RuntimeError:
        return False
