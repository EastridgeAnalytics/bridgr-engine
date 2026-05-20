"""Smoke tests for the benchmark harness.

Runs both benchmarks on small graphs (100-500 nodes) to verify correctness
without waiting for full-scale runs. Always runs in CI — no env var gate.

Usage:
    pytest benchmarks/test_benchmarks.py -v -s
"""

from __future__ import annotations

import pytest

from bridgr.database import Database

from benchmarks.bench_utils import (
    AlgorithmResult,
    TimingResult,
    algo_extension_available,
    format_algorithm_table,
    format_timing_table,
    generate_power_law_graph,
    generate_social_network,
    load_algo_extension,
    time_callable,
    time_query,
)


# ---------------------------------------------------------------------------
# bench_utils tests
# ---------------------------------------------------------------------------


class TestGenerateSocialNetwork:
    """Verify social network generation produces correct schema and counts."""

    def test_small_graph_creates_nodes_and_edges(self):
        db = Database(":memory:")
        stats = generate_social_network(db, node_count=100, edge_count=200, seed=42)
        assert stats["nodes"] == 100
        assert stats["edges"] > 0
        assert stats["generation_time_s"] > 0

        # Verify Person nodes exist
        rows = db.query("MATCH (p:Person) RETURN count(p) AS cnt")
        assert rows[0]["cnt"] == 100

        # Verify KNOWS edges exist
        rows = db.query("MATCH ()-[k:KNOWS]->() RETURN count(k) AS cnt")
        assert rows[0]["cnt"] > 0

        db.close()

    def test_node_properties_are_populated(self):
        db = Database(":memory:")
        generate_social_network(db, node_count=50, edge_count=100, seed=99)

        rows = db.query("MATCH (p:Person) RETURN p.id, p.name, p.age, p.city LIMIT 5")
        assert len(rows) == 5
        for row in rows:
            assert row["p.id"] is not None
            assert isinstance(row["p.name"], str)
            assert 18 <= row["p.age"] <= 80
            assert isinstance(row["p.city"], str)

        db.close()

    def test_no_self_loops(self):
        db = Database(":memory:")
        generate_social_network(db, node_count=100, edge_count=300, seed=42)

        # Check no self-loops in first 100 edges
        rows = db.query(
            "MATCH (a:Person)-[k:KNOWS]->(b:Person) "
            "WHERE a.id = b.id RETURN count(*) AS cnt"
        )
        assert rows[0]["cnt"] == 0

        db.close()


class TestGeneratePowerLawGraph:
    """Verify power-law graph generation."""

    def test_small_scale_graph(self):
        db = Database(":memory:")
        stats = generate_power_law_graph(db, scale=8, edge_factor=8, seed=42)
        assert stats["nodes"] == 256  # 2^8
        assert stats["edges"] > 0
        assert stats["scale"] == 8

        rows = db.query("MATCH (n:Node) RETURN count(n) AS cnt")
        assert rows[0]["cnt"] == 256

        db.close()

    def test_edges_exist(self):
        db = Database(":memory:")
        generate_power_law_graph(db, scale=8, edge_factor=4, seed=42)

        rows = db.query("MATCH ()-[e:EDGE]->() RETURN count(e) AS cnt")
        assert rows[0]["cnt"] > 0

        db.close()

    def test_power_law_distribution(self):
        """High-degree nodes should exist (power-law characteristic)."""
        db = Database(":memory:")
        generate_power_law_graph(db, scale=10, edge_factor=8, seed=42)

        # Check degree distribution has variance (not uniform)
        rows = db.query(
            "MATCH (n:Node)-[:EDGE]-(m) "
            "RETURN n.id AS id, count(m) AS degree "
            "ORDER BY degree DESC LIMIT 5"
        )
        if len(rows) >= 2:
            top_degree = rows[0]["degree"]
            fifth_degree = rows[-1]["degree"]
            # Power-law: top degree should be significantly higher than 5th
            assert top_degree > fifth_degree, "Expected power-law skew in degree distribution"

        db.close()


class TestTimeQuery:
    """Verify query timing utility."""

    def test_returns_percentiles(self):
        db = Database(":memory:")
        db.create_node_table("T", {"id": "INT64 PRIMARY KEY"})
        db.execute("CREATE (:T {id: 1})")

        result = time_query(db, "MATCH (t:T) RETURN t.id", warmup=1, runs=3)
        assert "median_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert result["runs"] == 3
        assert result["warmup"] == 1
        assert result["median_ms"] >= 0
        assert result["p95_ms"] >= result["median_ms"]
        assert result["p99_ms"] >= result["p95_ms"]

        db.close()

    def test_with_params(self):
        db = Database(":memory:")
        db.create_node_table("T", {"id": "INT64 PRIMARY KEY", "val": "STRING"})
        db.execute("CREATE (:T {id: 1, val: 'hello'})")

        result = time_query(
            db,
            "MATCH (t:T {id: $id}) RETURN t.val",
            warmup=1,
            runs=2,
            params={"id": 1},
        )
        assert result["last_result_count"] == 1

        db.close()


class TestTimeCallable:
    """Verify callable timing utility."""

    def test_times_lambda(self):
        counter = {"n": 0}

        def fn():
            counter["n"] += 1

        result = time_callable(fn, warmup=2, runs=5)
        assert counter["n"] == 7  # 2 warmup + 5 runs
        assert result["median_ms"] >= 0
        assert result["runs"] == 5


class TestFormatting:
    """Verify result formatting produces valid markdown tables."""

    def test_timing_table(self):
        results = [
            TimingResult("Q1: Lookup", 1.5, 2.0, 2.5),
            TimingResult("Q2: Traverse", 10.0, 15.0, 20.0),
        ]
        output = format_timing_table(results, "Test Results")
        assert "Test Results" in output
        assert "Q1: Lookup" in output
        assert "Q2: Traverse" in output
        assert "Median" in output
        assert "P95" in output
        # Should have table separator
        assert "---" in output

    def test_algorithm_table(self):
        results = [
            AlgorithmResult("BFS", 0.5, 1000, 5000, "Reached 800 nodes"),
        ]
        output = format_algorithm_table(results, "Algo Results")
        assert "Algo Results" in output
        assert "BFS" in output
        assert "Reached 800 nodes" in output


# ---------------------------------------------------------------------------
# Rao benchmark smoke test
# ---------------------------------------------------------------------------


class TestRaoBenchmarkSmoke:
    """Run the Rao benchmark at tiny scale to verify it completes."""

    def test_full_run_at_100_nodes(self):
        from benchmarks.rao_benchmark import run_benchmark

        report = run_benchmark(
            node_count=100,
            edge_count=300,
            warmup=1,
            runs=2,
            seed=42,
        )
        assert "Bridgr Rao Graph-Benchmark Results" in report
        assert "Q1: Single node lookup" in report
        assert "Q5: Shortest path" in report
        assert "Q8: Triangle count" in report
        assert "Q9: Pattern match" in report
        # Should have timing numbers (at least one decimal)
        assert "." in report

    def test_run_at_500_nodes(self):
        """Slightly larger to exercise more query paths."""
        from benchmarks.rao_benchmark import run_benchmark

        report = run_benchmark(
            node_count=500,
            edge_count=2000,
            warmup=1,
            runs=2,
            seed=123,
        )
        assert "Q2: 1-hop neighbors" in report
        assert "Q3: 2-hop" in report


# ---------------------------------------------------------------------------
# Graphalytics benchmark smoke test
# ---------------------------------------------------------------------------


class TestGraphanalyticsBenchmarkSmoke:
    """Run the Graphalytics benchmark at tiny scale."""

    def test_full_run_at_scale_8(self):
        from benchmarks.graphalytics_benchmark import run_benchmark

        report = run_benchmark(scale=8, edge_factor=4, seed=42)
        assert "Bridgr LDBC Graphalytics" in report
        assert "BFS" in report
        assert "PageRank" in report
        assert "WCC" in report
        assert "CDLP" in report
        assert "LCC" in report
        assert "SSSP" in report

    def test_run_at_scale_7(self):
        """Even smaller for fast CI."""
        from benchmarks.graphalytics_benchmark import run_benchmark

        report = run_benchmark(scale=7, edge_factor=4, seed=99)
        assert "128 nodes" in report  # 2^7 = 128


# ---------------------------------------------------------------------------
# Integration: verify algo extension detection
# ---------------------------------------------------------------------------


class TestAlgoExtensionDetection:
    """Verify the algo extension detection utility works."""

    def test_detection_returns_bool(self):
        result = algo_extension_available()
        assert isinstance(result, bool)

    def test_load_on_db(self):
        db = Database(":memory:")
        result = load_algo_extension(db)
        assert isinstance(result, bool)
        db.close()
