"""Bridgr Engine benchmarks: import, traversal, algorithms at scale.

Tests import throughput, query latency, and algorithm performance on
realistic graph data (Customer → Household → ZipCode) at 1M/5M/10M.

Run with: RUN_BENCHMARKS=1 pytest benchmarks/test_engine_benchmarks.py -v -s
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import tracemalloc
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bridgr.database import Database
from bridgr.export import _cypher_path

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_BENCHMARKS"),
    reason="benchmark test — set RUN_BENCHMARKS=1 to include",
)

SCALES = [1_000_000]  # Start with 1M; extend to [1_000_000, 5_000_000, 10_000_000]


@pytest.fixture(scope="module", params=SCALES, ids=[f"{s//1_000_000}M" for s in SCALES])
def engine_fixture(request):
    """Generate graph data and import into Bridgr for benchmarking."""
    n_customers = request.param
    label = f"{n_customers // 1_000_000}M"

    # Generate data
    from benchmarks.generate_er_graph import generate

    tmpdir = tempfile.mkdtemp(prefix=f"engine_bench_{label}_")
    print(f"\nGenerating {label} graph data...")
    data_dir = generate(n_customers, output_dir=tmpdir)

    # Create database and import
    db_path = os.path.join(tmpdir, "benchmark.lbug")
    db = Database(db_path)

    # Install algo extension for algorithm benchmarks
    try:
        db.execute("INSTALL algo")
        db.execute("LOAD EXTENSION algo")
    except Exception:
        pass

    # Schema
    db.create_node_table("Customer", {
        "customer_id": "STRING PRIMARY KEY",
        "name": "STRING",
        "email": "STRING",
        "phone": "STRING",
        "zip": "STRING",
    })
    db.create_node_table("Household", {
        "household_id": "STRING PRIMARY KEY",
        "zip": "STRING",
        "member_count": "INT64",
        "primary_phone": "STRING",
    })
    db.create_node_table("ZipCode", {
        "zip": "STRING PRIMARY KEY",
        "city": "STRING",
        "state": "STRING",
    })
    db.create_edge_table("MEMBER_OF", "Customer", "Household")
    db.create_edge_table("LOCATED_IN", "Household", "ZipCode")

    # Import nodes
    tracemalloc.start()
    t0 = time.monotonic()

    for table, parquet in [
        ("Customer", "customers.parquet"),
        ("Household", "households.parquet"),
        ("ZipCode", "zipcodes.parquet"),
    ]:
        ppath = data_dir / parquet
        if ppath.exists():
            cp = _cypher_path(str(ppath))
            db.execute(f'COPY {table} FROM "{cp}"')

    import_nodes_time = time.monotonic() - t0

    # Import edges via COPY FROM (positional: col1=source PK, col2=target PK)
    t1 = time.monotonic()
    for edge_table, parquet in [
        ("MEMBER_OF", "member_of.parquet"),
        ("LOCATED_IN", "located_in.parquet"),
    ]:
        ppath = data_dir / parquet
        if ppath.exists():
            cp = _cypher_path(str(ppath))
            db.execute(f'COPY {edge_table} FROM "{cp}"')

    import_edges_time = time.monotonic() - t1
    import_total_time = time.monotonic() - t0

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Get counts
    cust_count = db.query("MATCH (c:Customer) RETURN count(c) AS cnt")[0]["cnt"]
    hh_count = db.query("MATCH (h:Household) RETURN count(h) AS cnt")[0]["cnt"]

    print(f"\n{label} engine import:")
    print(f"  Customers: {cust_count:,}")
    print(f"  Households: {hh_count:,}")
    print(f"  Node import: {import_nodes_time:.1f}s ({cust_count/import_nodes_time:.0f} rec/s)")
    print(f"  Edge import: {import_edges_time:.1f}s")
    print(f"  Total import: {import_total_time:.1f}s")
    print(f"  Peak memory: {peak_bytes / (1024**3):.2f} GB")

    yield {
        "db": db,
        "label": label,
        "n_customers": cust_count,
        "import_nodes_time": import_nodes_time,
        "import_edges_time": import_edges_time,
        "import_total_time": import_total_time,
        "peak_memory_gb": peak_bytes / (1024 ** 3),
        "tmpdir": tmpdir,
    }

    db.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestImport:
    def test_import_completes(self, engine_fixture):
        assert engine_fixture["n_customers"] > 0

    def test_import_throughput(self, engine_fixture):
        rate = engine_fixture["n_customers"] / engine_fixture["import_nodes_time"]
        print(f"\n  Import throughput: {rate:,.0f} records/second")
        assert rate > 10_000, f"Import too slow: {rate:.0f} rec/s"


class TestQueries:
    def test_point_lookup(self, engine_fixture):
        db = engine_fixture["db"]
        t0 = time.monotonic()
        for _ in range(100):
            db.query("MATCH (c:Customer {customer_id: 'C00000042'}) RETURN c.name")
        avg_ms = (time.monotonic() - t0) / 100 * 1000
        print(f"\n  Point lookup: {avg_ms:.2f} ms avg")
        assert avg_ms < 10, f"Point lookup too slow: {avg_ms:.2f} ms"

    def test_1hop_traversal(self, engine_fixture):
        db = engine_fixture["db"]
        t0 = time.monotonic()
        for _ in range(50):
            db.query(
                "MATCH (c:Customer {customer_id: 'C00000042'})-[:MEMBER_OF]->(h:Household)<-[:MEMBER_OF]-(other:Customer) "
                "RETURN other.customer_id LIMIT 10"
            )
        avg_ms = (time.monotonic() - t0) / 50 * 1000
        print(f"\n  1-hop traversal (household members): {avg_ms:.2f} ms avg")
        assert avg_ms < 50, f"1-hop too slow: {avg_ms:.2f} ms"

    def test_2hop_traversal(self, engine_fixture):
        db = engine_fixture["db"]
        t0 = time.monotonic()
        for _ in range(10):
            db.query(
                "MATCH (c:Customer {customer_id: 'C00000042'})-[:MEMBER_OF]->(h:Household)"
                "-[:LOCATED_IN]->(z:ZipCode)<-[:LOCATED_IN]-(other_h:Household) "
                "RETURN count(other_h) AS cnt"
            )
        avg_ms = (time.monotonic() - t0) / 10 * 1000
        print(f"\n  2-hop traversal (zip neighbors): {avg_ms:.2f} ms avg")
        assert avg_ms < 500, f"2-hop too slow: {avg_ms:.2f} ms"

    def test_aggregation(self, engine_fixture):
        db = engine_fixture["db"]
        t0 = time.monotonic()
        result = db.query(
            "MATCH (h:Household) "
            "RETURN h.zip AS zip, count(h) AS household_count "
            "ORDER BY household_count DESC LIMIT 10"
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f"\n  Aggregation (top 10 zips): {elapsed_ms:.2f} ms, top: {result[0] if result else 'none'}")
        assert len(result) > 0, "Aggregation returned no results"
        assert elapsed_ms < 5000, f"Aggregation too slow: {elapsed_ms:.2f} ms"


class TestAlgorithms:
    def test_pagerank(self, engine_fixture):
        db = engine_fixture["db"]
        db.execute("CALL PROJECT_GRAPH('_bench_pr', ['Customer', 'Household'], ['MEMBER_OF'])")
        t0 = time.monotonic()
        result = db.query("CALL PAGE_RANK('_bench_pr') RETURN node, rank AS score ORDER BY score DESC LIMIT 10")
        elapsed = time.monotonic() - t0
        db.execute("CALL DROP_PROJECTED_GRAPH('_bench_pr')")
        print(f"\n  PageRank: {elapsed:.2f}s, {len(result)} results, top score: {result[0]['score'] if result else 'none'}")
        assert elapsed < 120, f"PageRank too slow: {elapsed:.2f}s"
        assert len(result) > 0, "PageRank returned no results"

    def test_fraud_community_detection_louvain(self, engine_fixture):
        """Louvain community detection via similarity graph — the query Josh cares about."""
        db = engine_fixture["db"]
        from bridgr.similarity_graph import SimilarityGraphBuilder

        # Build similarity edges from shared zip + phone
        builder = SimilarityGraphBuilder(db)
        t0 = time.monotonic()
        stats = builder.build("Customer", ["zip", "phone"], edge_label="SIMILAR_TO", min_weight=1, max_bucket_size=5000)
        sim_time = time.monotonic() - t0

        if stats["edges_created"] == 0:
            pytest.skip("No similarity edges created (attributes too unique)")

        # Run Louvain on unipartite similarity graph
        db.execute("CALL PROJECT_GRAPH('_bench_louv', ['Customer'], ['SIMILAR_TO'])")
        t1 = time.monotonic()
        result = db.query("CALL LOUVAIN('_bench_louv') RETURN louvain_id, count(*) AS size ORDER BY size DESC LIMIT 10")
        louvain_time = time.monotonic() - t1
        db.execute("CALL DROP_PROJECTED_GRAPH('_bench_louv')")

        print(f"\n  Similarity graph: {stats['edges_created']} edges in {sim_time:.2f}s")
        print(f"  Fraud community detection (Louvain): {louvain_time:.2f}s, top: {result[:3] if result else 'none'}")
        assert louvain_time < 120, f"Louvain too slow: {louvain_time:.2f}s"
        assert len(result) > 0, "Louvain returned no communities"

    def test_fraud_community_detection_wcc(self, engine_fixture):
        """WCC on bipartite graph as baseline comparison."""
        db = engine_fixture["db"]
        db.execute("CALL PROJECT_GRAPH('_bench_comm', ['Customer', 'Household'], ['MEMBER_OF'])")
        t0 = time.monotonic()
        result = db.query("CALL WEAKLY_CONNECTED_COMPONENTS('_bench_comm') RETURN group_id, count(*) AS size ORDER BY size DESC LIMIT 10")
        elapsed = time.monotonic() - t0
        db.execute("CALL DROP_PROJECTED_GRAPH('_bench_comm')")
        print(f"\n  WCC (bipartite baseline): {elapsed:.2f}s, top: {result[:3] if result else 'none'}")
        assert elapsed < 120, f"WCC too slow: {elapsed:.2f}s"
        assert len(result) > 0, "WCC returned no results"

    def test_wcc(self, engine_fixture):
        db = engine_fixture["db"]
        db.execute("CALL PROJECT_GRAPH('_bench_wcc2', ['Customer', 'Household'], ['MEMBER_OF'])")
        t0 = time.monotonic()
        result = db.query("CALL WEAKLY_CONNECTED_COMPONENTS('_bench_wcc2') RETURN group_id, count(*) AS size ORDER BY size DESC LIMIT 5")
        elapsed = time.monotonic() - t0
        db.execute("CALL DROP_PROJECTED_GRAPH('_bench_wcc2')")
        print(f"\n  WCC: {elapsed:.2f}s, largest components: {result[:3] if result else 'none'}")
        assert elapsed < 120, f"WCC too slow: {elapsed:.2f}s"
        assert len(result) > 0, "WCC returned no results"
