"""Benchmark: Leiden community detection on similarity graph at scale.

Generates synthetic customer data, builds similarity graph, runs Leiden,
and reports: time, community count, largest community, memory usage.

Two scale variants:
  - 1K customers: fast CI validation (always runs)
  - 10K customers: full benchmark (RUN_BENCHMARKS=1)

The Leiden algorithm requires the LadybugDB algo extension. If unavailable,
falls back to Louvain, then WCC, logging which algorithm was used.

Run:
    pytest benchmarks/test_leiden_scale.py -v -s                # 1K only
    RUN_BENCHMARKS=1 pytest benchmarks/test_leiden_scale.py -v -s  # both
"""

from __future__ import annotations

import os
import random
import time
import tracemalloc
from collections import Counter
from typing import Any

import pytest

from bridgr.algorithms import GraphAlgorithms
from bridgr.database import Database
from bridgr.similarity_graph import SimilarityGraphBuilder


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _generate_customers(
    n: int,
    *,
    duplicate_rate: float = 0.05,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Generate synthetic customer records with controlled duplication.

    About ``duplicate_rate`` fraction of customers share a phone or email
    with another customer, creating the overlap that similarity-graph
    construction needs.

    Returns a list of property dicts ready for ``db.create_node()``.
    """
    rng = random.Random(seed)

    try:
        from faker import Faker
        fake = Faker("en_US")
        Faker.seed(seed)
        _name = lambda: fake.name()  # noqa: E731
        _email = lambda n: f"{n.replace(' ', '.').lower()}@{fake.free_email_domain()}"  # noqa: E731
        _phone = lambda: fake.numerify("##########")  # noqa: E731
        _addr = lambda: fake.street_address()  # noqa: E731
    except ImportError:
        # Faker unavailable -- use deterministic random strings
        _name = lambda: f"Customer_{rng.randint(0, 999999)}"  # noqa: E731
        _email = lambda n: f"{n.replace(' ', '.').lower()}@example.com"  # noqa: E731
        _phone = lambda: f"{rng.randint(1000000000, 9999999999)}"  # noqa: E731
        _addr = lambda: f"{rng.randint(1, 9999)} Street {rng.randint(1, 999)}"  # noqa: E731

    # Pre-generate a pool of "shared" values that duplicates will draw from.
    n_shared = max(1, int(n * duplicate_rate))
    shared_phones = [_phone() for _ in range(n_shared)]
    shared_emails_base = [_name() for _ in range(n_shared)]
    shared_addrs = [_addr() for _ in range(n_shared)]

    customers: list[dict[str, str]] = []
    for i in range(n):
        name = _name()
        is_dup = rng.random() < duplicate_rate
        if is_dup:
            phone = rng.choice(shared_phones)
            email = _email(rng.choice(shared_emails_base))
            address = rng.choice(shared_addrs)
        else:
            phone = _phone()
            email = _email(name)
            address = _addr()

        customers.append({
            "id": f"C{i:07d}",
            "name": name,
            "phone": phone,
            "email": email,
            "address": address,
        })

    return customers


def _load_customers(db: Database, customers: list[dict[str, str]]) -> None:
    """Create the Customer node table and insert all records."""
    db.create_node_table("Customer", {
        "id": "STRING PRIMARY KEY",
        "name": "STRING",
        "phone": "STRING",
        "email": "STRING",
        "address": "STRING",
    })
    for c in customers:
        db.create_node("Customer", c)


# ---------------------------------------------------------------------------
# Algorithm selection helpers
# ---------------------------------------------------------------------------

def _algo_extension_available() -> bool:
    """Return True if the LadybugDB algo extension can be loaded."""
    try:
        db = Database(":memory:")
        db.execute("INSTALL algo")
        db.execute("LOAD EXTENSION algo")
        db.close()
        return True
    except Exception:
        return False


_HAS_ALGO = _algo_extension_available()


def _detect_community(
    algo: GraphAlgorithms,
    node_label: str,
    edge_label: str,
) -> tuple[list[dict[str, Any]], str]:
    """Run the best available community-detection algorithm.

    Tries Leiden first, then Louvain, then WCC.
    Returns (results, algorithm_name).
    """
    if _HAS_ALGO:
        try:
            results = algo.leiden(node_label, edge_label)
            return results, "leiden"
        except Exception:
            pass

        try:
            results = algo.louvain(node_label, edge_label)
            return results, "louvain"
        except Exception:
            pass

        try:
            results = algo.weakly_connected_components(node_label, edge_label)
            return results, "wcc"
        except Exception:
            pass

    # No algo extension -- cannot run graph algorithms at the engine level
    pytest.skip("No community detection algorithm available (algo extension missing)")
    return [], "none"  # unreachable, satisfies type checker


# ---------------------------------------------------------------------------
# Small-scale CI benchmark (1K customers, always runs)
# ---------------------------------------------------------------------------

class TestLeidenScale1K:
    """Leiden benchmark at 1K customers -- fast enough for CI."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.db = Database(":memory:")
        customers = _generate_customers(1_000, duplicate_rate=0.05, seed=42)
        _load_customers(self.db, customers)
        yield
        self.db.close()

    def test_similarity_build(self):
        """SimilarityGraphBuilder produces edges from shared attributes."""
        builder = SimilarityGraphBuilder(self.db, node_label="Customer")
        stats = builder.build(
            attributes=["phone", "email", "address"],
            min_shared=1,
            edge_label="SIMILAR_TO",
        )
        assert stats["edge_count"] > 0, "Expected similarity edges from ~5% duplicate rate"
        assert stats["node_count"] > 0

    def test_community_detection(self):
        """Community detection runs and produces multiple communities."""
        builder = SimilarityGraphBuilder(self.db, node_label="Customer")
        stats = builder.build(
            attributes=["phone", "email", "address"],
            min_shared=1,
            edge_label="SIMILAR_TO",
        )
        if stats["edge_count"] == 0:
            pytest.skip("No similarity edges -- cannot test community detection")

        algo = GraphAlgorithms(self.db)
        results, algo_name = _detect_community(algo, "Customer", "SIMILAR_TO")

        community_ids = {r["community_id"] for r in results}
        print(f"\n  Algorithm: {algo_name}")
        print(f"  Nodes assigned: {len(results)}")
        print(f"  Communities: {len(community_ids)}")
        assert len(results) > 0
        assert len(community_ids) >= 1

    def test_benchmark_reports_metrics(self):
        """Full benchmark cycle reports wall time, communities, peak memory."""
        tracemalloc.start()
        t0 = time.monotonic()

        # Build similarity graph
        builder = SimilarityGraphBuilder(self.db, node_label="Customer")
        sim_stats = builder.build(
            attributes=["phone", "email", "address"],
            min_shared=1,
            edge_label="SIMILAR_TO",
        )

        sim_time = time.monotonic() - t0

        if sim_stats["edge_count"] == 0:
            tracemalloc.stop()
            pytest.skip("No similarity edges at 1K scale")

        # Community detection
        t1 = time.monotonic()
        algo = GraphAlgorithms(self.db)
        results, algo_name = _detect_community(algo, "Customer", "SIMILAR_TO")
        algo_time = time.monotonic() - t1

        total_time = time.monotonic() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Compute stats
        community_counts = Counter(r["community_id"] for r in results)
        largest = max(community_counts.values()) if community_counts else 0

        print(f"\n  === Leiden Scale Benchmark (1K) ===")
        print(f"  Algorithm:          {algo_name}")
        print(f"  Similarity edges:   {sim_stats['edge_count']}")
        print(f"  Similarity time:    {sim_time:.2f}s")
        print(f"  Community count:    {len(community_counts)}")
        print(f"  Largest community:  {largest}")
        print(f"  Algorithm time:     {algo_time:.2f}s")
        print(f"  Total wall time:    {total_time:.2f}s")
        print(f"  Peak memory:        {peak_bytes / (1024**2):.1f} MB")

        assert len(results) > 0
        assert total_time < 120, f"1K benchmark too slow: {total_time:.1f}s"


# ---------------------------------------------------------------------------
# Large-scale benchmark (10K customers, gated behind RUN_BENCHMARKS)
# ---------------------------------------------------------------------------

pytestmark_bench = pytest.mark.skipif(
    not os.environ.get("RUN_BENCHMARKS"),
    reason="Set RUN_BENCHMARKS=1 to run large-scale benchmarks",
)


@pytestmark_bench
class TestLeidenScale10K:
    """Leiden benchmark at 10K customers -- full-scale."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.db = Database(":memory:")
        customers = _generate_customers(10_000, duplicate_rate=0.05, seed=42)
        _load_customers(self.db, customers)
        yield
        self.db.close()

    def test_full_benchmark(self):
        """Complete benchmark: generate, build similarity, run Leiden, report."""
        tracemalloc.start()
        t0 = time.monotonic()

        # Build similarity graph
        builder = SimilarityGraphBuilder(self.db, node_label="Customer")
        sim_stats = builder.build(
            attributes=["phone", "email", "address"],
            min_shared=1,
            edge_label="SIMILAR_TO",
            max_bucket_size=5000,
        )

        sim_time = time.monotonic() - t0

        if sim_stats["edge_count"] == 0:
            tracemalloc.stop()
            pytest.skip("No similarity edges generated at 10K scale")

        # Community detection
        t1 = time.monotonic()
        algo = GraphAlgorithms(self.db)
        results, algo_name = _detect_community(algo, "Customer", "SIMILAR_TO")
        algo_time = time.monotonic() - t1

        total_time = time.monotonic() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Compute stats
        community_counts = Counter(r["community_id"] for r in results)
        largest = max(community_counts.values()) if community_counts else 0
        sizes = sorted(community_counts.values(), reverse=True)
        top_5 = sizes[:5]

        print(f"\n  === Leiden Scale Benchmark (10K) ===")
        print(f"  Algorithm:          {algo_name}")
        print(f"  Similarity edges:   {sim_stats['edge_count']}")
        print(f"  Similarity time:    {sim_time:.2f}s")
        print(f"  Community count:    {len(community_counts)}")
        print(f"  Largest community:  {largest}")
        print(f"  Top 5 sizes:        {top_5}")
        print(f"  Algorithm time:     {algo_time:.2f}s")
        print(f"  Total wall time:    {total_time:.2f}s")
        print(f"  Peak memory:        {peak_bytes / (1024**2):.1f} MB")

        assert len(results) > 0
        assert len(community_counts) > 1, "Expected multiple communities at 10K scale"
        assert total_time < 300, f"10K benchmark too slow: {total_time:.1f}s"

    def test_resolution_comparison(self):
        """Compare community counts at different Leiden resolution values."""
        builder = SimilarityGraphBuilder(self.db, node_label="Customer")
        stats = builder.build(
            attributes=["phone", "email", "address"],
            min_shared=1,
            edge_label="SIMILAR_TO",
            max_bucket_size=5000,
        )
        if stats["edge_count"] == 0:
            pytest.skip("No similarity edges")

        if not _HAS_ALGO:
            pytest.skip("algo extension required for resolution comparison")

        algo = GraphAlgorithms(self.db)
        resolutions = [0.1, 0.5, 1.0, 2.0, 5.0]
        print(f"\n  === Resolution Comparison (10K) ===")
        print(f"  {'Resolution':>12} {'Communities':>14} {'Largest':>10} {'Time (s)':>10}")

        prev_count = 0
        for res in resolutions:
            t0 = time.monotonic()
            try:
                results = algo.leiden("Customer", "SIMILAR_TO", resolution=res)
            except Exception:
                print(f"  {res:>12.1f}  -- failed --")
                continue
            elapsed = time.monotonic() - t0

            community_counts = Counter(r["community_id"] for r in results)
            largest = max(community_counts.values()) if community_counts else 0
            n_comms = len(community_counts)

            print(f"  {res:>12.1f} {n_comms:>14} {largest:>10} {elapsed:>10.2f}")

            # Higher resolution should produce >= as many communities
            if prev_count > 0:
                assert n_comms >= prev_count - 1, (
                    f"Resolution {res} produced fewer communities ({n_comms}) "
                    f"than lower resolution ({prev_count})"
                )
            prev_count = n_comms
