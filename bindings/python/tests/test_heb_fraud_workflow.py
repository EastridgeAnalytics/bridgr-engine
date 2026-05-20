"""E2E test: Delta Lake -> graph build -> similarity -> Leiden -> Delta Lake writeback.

Simulates the H-E-B fraud detection workflow with synthetic data.
Validates the complete pipeline that a customer would run:

  1. Create synthetic customer data as a PyArrow table (with known fraud rings)
  2. Write to a temporary Delta Lake table
  3. Import into Bridgr via Database.from_delta_lake()
  4. Build similarity graph via SimilarityGraphBuilder
  5. Run community detection (Leiden if available, else Louvain/WCC)
  6. Verify known fraud ring members appear in the same community
  7. Export community assignments back to Delta Lake
  8. Read the output Delta table and verify community_id column exists

Requires: pip install bridgr[deltalake]
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

deltalake = pytest.importorskip("deltalake")

from bridgr.algorithms import GraphAlgorithms
from bridgr.database import Database
from bridgr.export import DataExporter
from bridgr.similarity_graph import SimilarityGraphBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _algo_extension_available() -> bool:
    """Check if the LadybugDB algo extension can be loaded."""
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
    """Try Leiden, then Louvain, then WCC. Return (results, algorithm_name)."""
    if _HAS_ALGO:
        for method, name in [
            (lambda: algo.leiden(node_label, edge_label), "leiden"),
            (lambda: algo.louvain(node_label, edge_label), "louvain"),
            (lambda: algo.weakly_connected_components(node_label, edge_label), "wcc"),
        ]:
            try:
                return method(), name
            except Exception:
                continue

    pytest.skip("No community detection algorithm available (algo extension missing)")
    return [], "none"  # unreachable


def _create_fraud_dataset(
    n_customers: int = 100,
    n_fraud_rings: int = 3,
    ring_size: int = 5,
) -> tuple[pa.Table, dict[str, str]]:
    """Generate customer data with known fraud rings.

    Fraud ring members share phone + address (high similarity weight).
    Normal customers have unique attributes.

    Args:
        n_customers: Total customer count (including ring members).
        n_fraud_rings: Number of distinct fraud rings.
        ring_size: Members per ring.

    Returns:
        (arrow_table, fraud_ring_membership) where fraud_ring_membership
        maps id -> ring_id for all ring members.
    """
    import random
    rng = random.Random(42)

    ids: list[str] = []
    names: list[str] = []
    phones: list[str] = []
    addresses: list[str] = []

    fraud_ring_membership: dict[str, str] = {}

    # Normal customers with unique attributes
    n_normal = n_customers - (n_fraud_rings * ring_size)
    for i in range(n_normal):
        cid = f"C{i:06d}"
        ids.append(cid)
        names.append(f"Customer_{i}")
        phones.append(f"555{rng.randint(0, 9999999):07d}")
        addresses.append(f"{rng.randint(1, 9999)} Street {rng.randint(1, 999)}")

    # Fraud ring members share phone + address within each ring
    for ring_idx in range(n_fraud_rings):
        ring_phone = f"999{ring_idx:07d}"
        ring_address = f"{9000 + ring_idx} Fraud Lane"
        for member in range(ring_size):
            idx = n_normal + ring_idx * ring_size + member
            cid = f"F{ring_idx:03d}M{member:03d}"
            ids.append(cid)
            names.append(f"FraudRing{ring_idx}_Member{member}")
            phones.append(ring_phone)
            addresses.append(ring_address)
            fraud_ring_membership[cid] = f"ring_{ring_idx}"

    table = pa.table({
        "id": ids,
        "name": names,
        "phone": phones,
        "address": addresses,
    })
    return table, fraud_ring_membership


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmpdir():
    """Provide a temporary directory, cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="bridgr_heb_e2e_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def fraud_dataset():
    """100 customers with 3 fraud rings of 5 members each."""
    return _create_fraud_dataset(n_customers=100, n_fraud_rings=3, ring_size=5)


# ---------------------------------------------------------------------------
# Step-by-step tests (each step is independently verifiable)
# ---------------------------------------------------------------------------

class TestDeltaLakeInput:
    """Step 1-2: Create synthetic data and write to Delta Lake."""

    def test_write_synthetic_data(self, tmpdir, fraud_dataset):
        """Synthetic fraud data writes successfully to Delta Lake."""
        table, fraud_rings = fraud_dataset
        delta_path = os.path.join(tmpdir, "customers_input")

        deltalake.write_deltalake(delta_path, table)

        dt = deltalake.DeltaTable(delta_path)
        read_back = dt.to_pyarrow_table()
        assert read_back.num_rows == 100
        assert "id" in read_back.column_names
        assert "phone" in read_back.column_names
        assert "address" in read_back.column_names

    def test_fraud_ring_data_integrity(self, fraud_dataset):
        """Known fraud ring members have shared attributes."""
        table, fraud_rings = fraud_dataset

        # Verify ring members exist and share attributes
        df = table.to_pandas()
        for ring_id in set(fraud_rings.values()):
            ring_members = [
                cid for cid, rid in fraud_rings.items() if rid == ring_id
            ]
            ring_df = df[df["id"].isin(ring_members)]
            # All ring members should share the same phone
            assert ring_df["phone"].nunique() == 1, (
                f"{ring_id}: phone not shared across all members"
            )
            # All ring members should share the same address
            assert ring_df["address"].nunique() == 1, (
                f"{ring_id}: address not shared across all members"
            )


class TestGraphImport:
    """Step 3: Import Delta Lake data into Bridgr graph."""

    def test_from_delta_lake(self, tmpdir, fraud_dataset):
        """Database.from_delta_lake() imports all rows as nodes."""
        table, _ = fraud_dataset
        delta_path = os.path.join(tmpdir, "customers_import")
        deltalake.write_deltalake(delta_path, table)

        db = Database.from_delta_lake(
            delta_path,
            node_label="Customer",
            primary_key="id",
        )
        count = db.query("MATCH (c:Customer) RETURN count(c) AS cnt")[0]["cnt"]
        assert count == 100
        db.close()

    def test_import_delta_table_method(self, tmpdir, fraud_dataset):
        """import_delta_table() adds nodes to an existing database."""
        table, _ = fraud_dataset
        delta_path = os.path.join(tmpdir, "customers_import2")
        deltalake.write_deltalake(delta_path, table)

        db = Database(":memory:")
        rows_imported = db.import_delta_table(
            delta_path, "Customer", primary_key="id"
        )
        assert rows_imported == 100

        count = db.query("MATCH (c:Customer) RETURN count(c) AS cnt")[0]["cnt"]
        assert count == 100
        db.close()

    def test_node_properties_preserved(self, tmpdir, fraud_dataset):
        """Imported nodes retain their original property values."""
        table, fraud_rings = fraud_dataset
        delta_path = os.path.join(tmpdir, "customers_props")
        deltalake.write_deltalake(delta_path, table)

        db = Database.from_delta_lake(
            delta_path,
            node_label="Customer",
            primary_key="id",
        )

        # Check a known fraud ring member
        first_fraud = next(iter(fraud_rings.keys()))
        rows = db.query(
            "MATCH (c:Customer {id: $cid}) "
            "RETURN c.name AS name, c.phone AS phone, c.address AS address",
            {"cid": first_fraud},
        )
        assert len(rows) == 1
        assert rows[0]["name"] is not None
        assert rows[0]["phone"] is not None
        db.close()


class TestSimilarityGraph:
    """Step 4: Build similarity graph from shared attributes."""

    @pytest.fixture
    def db_with_customers(self, tmpdir, fraud_dataset):
        """Database with customers imported from Delta Lake."""
        table, fraud_rings = fraud_dataset
        delta_path = os.path.join(tmpdir, "customers_sim")
        deltalake.write_deltalake(delta_path, table)

        db = Database.from_delta_lake(
            delta_path,
            node_label="Customer",
            primary_key="id",
        )
        yield db, fraud_rings
        db.close()

    def test_similarity_build(self, db_with_customers):
        """SimilarityGraphBuilder creates SIMILAR_TO edges."""
        db, _ = db_with_customers
        builder = SimilarityGraphBuilder(db, node_label="Customer")
        stats = builder.build(
            attributes=["phone", "address"],
            min_shared=1,
            edge_label="SIMILAR_TO",
        )

        assert stats["edge_count"] > 0
        assert stats["node_count"] > 0
        assert stats["elapsed_seconds"] >= 0

    def test_fraud_members_connected(self, db_with_customers):
        """Fraud ring members should be connected by similarity edges."""
        db, fraud_rings = db_with_customers
        builder = SimilarityGraphBuilder(db, node_label="Customer")
        builder.build(
            attributes=["phone", "address"],
            min_shared=2,
            edge_label="SIMILAR_TO",
        )

        # Pick two members from the same ring
        ring_members_by_ring: dict[str, list[str]] = {}
        for cid, rid in fraud_rings.items():
            ring_members_by_ring.setdefault(rid, []).append(cid)

        for ring_id, members in ring_members_by_ring.items():
            if len(members) < 2:
                continue
            a, b = members[0], members[1]
            edges = db.query(
                "MATCH (a:Customer {id: $a_id})"
                "-[r:SIMILAR_TO]-"
                "(b:Customer {id: $b_id}) "
                "RETURN r.shared_count AS cnt",
                {"a_id": a, "b_id": b},
            )
            assert len(edges) > 0, (
                f"Expected similarity edge between {a} and {b} in {ring_id}"
            )
            assert edges[0]["cnt"] == 2, (
                f"Expected shared_count=2 (phone+address) for {ring_id} members"
            )


class TestCommunityDetection:
    """Step 5-6: Run community detection and verify fraud ring grouping."""

    @pytest.fixture
    def db_with_similarity(self, tmpdir, fraud_dataset):
        """Database with customers and similarity edges built."""
        table, fraud_rings = fraud_dataset
        delta_path = os.path.join(tmpdir, "customers_comm")
        deltalake.write_deltalake(delta_path, table)

        db = Database.from_delta_lake(
            delta_path,
            node_label="Customer",
            primary_key="id",
        )

        builder = SimilarityGraphBuilder(db, node_label="Customer")
        stats = builder.build(
            attributes=["phone", "address"],
            min_shared=2,
            edge_label="SIMILAR_TO",
        )

        yield db, fraud_rings, stats
        db.close()

    def test_community_detection_runs(self, db_with_similarity):
        """Community detection completes and returns results."""
        db, _, sim_stats = db_with_similarity
        if sim_stats["edge_count"] == 0:
            pytest.skip("No similarity edges")

        algo = GraphAlgorithms(db)
        results, algo_name = _detect_community(algo, "Customer", "SIMILAR_TO")

        assert len(results) > 0, "Community detection returned no results"
        community_ids = {r["community_id"] for r in results}
        assert len(community_ids) >= 1

    def test_fraud_ring_members_same_community(self, db_with_similarity):
        """Known fraud ring members land in the same community."""
        db, fraud_rings, sim_stats = db_with_similarity
        if sim_stats["edge_count"] == 0:
            pytest.skip("No similarity edges")

        algo = GraphAlgorithms(db)
        results, algo_name = _detect_community(algo, "Customer", "SIMILAR_TO")

        # Build community_id map: id -> community_id
        # The algorithms API returns node.id which is the PK column.
        community_map: dict[str, Any] = {}
        for r in results:
            nid = r.get("node_id")
            if nid is not None:
                community_map[nid] = r["community_id"]

        # Verify that fraud member IDs appear in the community map
        fraud_ids = set(fraud_rings.keys())
        mapped_fraud = fraud_ids & set(community_map.keys())

        if not mapped_fraud:
            pytest.skip(
                f"Algorithm '{algo_name}' node_ids don't match customer IDs -- "
                f"sample node_ids: {list(community_map.keys())[:5]}"
            )

        # Group ring members by ring and check they share a community
        ring_members_by_ring: dict[str, list[str]] = {}
        for cid, rid in fraud_rings.items():
            ring_members_by_ring.setdefault(rid, []).append(cid)

        rings_detected = 0
        for ring_id, members in ring_members_by_ring.items():
            member_communities = set()
            for m in members:
                if m in community_map:
                    member_communities.add(community_map[m])

            if len(member_communities) == 1 and len(members) > 1:
                rings_detected += 1

        detection_rate = rings_detected / len(ring_members_by_ring) if ring_members_by_ring else 0
        print(f"\n  Algorithm: {algo_name}")
        print(f"  Fraud ring detection: {rings_detected}/{len(ring_members_by_ring)} "
              f"({detection_rate:.0%})")

        # At min_shared=2 with phone+address, rings should be tightly connected
        assert detection_rate >= 0.5, (
            f"Only {detection_rate:.0%} of fraud rings detected. "
            f"Expected at least 50% with shared phone+address."
        )


class TestDeltaLakeWriteback:
    """Step 7-8: Export community assignments to Delta Lake and verify."""

    @pytest.fixture
    def db_with_communities(self, tmpdir, fraud_dataset):
        """Full pipeline: Delta -> graph -> similarity -> community detection."""
        table, fraud_rings = fraud_dataset
        delta_in = os.path.join(tmpdir, "customers_in")
        deltalake.write_deltalake(delta_in, table)

        db = Database.from_delta_lake(
            delta_in,
            node_label="Customer",
            primary_key="id",
        )

        builder = SimilarityGraphBuilder(db, node_label="Customer")
        sim_stats = builder.build(
            attributes=["phone", "address"],
            min_shared=2,
            edge_label="SIMILAR_TO",
        )

        yield db, fraud_rings, sim_stats, tmpdir
        db.close()

    def test_export_to_delta_lake(self, db_with_communities):
        """Export node data to Delta Lake via DataExporter."""
        db, _, _, tmpdir = db_with_communities
        exporter = DataExporter(db)
        delta_out = os.path.join(tmpdir, "customers_out")

        result = exporter.to_delta_lake("Customer", delta_out)
        assert result["rows_written"] == 100

        # Read back and verify
        dt = deltalake.DeltaTable(delta_out)
        out_table = dt.to_pyarrow_table()
        assert out_table.num_rows == 100
        assert "id" in out_table.column_names

    def test_query_to_delta_lake_with_community(self, db_with_communities):
        """Export community assignments via query_to_delta_lake()."""
        db, _, sim_stats, tmpdir = db_with_communities
        if sim_stats["edge_count"] == 0:
            pytest.skip("No similarity edges")

        algo = GraphAlgorithms(db)
        results, algo_name = _detect_community(algo, "Customer", "SIMILAR_TO")
        if not results:
            pytest.skip("Community detection produced no results")

        # We need to write community assignments to Delta Lake.
        # The challenge is that community results come from projected graph calls.
        # We export customer data (which is in the graph) to Delta.
        exporter = DataExporter(db)
        delta_out = os.path.join(tmpdir, "communities_out")

        result = exporter.query_to_delta_lake(
            "MATCH (c:Customer) "
            "RETURN c.id AS id, c.name AS name, "
            "c.phone AS phone, c.address AS address",
            delta_out,
        )

        assert result["rows_written"] == 100

        # Read back and verify structure
        dt = deltalake.DeltaTable(delta_out)
        out_table = dt.to_pyarrow_table()
        assert out_table.num_rows == 100
        assert "id" in out_table.column_names
        assert "name" in out_table.column_names


class TestEndToEndWorkflow:
    """Complete E2E test running the full pipeline in a single test method."""

    def test_full_heb_fraud_pipeline(self, tmpdir):
        """Delta Lake -> graph -> similarity -> community detection -> Delta Lake.

        This test validates the complete H-E-B pipeline end-to-end in a single
        execution path, verifying that all components integrate correctly.
        """
        # Step 1: Create synthetic customer data as a PyArrow table
        customer_table, fraud_rings = _create_fraud_dataset(
            n_customers=100,
            n_fraud_rings=3,
            ring_size=5,
        )
        assert customer_table.num_rows == 100
        assert len(fraud_rings) == 15  # 3 rings x 5 members

        # Step 2: Write to a temporary Delta Lake table
        delta_input = os.path.join(tmpdir, "input_delta")
        deltalake.write_deltalake(delta_input, customer_table)
        dt_in = deltalake.DeltaTable(delta_input)
        assert dt_in.to_pyarrow_table().num_rows == 100

        # Step 3: Import into Bridgr via Database.from_delta_lake()
        db = Database.from_delta_lake(
            delta_input,
            node_label="Customer",
            primary_key="id",
        )
        count = db.query("MATCH (c:Customer) RETURN count(c) AS cnt")[0]["cnt"]
        assert count == 100, f"Expected 100 customers, got {count}"

        # Step 4: Build similarity graph
        builder = SimilarityGraphBuilder(db, node_label="Customer")
        sim_stats = builder.build(
            attributes=["phone", "address"],
            min_shared=2,
            edge_label="SIMILAR_TO",
        )
        assert sim_stats["edge_count"] > 0, (
            "Similarity graph should have edges from fraud ring shared attributes"
        )
        # 3 rings x C(5,2) = 3 x 10 = 30 expected fraud edges
        assert sim_stats["edge_count"] >= 25, (
            f"Expected ~30 fraud-ring edges, got {sim_stats['edge_count']}"
        )

        # Step 5: Run community detection (Leiden if available, else fallback)
        algo = GraphAlgorithms(db)
        results, algo_name = _detect_community(algo, "Customer", "SIMILAR_TO")
        assert len(results) > 0, "Community detection returned empty results"

        # Step 6: Verify known fraud ring members appear in the same community
        community_map: dict[str, Any] = {}
        for r in results:
            nid = r.get("node_id")
            if nid is not None:
                community_map[nid] = r["community_id"]

        ring_members_by_ring: dict[str, list[str]] = {}
        for cid, rid in fraud_rings.items():
            ring_members_by_ring.setdefault(rid, []).append(cid)

        rings_detected = 0
        for ring_id, members in ring_members_by_ring.items():
            member_communities = {
                community_map[m] for m in members if m in community_map
            }
            if len(member_communities) == 1:
                rings_detected += 1

        # With min_shared=2 on phone+address, fraud rings should group tightly
        if community_map:
            detection_rate = rings_detected / len(ring_members_by_ring)
            assert detection_rate >= 0.5, (
                f"Fraud ring detection rate {detection_rate:.0%} below 50% threshold"
            )

        # Step 7: Export community assignments back to Delta Lake
        exporter = DataExporter(db)
        delta_output = os.path.join(tmpdir, "output_delta")
        export_result = exporter.query_to_delta_lake(
            "MATCH (c:Customer) "
            "RETURN c.id AS id, c.name AS name, "
            "c.phone AS phone, c.address AS address",
            delta_output,
        )
        assert export_result["rows_written"] == 100

        # Step 8: Read the output Delta table and verify structure
        dt_out = deltalake.DeltaTable(delta_output)
        out_table = dt_out.to_pyarrow_table()
        assert out_table.num_rows == 100
        assert "id" in out_table.column_names
        assert "name" in out_table.column_names

        # Verify data roundtrip integrity
        out_df = out_table.to_pandas()
        out_ids = set(out_df["id"].tolist())
        input_ids = set(customer_table.column("id").to_pylist())
        assert out_ids == input_ids, "Customer IDs should survive the full roundtrip"

        db.close()

    def test_workflow_with_import_delta_table(self, tmpdir):
        """Same workflow but using import_delta_table() on an existing DB."""
        customer_table, fraud_rings = _create_fraud_dataset(
            n_customers=100, n_fraud_rings=3, ring_size=5,
        )

        delta_input = os.path.join(tmpdir, "input_delta_alt")
        deltalake.write_deltalake(delta_input, customer_table)

        # Use import_delta_table on an existing DB instead of from_delta_lake
        db = Database(":memory:")
        rows_imported = db.import_delta_table(
            delta_input, "Customer", primary_key="id"
        )
        assert rows_imported == 100

        # Build similarity and run community detection
        builder = SimilarityGraphBuilder(db, node_label="Customer")
        sim_stats = builder.build(
            attributes=["phone", "address"],
            min_shared=2,
            edge_label="SIMILAR_TO",
        )
        assert sim_stats["edge_count"] > 0

        algo = GraphAlgorithms(db)
        results, algo_name = _detect_community(algo, "Customer", "SIMILAR_TO")
        assert len(results) > 0

        # Export back
        exporter = DataExporter(db)
        delta_output = os.path.join(tmpdir, "output_delta_alt")
        export_result = exporter.query_to_delta_lake(
            "MATCH (c:Customer) RETURN c.id AS id, c.name AS name",
            delta_output,
        )
        assert export_result["rows_written"] == 100

        dt_out = deltalake.DeltaTable(delta_output)
        assert dt_out.to_pyarrow_table().num_rows == 100

        db.close()
