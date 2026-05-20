"""End-to-end HEB fraud workflow: Delta read → graph → similarity → Louvain → Delta write.

Covers D2 (Louvain on unipartite graph) and D4 (full workflow test).

Run with: RUN_BENCHMARKS=1 pytest benchmarks/test_heb_fraud_workflow.py -v -s
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

deltalake = pytest.importorskip("deltalake")

from bridgr.database import Database
from bridgr.export import DataExporter
from bridgr.similarity_graph import SimilarityGraphBuilder


def _create_fraud_dataset(n_customers: int = 10_000, n_fraud_rings: int = 10, ring_size: int = 30):
    """Generate customer data with known fraud rings for testing.

    Fraud ring members share zip + phone (high similarity weight).
    Normal customers have unique attributes.
    Returns (Arrow table, ground_truth dict).
    """
    import random
    random.seed(42)

    ids, names, phones, zips, address_hashes = [], [], [], [], []
    fraud_ring_membership = {}

    # Normal customers (unique attributes)
    n_normal = n_customers - (n_fraud_rings * ring_size)
    for i in range(n_normal):
        cid = f"C{i:07d}"
        ids.append(cid)
        names.append(f"Customer_{i}")
        phones.append(f"555{i:07d}")
        zips.append(f"{70000 + (i % 1000):05d}")
        address_hashes.append(f"addr_{i}")

    # Fraud ring members (shared zip + phone + address_hash within ring)
    for ring_idx in range(n_fraud_rings):
        ring_phone = f"999{ring_idx:07d}"
        ring_zip = f"{99000 + ring_idx:05d}"
        ring_addr = f"fraud_addr_{ring_idx}"
        for member in range(ring_size):
            idx = n_normal + ring_idx * ring_size + member
            cid = f"F{ring_idx:03d}M{member:03d}"
            ids.append(cid)
            names.append(f"FraudRing{ring_idx}_Member{member}")
            phones.append(ring_phone)
            zips.append(ring_zip)
            address_hashes.append(ring_addr)
            fraud_ring_membership[cid] = f"ring_{ring_idx}"

    table = pa.table({
        "customer_id": ids,
        "name": names,
        "phone": phones,
        "zip": zips,
        "address_hash": address_hashes,
    })
    return table, fraud_ring_membership


@pytest.fixture(scope="module")
def fraud_fixture():
    """Set up the fraud dataset and database for all tests."""
    tmpdir = tempfile.mkdtemp(prefix="heb_fraud_")
    delta_input = os.path.join(tmpdir, "customers_delta")
    delta_output = os.path.join(tmpdir, "communities_delta")
    db_path = os.path.join(tmpdir, "fraud.lbug")

    # Generate data with known fraud rings
    customer_table, fraud_rings = _create_fraud_dataset(
        n_customers=10_000, n_fraud_rings=10, ring_size=30,
    )
    deltalake.write_deltalake(delta_input, customer_table)
    print(f"\n  Dataset: {customer_table.num_rows} customers, {len(fraud_rings)} fraud ring members")

    yield {
        "tmpdir": tmpdir,
        "delta_input": delta_input,
        "delta_output": delta_output,
        "db_path": db_path,
        "customer_table": customer_table,
        "fraud_rings": fraud_rings,
    }

    shutil.rmtree(tmpdir, ignore_errors=True)


class TestSimilarityGraph:
    """D1 + D2: Build similarity graph and run Louvain."""

    def test_similarity_build(self, fraud_fixture):
        """D1: SimilarityGraphBuilder creates SIMILAR_TO edges."""
        db = Database(fraud_fixture["db_path"])
        try:
            db.execute("INSTALL algo")
            db.execute("LOAD EXTENSION algo")
        except Exception:
            pass

        # Import from Delta
        db.import_delta_table(fraud_fixture["delta_input"], "Customer", primary_key="customer_id")
        count = db.query("MATCH (c:Customer) RETURN count(c) AS cnt")[0]["cnt"]
        print(f"\n  Imported {count} customers")
        assert count == 10_000

        # Build similarity graph
        builder = SimilarityGraphBuilder(db)
        t0 = time.monotonic()
        stats = builder.build(
            node_label="Customer",
            attributes=["zip", "phone", "address_hash"],
            edge_label="SIMILAR_TO",
            min_weight=2,
        )
        elapsed = time.monotonic() - t0
        print(f"  Similarity graph: {stats['edges_created']} edges in {elapsed:.1f}s")
        print(f"  Total pairs found: {stats['total_pairs']}, above threshold: {stats['pairs_above_threshold']}")

        assert stats["edges_created"] > 0
        # Fraud ring members share 3 attributes (zip+phone+addr), so they should all have weight >= 2
        # 10 rings × C(30,2) = 10 × 435 = 4350 fraud edges expected
        assert stats["edges_created"] >= 4000

        fraud_fixture["db"] = db

    def test_louvain_on_similarity(self, fraud_fixture):
        """D2: Louvain community detection on the unipartite similarity graph."""
        db = fraud_fixture.get("db")
        if db is None:
            pytest.skip("Requires test_similarity_build to run first")

        # Project and run Louvain
        db.execute("CALL PROJECT_GRAPH('_fraud_louv', ['Customer'], ['SIMILAR_TO'])")
        t0 = time.monotonic()
        communities = db.query(
            "CALL LOUVAIN('_fraud_louv') "
            "RETURN node.customer_id AS customer_id, louvain_id AS community_id"
        )
        louvain_time = time.monotonic() - t0
        db.execute("CALL DROP_PROJECTED_GRAPH('_fraud_louv')")

        # Analyze results
        community_map = {r["customer_id"]: r["community_id"] for r in communities if r["customer_id"]}
        unique_communities = set(community_map.values())
        print(f"\n  Louvain: {len(unique_communities)} communities in {louvain_time:.2f}s")

        assert louvain_time < 30, f"Louvain too slow: {louvain_time:.2f}s"
        assert len(unique_communities) > 1

        # Verify fraud ring detection: members of the same ring should be in the same community
        fraud_rings = fraud_fixture["fraud_rings"]
        rings_by_id: dict[str, list[str]] = {}
        for cid, ring_id in fraud_rings.items():
            rings_by_id.setdefault(ring_id, []).append(cid)

        rings_detected = 0
        for ring_id, members in rings_by_id.items():
            member_communities = set()
            for m in members:
                if m in community_map:
                    member_communities.add(community_map[m])
            # Ring is "detected" if all members are in the same community
            if len(member_communities) == 1:
                rings_detected += 1

        detection_rate = rings_detected / len(rings_by_id)
        print(f"  Fraud ring detection rate: {rings_detected}/{len(rings_by_id)} ({detection_rate:.0%})")
        assert detection_rate >= 0.8, f"Only {detection_rate:.0%} of fraud rings detected"

        fraud_fixture["community_map"] = community_map
        fraud_fixture["louvain_time"] = louvain_time


class TestDeltaWriteback:
    """D3: Export results to Delta Lake."""

    def test_to_delta_lake(self, fraud_fixture):
        """D3: Write node data to Delta Lake."""
        db = fraud_fixture.get("db")
        if db is None:
            pytest.skip("Requires similarity graph tests to run first")

        exporter = DataExporter(db)
        delta_out = fraud_fixture["delta_output"]

        rows_written = exporter.to_delta_lake("Customer", delta_out)
        print(f"\n  Wrote {rows_written} rows to Delta Lake at {delta_out}")
        assert rows_written == 10_000

        # Read back and verify
        dt = deltalake.DeltaTable(delta_out)
        df = dt.to_pandas()
        assert len(df) == 10_000
        assert "customer_id" in df.columns

    def test_query_to_delta_lake(self, fraud_fixture):
        """D3: Write Cypher query results to Delta Lake."""
        db = fraud_fixture.get("db")
        if db is None:
            pytest.skip("Requires similarity graph tests to run first")

        exporter = DataExporter(db)
        delta_out = os.path.join(fraud_fixture["tmpdir"], "query_delta")

        rows_written = exporter.query_to_delta_lake(
            "MATCH (c:Customer) WHERE c.zip STARTS WITH '990' "
            "RETURN c.customer_id AS customer_id, c.name AS name, c.zip AS zip",
            delta_out,
        )
        print(f"\n  Wrote {rows_written} fraud-zip customers to Delta Lake")
        assert rows_written > 0  # fraud ring members have zip 990xx

        dt = deltalake.DeltaTable(delta_out)
        df = dt.to_pandas()
        assert "customer_id" in df.columns
        assert all(df["zip"].str.startswith("990"))


class TestEndToEndWorkflow:
    """D4: Full HEB fraud detection workflow."""

    def test_full_workflow(self):
        """Delta read → graph → similarity → Louvain → Delta write → verify."""
        tmpdir = tempfile.mkdtemp(prefix="heb_e2e_")
        try:
            t0 = time.monotonic()

            # Step 1: Generate synthetic fraud data and write to Delta Lake
            customer_table, fraud_rings = _create_fraud_dataset(
                n_customers=10_000, n_fraud_rings=10, ring_size=30,
            )
            delta_input = os.path.join(tmpdir, "input_delta")
            deltalake.write_deltalake(delta_input, customer_table)

            # Step 2: Read from Delta Lake into Bridgr
            db = Database(os.path.join(tmpdir, "workflow.lbug"))
            try:
                db.execute("INSTALL algo")
                db.execute("LOAD EXTENSION algo")
            except Exception:
                pass
            db.import_delta_table(delta_input, "Customer", primary_key="customer_id")

            # Step 3: Build similarity graph
            builder = SimilarityGraphBuilder(db)
            sim_stats = builder.build(
                node_label="Customer",
                attributes=["zip", "phone", "address_hash"],
                edge_label="SIMILAR_TO",
                min_weight=2,
            )

            # Step 4: Run Louvain — export results directly (no need to ALTER TABLE)
            db.execute("CALL PROJECT_GRAPH('_e2e_louv', ['Customer'], ['SIMILAR_TO'])")

            # Step 5: Export Louvain results joined with customer data to Delta Lake
            exporter = DataExporter(db)
            delta_output = os.path.join(tmpdir, "output_delta")
            rows = exporter.query_to_delta_lake(
                "CALL LOUVAIN('_e2e_louv') "
                "RETURN node.customer_id AS customer_id, node.name AS name, "
                "louvain_id AS community_id",
                delta_output,
            )
            db.execute("CALL DROP_PROJECTED_GRAPH('_e2e_louv')")

            elapsed = time.monotonic() - t0

            # Step 6: Verify
            dt = deltalake.DeltaTable(delta_output)
            result_df = dt.to_pandas()
            n_communities = result_df["community_id"].nunique()

            print(f"\n  Full workflow completed in {elapsed:.1f}s")
            print(f"  Customers: {len(result_df)}")
            print(f"  Communities: {n_communities}")
            print(f"  Similarity edges: {sim_stats['edges_created']}")
            print(f"  Rows written to Delta: {rows}")

            assert len(result_df) == 10_000
            assert n_communities > 1
            assert rows == 10_000

            # Verify fraud rings detected
            ring_members = {cid for cid in fraud_rings}
            fraud_df = result_df[result_df["customer_id"].isin(ring_members)]
            # Check that fraud ring members are concentrated in few communities
            fraud_communities = fraud_df["community_id"].nunique()
            print(f"  Fraud ring members span {fraud_communities} communities (ideal: ~{len(set(fraud_rings.values()))})")

            db.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
