"""Tests for SimilarityGraphBuilder — builds SIMILAR_TO edges from shared attributes."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import bridgr
from bridgr.similarity_graph import SimilarityGraphBuilder


@pytest.fixture
def db():
    """Create an in-memory database with 5 Customer nodes having overlapping attributes."""
    d = bridgr.open(":memory:")
    d.create_node_table("Customer", {
        "id": "STRING PRIMARY KEY",
        "name": "STRING",
        "phone": "STRING",
        "email": "STRING",
        "zip": "STRING",
    })
    # Customer data with deliberate overlaps:
    # C1 and C2 share phone "555-0100" and email "shared@example.com"
    # C1 and C3 share phone "555-0100"
    # C2 and C3 share zip "10001"
    # C4 and C5 share email "duo@example.com"
    # C4 and C5 share zip "90210"
    d.create_node("Customer", {
        "id": "C1", "name": "Alice", "phone": "555-0100",
        "email": "shared@example.com", "zip": "30301",
    })
    d.create_node("Customer", {
        "id": "C2", "name": "Bob", "phone": "555-0100",
        "email": "shared@example.com", "zip": "10001",
    })
    d.create_node("Customer", {
        "id": "C3", "name": "Carol", "phone": "555-0100",
        "email": "carol@example.com", "zip": "10001",
    })
    d.create_node("Customer", {
        "id": "C4", "name": "Dave", "phone": "555-0200",
        "email": "duo@example.com", "zip": "90210",
    })
    d.create_node("Customer", {
        "id": "C5", "name": "Eve", "phone": "555-0300",
        "email": "duo@example.com", "zip": "90210",
    })
    yield d
    d.close()


class TestSimilarityGraphBuilder:
    """Tests for building similarity graphs from shared attributes."""

    def test_build_with_min_shared_1(self, db):
        """With min_shared=1, all pairs sharing at least one attribute get edges."""
        builder = SimilarityGraphBuilder(db, node_label="Customer")
        stats = builder.build(
            attributes=["phone", "email"],
            min_shared=1,
        )
        assert stats["edge_count"] > 0
        assert stats["node_count"] > 0
        assert stats["elapsed_seconds"] >= 0
        assert stats["attributes_checked"] == 2

        # Verify edges exist in the graph
        edges = db.query(
            "MATCH (a:Customer)-[r:SIMILAR_TO]->(b:Customer) "
            "RETURN a.id AS src, b.id AS dst, r.shared_count AS cnt, "
            "r.shared_attributes AS attrs ORDER BY a.id, b.id"
        )
        assert len(edges) > 0

    def test_shared_count_and_attributes(self, db):
        """Edges store correct shared_count and shared_attributes."""
        builder = SimilarityGraphBuilder(db, node_label="Customer")
        stats = builder.build(
            attributes=["phone", "email"],
            min_shared=1,
        )

        # C1-C2 share both phone and email => shared_count=2
        c1_c2 = db.query(
            "MATCH (a:Customer {id: 'C1'})-[r:SIMILAR_TO]->(b:Customer {id: 'C2'}) "
            "RETURN r.shared_count AS cnt, r.shared_attributes AS attrs"
        )
        assert len(c1_c2) == 1
        assert c1_c2[0]["cnt"] == 2
        # shared_attributes is a comma-separated sorted string
        attrs = set(c1_c2[0]["attrs"].split(","))
        assert attrs == {"email", "phone"}

    def test_min_shared_filters_weak_pairs(self, db):
        """min_shared=2 excludes pairs that only share one attribute."""
        builder = SimilarityGraphBuilder(db, node_label="Customer")
        stats = builder.build(
            attributes=["phone", "email"],
            min_shared=2,
        )

        # C1-C2 share 2 attributes (phone + email) => should have edge
        c1_c2 = db.query(
            "MATCH (a:Customer {id: 'C1'})-[r:SIMILAR_TO]->(b:Customer {id: 'C2'}) "
            "RETURN r.shared_count AS cnt"
        )
        assert len(c1_c2) == 1
        assert c1_c2[0]["cnt"] == 2

        # C1-C3 share only phone => should NOT have edge with min_shared=2
        c1_c3 = db.query(
            "MATCH (a:Customer {id: 'C1'})-[r:SIMILAR_TO]->(b:Customer {id: 'C3'}) "
            "RETURN r.shared_count AS cnt"
        )
        assert len(c1_c3) == 0

    def test_three_attributes(self, db):
        """Building with three attributes accumulates correctly."""
        builder = SimilarityGraphBuilder(db, node_label="Customer")
        stats = builder.build(
            attributes=["phone", "email", "zip"],
            min_shared=1,
        )
        assert stats["attributes_checked"] == 3

        # C4-C5 share email + zip => shared_count=2
        c4_c5 = db.query(
            "MATCH (a:Customer {id: 'C4'})-[r:SIMILAR_TO]->(b:Customer {id: 'C5'}) "
            "RETURN r.shared_count AS cnt, r.shared_attributes AS attrs"
        )
        assert len(c4_c5) == 1
        assert c4_c5[0]["cnt"] == 2

    def test_no_edges_when_no_shared(self, db):
        """No edges created if nodes don't share any of the specified attributes."""
        # Use an attribute that no two nodes share
        builder = SimilarityGraphBuilder(db, node_label="Customer")
        stats = builder.build(
            attributes=["name"],  # All names are unique
            min_shared=1,
            edge_label="NAME_SIM",
        )
        assert stats["edge_count"] == 0

    def test_backward_compat_node_label_in_build(self, db):
        """node_label can be passed to build() for backward compatibility."""
        builder = SimilarityGraphBuilder(db)  # default label is "Customer"
        stats = builder.build(
            node_label="Customer",
            attributes=["phone"],
            min_shared=1,
            edge_label="COMPAT_SIM",
        )
        assert stats["edge_count"] > 0

    def test_stats_dict_has_required_fields(self, db):
        """Return dict has all spec-required fields."""
        builder = SimilarityGraphBuilder(db, node_label="Customer")
        stats = builder.build(attributes=["phone"], min_shared=1, edge_label="STAT_SIM")

        assert "edge_count" in stats
        assert "node_count" in stats
        assert "elapsed_seconds" in stats
        assert isinstance(stats["edge_count"], int)
        assert isinstance(stats["node_count"], int)
        assert isinstance(stats["elapsed_seconds"], float)

    def test_custom_edge_label(self, db):
        """Custom edge label is used for the created edges."""
        builder = SimilarityGraphBuilder(db, node_label="Customer")
        builder.build(
            attributes=["email"],
            min_shared=1,
            edge_label="SHARES_EMAIL",
        )
        edges = db.query(
            "MATCH ()-[r:SHARES_EMAIL]->() RETURN count(r) AS cnt"
        )
        assert edges[0]["cnt"] > 0
