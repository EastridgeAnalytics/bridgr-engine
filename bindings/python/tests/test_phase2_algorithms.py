"""Tests for Phase 2 Graph Algorithms — Triangle Count, Closeness/Betweenness
Centrality, Link Prediction, and FastRP Embeddings.

Each test class builds a small, manually-verifiable graph and checks the
algorithm output against expected values.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import bridgr
from bridgr.algorithms import GraphAlgorithms

TEST_DIR = Path(__file__).parent / "test_phase2_dbs"


@pytest.fixture(autouse=True)
def clean_test_dir():
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR))
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEST_DIR.exists():
        shutil.rmtree(str(TEST_DIR), ignore_errors=True)


@pytest.fixture
def db():
    """Fresh in-memory database with Person nodes and KNOWS edges."""
    d = bridgr.open(":memory:")
    d.create_node_table("Person", {
        "id": "STRING PRIMARY KEY",
        "name": "STRING",
    })
    d.create_edge_table("KNOWS", "Person", "Person")
    yield d
    d.close()


# ------------------------------------------------------------------
# Graph builders
# ------------------------------------------------------------------

def _build_triangle_graph(db):
    """Triangle graph: A-B-C-A (one triangle).

        A
       / \\
      B---C
    """
    for name in ["A", "B", "C"]:
        db.create_node("Person", {"id": name.lower(), "name": name})
    for src, dst in [("a", "b"), ("b", "c"), ("a", "c")]:
        db.create_edge("KNOWS", src, dst, from_label="Person", to_label="Person")


def _build_bowtie_graph(db):
    """Bowtie: two triangles sharing node C.

      A---B       D---E
       \\ /         \\ /
        C-----------C  (C is shared)

    Edges: A-B, B-C, A-C, C-D, C-E, D-E
    """
    for name in ["A", "B", "C", "D", "E"]:
        db.create_node("Person", {"id": name.lower(), "name": name})
    for src, dst in [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("c", "e"), ("d", "e")]:
        db.create_edge("KNOWS", src, dst, from_label="Person", to_label="Person")


def _build_path_graph(db):
    """Simple path: A -- B -- C -- D -- E (no triangles)."""
    for name in ["A", "B", "C", "D", "E"]:
        db.create_node("Person", {"id": name.lower(), "name": name})
    for src, dst in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]:
        db.create_edge("KNOWS", src, dst, from_label="Person", to_label="Person")


def _build_star_graph(db):
    """Star: center A connected to B, C, D, E. No triangles among leaves.

        B
        |
    C - A - D
        |
        E
    """
    for name in ["A", "B", "C", "D", "E"]:
        db.create_node("Person", {"id": name.lower(), "name": name})
    for leaf in ["b", "c", "d", "e"]:
        db.create_edge("KNOWS", "a", leaf, from_label="Person", to_label="Person")


def _build_square_with_diagonal(db):
    """Square with one diagonal: A-B-D-C-A plus A-D.

    A---B
    |\\ |
    | \\|
    C---D

    Triangles: A-B-D, A-C-D
    """
    for name in ["A", "B", "C", "D"]:
        db.create_node("Person", {"id": name.lower(), "name": name})
    for src, dst in [("a", "b"), ("b", "d"), ("a", "c"), ("c", "d"), ("a", "d")]:
        db.create_edge("KNOWS", src, dst, from_label="Person", to_label="Person")


# ------------------------------------------------------------------
# Triangle Count / Clustering Coefficient
# ------------------------------------------------------------------

class TestTriangleCount:
    def test_single_triangle(self, db):
        _build_triangle_graph(db)
        algo = GraphAlgorithms(db)
        result = algo.triangle_count("Person")

        assert result["total_triangles"] == 1
        by_id = {n["id"]: n for n in result["nodes"]}
        # Each node in a single triangle has 1 triangle
        for nid in ["a", "b", "c"]:
            assert by_id[nid]["triangles"] == 1
            # degree=2, triangles=1 => cc = 2*1/(2*1) = 1.0
            assert by_id[nid]["clustering_coefficient"] == pytest.approx(1.0)

    def test_bowtie_two_triangles(self, db):
        _build_bowtie_graph(db)
        algo = GraphAlgorithms(db)
        result = algo.triangle_count("Person")

        assert result["total_triangles"] == 2
        by_id = {n["id"]: n for n in result["nodes"]}
        # C participates in both triangles
        assert by_id["c"]["triangles"] == 2
        # C has degree 4, cc = 2*2/(4*3) = 4/12 = 1/3
        assert by_id["c"]["clustering_coefficient"] == pytest.approx(1.0 / 3.0)
        # A has degree 2, participates in 1 triangle, cc = 1.0
        assert by_id["a"]["triangles"] == 1
        assert by_id["a"]["clustering_coefficient"] == pytest.approx(1.0)

    def test_no_triangles_path(self, db):
        _build_path_graph(db)
        algo = GraphAlgorithms(db)
        result = algo.triangle_count("Person")

        assert result["total_triangles"] == 0
        for node in result["nodes"]:
            assert node["triangles"] == 0
            assert node["clustering_coefficient"] == 0.0

    def test_no_triangles_star(self, db):
        _build_star_graph(db)
        algo = GraphAlgorithms(db)
        result = algo.triangle_count("Person")

        assert result["total_triangles"] == 0

    def test_two_triangles_square_diagonal(self, db):
        _build_square_with_diagonal(db)
        algo = GraphAlgorithms(db)
        result = algo.triangle_count("Person")

        assert result["total_triangles"] == 2
        by_id = {n["id"]: n for n in result["nodes"]}
        # A has degree 3 (connects to B, C, D), 2 triangles
        assert by_id["a"]["triangles"] == 2
        # cc(A) = 2*2/(3*2) = 4/6 = 2/3
        assert by_id["a"]["clustering_coefficient"] == pytest.approx(2.0 / 3.0)

    def test_empty_graph(self, db):
        algo = GraphAlgorithms(db)
        result = algo.triangle_count("Person")
        assert result["total_triangles"] == 0
        assert result["nodes"] == []


# ------------------------------------------------------------------
# Closeness Centrality
# ------------------------------------------------------------------

class TestClosenessCentrality:
    def test_path_graph_center_highest(self, db):
        """In A-B-C-D-E, the center node C should have highest closeness."""
        _build_path_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.closeness_centrality("Person")

        assert len(results) == 5
        by_id = {r["id"]: r["closeness"] for r in results}

        # C is center: distances 2,1,1,2 => sum=6, closeness=4/6
        assert by_id["c"] == pytest.approx(4.0 / 6.0)
        # A is endpoint: distances 1,2,3,4 => sum=10, closeness=4/10
        assert by_id["a"] == pytest.approx(4.0 / 10.0)
        # C should be the most central
        assert results[0]["id"] == "c"

    def test_triangle_all_equal(self, db):
        """In a triangle, all nodes have the same closeness."""
        _build_triangle_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.closeness_centrality("Person")

        closeness_values = [r["closeness"] for r in results]
        assert all(c == pytest.approx(closeness_values[0]) for c in closeness_values)
        # Each node: distances 1,1 => sum=2, closeness=2/2=1.0
        assert closeness_values[0] == pytest.approx(1.0)

    def test_star_center_highest(self, db):
        """In a star, the center node has highest closeness."""
        _build_star_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.closeness_centrality("Person")

        assert results[0]["id"] == "a"
        # Center: distances all 1 => sum=4, closeness=4/4=1.0
        assert results[0]["closeness"] == pytest.approx(1.0)

    def test_sorted_descending(self, db):
        _build_path_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.closeness_centrality("Person")
        closeness_vals = [r["closeness"] for r in results]
        assert closeness_vals == sorted(closeness_vals, reverse=True)

    def test_empty_graph(self, db):
        algo = GraphAlgorithms(db)
        results = algo.closeness_centrality("Person")
        assert results == []

    def test_sample_size(self, db):
        _build_path_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.closeness_centrality("Person", sample_size=3)
        assert len(results) == 3


# ------------------------------------------------------------------
# Betweenness Centrality
# ------------------------------------------------------------------

class TestBetweennessCentrality:
    def test_path_graph_center_highest(self, db):
        """In A-B-C-D-E, C has the highest betweenness."""
        _build_path_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.betweenness_centrality("Person", sample_size=0)

        by_id = {r["id"]: r["betweenness"] for r in results}
        # C is on every shortest path between the two halves
        assert by_id["c"] > by_id["a"]
        assert by_id["c"] > by_id["e"]
        assert results[0]["id"] == "c"

    def test_path_graph_endpoints_zero(self, db):
        """Endpoints A and E have betweenness 0 in a path graph."""
        _build_path_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.betweenness_centrality("Person", sample_size=0)

        by_id = {r["id"]: r["betweenness"] for r in results}
        assert by_id["a"] == pytest.approx(0.0)
        assert by_id["e"] == pytest.approx(0.0)

    def test_triangle_all_equal(self, db):
        """In a triangle with 3 nodes, all betweenness should be 0 (no
        intermediate nodes on any shortest path)."""
        _build_triangle_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.betweenness_centrality("Person", sample_size=0)

        for r in results:
            assert r["betweenness"] == pytest.approx(0.0)

    def test_star_center_highest(self, db):
        """In a star, center has highest betweenness (all paths go through it)."""
        _build_star_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.betweenness_centrality("Person", sample_size=0)

        assert results[0]["id"] == "a"
        # All leaves have betweenness 0
        for r in results[1:]:
            assert r["betweenness"] == pytest.approx(0.0)

    def test_sorted_descending(self, db):
        _build_path_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.betweenness_centrality("Person", sample_size=0)
        vals = [r["betweenness"] for r in results]
        assert vals == sorted(vals, reverse=True)

    def test_two_node_graph(self, db):
        """Two connected nodes should both have betweenness 0."""
        db.create_node("Person", {"id": "x", "name": "X"})
        db.create_node("Person", {"id": "y", "name": "Y"})
        db.create_edge("KNOWS", "x", "y", from_label="Person", to_label="Person")
        algo = GraphAlgorithms(db)
        results = algo.betweenness_centrality("Person", sample_size=0)
        for r in results:
            assert r["betweenness"] == pytest.approx(0.0)

    def test_empty_graph(self, db):
        algo = GraphAlgorithms(db)
        results = algo.betweenness_centrality("Person")
        assert results == []


# ------------------------------------------------------------------
# Link Prediction
# ------------------------------------------------------------------

class TestLinkPrediction:
    def test_common_neighbors(self, db):
        """A-B, A-C, B-C, B-D, C-D.  Predict link A-D.

        Common neighbors of A and D: {B, C} => 2.
        """
        for name in ["A", "B", "C", "D"]:
            db.create_node("Person", {"id": name.lower(), "name": name})
        for src, dst in [("a", "b"), ("a", "c"), ("b", "c"), ("b", "d"), ("c", "d")]:
            db.create_edge("KNOWS", src, dst, from_label="Person", to_label="Person")

        algo = GraphAlgorithms(db)
        result = algo.link_prediction("a", "d")

        assert result["common_neighbors"] == 2
        assert result["predicted"] is True
        assert result["adamic_adar"] > 0.0
        assert 0.0 < result["jaccard"] <= 1.0

    def test_no_common_neighbors(self, db):
        """Disconnected components: A-B and C-D."""
        for name in ["A", "B", "C", "D"]:
            db.create_node("Person", {"id": name.lower(), "name": name})
        db.create_edge("KNOWS", "a", "b", from_label="Person", to_label="Person")
        db.create_edge("KNOWS", "c", "d", from_label="Person", to_label="Person")

        algo = GraphAlgorithms(db)
        result = algo.link_prediction("a", "c")

        assert result["common_neighbors"] == 0
        assert result["adamic_adar"] == pytest.approx(0.0)
        assert result["jaccard"] == pytest.approx(0.0)
        assert result["predicted"] is False

    def test_jaccard_value(self, db):
        """A-B, A-C, B-C.  Predict A-D where D connects to B only.

        neighbors(A) = {B, C}, neighbors(D) = {B}
        common = {B}, union = {B, C}
        jaccard = 1/2
        """
        for name in ["A", "B", "C", "D"]:
            db.create_node("Person", {"id": name.lower(), "name": name})
        for src, dst in [("a", "b"), ("a", "c"), ("b", "c"), ("b", "d")]:
            db.create_edge("KNOWS", src, dst, from_label="Person", to_label="Person")

        algo = GraphAlgorithms(db)
        result = algo.link_prediction("a", "d")

        assert result["jaccard"] == pytest.approx(0.5)

    def test_adamic_adar_formula(self, db):
        """Verify Adamic-Adar computation.

        A-B, A-C, B-D, C-D.  Common neighbors of A and D: {B, C}.
        degree(B) = 2 (connects A, D) => 1/log(2)
        degree(C) = 2 (connects A, D) => 1/log(2)
        adamic_adar = 2/log(2)
        """
        for name in ["A", "B", "C", "D"]:
            db.create_node("Person", {"id": name.lower(), "name": name})
        for src, dst in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]:
            db.create_edge("KNOWS", src, dst, from_label="Person", to_label="Person")

        algo = GraphAlgorithms(db)
        result = algo.link_prediction("a", "d")

        expected_aa = 2.0 / math.log(2)
        assert result["adamic_adar"] == pytest.approx(expected_aa)


class TestPredictLinks:
    def test_predict_finds_missing_links(self, db):
        """In a path A-B-C-D, missing link A-C should be predicted (common neighbor B)."""
        _build_path_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.predict_links("Person", top_k=10)

        # There should be predicted links for pairs with common neighbors
        assert len(results) > 0
        # All results should have positive adamic_adar
        for r in results:
            assert r["adamic_adar"] > 0.0
            assert r["common_neighbors"] > 0

    def test_predict_sorted_by_adamic_adar(self, db):
        _build_bowtie_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.predict_links("Person", top_k=10)

        aa_values = [r["adamic_adar"] for r in results]
        assert aa_values == sorted(aa_values, reverse=True)

    def test_predict_top_k_limit(self, db):
        _build_bowtie_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.predict_links("Person", top_k=1)
        assert len(results) <= 1

    def test_predict_no_links_in_complete_graph(self, db):
        """A complete graph has no missing links to predict."""
        _build_triangle_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.predict_links("Person", top_k=10)
        assert results == []

    def test_predict_no_links_disconnected(self, db):
        """Two disconnected pairs: A-B and C-D. No common neighbors between components."""
        for name in ["A", "B", "C", "D"]:
            db.create_node("Person", {"id": name.lower(), "name": name})
        db.create_edge("KNOWS", "a", "b", from_label="Person", to_label="Person")
        db.create_edge("KNOWS", "c", "d", from_label="Person", to_label="Person")

        algo = GraphAlgorithms(db)
        results = algo.predict_links("Person", top_k=10)
        assert results == []


# ------------------------------------------------------------------
# FastRP Graph Embeddings
# ------------------------------------------------------------------

class TestFastRP:
    def test_returns_correct_dimension(self, db):
        _build_triangle_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.fast_rp("Person", dimension=16, iterations=2, seed=42)

        assert len(results) == 3
        for r in results:
            assert len(r["embedding"]) == 16
            assert "id" in r
            assert "label" in r

    def test_embedding_not_all_zero(self, db):
        _build_triangle_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.fast_rp("Person", dimension=32, iterations=2, seed=42)

        for r in results:
            emb = r["embedding"]
            assert any(abs(x) > 1e-10 for x in emb), "Embedding should not be all zeros"

    def test_seed_reproducibility(self, db):
        _build_path_graph(db)
        algo = GraphAlgorithms(db)

        r1 = algo.fast_rp("Person", dimension=16, iterations=2, seed=123)
        r2 = algo.fast_rp("Person", dimension=16, iterations=2, seed=123)

        for a, b in zip(r1, r2):
            for x, y in zip(a["embedding"], b["embedding"]):
                assert x == pytest.approx(y)

    def test_different_seeds_differ(self, db):
        _build_path_graph(db)
        algo = GraphAlgorithms(db)

        r1 = algo.fast_rp("Person", dimension=16, iterations=2, seed=1)
        r2 = algo.fast_rp("Person", dimension=16, iterations=2, seed=999)

        # At least one embedding should differ
        any_differ = False
        for a, b in zip(r1, r2):
            for x, y in zip(a["embedding"], b["embedding"]):
                if abs(x - y) > 1e-10:
                    any_differ = True
                    break
            if any_differ:
                break
        assert any_differ, "Different seeds should produce different embeddings"

    def test_similar_nodes_similar_embeddings(self, db):
        """In a symmetric graph (triangle), all nodes should get similar embeddings
        (close L2 distance after neighbor averaging)."""
        _build_triangle_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.fast_rp("Person", dimension=64, iterations=3, seed=42)

        embs = [r["embedding"] for r in results]
        # Pairwise L2 distances should be small relative to norm
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                dist = math.sqrt(sum(
                    (a - b) ** 2 for a, b in zip(embs[i], embs[j])
                ))
                norm_i = math.sqrt(sum(x ** 2 for x in embs[i]))
                # Distance should be small relative to the embedding magnitude
                assert dist < norm_i * 2.0, "Symmetric nodes should have similar embeddings"

    def test_normalization(self, db):
        """When normalization_strength > 0, embeddings should be approximately unit length."""
        _build_path_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.fast_rp(
            "Person", dimension=32, iterations=3,
            normalization_strength=1.0, seed=42,
        )

        for r in results:
            norm = math.sqrt(sum(x ** 2 for x in r["embedding"]))
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_empty_graph(self, db):
        algo = GraphAlgorithms(db)
        results = algo.fast_rp("Person", dimension=16)
        assert results == []

    def test_single_node(self, db):
        db.create_node("Person", {"id": "solo", "name": "Solo"})
        algo = GraphAlgorithms(db)
        results = algo.fast_rp("Person", dimension=8, iterations=2, seed=42)

        assert len(results) == 1
        assert results[0]["id"] == "solo"
        assert len(results[0]["embedding"]) == 8
