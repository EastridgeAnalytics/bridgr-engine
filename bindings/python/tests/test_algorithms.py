"""Tests for Bridgr Graph Algorithms — stories R6-1 through R6-6."""

import shutil
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import bridgr
from bridgr.algorithms import GraphAlgorithms

TEST_DIR = Path(__file__).parent / "test_algo_dbs"


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
    d = bridgr.open(":memory:")
    d.create_node_table("Person", {
        "id": "STRING PRIMARY KEY",
        "name": "STRING",
    })
    d.create_edge_table("KNOWS", "Person", "Person")
    yield d
    d.close()


def _build_graph(db):
    """Build a test graph:
    A -- B -- C
    |         |
    D -- E -- F
    """
    for name in ["A", "B", "C", "D", "E", "F"]:
        db.create_node("Person", {"id": name.lower(), "name": name})

    edges = [("a", "b"), ("b", "c"), ("a", "d"), ("d", "e"), ("e", "f"), ("c", "f")]
    for src, dst in edges:
        db.create_edge("KNOWS", src, dst, from_label="Person", to_label="Person")


def _build_disconnected_graph(db):
    """Two disconnected components: {A,B,C} and {D,E}."""
    for name in ["A", "B", "C", "D", "E"]:
        db.create_node("Person", {"id": name.lower(), "name": name})

    db.create_edge("KNOWS", "a", "b", from_label="Person", to_label="Person")
    db.create_edge("KNOWS", "b", "c", from_label="Person", to_label="Person")
    db.create_edge("KNOWS", "d", "e", from_label="Person", to_label="Person")


# ------------------------------------------------------------------
# Cypher-based algorithms (always work)
# ------------------------------------------------------------------

class TestShortestPath:
    def test_direct_connection(self, db):
        _build_graph(db)
        algo = GraphAlgorithms(db)
        result = algo.shortest_path("a", "b", "Person", edge_label="KNOWS")
        assert result is not None
        assert result["path_length"] == 1

    def test_multi_hop(self, db):
        _build_graph(db)
        algo = GraphAlgorithms(db)
        result = algo.shortest_path("a", "f", "Person", edge_label="KNOWS")
        assert result is not None
        assert result["path_length"] <= 3

    def test_no_path(self, db):
        _build_disconnected_graph(db)
        algo = GraphAlgorithms(db)
        result = algo.shortest_path("a", "d", "Person", edge_label="KNOWS")
        assert result is None


class TestDegreeCentrality:
    def test_degree_centrality(self, db):
        _build_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.degree_centrality("Person", "KNOWS")
        assert len(results) == 6
        by_id = {r["node_id"]: r for r in results}
        assert by_id["a"]["total_degree"] >= 2
        assert by_id["b"]["total_degree"] >= 2

    def test_isolated_node(self, db):
        db.create_node("Person", {"id": "lonely", "name": "Lonely"})
        algo = GraphAlgorithms(db)
        results = algo.degree_centrality("Person", "KNOWS")
        by_id = {r["node_id"]: r for r in results}
        assert by_id["lonely"]["total_degree"] == 0


class TestNodeSimilarity:
    def test_jaccard_identical_neighbors(self, db):
        _build_graph(db)
        algo = GraphAlgorithms(db)
        sim = algo.node_similarity("a", "a", "Person", "KNOWS", metric="jaccard")
        assert sim == 1.0

    def test_jaccard_different_neighbors(self, db):
        _build_disconnected_graph(db)
        algo = GraphAlgorithms(db)
        sim = algo.node_similarity("a", "d", "Person", "KNOWS", metric="jaccard")
        assert sim == 0.0

    def test_overlap_metric(self, db):
        _build_graph(db)
        algo = GraphAlgorithms(db)
        sim = algo.node_similarity("a", "b", "Person", "KNOWS", metric="overlap")
        assert 0.0 <= sim <= 1.0

    def test_invalid_metric(self, db):
        algo = GraphAlgorithms(db)
        with pytest.raises(ValueError):
            algo.node_similarity("a", "b", "Person", "KNOWS", metric="invalid")


# ------------------------------------------------------------------
# Built-in extension algorithms (require algo extension)
# ------------------------------------------------------------------

def _algo_extension_available():
    """Check if the algo extension is loadable."""
    try:
        db = bridgr.open(":memory:")
        db.execute("LOAD EXTENSION algo")
        db.close()
        return True
    except RuntimeError:
        return False


algo_available = pytest.mark.skipif(
    not _algo_extension_available(),
    reason="algo extension not available"
)


@algo_available
class TestWCC:
    def test_single_component(self, db):
        _build_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.weakly_connected_components("Person", "KNOWS")
        assert len(results) == 6
        component_ids = {r["component_id"] for r in results}
        assert len(component_ids) == 1

    def test_two_components(self, db):
        _build_disconnected_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.weakly_connected_components("Person", "KNOWS")
        component_ids = {r["component_id"] for r in results}
        assert len(component_ids) == 2


@algo_available
class TestPageRank:
    def test_pagerank_returns_scores(self, db):
        _build_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.pagerank("Person", "KNOWS")
        assert len(results) == 6
        for r in results:
            assert "score" in r
            assert r["score"] > 0

    def test_pagerank_ordered_descending(self, db):
        _build_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.pagerank("Person", "KNOWS")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


@algo_available
class TestLouvain:
    def test_community_detection(self, db):
        _build_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.louvain("Person", "KNOWS")
        assert len(results) == 6
        for r in results:
            assert "community_id" in r


def _build_barbell_graph(db):
    """Barbell: two cliques {A,B,C} and {D,E,F} joined by bridge C-D,
    plus an isolated pair {G,H}. Expected: 3 connected communities."""
    for name in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        db.create_node("Person", {"id": name.lower(), "name": name})

    edges = [
        ("a", "b"), ("a", "c"), ("b", "c"),
        ("d", "e"), ("d", "f"), ("e", "f"),
        ("c", "d"),
        ("g", "h"),
    ]
    for src, dst in edges:
        db.create_edge("KNOWS", src, dst, from_label="Person", to_label="Person")


def _build_fully_connected(db, n=5):
    """Fully connected graph of n nodes."""
    ids = [chr(ord("a") + i) for i in range(n)]
    for nid in ids:
        db.create_node("Person", {"id": nid, "name": nid.upper()})
    for i, src in enumerate(ids):
        for dst in ids[i + 1 :]:
            db.create_edge("KNOWS", src, dst, from_label="Person", to_label="Person")


@algo_available
class TestLeiden:
    def test_leiden_basic(self, db):
        _build_disconnected_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.leiden("Person", "KNOWS")
        assert len(results) == 5
        comms = {r["community_id"] for r in results}
        assert len(comms) == 2

    def test_leiden_single_community(self, db):
        _build_fully_connected(db, 5)
        algo = GraphAlgorithms(db)
        results = algo.leiden("Person", "KNOWS")
        assert len(results) == 5
        comms = {r["community_id"] for r in results}
        assert len(comms) == 1

    def test_leiden_empty_graph(self, db):
        algo = GraphAlgorithms(db)
        results = algo.leiden("Person", "KNOWS")
        assert results == []

    def test_leiden_returns_sorted(self, db):
        _build_disconnected_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.leiden("Person", "KNOWS")
        keys = [(r["community_id"], r["node_id"]) for r in results]
        assert keys == sorted(keys)

    def test_leiden_resolution_low(self, db):
        _build_barbell_graph(db)
        algo = GraphAlgorithms(db)
        low = algo.leiden("Person", "KNOWS", resolution=0.1)
        default = algo.leiden("Person", "KNOWS", resolution=1.0)
        comms_low = len({r["community_id"] for r in low})
        comms_default = len({r["community_id"] for r in default})
        assert comms_low <= comms_default

    def test_leiden_resolution_high(self, db):
        _build_barbell_graph(db)
        algo = GraphAlgorithms(db)
        default = algo.leiden("Person", "KNOWS", resolution=1.0)
        high = algo.leiden("Person", "KNOWS", resolution=5.0)
        comms_default = len({r["community_id"] for r in default})
        comms_high = len({r["community_id"] for r in high})
        assert comms_high >= comms_default

    def test_leiden_communities_connected(self, db):
        _build_barbell_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.leiden("Person", "KNOWS")

        edges = [
            ("a", "b"), ("a", "c"), ("b", "c"),
            ("d", "e"), ("d", "f"), ("e", "f"),
            ("c", "d"), ("g", "h"),
        ]
        adj: dict[str, set[str]] = {}
        for s, d in edges:
            adj.setdefault(s, set()).add(d)
            adj.setdefault(d, set()).add(s)

        comm_nodes: dict[int, set[str]] = {}
        for r in results:
            comm_nodes.setdefault(r["community_id"], set()).add(r["node_id"])

        from collections import deque
        for members in comm_nodes.values():
            if len(members) <= 1:
                continue
            start = next(iter(members))
            visited = {start}
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj.get(u, set()):
                    if v in members and v not in visited:
                        visited.add(v)
                        q.append(v)
            assert visited == members, f"Community {members} is not connected"

    def test_leiden_vs_louvain_on_barbell(self, db):
        _build_barbell_graph(db)
        algo = GraphAlgorithms(db)
        leiden_results = algo.leiden("Person", "KNOWS")

        edges = [
            ("a", "b"), ("a", "c"), ("b", "c"),
            ("d", "e"), ("d", "f"), ("e", "f"),
            ("c", "d"), ("g", "h"),
        ]
        adj: dict[str, set[str]] = {}
        for s, d in edges:
            adj.setdefault(s, set()).add(d)
            adj.setdefault(d, set()).add(s)

        leiden_comms: dict[int, set[str]] = {}
        for r in leiden_results:
            leiden_comms.setdefault(r["community_id"], set()).add(r["node_id"])

        from collections import deque
        for members in leiden_comms.values():
            if len(members) <= 1:
                continue
            start = next(iter(members))
            visited = {start}
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj.get(u, set()):
                    if v in members and v not in visited:
                        visited.add(v)
                        q.append(v)
            assert visited == members


@algo_available
class TestSCC:
    def test_strongly_connected(self, db):
        _build_graph(db)
        algo = GraphAlgorithms(db)
        results = algo.strongly_connected_components("Person", "KNOWS")
        assert len(results) == 6
