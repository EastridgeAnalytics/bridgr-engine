"""Regression: graph-algorithm wrappers must use the node table's declared
primary key, not a hardcoded ``id``.

Tables whose primary key isn't named ``id`` (e.g. H-E-B's ``record_id``)
previously raised "Cannot find property id for node" — or returned NULL
``node_id`` — from the CALL-based algorithm wrappers, which hardcoded
``RETURN node.id``. The wrappers now derive the PK via ``table_info``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bridgr.algorithms import GraphAlgorithms
from bridgr.database import Database


def _algo_available() -> bool:
    """True iff the algo extension actually runs (not just installs)."""
    try:
        db = Database(":memory:")
        db.execute("CREATE NODE TABLE _P(id INT64 PRIMARY KEY)")
        db.execute("CREATE REL TABLE _E(FROM _P TO _P)")
        db.execute("CREATE (:_P {id: 0})")
        algo = GraphAlgorithms(db)
        algo.weakly_connected_components("_P", "_E")
        db.close()
        return True
    except Exception:
        return False


_HAS_ALGO = _algo_available()
skip_no_algo = pytest.mark.skipif(not _HAS_ALGO, reason="algo extension unavailable")


@pytest.fixture
def db_record_id():
    """Two disjoint triangles whose node table's PK is ``record_id`` (not ``id``)."""
    db = Database(":memory:")
    db.execute("CREATE NODE TABLE Rec(record_id STRING PRIMARY KEY, name STRING)")
    db.execute("CREATE REL TABLE LINK(FROM Rec TO Rec)")
    ids = {"r1", "r2", "r3", "r4", "r5", "r6"}
    for i in sorted(ids):
        db.execute(f"CREATE (:Rec {{record_id: '{i}', name: '{i}'}})")
    for a, b in [("r1", "r2"), ("r2", "r3"), ("r1", "r3"),
                 ("r4", "r5"), ("r5", "r6"), ("r4", "r6")]:
        db.execute(
            f"MATCH (a:Rec {{record_id:'{a}'}}), (b:Rec {{record_id:'{b}'}}) "
            f"CREATE (a)-[:LINK]->(b)"
        )
    yield db, ids
    db.close()


def test_primary_key_lookup(db_record_id):
    db, _ = db_record_id
    assert GraphAlgorithms(db)._primary_key("Rec") == "record_id"


def test_primary_key_falls_back_to_id_when_unknown(db_record_id):
    db, _ = db_record_id
    # A label with no table_info / no PK resolves to the historical default.
    assert GraphAlgorithms(db)._primary_key("DoesNotExist") == "id"


@skip_no_algo
def test_wcc_uses_declared_pk(db_record_id):
    db, ids = db_record_id
    results = GraphAlgorithms(db).weakly_connected_components("Rec", "LINK")
    node_ids = {r["node_id"] for r in results}
    assert None not in node_ids  # old `node.id` returned NULL / errored here
    assert node_ids == ids
    assert len({r["component_id"] for r in results}) == 2


@skip_no_algo
def test_louvain_uses_declared_pk(db_record_id):
    db, ids = db_record_id
    results = GraphAlgorithms(db).louvain("Rec", "LINK")
    assert {r["node_id"] for r in results} == ids


@skip_no_algo
def test_id_pk_table_still_works():
    """A table whose PK *is* ``id`` is unchanged (helper returns 'id')."""
    db = Database(":memory:")
    db.execute("CREATE NODE TABLE N(id STRING PRIMARY KEY)")
    db.execute("CREATE REL TABLE E(FROM N TO N)")
    for i in ("a", "b"):
        db.execute(f"CREATE (:N {{id: '{i}'}})")
    db.execute("MATCH (x:N {id:'a'}), (y:N {id:'b'}) CREATE (x)-[:E]->(y)")
    algo = GraphAlgorithms(db)
    assert algo._primary_key("N") == "id"
    results = algo.weakly_connected_components("N", "E")
    assert {r["node_id"] for r in results} == {"a", "b"}
    db.close()
