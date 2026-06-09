#!/usr/bin/env python3
"""Phase 2 (#1 action) + Phase 6 on REAL CALL LEIDEN, using a from-source
real_ladybug (the installed wheel segfaults on a from-source extension).

Requires:
  PYTHONPATH includes the built python module (build/release/tools/python_api/build)
  ALGO_EXT = path to the freshly-built libalgo.lbug_extension (default below)

Answers:
  (#1 action) Does the REAL bespoke Leiden inherit the over-merge, and does its
              refinement help vs Louvain? (compare on the fragmented ER fixture)
  (Phase 6)   Does weighting help ER? (weighted vs unweighted CALL LEIDEN on a
              weighted fragmented graph: strong intra edges, weak spurious bridges)
"""
from __future__ import annotations
import os
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from leiden_verify import (fragmented_er, bcubed_prf, oce_uce, size_diag,  # noqa: E402
                           run_igraph_leiden, run_igraph_louvain)

from bridgr.database import Database  # noqa: E402

EXT = os.environ.get(
    "ALGO_EXT",
    "C:/Users/eastr/Projects/bridgr-engine/extension/algo/build/libalgo.lbug_extension",
).replace("\\", "/")


def _pq(tbl, suffix):
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    pq.write_table(tbl, p)
    return p


def _engine_db(n, edges, weights=None):
    db = Database(":memory:")
    db.execute(f"LOAD EXTENSION '{EXT}'")
    db.execute("CREATE NODE TABLE Node(id INT64 PRIMARY KEY)")
    if weights is None:
        db.execute("CREATE REL TABLE Edge(FROM Node TO Node)")
    else:
        db.execute("CREATE REL TABLE Edge(FROM Node TO Node, w DOUBLE)")
    npath = _pq(pa.table({"id": pa.array(list(range(n)), pa.int64())}), "_n.parquet")
    cols = {"from": pa.array([e[0] for e in edges], pa.int64()),
            "to": pa.array([e[1] for e in edges], pa.int64())}
    if weights is not None:
        cols["w"] = pa.array(weights, pa.float64())
    epath = _pq(pa.table(cols), "_e.parquet")
    try:
        db.execute(f'COPY Node FROM "{npath}"')
        db.execute(f'COPY Edge FROM "{epath}"')
    finally:
        for p in (npath, epath):
            try:
                os.unlink(p)
            except OSError:
                pass
    db.execute("CALL PROJECT_GRAPH('g', ['Node'], ['Edge'])")
    return db


def run_real_leiden(n, edges, resolution=1.0, weighted=False, weights=None):
    db = _engine_db(n, edges, weights if weighted else None)
    try:
        wp = ", weight_property:='w'" if weighted else ""
        rows = db.query(f"CALL LEIDEN('g', resolution := {resolution}, maxphases := 50, "
                        f"maxiterations := 100{wp}) RETURN node.id AS nid, community_id AS c")
        return {int(r["nid"]): int(r["c"]) for r in rows}
    finally:
        db.close()


def run_real_louvain(n, edges):
    db = _engine_db(n, edges, None)
    try:
        rows = db.query("CALL LOUVAIN('g', maxphases := 50, maxiterations := 100) "
                        "RETURN node.id AS nid, louvain_id AS c")
        return {int(r["nid"]): int(r["c"]) for r in rows}
    finally:
        db.close()


def weighted_fragmented():
    """fragmented_er topology, but with edge weights: strong intra-entity edges,
    weak spurious bridges/hub/ring-bridge edges. Truth = the entity labels.
    Returns (n, edges, weights, truth)."""
    n, edges, truth = fragmented_er()
    # An edge is 'intra' if both endpoints share a true entity, else spurious.
    weights = []
    for u, v in edges:
        weights.append(5.0 if truth[u] == truth[v] else 0.2)
    return n, edges, weights, truth


def score(lab, truth, label):
    bc = bcubed_prf(lab, truth)
    oce, uce = oce_uce(lab, truth)
    d = size_diag(lab)
    print(f"  {label:24s} B3 P={bc.precision:.3f} R={bc.recall:.3f} F1={bc.f1:.3f}  "
          f"OCE={oce:5d} UCE={uce:4d}  comms={d['n_comm']:3d} max={d['max_size']:3d}")
    return dict(label=label, f1=round(bc.f1, 3), p=round(bc.precision, 3), r=round(bc.recall, 3),
                oce=oce, uce=uce, n_comm=d["n_comm"], max_size=d["max_size"])


def main():
    print("=" * 74)
    print("REAL CALL LEIDEN — #1 action (over-merge) + Phase 6 (weighting helps ER?)")
    print(f"extension: {EXT}")
    print("=" * 74)

    n, edges, truth = fragmented_er()
    nt = len(set(truth.values()))
    print(f"\nFragmented ER graph: {n} nodes, {len(edges)} edges, {nt} true entities\n")
    print("--- #1 action: does REAL Leiden over-merge, and does refinement beat Louvain? ---")
    rows = []
    rows.append(score(run_real_leiden(n, edges), truth, "engine_LEIDEN(real)"))
    rows.append(score(run_real_louvain(n, edges), truth, "engine_LOUVAIN(real)"))
    rows.append(score(run_igraph_louvain(n, edges), truth, "igraph_louvain(ref)"))
    rows.append(score(run_igraph_leiden(n, edges, "modularity", 1.0), truth, "igraph_leiden_mod(ref)"))
    rows.append(score(run_igraph_leiden(n, edges, "CPM", 0.1), truth, "igraph_CPM@0.1(ref)"))

    print("\n--- Phase 6: weighted vs unweighted CALL LEIDEN on a weighted fragmented graph ---")
    wn, wedges, wweights, wtruth = weighted_fragmented()
    score(run_real_leiden(wn, wedges, weighted=False), wtruth, "LEIDEN unweighted")
    score(run_real_leiden(wn, wedges, weighted=True, weights=wweights), wtruth, "LEIDEN weighted(w)")

    import json
    with open(os.path.join(HERE, "results", "real_leiden.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nwrote results/real_leiden.json")


if __name__ == "__main__":
    main()
