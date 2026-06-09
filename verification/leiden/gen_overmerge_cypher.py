#!/usr/bin/env python3
"""Generate parquet + a .cypher script to OBSERVE real CALL LEIDEN over-merge in
the shell. Stores the true entity id on each node (`te`) so over-merge is measured
in Cypher: per community, how many distinct true entities were fused.

Emits, for the fragmented-ER graph:
  - LEIDEN (unweighted), LOUVAIN (unweighted)  -> #1 action (does real Leiden over-merge / beat Louvain)
  - LEIDEN unweighted vs weighted on a weighted copy (strong intra=5, weak bridge=0.2) -> Phase 6
"""
from __future__ import annotations
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from leiden_verify import fragmented_er  # noqa: E402

DATA = os.path.join(HERE, "_data")
os.makedirs(DATA, exist_ok=True)
EXT = "C:/Users/eastr/Projects/bridgr-engine/extension/algo/build/libalgo.lbug_extension"


def fwd(p):
    return p.replace("\\", "/")


def main():
    n, edges, truth = fragmented_er()
    nt = len(set(truth.values()))
    weights = [5.0 if truth[u] == truth[v] else 0.2 for (u, v) in edges]

    pq.write_table(pa.table({"id": pa.array(range(n), pa.int64()),
                             "te": pa.array([truth[i] for i in range(n)], pa.int64())}),
                   os.path.join(DATA, "nodes.parquet"))
    pq.write_table(pa.table({"from": pa.array([e[0] for e in edges], pa.int64()),
                             "to": pa.array([e[1] for e in edges], pa.int64())}),
                   os.path.join(DATA, "edges.parquet"))
    pq.write_table(pa.table({"from": pa.array([e[0] for e in edges], pa.int64()),
                             "to": pa.array([e[1] for e in edges], pa.int64()),
                             "w": pa.array(weights, pa.float64())}),
                   os.path.join(DATA, "edges_w.parquet"))

    np_, ep, ewp = (fwd(os.path.join(DATA, f)) for f in ("nodes.parquet", "edges.parquet", "edges_w.parquet"))

    def measure(call):
        # one-row summary: communities, impure (false-merge) communities, worst fusion, max size
        return (f"{call} WITH community_id, count(DISTINCT node.te) AS te_in_comm, count(*) AS sz "
                f"RETURN count(*) AS n_comms, "
                f"sum(CASE WHEN te_in_comm > 1 THEN 1 ELSE 0 END) AS impure_comms, "
                f"max(te_in_comm) AS worst_merge, max(sz) AS max_size;")

    def measure_louv(call):
        return (f"{call} WITH louvain_id, count(DISTINCT node.te) AS te_in_comm, count(*) AS sz "
                f"RETURN count(*) AS n_comms, "
                f"sum(CASE WHEN te_in_comm > 1 THEN 1 ELSE 0 END) AS impure_comms, "
                f"max(te_in_comm) AS worst_merge, max(sz) AS max_size;")

    lines = [
        f"LOAD EXTENSION '{EXT}';",
        "CREATE NODE TABLE Node(id INT64 PRIMARY KEY, te INT64);",
        "CREATE REL TABLE Edge(FROM Node TO Node);",
        "CREATE REL TABLE EdgeW(FROM Node TO Node, w DOUBLE);",
        f'COPY Node FROM "{np_}";',
        f'COPY Edge FROM "{ep}";',
        f'COPY EdgeW FROM "{ewp}";',
        "CALL PROJECT_GRAPH('g', ['Node'], ['Edge']);",
        "CALL PROJECT_GRAPH('gw', ['Node'], ['EdgeW']);",
        f"RETURN {nt} AS TRUE_ENTITY_COUNT, {n} AS NODES, {len(edges)} AS EDGES;",
        "// ---- #1 action: real LEIDEN vs LOUVAIN over-merge (unweighted) ----",
        measure("CALL LEIDEN('g', maxphases:=50, maxiterations:=100)"),
        measure_louv("CALL LOUVAIN('g', maxphases:=50, maxiterations:=100)"),
        "// ---- Phase 6: LEIDEN unweighted vs weighted on the weighted graph ----",
        measure("CALL LEIDEN('gw', maxphases:=50, maxiterations:=100)"),
        measure("CALL LEIDEN('gw', maxphases:=50, maxiterations:=100, weight_property:='w')"),
    ]
    script = os.path.join(DATA, "overmerge.cypher")
    with open(script, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", script)
    print(f"true_entities={nt} nodes={n} edges={len(edges)}")
    print("rows expected: TRUE_ENTITY_COUNT, then 4 measure rows "
          "(leiden, louvain, leiden-gw-unweighted, leiden-gw-weighted)")


if __name__ == "__main__":
    main()
