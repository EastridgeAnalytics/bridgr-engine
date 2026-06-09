#!/usr/bin/env python3
"""Phase 3 (partial) — honest scale/timing for the engine modularity core.

CALL LEIDEN is absent from the wheel, so we time CALL LOUVAIN (the shared core)
at increasing scale + a determinism check. Replaces the retracted Leiden timing
with a real, reproducible engine-modularity number. Re-point at CALL LEIDEN in CI.
"""
from __future__ import annotations
import json
import os
import random
import time

from leiden_verify import _engine_db, clusters_from_labeling, RESULTS


def scale_graph(n, n_blocks, avg_deg=8, cross_frac=0.03, seed=0):
    rng = random.Random(seed)
    block = [i % n_blocks for i in range(n)]
    by_block = {}
    for i in range(n):
        by_block.setdefault(block[i], []).append(i)
    seen = set()
    edges = []
    for i in range(n):
        peers = by_block[block[i]]
        for _ in range(avg_deg):
            if rng.random() < cross_frac:
                j = rng.randrange(n)
            else:
                j = peers[rng.randrange(len(peers))]
            if j == i:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            edges.append((i, j))
    return n, edges


def louvain_timed(n, edges):
    db = _engine_db(n, edges)
    try:
        t0 = time.time()
        rows = db.query("CALL LOUVAIN('g') RETURN node.id AS node_id, louvain_id AS community_id")
        dt = time.time() - t0
        lab = {int(r["node_id"]): int(r["community_id"]) for r in rows}
        sizes = [len(c) for c in clusters_from_labeling(lab).values()]
        return dt, len(sizes), max(sizes), lab
    finally:
        db.close()


def main():
    rows = []
    print(f"{'nodes':>8} {'edges':>9} {'louvain_s':>10} {'comms':>8} {'maxsz':>8}")
    print("-" * 50)
    for n, nb in [(10_000, 200), (50_000, 800), (100_000, 1500)]:
        gn, edges = scale_graph(n, nb, seed=1)
        dt, ncomm, mx, _ = louvain_timed(gn, edges)
        rows.append({"nodes": gn, "edges": len(edges), "louvain_s": round(dt, 3),
                     "comms": ncomm, "max_size": mx})
        print(f"{gn:8d} {len(edges):9d} {dt:10.3f} {ncomm:8d} {mx:8d}")

    # determinism at 50k
    gn, edges = scale_graph(50_000, 800, seed=2)
    _, _, _, l1 = louvain_timed(gn, edges)
    _, _, _, l2 = louvain_timed(gn, edges)
    deterministic = l1 == l2
    print(f"\n  determinism @50k (two runs identical): {deterministic}")

    with open(os.path.join(RESULTS, "scale.json"), "w") as fh:
        json.dump({"scale": rows, "deterministic_50k": deterministic}, fh, indent=2)
    print("\nWrote", os.path.join(RESULTS, "scale.json"))
    print("NOTE: this is the engine LOUVAIN core (Leiden binary absent). Timings are illustrative,")
    print("      single-machine; re-run against CALL LEIDEN in CI for the Leiden number.")


if __name__ == "__main__":
    main()
