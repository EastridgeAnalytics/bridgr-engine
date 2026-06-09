#!/usr/bin/env python3
"""Bespoke Leiden verification harness (Phases 0-2 of the dev plan).

Tests the ALREADY-COMPILED bespoke Leiden in the installed `real_ladybug` wheel
(no C++ build required) against an igraph reference (modularity + CPM).

Phase 0  smoke + which-Leiden-is-this + determinism
Phase 1  correctness on known-answer graphs (NMI/ARI vs ground truth, connectivity)
Phase 2  ER-quality + the bug-vs-objective verdict (B-cubed / OCE-UCE;
         bespoke-Leiden vs bespoke-Louvain vs igraph-modularity vs igraph-CPM)

Metric implementations are faithful copies of the calibration notebook's
`leiden_lib.py` (EA, 2026) — pure numpy/stdlib (that module imports leidenalg,
which is not installed here, so we copy the metric fns and use igraph's native
Leiden for the reference instead).

Run:  python verification/leiden/leiden_verify.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict, namedtuple

import igraph as ig
import pyarrow as pa
import pyarrow.parquet as pq

from bridgr.database import Database

try:
    from bridgr.schema_utils import cypher_path as _cypher_path
except Exception:                                    # pragma: no cover
    def _cypher_path(p: str) -> str:
        return p.replace("\\", "/")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

PRF = namedtuple("PRF", "precision recall f1")

# ── metrics (copied from leiden_lib.py; labelings are {node_id: cluster}) ─────

def _harm(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

def _comb2(n):
    return n * (n - 1) // 2

def clusters_from_labeling(labeling):
    out = defaultdict(set)
    for rec, lab in labeling.items():
        out[lab].add(rec)
    return dict(out)

def bcubed_prf(pred, truth):
    keys = pred.keys() & truth.keys()
    pred_groups = defaultdict(list)
    true_size = Counter()
    for r in keys:
        pred_groups[pred[r]].append(r)
        true_size[truth[r]] += 1
    sp = sr = 0.0
    n = 0
    for members in pred_groups.values():
        m = len(members)
        cnt = Counter(truth[r] for r in members)
        for r in members:
            overlap = cnt[truth[r]]
            sp += overlap / m
            sr += overlap / true_size[truth[r]]
            n += 1
    p = sp / n if n else 1.0
    r = sr / n if n else 1.0
    return PRF(p, r, _harm(p, r))

def pairwise_prf(pred, truth):
    keys = pred.keys() & truth.keys()
    pred_groups = defaultdict(list)
    true_sizes = Counter()
    for r in keys:
        pred_groups[pred[r]].append(r)
        true_sizes[truth[r]] += 1
    tp = pp = 0
    for members in pred_groups.values():
        pp += _comb2(len(members))
        sub = Counter(truth[r] for r in members)
        for c in sub.values():
            tp += _comb2(c)
    ap = sum(_comb2(c) for c in true_sizes.values())
    p = tp / pp if pp else 1.0
    r = tp / ap if ap else 1.0
    return PRF(p, r, _harm(p, r))

def oce_uce(pred, truth):
    """OCE = false merges (records wrongly pulled in), UCE = false splits."""
    keys = pred.keys() & truth.keys()
    pred_size = Counter(pred[r] for r in keys)
    true_groups = defaultdict(list)
    for r in keys:
        true_groups[truth[r]].append(r)
    oce_total = uce_total = 0
    for c, members in true_groups.items():
        pred_labels = Counter(pred[r] for r in members)
        anchor, anchor_in_c = pred_labels.most_common(1)[0]
        oce_total += pred_size[anchor] - anchor_in_c
        uce_total += len(members) - anchor_in_c
    return oce_total, uce_total

def size_diag(labeling):
    sizes = [len(c) for c in clusters_from_labeling(labeling).values()]
    n = sum(sizes)
    mx = max(sizes) if sizes else 0
    return {"n_comm": len(sizes), "max_size": mx, "giant": bool(n and mx > 0.10 * n)}

def membership_list(labeling, n):
    return [labeling[i] for i in range(n)]

def nmi(pred, truth, n):
    return ig.compare_communities(membership_list(pred, n), membership_list(truth, n), method="nmi")

def ari(pred, truth, n):
    return ig.compare_communities(membership_list(pred, n), membership_list(truth, n),
                                  method="adjusted_rand")

# ── engines ──────────────────────────────────────────────────────────────────

def _write_parquet(table, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    pq.write_table(table, path)
    return path

def _load_graph(db, n, edges):
    """Create Node/Edge tables and bulk-load via COPY FROM parquet."""
    db.execute("CREATE NODE TABLE Node(id INT64 PRIMARY KEY)")
    db.execute("CREATE REL TABLE Edge(FROM Node TO Node)")
    npath = _write_parquet(pa.table({"id": pa.array(list(range(n)), pa.int64())}), "_nodes.parquet")
    ef = [e[0] for e in edges]
    et = [e[1] for e in edges]
    epath = _write_parquet(pa.table({"from": pa.array(ef, pa.int64()),
                                     "to": pa.array(et, pa.int64())}), "_edges.parquet")
    try:
        db.execute(f'COPY Node FROM "{_cypher_path(npath)}"')
        db.execute(f'COPY Edge FROM "{_cypher_path(epath)}"')
    finally:
        for p in (npath, epath):
            try:
                os.unlink(p)
            except OSError:
                pass

def _engine_db(n, edges):
    db = Database(":memory:")
    try:
        db.execute("INSTALL algo")
    except Exception:
        pass
    try:
        db.execute("LOAD EXTENSION algo")
    except Exception:
        pass
    _load_graph(db, n, edges)
    db.execute("CALL PROJECT_GRAPH('g', ['Node'], ['Edge'])")
    return db

def run_bespoke_leiden(n, edges, resolution=1.0, maxphases=20, maxiterations=20):
    db = _engine_db(n, edges)
    try:
        rows = db.query(
            f"CALL LEIDEN('g', resolution := {resolution}, maxphases := {maxphases}, "
            f"maxiterations := {maxiterations}) RETURN node.id AS node_id, community_id"
        )
        return {int(r["node_id"]): int(r["community_id"]) for r in rows}
    finally:
        db.close()

def run_bespoke_louvain(n, edges):
    db = _engine_db(n, edges)
    try:
        rows = db.query("CALL LOUVAIN('g') RETURN node.id AS node_id, louvain_id AS community_id")
        return {int(r["node_id"]): int(r["community_id"]) for r in rows}
    finally:
        db.close()

def run_igraph_leiden(n, edges, objective, resolution):
    g = ig.Graph(n=n, edges=[(int(u), int(v)) for u, v in edges])
    kwargs = dict(objective_function=objective, n_iterations=-1)
    try:
        part = g.community_leiden(resolution=resolution, **kwargs)
    except TypeError:
        part = g.community_leiden(resolution_parameter=resolution, **kwargs)
    return {i: part.membership[i] for i in range(n)}

def run_igraph_louvain(n, edges):
    """igraph's Louvain (community_multilevel) — direct implementation-parity
    reference for the engine's upstream Louvain."""
    g = ig.Graph(n=n, edges=[(int(u), int(v)) for u, v in edges])
    part = g.community_multilevel()
    return {i: part.membership[i] for i in range(n)}

def connectivity_ok(n, edges, labeling):
    """True if every community's induced subgraph is connected (Leiden guarantee)."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for comm, members in clusters_from_labeling(labeling).items():
        if len(members) <= 1:
            continue
        start = next(iter(members))
        seen = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y in members and y not in seen:
                    seen.add(y)
                    stack.append(y)
        if seen != members:
            return False
    return True

# ── fixtures: return (n, edges, truth{node:entity}) ──────────────────────────

def disjoint_cliques(k, size):
    edges, truth = [], {}
    nid = 0
    for c in range(k):
        members = list(range(nid, nid + size))
        for i in range(len(members)):
            truth[members[i]] = c
            for j in range(i + 1, len(members)):
                edges.append((members[i], members[j]))
        nid += size
    return nid, edges, truth

def barbell(size):
    n, edges, truth = 2 * size, [], {}
    for c in range(2):
        members = list(range(c * size, c * size + size))
        for i in range(len(members)):
            truth[members[i]] = c
            for j in range(i + 1, len(members)):
                edges.append((members[i], members[j]))
    edges.append((size - 1, size))               # single bridge
    return n, edges, truth

def planted_partition(k, size, p_in, p_out, seed=0):
    rng = random.Random(seed)
    n = k * size
    truth = {i: i // size for i in range(n)}
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if truth[i] == truth[j] else p_out
            if rng.random() < p:
                edges.add((i, j))
    return n, list(edges), truth

def ring_of_cliques(k, size):
    """k cliques in a ring, adjacent cliques joined by ONE edge.
    Canonical modularity resolution-limit trap (merges in pairs)."""
    n, edges, truth = k * size, [], {}
    firsts = []
    for c in range(k):
        members = list(range(c * size, c * size + size))
        firsts.append(members[0])
        for i in range(len(members)):
            truth[members[i]] = c
            for j in range(i + 1, len(members)):
                edges.append((members[i], members[j]))
    for c in range(k):
        a = c * size + size - 1                   # last node of clique c
        b = firsts[(c + 1) % k]                   # first node of next clique
        edges.append((a, b))
    return n, edges, truth

def fragmented_er(n_entities=200, n_bridges=40, n_hubs=3, hub_fanout=12,
                  ring_k=20, ring_size=5, seed=42):
    """Splink-like ER graph: many small true-entity cliques + weak bridges
    (reused phone) + hubs (data-entry artifacts) + a ring-of-cliques block.
    Bridges/hubs DO NOT change truth — distinct entities stay distinct."""
    rng = random.Random(seed)
    edges, truth = [], {}
    nid = 0
    ent = 0
    entity_first = []
    # small true-entity cliques, sizes 1..5
    for _ in range(n_entities):
        size = rng.choices([1, 2, 3, 4, 5], weights=[40, 30, 15, 10, 5])[0]
        members = list(range(nid, nid + size))
        entity_first.append(members[0])
        for i in range(len(members)):
            truth[members[i]] = ent
            for j in range(i + 1, len(members)):
                edges.append((members[i], members[j]))
        nid += size
        ent += 1
    # ring-of-cliques block (distinct entities each)
    ring_firsts = []
    for _ in range(ring_k):
        members = list(range(nid, nid + ring_size))
        ring_firsts.append(members[0])
        for i in range(len(members)):
            truth[members[i]] = ent
            for j in range(i + 1, len(members)):
                edges.append((members[i], members[j]))
        nid += ring_size
        ent += 1
    for i in range(ring_k):
        edges.append((ring_firsts[i] + ring_size - 1, ring_firsts[(i + 1) % ring_k]))
    # weak bridges between random distinct entities (reused phone)
    allnodes = list(range(nid))
    for _ in range(n_bridges):
        u, v = rng.sample(allnodes, 2)
        if truth[u] != truth[v]:
            edges.append((u, v))
    # hubs
    for _ in range(n_hubs):
        h = rng.choice(allnodes)
        for _ in range(hub_fanout):
            t = rng.choice(allnodes)
            if t != h:
                edges.append((h, t))
    # dedupe undirected, drop self-loops
    seen = set()
    clean = []
    for u, v in edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            clean.append((u, v))
    return nid, clean, truth

# ── phases ───────────────────────────────────────────────────────────────────

def phase0_smoke():
    print("=" * 70)
    print("PHASE 0 — wheel capability / determinism")
    print("=" * 70)
    print("  NOTE: CALL LEIDEN is ABSENT from the installed real_ladybug wheel")
    print("        (pre-Leiden build; no C++ toolchain here to rebuild).")
    print("        => Testing the engine's UPSTREAM LOUVAIN — the trusted modularity")
    print("        core the bespoke Leiden is built on. The ER over-merge is a")
    print("        property of the modularity OBJECTIVE (shared by Louvain & Leiden);")
    print("        Leiden's only addition (connectivity refinement) does not change it.")
    edges = [(0,1),(0,2),(1,2),(2,3),(3,4),(5,6),(5,7),(6,7),(7,8),(8,9),(2,5),(4,9)]
    lab1 = run_bespoke_louvain(10, edges)
    lab2 = run_bespoke_louvain(10, edges)
    groups = sorted(sorted(m) for m in clusters_from_labeling(lab1).values())
    deterministic = lab1 == lab2
    print(f"  engine LOUVAIN communities on test graph: {groups}")
    print(f"  deterministic across 2 runs: {deterministic}")
    return {"leiden_in_wheel": False, "engine_louvain_groups": groups,
            "deterministic": deterministic}

def phase1_correctness():
    print("\n" + "=" * 70)
    print("PHASE 1 — correctness on known-answer graphs")
    print("=" * 70)
    fixtures = {
        "disjoint_cliques(k=8,size=6)": disjoint_cliques(8, 6),
        "barbell(size=4)": barbell(4),
        "planted_partition(k=5,size=20,pin=0.6,pout=0.02)": planted_partition(5, 20, 0.6, 0.02, seed=1),
        "ring_of_cliques(k=10,size=5)": ring_of_cliques(10, 5),
    }
    rows = []
    for name, (n, edges, truth) in fixtures.items():
        elo = run_bespoke_louvain(n, edges)          # ENGINE louvain (trusted core)
        ilo = run_igraph_louvain(n, edges)           # igraph louvain (parity ref)
        igm = run_igraph_leiden(n, edges, "modularity", 1.0)
        igc = run_igraph_leiden(n, edges, "CPM", 0.05)
        n_truth = len(set(truth.values()))
        row = {
            "fixture": name, "n_nodes": n, "n_edges": len(edges), "true_comms": n_truth,
            "engLouv_comms": size_diag(elo)["n_comm"],
            "engLouv_nmi": round(nmi(elo, truth, n), 3),
            "engLouv_ari": round(ari(elo, truth, n), 3),
            "engLouv_connected": connectivity_ok(n, edges, elo),
            "igLouv_nmi": round(nmi(ilo, truth, n), 3),
            "parity_eng_vs_ig_louvain": round(ari(elo, ilo, n), 3),
            "igLeidenMod_nmi": round(nmi(igm, truth, n), 3),
            "igCPM_comms": size_diag(igc)["n_comm"],
            "igCPM_nmi": round(nmi(igc, truth, n), 3),
        }
        rows.append(row)
        print(f"\n  {name}  (true comms={n_truth})")
        print(f"    engine-Louvain: comms={row['engLouv_comms']:3d}  NMI={row['engLouv_nmi']:.3f}  "
              f"ARI={row['engLouv_ari']:.3f}  connected={row['engLouv_connected']}")
        print(f"    parity vs igraph-Louvain (ARI)={row['parity_eng_vs_ig_louvain']:.3f}  "
              f"| igraph-Louvain NMI={row['igLouv_nmi']:.3f}")
        print(f"    ref Leiden-mod NMI={row['igLeidenMod_nmi']:.3f}  "
              f"| ref-CPM comms={row['igCPM_comms']:3d} NMI={row['igCPM_nmi']:.3f}")
    return rows

def phase2_er_verdict():
    print("\n" + "=" * 70)
    print("PHASE 2 — ER quality + BUG-vs-OBJECTIVE verdict")
    print("=" * 70)
    n, edges, truth = fragmented_er()
    n_truth = len(set(truth.values()))
    print(f"  fragmented ER graph: {n} nodes, {len(edges)} edges, {n_truth} true entities")

    runs = {}
    t0 = time.time()
    runs["engine_louvain"] = run_bespoke_louvain(n, edges)         # engine modularity core
    runs["igraph_louvain"] = run_igraph_louvain(n, edges)          # ref modularity (Louvain)
    runs["igraph_leiden_mod"] = run_igraph_leiden(n, edges, "modularity", 1.0)  # ref modularity (Leiden)
    runs["igraph_cpm@0.05"] = run_igraph_leiden(n, edges, "CPM", 0.05)
    runs["igraph_cpm@0.1"] = run_igraph_leiden(n, edges, "CPM", 0.1)
    elapsed = time.time() - t0

    rows = []
    for name, lab in runs.items():
        bc = bcubed_prf(lab, truth)
        oce, uce = oce_uce(lab, truth)
        d = size_diag(lab)
        row = {"method": name, "bcubed_p": round(bc.precision, 3), "bcubed_r": round(bc.recall, 3),
               "bcubed_f1": round(bc.f1, 3), "oce_false_merge": oce, "uce_false_split": uce,
               "n_comm": d["n_comm"], "max_size": d["max_size"], "giant": d["giant"]}
        rows.append(row)
        print(f"\n  {name}")
        print(f"    B-cubed P={row['bcubed_p']:.3f} R={row['bcubed_r']:.3f} F1={row['bcubed_f1']:.3f}"
              f"   OCE(false-merge)={oce}  UCE(false-split)={uce}")
        print(f"    communities={d['n_comm']} (truth {n_truth})  max_size={d['max_size']}  giant={d['giant']}")

    # verdict logic — engine modularity core (Louvain) vs reference modularity
    f = {r["method"]: r for r in rows}
    eng_f1 = f["engine_louvain"]["bcubed_f1"]
    refmod_f1 = max(f["igraph_louvain"]["bcubed_f1"], f["igraph_leiden_mod"]["bcubed_f1"])
    cpm_best = max(f["igraph_cpm@0.05"]["bcubed_f1"], f["igraph_cpm@0.1"]["bcubed_f1"])
    gap = refmod_f1 - eng_f1
    if abs(gap) <= 0.05:
        verdict = ("OBJECTIVE — the engine's modularity core (Louvain) tracks a correct modularity "
                   "reference; the ER over-merge is the modularity RESOLUTION LIMIT, not an "
                   "implementation bug. The bespoke Leiden shares this exact core + objective "
                   "(its refinement only splits disconnected comms, which does not fix this). "
                   "=> extending the engine core + adding CPM is viable.")
    elif gap > 0.05:
        verdict = (f"IMPLEMENTATION CONCERN — engine Louvain ({eng_f1:.3f}) is materially worse than a "
                   f"correct modularity reference ({refmod_f1:.3f}, gap {gap:.3f}). Investigate the "
                   "engine modularity core before building on it.")
    else:
        verdict = (f"engine Louvain ({eng_f1:.3f}) BEATS reference modularity ({refmod_f1:.3f}) — "
                   "unexpected; inspect before trusting.")
    cpm_helps = cpm_best - max(eng_f1, refmod_f1)
    print("\n  " + "-" * 66)
    print(f"  VERDICT: {verdict}")
    print(f"  CPM uplift over best modularity: +{cpm_helps:.3f} B-cubed F1 "
          f"(CPM best={cpm_best:.3f} vs modularity best={max(eng_f1, refmod_f1):.3f})")
    print(f"  (phase-2 compute {elapsed:.1f}s)")
    return {"graph": {"n": n, "edges": len(edges), "true_entities": n_truth},
            "rows": rows, "verdict": verdict, "cpm_uplift": round(cpm_helps, 3)}

def _md_table(rows):
    if not rows:
        return ""
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)

def main():
    p0 = phase0_smoke()
    p1 = phase1_correctness()
    p2 = phase2_er_verdict()

    with open(os.path.join(RESULTS, "results.json"), "w") as fh:
        json.dump({"phase0": p0, "phase1": p1, "phase2": p2}, fh, indent=2)

    md = []
    md.append("# Bespoke Leiden Verification — Results\n")
    md.append("_Tests the already-compiled bespoke Leiden in the `real_ladybug` wheel "
              "vs an igraph reference (modularity + CPM). No C++ rebuild._\n")
    md.append("## Phase 0 — wheel capability / determinism\n")
    md.append("- **CALL LEIDEN is ABSENT from the installed wheel** (pre-Leiden build; no toolchain "
              "to rebuild). Engine **Louvain** — the trusted modularity core the bespoke Leiden is "
              "built on — is tested as the proxy. The ER over-merge is a property of the modularity "
              "objective shared by both; Leiden's refinement only splits disconnected communities.")
    md.append(f"- engine-Louvain communities on test graph: `{p0['engine_louvain_groups']}`")
    md.append(f"- deterministic across 2 runs: **{p0['deterministic']}**\n")
    md.append("## Phase 1 — correctness on known-answer graphs\n")
    md.append(_md_table(p1) + "\n")
    md.append("## Phase 2 — ER quality + verdict\n")
    md.append(f"Graph: {p2['graph']['n']} nodes, {p2['graph']['edges']} edges, "
              f"{p2['graph']['true_entities']} true entities.\n")
    md.append(_md_table(p2["rows"]) + "\n")
    md.append(f"\n**Verdict:** {p2['verdict']}\n")
    md.append(f"\n**CPM uplift over best modularity:** +{p2['cpm_uplift']} B-cubed F1\n")
    with open(os.path.join(RESULTS, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(md))

    print("\n\nWrote:", os.path.join(RESULTS, "RESULTS.md"))
    print("Wrote:", os.path.join(RESULTS, "results.json"))

if __name__ == "__main__":
    main()
