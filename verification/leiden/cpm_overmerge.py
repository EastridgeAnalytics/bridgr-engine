#!/usr/bin/env python3
"""Prove the CPM objective fixes the modularity resolution-limit over-merge on a
fragmented entity-resolution graph, independent of the engine/wheel.

Reuses fragmented_er() (the Splink-like ER graph: small true-entity cliques +
weak bridges + hubs + a ring-of-cliques) and the B-cubed / pairwise metrics from
leiden_verify.py. Runs the GVE bridge directly via cpm_overmerge_driver for
modularity (resolution=1) and CPM (several gammas), then reports, per run:
  - n_comms        : number of predicted communities
  - worst_merge    : max distinct TRUE entities fused into one predicted community
                     (1 == no false merge; the bespoke engine Leiden hit 67)
  - max_size       : largest predicted community
  - bcubed F1      : element-wise ER cluster quality (the "r^3" / B^3 metric)
  - pairwise F1
  - OCE / UCE      : false-merge / false-split error counts

Run inside the container:
  python3 verification/leiden/cpm_overmerge.py
(the driver is compiled on first use if /tmp/cpm_overmerge is absent)
"""
from __future__ import annotations
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict, namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# Inlined from leiden_verify.py so this verifier has no igraph dependency (that
# module imports igraph at load time, only for NMI/ARI which we do not use here).
PRF = namedtuple("PRF", "precision recall f1")


def _harm(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def _comb2(k):
    return k * (k - 1) // 2


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


def fragmented_er(n_entities=200, n_bridges=40, n_hubs=3, hub_fanout=12,
                  ring_k=20, ring_size=5, seed=42):
    """Splink-like ER graph: many small true-entity cliques + weak bridges
    (reused phone) + hubs (data-entry artifacts) + a ring-of-cliques block.
    Bridges/hubs DO NOT change truth — distinct entities stay distinct.
    (Inlined verbatim from leiden_verify.py.)"""
    rng = random.Random(seed)
    edges, truth = [], {}
    nid = 0
    ent = 0
    for _ in range(n_entities):
        size = rng.choices([1, 2, 3, 4, 5], weights=[40, 30, 15, 10, 5])[0]
        members = list(range(nid, nid + size))
        for i in range(len(members)):
            truth[members[i]] = ent
            for j in range(i + 1, len(members)):
                edges.append((members[i], members[j]))
        nid += size
        ent += 1
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
    allnodes = list(range(nid))
    for _ in range(n_bridges):
        u, v = rng.sample(allnodes, 2)
        if truth[u] != truth[v]:
            edges.append((u, v))
    for _ in range(n_hubs):
        h = rng.choice(allnodes)
        for _ in range(hub_fanout):
            t = rng.choice(allnodes)
            if t != h:
                edges.append((h, t))
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
DRIVER = "/tmp/cpm_overmerge"
DATA = os.path.join(HERE, "_data")
os.makedirs(DATA, exist_ok=True)


def ensure_driver():
    if os.path.exists(DRIVER):
        return
    cmd = [
        "g++", "-std=c++17", "-O3", "-fopenmp", "-DTYPE=float",
        "-I" + os.path.join(ROOT, "extension/algo/src/include"),
        os.path.join(ROOT, "extension/algo/src/function/gve_leiden.cpp"),
        os.path.join(HERE, "cpm_overmerge_driver.cpp"),
        "-o", DRIVER,
    ]
    print("compiling driver:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def worst_merge(pred, truth):
    """Max distinct true entities fused into any one predicted community."""
    worst = 0
    for members in clusters_from_labeling(pred).values():
        worst = max(worst, len(set(truth[m] for m in members)))
    return worst


def run(n, edges_csv, mode, param):
    out = os.path.join(DATA, f"cpm_membership_{mode}_{param}.csv")
    subprocess.run([DRIVER, edges_csv, str(n), mode, str(param), out], check=True,
                   stderr=subprocess.DEVNULL)
    pred = {}
    with open(out) as fh:
        next(fh)
        for line in fh:
            node, comm = line.strip().split(",")
            pred[int(node)] = int(comm)
    return pred


def main():
    ensure_driver()
    n, edges, truth = fragmented_er()
    nt = len(set(truth.values()))
    edges_csv = os.path.join(DATA, "cpm_edges.csv")
    with open(edges_csv, "w") as fh:
        for u, v in edges:
            fh.write(f"{u},{v}\n")

    print(f"fragmented ER graph: nodes={n} edges={len(edges)} true_entities={nt}")
    print(f"{'method':<22}{'n_comms':>8}{'worst_merge':>13}{'max_size':>10}"
          f"{'b3_P':>7}{'b3_R':>7}{'b3_F1':>7}{'pair_F1':>9}{'OCE':>6}{'UCE':>6}")
    print("-" * 103)

    configs = [
        ("modularity res=1.0", "mod", 1.0),
        ("cpm gamma=0.05", "cpm", 0.05),
        ("cpm gamma=0.1", "cpm", 0.1),
        ("cpm gamma=0.25", "cpm", 0.25),
        ("cpm gamma=0.5", "cpm", 0.5),
        ("cpm gamma=1.0", "cpm", 1.0),
    ]
    for label, mode, param in configs:
        pred = run(n, edges_csv, mode, param)
        b3 = bcubed_prf(pred, truth)
        pw = pairwise_prf(pred, truth)
        oce, uce = oce_uce(pred, truth)
        sizes = [len(c) for c in clusters_from_labeling(pred).values()]
        print(f"{label:<22}{len(set(pred.values())):>8}{worst_merge(pred, truth):>13}"
              f"{max(sizes):>10}{b3.precision:>7.3f}{b3.recall:>7.3f}{b3.f1:>7.3f}"
              f"{pw.f1:>9.3f}{oce:>6}{uce:>6}")


if __name__ == "__main__":
    main()
