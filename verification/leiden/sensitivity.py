#!/usr/bin/env python3
"""Localize the engine-Louvain excess over-merge (Phase 2 follow-up).

Vary the fragmented-ER structure to see WHERE engine Louvain diverges from a
correct modularity reference (igraph Louvain): hubs? bridges? scale?
"""
from __future__ import annotations
import json
import os

from leiden_verify import (fragmented_er, run_bespoke_louvain, run_igraph_louvain,
                           run_igraph_leiden, bcubed_prf, oce_uce, size_diag, RESULTS)


def score(lab, truth):
    bc = bcubed_prf(lab, truth)
    oce, uce = oce_uce(lab, truth)
    d = size_diag(lab)
    return dict(f1=round(bc.f1, 3), p=round(bc.precision, 3), r=round(bc.recall, 3),
                oce=oce, n_comm=d["n_comm"], max_size=d["max_size"], giant=d["giant"])


VARIANTS = {
    "full(default)":        dict(),
    "no_hubs":              dict(n_hubs=0),
    "no_bridges":           dict(n_bridges=0),
    "no_hubs_no_bridges":   dict(n_hubs=0, n_bridges=0),
    "hubs_only(no_bridge)": dict(n_bridges=0),
    "bigger(400 ent)":      dict(n_entities=400),
    "many_bridges(120)":    dict(n_bridges=120),
}

rows = []
print(f"{'variant':22s} {'true':>5} | {'engLouv F1':>10} {'maxsz':>6} {'OCE':>6} | "
      f"{'igLouv F1':>10} {'maxsz':>6} {'OCE':>6} | {'CPM.1 F1':>9} | {'gap':>6}")
print("-" * 110)
for name, kw in VARIANTS.items():
    n, edges, truth = fragmented_er(**kw)
    nt = len(set(truth.values()))
    el = score(run_bespoke_louvain(n, edges), truth)
    il = score(run_igraph_louvain(n, edges), truth)
    cp = score(run_igraph_leiden(n, edges, "CPM", 0.1), truth)
    gap = round(il["f1"] - el["f1"], 3)
    rows.append(dict(variant=name, n=n, true=nt, eng=el, ig=il, cpm=cp, gap=gap))
    print(f"{name:22s} {nt:5d} | {el['f1']:10.3f} {el['max_size']:6d} {el['oce']:6d} | "
          f"{il['f1']:10.3f} {il['max_size']:6d} {il['oce']:6d} | {cp['f1']:9.3f} | {gap:6.3f}")

with open(os.path.join(RESULTS, "sensitivity.json"), "w") as fh:
    json.dump(rows, fh, indent=2)
print("\nWrote", os.path.join(RESULTS, "sensitivity.json"))
print("\nINTERPRETATION:")
print("  gap>0 => engine Louvain over-merges MORE than correct modularity for that structure.")
print("  Compare 'no_hubs_no_bridges' (should be ~0 gap) vs others to localize the cause.")
