# Real CALL LEIDEN — over-merge (#1 action) + weighting (Phase 6), observed in the shell

Run via the from-source `lbug_shell.exe` (the installed wheel segfaults on a from-source
extension; `tools/python_api` is vendored out so the Python harness can't be built). Over-merge
is measured in Cypher: each node carries its true entity id (`te`); per community we count how
many **distinct** true entities were fused. Graph: fragmented ER, **511 nodes, 655 edges, 220 true entities**.
Script + data: `verification/leiden/_data/overmerge.cypher`.

| measurement | communities | impure comms (≥2 entities = false merges) | worst_merge (max entities fused into 1) |
|---|---|---|---|
| **ground truth** | 220 | 0 | 1 |
| **CALL LEIDEN** (unweighted) | 89 | 25 | **67** |
| CALL LOUVAIN (unweighted) | 91 | 26 | 67 |
| CALL LEIDEN unweighted (weighted-graph copy) | 89 | 25 | 67 |
| **CALL LEIDEN weighted** (`weight_property:='w'`) | **138** | **15** | 67 |

*(`max_size` column returned 0 — a query artifact from reusing `count(*)` in the aggregate; `worst_merge` is the working over-merge metric.)*

## #1 action — does the REAL bespoke Leiden over-merge, and does refinement help? — CONFIRMED

- **Real `CALL LEIDEN` over-merges hard:** 89 communities for 220 true entities; **25 communities are false merges**; the worst single community **fuses 67 distinct customers** into one false identity.
- **Leiden's refinement does NOT mitigate it:** Leiden (89 comms / 25 impure / 67 worst) ≈ Louvain (91 / 26 / 67). The BFS refinement only splits *disconnected* communities; the over-merge here is *connected* (weak bridges / ring), so refinement can't touch it. The bespoke Leiden inherits the shared modularity-core over-merge — exactly as the Louvain-proxy analysis predicted, now confirmed on the real binary.

## Phase 6 — does weighting help ER? — YES (partially); CPM still needed

Weighted graph: strong intra-entity edges (w=5), weak spurious bridges (w=0.2).
- **Weighting helps:** false-merge communities drop **25 → 15** (−40%); communities rise **89 → 138** (closer to the 220 truth — fewer entities lost to fusion).
- **But weighting does NOT fix the worst case:** `worst_merge` stays **67** — the giant resolution-limit blob survives, because that fusion is driven by the *objective* (modularity's resolution limit), not by bridge weight. Removing it requires **CPM** (resolution-limit-free), confirming the layered conclusion: *weights reduce spurious-bridge merges; CPM is required for the objective-driven over-merge.*

## Bottom line
Every headline now holds on the **real `CALL LEIDEN` binary** (not the Louvain proxy): the engine over-merges fragmented ER (objective limit + the engine's heuristic), refinement doesn't save it, weighting helps but isn't sufficient, and CPM is the missing piece. The path stands: **CPM on a correct base + weighted ingestion** (weighted ingestion now BUILT + GREEN).
