# Bespoke Leiden Verification — Results

_Tests the already-compiled bespoke Leiden in the `real_ladybug` wheel vs an igraph reference (modularity + CPM). No C++ rebuild._

## Phase 0 — wheel capability / determinism

- **CALL LEIDEN is ABSENT from the installed wheel** (pre-Leiden build; no toolchain to rebuild). Engine **Louvain** — the trusted modularity core the bespoke Leiden is built on — is tested as the proxy. The ER over-merge is a property of the modularity objective shared by both; Leiden's refinement only splits disconnected communities.
- engine-Louvain communities on test graph: `[[0, 1, 2], [3, 4, 8, 9], [5, 6, 7]]`
- deterministic across 2 runs: **True**

## Phase 1 — correctness on known-answer graphs

| fixture | n_nodes | n_edges | true_comms | engLouv_comms | engLouv_nmi | engLouv_ari | engLouv_connected | igLouv_nmi | parity_eng_vs_ig_louvain | igLeidenMod_nmi | igCPM_comms | igCPM_nmi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| disjoint_cliques(k=8,size=6) | 48 | 120 | 8 | 8 | 1.0 | 1.0 | True | 1.0 | 1.0 | 1.0 | 8 | 1.0 |
| barbell(size=4) | 8 | 13 | 2 | 2 | 1.0 | 1.0 | True | 1.0 | 1.0 | 1.0 | 1 | 0.0 |
| planted_partition(k=5,size=20,pin=0.6,pout=0.02) | 100 | 678 | 5 | 5 | 1.0 | 1.0 | True | 1.0 | 1.0 | 1.0 | 5 | 1.0 |
| ring_of_cliques(k=10,size=5) | 50 | 110 | 10 | 10 | 1.0 | 1.0 | True | 1.0 | 1.0 | 1.0 | 10 | 1.0 |

## Phase 2 — ER quality + verdict

Graph: 511 nodes, 655 edges, 220 true entities.

| method | bcubed_p | bcubed_r | bcubed_f1 | oce_false_merge | uce_false_split | n_comm | max_size | giant |
|---|---|---|---|---|---|---|---|---|
| engine_louvain | 0.533 | 1.0 | 0.695 | 5610 | 0 | 91 | 67 | True |
| igraph_louvain | 0.661 | 1.0 | 0.796 | 1103 | 0 | 157 | 35 | False |
| igraph_leiden_mod | 0.653 | 1.0 | 0.79 | 1077 | 0 | 156 | 33 | False |
| igraph_cpm@0.05 | 0.842 | 0.996 | 0.912 | 266 | 2 | 181 | 11 | False |
| igraph_cpm@0.1 | 0.912 | 1.0 | 0.954 | 137 | 0 | 193 | 7 | False |


**Verdict:** IMPLEMENTATION CONCERN — engine Louvain (0.695) is materially worse than a correct modularity reference (0.796, gap 0.101). Investigate the engine modularity core before building on it.


**CPM uplift over best modularity:** +0.158 B-cubed F1
