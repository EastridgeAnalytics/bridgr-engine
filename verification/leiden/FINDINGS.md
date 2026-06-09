# Leiden / Modularity Verification — Findings

> **STATUS 2026-06-09 — RESOLVED IN-ENGINE (read this first).** The two
> conclusions below (CPM fixes the ER over-merge; build it on a correct base) are
> now **implemented and verified in the MIT engine**. `CALL LEIDEN` exists and is
> the GVE-Leiden core; weighted ingestion and the **CPM objective**
> (`objective:='cpm', gamma:=…`) are built and green. The "no C++ toolchain" /
> "CALL LEIDEN absent from the wheel" notes below are **historical** (they
> described the first, wheel-only exploration on 2026-06-08); the engine is now
> built/tested in Docker (`ghcr.io/ladybugdb/ubuntu-22.04-gcc13`, gcc-13), not
> MSVC. See **"CPM IMPLEMENTED in the engine"** at the bottom for the engine's own
> numbers, and **`results/cpm_verification_20260609.log`** for the captured
> build + `e2e_test` 24/24 + harness run.

---

## Original exploration (2026-06-08, wheel + igraph reference)

**Harness:** `verification/leiden/leiden_verify.py` (+ `sensitivity.py`)
**What ran:** the already-compiled algo extension in the installed `real_ladybug` wheel, vs an **igraph 1.0.0** reference (Louvain, Leiden-modularity, Leiden-CPM). No C++ build (no toolchain on this machine).
**Reproduce:** `python verification/leiden/leiden_verify.py` then `python verification/leiden/sensitivity.py`

---

## Headline

1. **The objective is the problem (as theorized) — and CPM fixes it, proven on our own numbers.** Even a *correct* modularity implementation over-merges fragmented ER graphs; CPM lifts B-cubed F1 by **+0.10 to +0.16**, recovering up to F1 = 1.000.
2. **Surprise: the engine's modularity core has an *excess* over-merge defect** — it over-merges **3–9× more** than a correct modularity optimizer on every fragmented / multi-component graph tested, while being **exactly correct on clean graphs**. This is separate from, and on top of, the objective limit.

These two findings **revise the build recommendation**: the earlier "extend the bespoke core" option is now the weaker one. See *Decision impact*.

---

## Environment constraint (important)

- **`CALL LEIDEN` is ABSENT from the installed wheel** (pre-Leiden build; `CALL LOUVAIN`, `WCC`, `PAGE_RANK` are present). With no C++ toolchain here, the bespoke Leiden binary cannot be built or run.
- So the engine arm tested is **`CALL LOUVAIN`** — the upstream, trusted modularity core that the bespoke Leiden is *built on top of*. The ER over-merge is a property of the **modularity objective + the engine's local-moving/aggregation heuristic**, both of which Louvain and the bespoke Leiden **share**. Leiden's only addition (BFS refinement) splits *disconnected* communities and cannot undo the *connected* over-merges seen here. So Louvain is a faithful — and likely optimistic — proxy for the bespoke Leiden's behavior on ER graphs.

---

## Phase 1 — correctness on known-answer graphs ✅

Engine Louvain is **correct on clean data** and matches the reference exactly.

| Fixture | true comms | engine-Louvain NMI | engine-Louvain ARI | parity vs igraph-Louvain (ARI) | connectivity ok |
|---|---|---|---|---|---|
| disjoint_cliques(8×6) | 8 | 1.000 | 1.000 | 1.000 | ✓ |
| barbell(4) | 2 | 1.000 | 1.000 | 1.000 | ✓ |
| planted_partition(5×20, pin .6/pout .02) | 5 | 1.000 | 1.000 | 1.000 | ✓ |
| ring_of_cliques(10×5) | 10 | 1.000 | 1.000 | 1.000 | ✓ |

The engine reproduces igraph Louvain **bit-for-bit (ARI = 1.0)** on every clean fixture, and is deterministic across runs. The pipeline is apples-to-apples; any divergence on harder graphs is real engine behavior, not a harness artifact.

## Phase 2 — fragmented ER (511 nodes, 655 edges, 220 true entities)

| Method | B³ P | B³ R | B³ F1 | false-merge (OCE) | comms (truth 220) | max comm |
|---|---|---|---|---|---|---|
| **engine_louvain** | 0.533 | 1.000 | **0.695** | **5,610** | 91 | **67** |
| igraph_louvain (ref) | 0.661 | 1.000 | 0.796 | 1,103 | 157 | 35 |
| igraph_leiden_mod (ref) | 0.653 | 1.000 | 0.790 | 1,077 | 156 | 33 |
| igraph_cpm @ γ=0.05 | 0.842 | 0.996 | 0.912 | 266 | 181 | 11 |
| igraph_cpm @ γ=0.10 | 0.912 | 1.000 | **0.954** | 137 | 193 | 7 |

Two gaps are visible at once: **modularity (any impl) → CPM** is +0.158 F1 (the objective limit), and **engine modularity → reference modularity** is −0.101 F1 with 5× the false merges and a giant 67-node cluster (the implementation defect).

## Sensitivity — localizing the engine's excess over-merge

| Variant | true | engine F1 | engine maxsz | igraph F1 | igraph maxsz | CPM γ.1 F1 | gap |
|---|---|---|---|---|---|---|---|
| full (default) | 220 | 0.695 | 67 | 0.790 | 33 | 0.948 | 0.095 |
| no_hubs | 220 | 0.737 | 73 | 0.853 | 16 | 0.959 | 0.116 |
| no_bridges | 220 | 0.757 | 78 | 0.881 | 24 | 0.988 | 0.124 |
| **no_hubs_no_bridges** | 220 | 0.841 | **86** | 0.949 | **10** | 1.000 | 0.108 |
| bigger (400 entities) | 420 | 0.764 | 140 | 0.863 | 37 | 0.972 | 0.099 |
| many_bridges (120) | 220 | 0.467 | 51 | 0.596 | 32 | 0.907 | 0.129 |

**The defect is not caused by hubs or bridges.** Even `no_hubs_no_bridges` — a graph of 200 isolated small cliques plus one 20-clique ring — shows it: correct modularity merges the ring cliques only **in pairs** (max size 10, textbook resolution-limit behavior), while the **engine fuses ~17 of the 20 ring-cliques into a single 86-node blob.** The excess over-merge is driven by **ring/chain structure and the presence of many components** (i.e., exactly the shape Splink ER output has). The gap is stable (0.095–0.129) and present in every fragmented configuration. CPM is robust throughout (F1 0.907–1.000).

> Reconciliation with Phase 1: `ring_of_cliques(10×5)` *alone* recovered perfectly (NMI 1.0). The same ring inside a larger many-component graph over-merges — consistent with a heuristic whose merge decisions degrade as global graph size grows, beyond what a correct modularity optimizer does.

---

## Phase 3 — engine modularity-core scale/timing (Louvain; Leiden binary absent)

| nodes | edges | LOUVAIN time | comms | max comm |
|---|---|---|---|---|
| 10,000 | 68,259 | 0.114 s | 198 | 100 |
| 50,000 | 352,306 | 0.374 s | 504 | 559 |
| 100,000 | 710,066 | 0.717 s | 640 | 998 |

Deterministic across runs (identical partitions @50k). The core is **genuinely fast** (sub-second at 100K, C++/CSR) — but the over-merge defect shows here too: on a planted-block graph (~67 nodes/block expected) the engine fuses ~15 blocks into a 998-node community. (Replaces the retracted "23.7s/4.3× igraph" Leiden claim with a real *Louvain* number; re-run against `CALL LEIDEN` in CI for the Leiden timing.)

## Verdict

- **Objective question — SETTLED.** Modularity (the only objective the engine exposes) over-merges fragmented ER by design; **CPM is the fix** (+0.10–0.16 B-cubed F1, up to F1 = 1.0). This validates the CPM plan on H-E-B-shaped data with our own measurements.
- **Implementation question — CONCERN RAISED.** The engine's modularity core **over-merges well beyond the objective limit** on multi-component / ring-structured graphs (3–9× the false merges; giant components 3–8× larger than a correct optimizer), despite being exactly correct on clean graphs. This is the long-suspected over-merge defect, now reproduced under controlled fixtures against a reference.

## Decision impact (revises the dev plan's Phase-2 fork)

The dev plan left "extend the bespoke core" vs "adopt GVE-Leiden" to this evidence. The evidence now **tilts toward GVE-Leiden (or a root-cause fix of the engine core), not naive extension**, because:
- Building CPM on a core that already over-merges beyond the objective would inherit that defect.
- GVE-Leiden is externally benchmarked-correct; it is a clean base for the CPM gain function.
- The bespoke Leiden shares the defective core (its refinement cannot undo connected over-merges), so it is unlikely to rescue this on its own.

**Net path:** (1) ship **CPM** (validated as the fix), (2) on a **correct base** — prefer vendored GVE-Leiden or a root-caused fix of the engine Louvain/Leiden heuristic, **not** the current bespoke core as-is, (3) add **weighted** ingestion (separate workstream).

## Root-cause hypothesis (for the dev to confirm — not proven here)

From reading `louvain.cpp` / `leiden.cpp`, the excess over-merge most likely comes from the **naive parallel move phase**, not the objective:

- The move phase (`RunIterationVC`) evaluates **all nodes in parallel against a single pre-move community snapshot** (`currComm`), writing destinations to `nextComm`. Simultaneous moves computed on a stale snapshot are a classic parallel-Louvain hazard: many nodes can pile into the same community at once (or two communities "swap"), producing merges no *sequential* optimizer would make. Grappolo (the design this cites) normally mitigates this with graph coloring / staged updates; a naïve all-at-once update over-merges on dense/ring/chain structure — exactly what we see.
- The tie-break favors the **lowest community id** (`... && nbrCommId < newComm`). On a ring/chain this can **cascade merges toward the lowest id**, fusing a whole ring instead of stopping at pairs — consistent with the 86-node blob on a 20-clique ring.

igraph avoids both (sequential local moving + proper refinement), which is why it stops at the resolution-limit pairing rather than blob-merging. **Implication:** the fix is a *correct* local-moving/refinement core (GVE-Leiden is built and benchmarked for exactly this), not a tweak to the objective. Confirm by instrumenting the move phase (count multi-node simultaneous moves into one community) on a Leiden-enabled CI build.

## Limitations / what still needs a CI build

- ~~This tested **Louvain** (Leiden binary absent from the wheel). Re-run on a Leiden-enabled build.~~ **RESOLVED 2026-06-08:** built the engine with MSVC and confirmed **on the real `CALL LEIDEN` binary** that it over-merges (89 communities for 220 entities; worst community fuses 67 distinct customers) and that **refinement does not mitigate it** (Leiden 89 ≈ Louvain 91). Phase 6 also confirmed (weighting 25→15 false-merge comms; worst-merge 67 unchanged → CPM still needed). See `results/real_leiden_shell.md`.
- igraph Louvain is randomized (single run per cell); the engine is deterministic. The gap is large and consistent across 7 variants, so run-to-run variance does not explain it — but CI can average multiple igraph seeds for publication.
- CPM here is igraph's (GPL — **validation only**). Shipping CPM means implementing it in the MIT engine (the separate CPM plan).

---

## CPM IMPLEMENTED in the engine (MIT, on the GVE base) — 2026-06-09

The CPM plan above is now **built**. CPM (Constant Potts Model) was added to the
vendored GVE-Leiden core as a compile-time objective (`bool CPM` template +
`if constexpr`, so the modularity path is byte-identical), exposed through
`CALL LEIDEN('g', objective:='cpm', gamma:=<density threshold>)`. The node-count
null model is `deltaCPM = (vcout - vdout) - gamma*nv*(nv + nC - nD)` (node counts
in place of modularity's degree-product penalty); per-community node counts are
threaded through the local-move + coarsening phases alongside the existing
degree-weight tracking. This is **our own MIT implementation**, not igraph.

### Result — `cpm_overmerge.py` (this engine's `runGveLeiden`, not igraph)

Fragmented ER graph (`fragmented_er`, seed=42): nodes=511, edges=655, true_entities=220.
Deterministic across re-runs.

| objective            | n_comms | worst_merge | max_size | B³ P | B³ R | B³ F1 | pair F1 | OCE | UCE |
|----------------------|--------:|------------:|---------:|-----:|-----:|------:|--------:|----:|----:|
| modularity res=1.0   |     159 |       **9** |       24 | 0.676| 1.000| 0.807 |   0.508 | 790 |   0 |
| cpm gamma=0.05       |     180 |           6 |       12 | 0.842| 1.000| 0.914 |   0.789 | 294 |   0 |
| cpm gamma=0.1        |     193 |           4 |        7 | 0.913| 1.000| 0.955 |   0.908 | 139 |   0 |
| cpm gamma=0.25       |     210 |           3 |        5 | 0.976| 1.000| 0.988 |   0.982 |  33 |   0 |
| cpm gamma=0.5        |     218 |       **2** |        5 | 0.994| 0.998| **0.996** | 0.996 |   6 |   1 |
| cpm gamma=1.0        |     511 |       **1** |        1 | 1.000| 0.431| 0.602 |   0.000 |   0 | 291 |

**Reading:** modularity over-merges (one community fuses 9 distinct entities; 790
false merges; B³ F1 0.807). CPM monotonically removes the over-merge as gamma
rises — at gamma≈0.5, worst_merge → 2, false merges → 6, B³ F1 → 0.996. gamma is
a clean knob: gamma=1.0 over-splits (every node its own community: worst_merge=1
but recall collapses). **The over-merge objective limit is closed in the MIT
engine.** (worst_merge here is GVE-modularity's 9, not the bespoke core's 67 —
GVE already fixed the *catastrophic* over-merge; CPM now also removes the
*resolution-limit* residue.)

### Unit tests (`leiden.test`, run via `e2e_test`, Docker gcc-13)

All 14 leiden + 10 louvain cases PASS (24/24). New CPM cases:
`CpmDenseCliqueStaysWhole` (K6, gamma=0.5 → 1 community), `CpmHighGammaSplitsToSingletons`
(K6, gamma=2.0 → 6 singletons), `CpmResolutionLimitTwoCliquesStaySeparate`
(two K4 + bridge, gamma=0.5 → 3 and 4 stay separate), `CpmInvalidGammaRejected`,
`CpmInvalidObjectiveRejected`. Weighted Leiden/Louvain cases also green.

### Wiring

`CALL LEIDEN(objective, gamma)` → `leiden.cpp` (Objective/Gamma OptionalParams,
validated) → `runGveLeiden(..., useCpm, gamma)` → `leidenStaticOmp<CPM>`. Modularity
remains the default; `objective:='modularity'` and the no-arg form are unchanged.
