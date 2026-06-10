# REL-table COPY: ~30-60× slowdown for non-compressible DOUBLE property values

**Status:** verified repro, pre-ticket write-up (2026-06-10). Found while building the H-E-B 100M
identity-resolution scale demo (ingest was 88s at 2M records instead of ~2s).

## Symptom

`COPY <RelTable> FROM '<parquet>'` where the rel table has a DOUBLE property is **~30-60× slower**
for some value distributions than others, at identical row counts, schema, topology, and file format.

## Receipts (serialized, clean container — bridgr-notebook source-build image, real_ladybug 0.16.1, 3M edges / 2M nodes, 12 CPU)

| weight column values | distinct | edge COPY | path |
|---|---|---|---|
| constant 0.9 | 1 | 1.6s | FAST |
| two distinct (0.96 / 0.60) | 2 | 1.6s | FAST |
| `np.round(uniform(0.5,1.0), 2)` | 51 | 1.6s | FAST |
| INT64 column (45 distinct) | 45 | 1.5s | FAST |
| `linspace(0.5,1.0,16)` sampled | **16** | **64.9s** | SLOW |
| `linspace`, 256 / 4096 / 65536 / ~1M distinct | … | 56-60s | SLOW |
| `np.round(uniform, 4)` | 5001 | 96.4s | SLOW |
| `uniform(0.5,1.0)` raw | ~3M | 47.2s | SLOW |

Node-table COPY is unaffected (~0.5s for 2M rows in all trials).

## What it is NOT

- **Not cardinality:** 16 distinct linspace values are slow; 51 distinct 2-decimal values are fast.
- **Not edge topology/order:** clique-local edges, shuffled edge order, and permuted node ids made
  no difference in earlier trials.
- **Not the parquet file:** same writer settings throughout; only the values differ.
- An earlier measurement of "even 2 distinct values slow / ~50×" was taken with concurrent load in
  the container and is **retracted** — clean serialized runs show 2-distinct is fast.

## Hypothesis

The DOUBLE column storage compression (ALP-style floating-point scheme) has a fast path for values
that round-trip as short decimals (≤2dp empirically) and falls into a pathological exception/fallback
path otherwise. The cliff between `round(…, 2)` (fast) and `round(…, 4)` (slow) suggests either a
precision bound or an exceptions-per-vector budget.

**Discriminating follow-up trial:** 4-decimal values restricted to 16 distinct — separates decimal
precision from distinct-count. Then profile the C++ COPY path on a slow case.

## Impact

Any weighted-graph import at scale (ER/Splink match probabilities, similarity scores, amounts) —
real-world weights are arbitrary doubles, i.e., the slow path is the *default* for customer data.
At 100M-edge scale this is the difference between ~2 min and ~1+ hour of ingest. Sibling
correctness ticket: WCC nondeterminism. Related minor limitation found in the same session:
`CALL LEIDEN` rejects multi-rel-table projections ("Leiden only supports operations on one edge
table"), which blocks the two-table workaround for this bug.

## Related observations (same sessions, candidate sibling tickets)

1. **Repeated COPY-appends are ~quadratic:** each `COPY <table> FROM <file>` costs ~linear in the
   table's **existing** row count (~16.6µs/existing-row measured at 2M and 10M scales, node and rel
   tables alike — looks like a full structure rebuild per COPY). N chunked appends therefore cost
   O(N²/chunk). Mitigation that works: stage all chunks, then ONE list-COPY
   (`COPY t FROM ['c0.parquet', 'c1.parquet', ...]`) — measured 141K-1.2M rows/s vs ~3K rows/s for
   appends onto a large table. A genuine *incremental* append (new data arriving monthly) cannot
   avoid this cost today.
2. `CALL LEIDEN` rejects multi-rel-table projections ("Leiden only supports operations on one edge
   table"), which blocks weighted multi-table workflows.

## Workarounds (validated)

1. Quantize DOUBLE weights to 2 decimals before COPY (fast path; F1 impact nil for ER use).
2. Or store weights as INT64 (e.g., probability × 100) — `weight_property` casts to double.

## Repro

`python repro.py` (inside an image with the engine binding; run trials serialized, nothing else on
the box). Fuller trial set: `bridgr-mono sdk/demos/identity-resolution-leiden/_dev/validate_copy_{fix,cardinality,decimals}.py`.
