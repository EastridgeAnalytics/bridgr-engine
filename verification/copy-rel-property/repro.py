"""Minimal repro: REL-table COPY is ~30-60x slower for DOUBLE property values that don't
compress (value-DEPENDENT, not cardinality-dependent).

Run inside any image with the engine Python binding (e.g. the bridgr-notebook source-build):
    python repro.py

Expected shape of results (3M edges / 2M nodes, 12 CPU, 2026-06-10):
    const_0.9          ~1.6s   FAST
    two_distinct       ~1.6s   FAST   (0.96 / 0.60)
    round2dp_51vals    ~1.6s   FAST   (np.round(uniform, 2))
    linspace_16vals   ~65s     SLOW   (16 DISTINCT values! not high cardinality)
    round4dp_5001vals ~96s     SLOW
    random_3M         ~47s     SLOW

Hypothesis: the FP-compression (ALP-style) fast path accepts short-decimal-friendly doubles;
everything else hits a pathological exception/fallback path. Suggested discriminating trial:
4-decimal values restricted to 16 distinct (separates precision from distinct-count).
"""
import sys, time, tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from bridgr.database import Database  # MIT SDK wrapper
    def connect(path):
        return Database(path)
except ImportError:  # raw engine binding
    import real_ladybug as lb
    class _Db:
        def __init__(self, path):
            self._db = lb.Database(path)
            self._conn = lb.Connection(self._db)
        def execute(self, q):
            return self._conn.execute(q)
        def close(self):
            self._conn.close(); self._db.close()
    def connect(path):
        return _Db(path)

NN, NE = 2_000_000, 3_000_000
rng = np.random.default_rng(7)
SRC = rng.integers(0, NN, NE, dtype=np.int64)
DST = rng.integers(0, NN, NE, dtype=np.int64)

CASES = {
    "const_0.9":         lambda r: np.full(NE, 0.9),
    "two_distinct":      lambda r: np.where(r.random(NE) < 0.9, 0.96, 0.60),
    "round2dp_51vals":   lambda r: np.round(r.uniform(0.5, 1.0, NE), 2),
    "linspace_16vals":   lambda r: np.linspace(0.5, 1.0, 16)[r.integers(0, 16, NE)],
    "round4dp_5001vals": lambda r: np.round(r.uniform(0.5, 1.0, NE), 4),
    "random_3M":         lambda r: r.uniform(0.5, 1.0, NE),
}


def trial(name):
    w = CASES[name](np.random.default_rng(7))
    sc = tempfile.mkdtemp(prefix=f"copyrepro_{name}_")
    pq.write_table(pa.table({"id": np.arange(NN, dtype=np.int64)}), f"{sc}/n.parquet")
    pq.write_table(pa.table({"src": SRC, "dst": DST, "w": w}), f"{sc}/e.parquet")
    d = connect(f"{sc}/b.lbug")
    d.execute("CREATE NODE TABLE Rec(id INT64 PRIMARY KEY)")
    d.execute("CREATE REL TABLE Match(FROM Rec TO Rec, w DOUBLE)")
    d.execute(f"COPY Rec FROM '{sc}/n.parquet'")
    t0 = time.perf_counter()
    d.execute(f"COPY Match FROM '{sc}/e.parquet'")
    te = time.perf_counter() - t0
    d.close()
    print(f"{name:<20} distinct={np.unique(w).size:>8,}  edge_copy={te:6.1f}s", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    print(f"{NE:,} edges / {NN:,} nodes — run trials one at a time, nothing else on the box", flush=True)
    for n in names:
        trial(n)
