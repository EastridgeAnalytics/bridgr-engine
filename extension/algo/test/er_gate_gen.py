"""Generate the fragmented ER match graph as engine-loadable CSVs + ground truth.

Reproduces EXACTLY the standalone GVE-Leiden over-merge gate (gen_graph.py):
  N_ID=50000, numpy default_rng(1), sizes=min(1+geometric(0.25),25),
  ident=repeat(arange(N_ID),sizes), chain edges between consecutive same-identity
  records, prob=uniform(0.75,0.99) with ~8% weakened to uniform(0.60,0.699),
  keep prob>=0.7. Yields ~249,740 records / ~183,612 edges / 50,000 identities.

Emits (all 0-indexed record ids, matching engine node primary keys):
  nodes.csv  : header 'id', one row per record 0..n-1 (so isolated records exist)
  edges.csv  : header 'src,dst', one row per kept undirected chain edge
  ident.npy  : ground-truth identity per record (for score.py)
"""

import numpy as np

N_ID = 50000
rng = np.random.default_rng(1)

sizes = np.minimum(1 + rng.geometric(0.25, N_ID), 25)
n = int(sizes.sum())
ident = np.repeat(np.arange(N_ID), sizes)

# Candidate chain edges: consecutive records sharing the same identity.
s = np.nonzero(ident[1:] == ident[:-1])[0]

prob = rng.uniform(0.75, 0.99, s.size)
weak = rng.random(s.size) < 0.08
prob[weak] = rng.uniform(0.60, 0.699, weak.sum())
keep = prob >= 0.7

u_kept = s[keep]        # record indices (0-based)
v_kept = s[keep] + 1    # next record in chain
n_edges = int(keep.sum())

OUT = "/root/work"

# nodes.csv: every record id 0..n-1 (isolated records remain valid nodes).
with open(f"{OUT}/nodes.csv", "w") as f:
    f.write("id\n")
    f.write("\n".join(str(i) for i in range(n)))
    f.write("\n")

# edges.csv: kept chain edges, 0-indexed src,dst.
with open(f"{OUT}/edges.csv", "w") as f:
    f.write("src,dst\n")
    lines = [f"{a},{b}" for a, b in zip(u_kept.tolist(), v_kept.tolist())]
    f.write("\n".join(lines))
    f.write("\n")

np.save(f"{OUT}/ident.npy", ident)

print(f"N_ID={N_ID}")
print(f"n (records)={n}")
print(f"candidate chain edges={s.size}")
print(f"weak edges={int(weak.sum())}")
print(f"kept edges={n_edges}")
print(f"wrote {OUT}/nodes.csv {OUT}/edges.csv {OUT}/ident.npy")
