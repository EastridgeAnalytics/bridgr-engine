"""Score the IN-ENGINE GVE-Leiden output (pairwise-F1 vs ground-truth identity).

Reads membership_engine.csv (record_id,community) produced by CALL LEIDEN through
the engine, and ident.npy (ground truth). Same metric + PASS gate as the
standalone score.py, so the in-engine result is directly comparable to the
verified standalone F1=0.831 baseline.
"""

import sys

import numpy as np
import pandas as pd

OUT = "/root/work"
mem_path = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/membership_engine.csv"

ident = np.load(f"{OUT}/ident.npy")
n = ident.size

df = pd.read_csv(mem_path).sort_values("record_id")
assert df.shape[0] == n, f"row count {df.shape[0]} != n {n}"
assert np.array_equal(df["record_id"].to_numpy(), np.arange(n)), "record_id not 0..n-1 contiguous"
comm = df["community"].to_numpy()

pair_df = pd.DataFrame({"t": ident, "c": comm})
cell = pair_df.groupby(["t", "c"]).size().to_numpy()
true_sizes = pair_df.groupby("t").size().to_numpy()
pred_sizes = pair_df.groupby("c").size().to_numpy()


def sum_choose2(arr):
    a = arr.astype(np.int64)
    return int(np.sum(a * (a - 1) // 2))


TP = sum_choose2(cell)
pred = sum_choose2(pred_sizes)
true = sum_choose2(true_sizes)

precision = TP / pred if pred else 0.0
recall = TP / true if true else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

n_comm = int(pred_sizes.size)
max_comm = int(pred_sizes.max())

print("================ IN-ENGINE GVE-Leiden pairwise-F1 ================")
print("records (n)        : {}".format(n))
print("TP   pairs         : {}".format(TP))
print("pred pairs         : {}".format(pred))
print("true pairs         : {}".format(true))
print("precision          : {:.6f}".format(precision))
print("recall             : {:.6f}".format(recall))
print("F1                 : {:.6f}".format(f1))
print("# communities      : {}".format(n_comm))
print("max community size : {}".format(max_comm))
print("----- standalone GVE baseline: F1=0.8309 prec=1.0 max_comm=17 -----")
print("----- broken engine Leiden  : F1~=0.03  prec=low max_comm~=9023 ---")

pass_f1 = f1 >= 0.80
pass_prec = precision >= 0.95
pass_size = max_comm <= 50
over_merge = max_comm > 1000

verdict = "PASS" if (pass_f1 and pass_prec and pass_size and not over_merge) else "FAIL"
print("F1>=0.80           : {}".format(pass_f1))
print("precision>=0.95    : {}".format(pass_prec))
print("max_comm<=50       : {}".format(pass_size))
print("over_merge(>1000)  : {}".format(over_merge))
print("VERDICT            : {}".format(verdict))
sys.exit(0 if verdict == "PASS" else 1)
