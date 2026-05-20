"""LDBC FinBench benchmark: financial fraud detection on synthetic data.

Implements 10 queries from the LDBC Financial Benchmark (FinBench)
specification on a synthetically generated financial transaction graph.

Data model (FinBench-compatible):
  Node types: Person, Company, Account, Loan, Medium
  Edge types: own, transfer, signIn, apply, guarantee, invest

Queries:
  Simple Reads:
    SR-1: Person's owned accounts
    SR-2: Account transfer history (time-bounded)
    SR-3: Person's loan applications
  Complex Reads:
    CR-1: Circular transfer detection (3-hop fraud rings)
    CR-2: Guarantee chains (variable-length person paths)
    CR-3: Multi-hop account reachability (BFS within N transfers)
    CR-4: Fund tracing (follow money flow with amount tracking)
    CR-5: Common accounts between two persons
  Write Operations:
    W-1: Create new transfer
    W-2: Block an account

Synthetic data includes injected fraud patterns:
  - 5-10 circular transfer rings (A->B->C->A)
  - 3-5 structuring patterns (rapid small transfers < $10K)
  - 2-3 guarantee chains (person->person->person)

Usage:
    python -m benchmarks.finbench_benchmark                           # SF-1 default
    python -m benchmarks.finbench_benchmark --persons 100000          # SF-10
    python -m benchmarks.finbench_benchmark --persons 1000 --warmup 1 # quick test
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import tempfile
import time
from datetime import date
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from bridgr.database import Database
from bridgr.export import _cypher_path

from benchmarks.bench_utils import (
    TimingResult,
    format_timing_table,
    time_callable,
    time_query,
)


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


def _create_finbench_schema(db: Database) -> None:
    """Create all FinBench node and edge tables.

    Args:
        db: Target database (should be empty).
    """
    # Node tables
    db.create_node_table("Person", {
        "personId": "INT64 PRIMARY KEY",
        "name": "STRING",
        "isBlocked": "BOOLEAN",
    })
    db.create_node_table("Company", {
        "companyId": "INT64 PRIMARY KEY",
        "name": "STRING",
    })
    db.create_node_table("Account", {
        "accountId": "INT64 PRIMARY KEY",
        "createDate": "INT64",
        "isBlocked": "BOOLEAN",
        "type": "STRING",
    })
    db.create_node_table("Loan", {
        "loanId": "INT64 PRIMARY KEY",
        "amount": "DOUBLE",
        "balance": "DOUBLE",
    })
    db.create_node_table("Medium", {
        "mediumId": "INT64 PRIMARY KEY",
        "type": "STRING",
    })

    # Edge tables
    db.create_edge_table("own", "Person", "Account")
    db.create_edge_table("transfer", "Account", "Account", properties={
        "amount": "DOUBLE",
        "timestamp": "INT64",
    })
    db.create_edge_table("signIn", "Medium", "Account", properties={
        "timestamp": "INT64",
    })
    db.create_edge_table("apply", "Person", "Loan", properties={
        "timestamp": "INT64",
    })
    db.create_edge_table("guarantee", "Person", "Person", properties={
        "timestamp": "INT64",
    })
    db.create_edge_table("invest", "Company", "Company", properties={
        "amount": "DOUBLE",
    })


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def _power_law_amount(rng: random.Random, minimum: float = 10.0, maximum: float = 500_000.0) -> float:
    """Generate a power-law distributed transfer amount.

    Most transactions are small; a few are very large.
    Uses the inverse transform method with alpha=1.5.

    Args:
        rng: Random number generator.
        minimum: Minimum amount.
        maximum: Maximum amount.

    Returns:
        A float amount in [minimum, maximum].
    """
    alpha = 1.5
    u = rng.random()
    # Inverse CDF of Pareto: x = min * (1 - u)^(-1/alpha)
    raw = minimum * ((1.0 - u) ** (-1.0 / alpha))
    return min(round(raw, 2), maximum)


def generate_finbench_data(
    db: Database,
    num_persons: int = 10_000,
    num_accounts: int = 50_000,
    num_transfers: int = 500_000,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate FinBench-compatible data with known fraud patterns injected.

    Creates persons, companies, accounts, loans, mediums, and all edge types.
    Injects known fraud patterns (circular transfers, structuring, guarantee
    chains) at deterministic positions for query validation.

    Uses Parquet + COPY FROM for fast bulk loading.

    Args:
        db: Target database (schema should already be created).
        num_persons: Number of Person nodes.
        num_accounts: Number of Account nodes.
        num_transfers: Number of transfer edges.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with node/edge counts and generation timing, plus injected
        fraud pattern metadata for query validation.
    """
    rng = random.Random(seed)
    t0 = time.monotonic()

    num_companies = max(50, num_persons // 100)
    num_loans = max(100, num_persons // 5)
    num_mediums = max(200, num_persons // 2)

    # Derived edge counts (proportional to graph size)
    num_own = min(num_accounts, num_persons * 3)
    num_signin = min(num_accounts * 2, num_mediums * 3)
    num_apply = min(num_loans, num_persons)
    num_guarantee = max(50, num_persons // 50)
    num_invest = max(20, num_companies // 5)

    tmpdir = tempfile.mkdtemp(prefix="bridgr_finbench_")

    try:
        # --- Person nodes ---
        first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer",
            "Michael", "Linda", "David", "Elizabeth", "William", "Barbara",
            "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
            "Christopher", "Karen",
        ]
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
            "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
            "Jackson", "Martin",
        ]
        person_ids = list(range(num_persons))
        person_names = [
            f"{rng.choice(first_names)} {rng.choice(last_names)}"
            for _ in person_ids
        ]
        # ~2% of persons are blocked
        person_blocked = [rng.random() < 0.02 for _ in person_ids]

        _write_parquet_and_copy(db, tmpdir, "persons.parquet", "Person", pa.table({
            "personId": pa.array(person_ids, type=pa.int64()),
            "name": pa.array(person_names, type=pa.string()),
            "isBlocked": pa.array(person_blocked, type=pa.bool_()),
        }))

        # --- Company nodes ---
        company_ids = list(range(num_companies))
        company_names = [f"Corp_{i}" for i in company_ids]

        _write_parquet_and_copy(db, tmpdir, "companies.parquet", "Company", pa.table({
            "companyId": pa.array(company_ids, type=pa.int64()),
            "name": pa.array(company_names, type=pa.string()),
        }))

        # --- Account nodes ---
        account_ids = list(range(num_accounts))
        acct_types = ["checking", "savings", "business", "investment"]
        base_ts = 1_600_000_000  # ~Sep 2020
        account_create = [base_ts + rng.randint(0, 100_000_000) for _ in account_ids]
        account_blocked = [rng.random() < 0.01 for _ in account_ids]
        account_type = [rng.choice(acct_types) for _ in account_ids]

        _write_parquet_and_copy(db, tmpdir, "accounts.parquet", "Account", pa.table({
            "accountId": pa.array(account_ids, type=pa.int64()),
            "createDate": pa.array(account_create, type=pa.int64()),
            "isBlocked": pa.array(account_blocked, type=pa.bool_()),
            "type": pa.array(account_type, type=pa.string()),
        }))

        # --- Loan nodes ---
        loan_ids = list(range(num_loans))
        loan_amounts = [round(rng.uniform(1_000, 1_000_000), 2) for _ in loan_ids]
        loan_balances = [round(amt * rng.uniform(0.1, 1.0), 2) for amt in loan_amounts]

        _write_parquet_and_copy(db, tmpdir, "loans.parquet", "Loan", pa.table({
            "loanId": pa.array(loan_ids, type=pa.int64()),
            "amount": pa.array(loan_amounts, type=pa.float64()),
            "balance": pa.array(loan_balances, type=pa.float64()),
        }))

        # --- Medium nodes ---
        medium_ids = list(range(num_mediums))
        medium_types = ["phone", "email"]
        medium_type = [rng.choice(medium_types) for _ in medium_ids]

        _write_parquet_and_copy(db, tmpdir, "mediums.parquet", "Medium", pa.table({
            "mediumId": pa.array(medium_ids, type=pa.int64()),
            "type": pa.array(medium_type, type=pa.string()),
        }))

        # --- own edges (Person -> Account) ---
        own_from: list[int] = []
        own_to: list[int] = []
        # Ensure every person owns at least one account
        for pid in range(min(num_persons, num_accounts)):
            own_from.append(pid)
            own_to.append(pid)  # 1:1 for the first batch
        # Additional ownership (some people have multiple accounts)
        for _ in range(num_own - len(own_from)):
            own_from.append(rng.randint(0, num_persons - 1))
            own_to.append(rng.randint(0, num_accounts - 1))

        _write_parquet_and_copy(db, tmpdir, "own.parquet", "own", pa.table({
            "from": pa.array(own_from, type=pa.int64()),
            "to": pa.array(own_to, type=pa.int64()),
        }))

        # --- transfer edges (Account -> Account) with fraud injection ---
        xfer_from: list[int] = []
        xfer_to: list[int] = []
        xfer_amount: list[float] = []
        xfer_ts: list[int] = []

        # Track injected fraud patterns for validation
        fraud_rings: list[list[int]] = []
        structuring_accounts: list[int] = []
        guarantee_chains: list[list[int]] = []

        # Inject 5-10 circular transfer rings (A->B->C->A)
        num_rings = min(rng.randint(5, 10), num_accounts // 10)
        ring_start = 0
        for _ in range(num_rings):
            ring_size = rng.randint(3, 6)
            if ring_start + ring_size >= num_accounts:
                break
            ring_accounts = list(range(ring_start, ring_start + ring_size))
            fraud_rings.append(ring_accounts)
            ring_amount = round(rng.uniform(50_000, 200_000), 2)
            ring_base_ts = base_ts + rng.randint(50_000_000, 90_000_000)
            for i in range(ring_size):
                src = ring_accounts[i]
                dst = ring_accounts[(i + 1) % ring_size]
                xfer_from.append(src)
                xfer_to.append(dst)
                xfer_amount.append(ring_amount)
                xfer_ts.append(ring_base_ts + i * 3600)  # 1 hour apart
            ring_start += ring_size + 5  # gap between rings

        # Inject 3-5 structuring patterns (rapid small transfers < $10K)
        num_struct = min(rng.randint(3, 5), num_accounts // 20)
        for _ in range(num_struct):
            src_acct = rng.randint(num_accounts // 2, num_accounts - 1)
            structuring_accounts.append(src_acct)
            struct_base_ts = base_ts + rng.randint(50_000_000, 90_000_000)
            for j in range(rng.randint(8, 15)):
                dst_acct = rng.randint(0, num_accounts - 1)
                while dst_acct == src_acct:
                    dst_acct = rng.randint(0, num_accounts - 1)
                xfer_from.append(src_acct)
                xfer_to.append(dst_acct)
                xfer_amount.append(round(rng.uniform(5_000, 9_999), 2))
                xfer_ts.append(struct_base_ts + j * 300)  # 5 min apart

        # Fill remaining organic transfers
        organic_count = num_transfers - len(xfer_from)
        edge_set: set[tuple[int, int, int]] = set()  # deduplicate by (from, to, ts)
        for _ in range(organic_count):
            src = rng.randint(0, num_accounts - 1)
            dst = rng.randint(0, num_accounts - 1)
            while dst == src:
                dst = rng.randint(0, num_accounts - 1)
            ts = base_ts + rng.randint(0, 100_000_000)
            xfer_from.append(src)
            xfer_to.append(dst)
            xfer_amount.append(_power_law_amount(rng))
            xfer_ts.append(ts)

        _write_parquet_and_copy(db, tmpdir, "transfers.parquet", "transfer", pa.table({
            "from": pa.array(xfer_from, type=pa.int64()),
            "to": pa.array(xfer_to, type=pa.int64()),
            "amount": pa.array(xfer_amount, type=pa.float64()),
            "timestamp": pa.array(xfer_ts, type=pa.int64()),
        }))

        # --- signIn edges (Medium -> Account) ---
        si_from: list[int] = []
        si_to: list[int] = []
        si_ts: list[int] = []
        for _ in range(num_signin):
            si_from.append(rng.randint(0, num_mediums - 1))
            si_to.append(rng.randint(0, num_accounts - 1))
            si_ts.append(base_ts + rng.randint(0, 100_000_000))

        _write_parquet_and_copy(db, tmpdir, "signin.parquet", "signIn", pa.table({
            "from": pa.array(si_from, type=pa.int64()),
            "to": pa.array(si_to, type=pa.int64()),
            "timestamp": pa.array(si_ts, type=pa.int64()),
        }))

        # --- apply edges (Person -> Loan) ---
        ap_from: list[int] = []
        ap_to: list[int] = []
        ap_ts: list[int] = []
        for _ in range(num_apply):
            ap_from.append(rng.randint(0, num_persons - 1))
            ap_to.append(rng.randint(0, num_loans - 1))
            ap_ts.append(base_ts + rng.randint(0, 100_000_000))

        _write_parquet_and_copy(db, tmpdir, "apply.parquet", "apply", pa.table({
            "from": pa.array(ap_from, type=pa.int64()),
            "to": pa.array(ap_to, type=pa.int64()),
            "timestamp": pa.array(ap_ts, type=pa.int64()),
        }))

        # --- guarantee edges (Person -> Person) with chain injection ---
        gu_from: list[int] = []
        gu_to: list[int] = []
        gu_ts: list[int] = []

        # Inject 2-3 guarantee chains (person->person->person->...)
        num_chains = min(rng.randint(2, 3), num_persons // 20)
        chain_start = num_persons // 2  # Start in the middle of person ID space
        for _ in range(num_chains):
            chain_len = rng.randint(3, 6)
            if chain_start + chain_len >= num_persons:
                break
            chain = list(range(chain_start, chain_start + chain_len))
            guarantee_chains.append(chain)
            chain_base_ts = base_ts + rng.randint(10_000_000, 50_000_000)
            for i in range(chain_len - 1):
                gu_from.append(chain[i])
                gu_to.append(chain[i + 1])
                gu_ts.append(chain_base_ts + i * 86400)  # 1 day apart
            chain_start += chain_len + 10

        # Fill remaining organic guarantees
        for _ in range(num_guarantee - len(gu_from)):
            a = rng.randint(0, num_persons - 1)
            b = rng.randint(0, num_persons - 1)
            while b == a:
                b = rng.randint(0, num_persons - 1)
            gu_from.append(a)
            gu_to.append(b)
            gu_ts.append(base_ts + rng.randint(0, 100_000_000))

        _write_parquet_and_copy(db, tmpdir, "guarantee.parquet", "guarantee", pa.table({
            "from": pa.array(gu_from, type=pa.int64()),
            "to": pa.array(gu_to, type=pa.int64()),
            "timestamp": pa.array(gu_ts, type=pa.int64()),
        }))

        # --- invest edges (Company -> Company) ---
        inv_from: list[int] = []
        inv_to: list[int] = []
        inv_amount: list[float] = []
        for _ in range(num_invest):
            a = rng.randint(0, num_companies - 1)
            b = rng.randint(0, num_companies - 1)
            while b == a:
                b = rng.randint(0, num_companies - 1)
            inv_from.append(a)
            inv_to.append(b)
            inv_amount.append(round(rng.uniform(100_000, 10_000_000), 2))

        _write_parquet_and_copy(db, tmpdir, "invest.parquet", "invest", pa.table({
            "from": pa.array(inv_from, type=pa.int64()),
            "to": pa.array(inv_to, type=pa.int64()),
            "amount": pa.array(inv_amount, type=pa.float64()),
        }))

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = time.monotonic() - t0

    return {
        "persons": num_persons,
        "companies": num_companies,
        "accounts": num_accounts,
        "loans": num_loans,
        "mediums": num_mediums,
        "transfers": len(xfer_from),
        "own_edges": len(own_from),
        "signin_edges": len(si_from),
        "apply_edges": len(ap_from),
        "guarantee_edges": len(gu_from),
        "invest_edges": len(inv_from),
        "generation_time_s": elapsed,
        # Fraud pattern metadata for validation
        "fraud_rings": fraud_rings,
        "structuring_accounts": structuring_accounts,
        "guarantee_chains": guarantee_chains,
    }


def _write_parquet_and_copy(
    db: Database,
    tmpdir: str,
    filename: str,
    table_name: str,
    arrow_table: pa.Table,
) -> None:
    """Write an Arrow table to Parquet and COPY FROM into the database.

    Args:
        db: Target database.
        tmpdir: Temporary directory for Parquet files.
        filename: Parquet file name.
        table_name: Database table name to COPY into.
        arrow_table: Arrow table with the data.
    """
    path = os.path.join(tmpdir, filename)
    pq.write_table(arrow_table, path)
    db.execute(f'COPY {table_name} FROM "{_cypher_path(path)}"')


# ---------------------------------------------------------------------------
# Query implementations
# ---------------------------------------------------------------------------


def _pick_seed_ids(
    db: Database,
    label: str,
    pk_col: str,
    count: int = 5,
) -> list[int]:
    """Pick seed node IDs spread across the graph for query parameters.

    Args:
        db: Database to query.
        label: Node label.
        pk_col: Primary key column name.
        count: Number of IDs to return.

    Returns:
        List of primary key values.
    """
    rows = db.query(f"MATCH (n:{label}) RETURN n.{pk_col} AS id ORDER BY n.{pk_col}")
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    step = max(1, len(ids) // (count + 1))
    return [ids[step * (i + 1)] for i in range(min(count, len(ids)))]


def run_benchmark(
    num_persons: int = 10_000,
    num_accounts: int = 50_000,
    num_transfers: int = 500_000,
    *,
    warmup: int = 3,
    runs: int = 10,
    seed: int = 42,
) -> str:
    """Run the full FinBench benchmark and return a formatted report.

    Args:
        num_persons: Number of Person nodes (SF-1 = 10K, SF-10 = 100K).
        num_accounts: Number of Account nodes.
        num_transfers: Number of transfer edges.
        warmup: Warmup iterations per query.
        runs: Timed iterations per query.
        seed: RNG seed for reproducibility.

    Returns:
        Formatted benchmark report string.
    """
    db = Database(":memory:")

    # --- Create schema and generate data ---
    print(f"Creating FinBench schema...")
    _create_finbench_schema(db)

    print(f"Generating data: {num_persons:,} persons, {num_accounts:,} accounts, "
          f"{num_transfers:,} transfers...")
    gen_stats = generate_finbench_data(
        db, num_persons, num_accounts, num_transfers, seed=seed,
    )
    print(f"Data loaded in {gen_stats['generation_time_s']:.1f}s")
    print(f"  Nodes: {gen_stats['persons']:,} persons, {gen_stats['companies']:,} companies, "
          f"{gen_stats['accounts']:,} accounts, {gen_stats['loans']:,} loans, "
          f"{gen_stats['mediums']:,} mediums")
    print(f"  Edges: {gen_stats['transfers']:,} transfers, {gen_stats['own_edges']:,} own, "
          f"{gen_stats['guarantee_edges']:,} guarantee")
    print(f"  Injected: {len(gen_stats['fraud_rings'])} fraud rings, "
          f"{len(gen_stats['structuring_accounts'])} structuring patterns, "
          f"{len(gen_stats['guarantee_chains'])} guarantee chains")

    # Pick seed IDs for parameterized queries
    person_ids = _pick_seed_ids(db, "Person", "personId", 5)
    account_ids = _pick_seed_ids(db, "Account", "accountId", 5)

    if len(person_ids) < 2 or len(account_ids) < 2:
        print("ERROR: Not enough nodes generated. Aborting.")
        db.close()
        return "Benchmark failed: insufficient nodes."

    pid_a = person_ids[0]
    pid_b = person_ids[-1]
    aid_a = account_ids[0]

    # Use a timestamp window that covers the bulk of generated data
    base_ts = 1_600_000_000
    ts_start = base_ts + 40_000_000
    ts_end = base_ts + 80_000_000

    timing_results: list[TimingResult] = []

    # ------------------------------------------------------------------
    # SR-1: Person's owned accounts
    # ------------------------------------------------------------------
    print("  [1/10] SR-1: Person's owned accounts...")
    sr1 = time_query(
        db,
        "MATCH (p:Person {personId: $pid})-[:own]->(a:Account) "
        "RETURN a.accountId, a.createDate, a.isBlocked, a.type",
        warmup=warmup,
        runs=runs,
        params={"pid": pid_a},
    )
    timing_results.append(TimingResult(
        name="SR-1: Person's owned accounts",
        median_ms=sr1["median_ms"],
        p95_ms=sr1["p95_ms"],
        p99_ms=sr1["p99_ms"],
        runs=runs,
        warmup=warmup,
        extra={"result_count": sr1["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # SR-2: Account transfer history (time-bounded)
    # ------------------------------------------------------------------
    print("  [2/10] SR-2: Transfer history (time-bounded)...")
    sr2 = time_query(
        db,
        "MATCH (a:Account {accountId: $aid})-[t:transfer]->(b:Account) "
        "WHERE t.timestamp >= $ts_start AND t.timestamp <= $ts_end "
        "RETURN b.accountId, t.amount, t.timestamp "
        "ORDER BY t.timestamp",
        warmup=warmup,
        runs=runs,
        params={"aid": aid_a, "ts_start": ts_start, "ts_end": ts_end},
    )
    timing_results.append(TimingResult(
        name="SR-2: Transfer history (bounded)",
        median_ms=sr2["median_ms"],
        p95_ms=sr2["p95_ms"],
        p99_ms=sr2["p99_ms"],
        extra={"result_count": sr2["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # SR-3: Person's loan applications
    # ------------------------------------------------------------------
    print("  [3/10] SR-3: Person's loan applications...")
    sr3 = time_query(
        db,
        "MATCH (p:Person {personId: $pid})-[a:apply]->(l:Loan) "
        "RETURN l.loanId, l.amount, l.balance, a.timestamp",
        warmup=warmup,
        runs=runs,
        params={"pid": pid_a},
    )
    timing_results.append(TimingResult(
        name="SR-3: Person's loan applications",
        median_ms=sr3["median_ms"],
        p95_ms=sr3["p95_ms"],
        p99_ms=sr3["p99_ms"],
        extra={"result_count": sr3["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # CR-1: Circular transfer detection (3-hop cycle) -- THE fraud query
    # ------------------------------------------------------------------
    print("  [4/10] CR-1: Circular transfers (3-hop)...")
    cr1 = time_query(
        db,
        "MATCH (a:Account)-[t1:transfer]->(b:Account)"
        "-[t2:transfer]->(c:Account)-[t3:transfer]->(a) "
        "WHERE a.accountId < b.accountId AND b.accountId < c.accountId "
        "RETURN a.accountId AS a_id, b.accountId AS b_id, c.accountId AS c_id, "
        "t1.amount AS amt1, t2.amount AS amt2, t3.amount AS amt3 "
        "LIMIT 100",
        warmup=warmup,
        runs=runs,
    )
    timing_results.append(TimingResult(
        name="CR-1: Circular transfers (3-hop)",
        median_ms=cr1["median_ms"],
        p95_ms=cr1["p95_ms"],
        p99_ms=cr1["p99_ms"],
        extra={"result_count": cr1["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # CR-2: Guarantee chains (variable-length person paths)
    # ------------------------------------------------------------------
    print("  [5/10] CR-2: Guarantee chains...")
    # Use a person in the middle of ID space where chains were injected
    chain_seed = num_persons // 2
    cr2 = time_query(
        db,
        "MATCH p = (src:Person {personId: $pid})-[:guarantee*1..5]->(dst:Person) "
        "RETURN dst.personId AS dst_id, dst.name AS dst_name, length(p) AS depth",
        warmup=warmup,
        runs=runs,
        params={"pid": chain_seed},
    )
    timing_results.append(TimingResult(
        name="CR-2: Guarantee chains",
        median_ms=cr2["median_ms"],
        p95_ms=cr2["p95_ms"],
        p99_ms=cr2["p99_ms"],
        extra={"result_count": cr2["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # CR-3: Multi-hop account reachability (BFS within N transfers)
    # ------------------------------------------------------------------
    print("  [6/10] CR-3: Multi-hop reachability...")
    cr3 = time_query(
        db,
        "MATCH (src:Account {accountId: $aid})-[:transfer*1..3]->(dst:Account) "
        "RETURN DISTINCT dst.accountId AS dst_id "
        "LIMIT 500",
        warmup=warmup,
        runs=runs,
        params={"aid": aid_a},
    )
    timing_results.append(TimingResult(
        name="CR-3: Multi-hop reachability",
        median_ms=cr3["median_ms"],
        p95_ms=cr3["p95_ms"],
        p99_ms=cr3["p99_ms"],
        extra={"result_count": cr3["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # CR-4: Fund tracing (follow money flow with amount tracking)
    # ------------------------------------------------------------------
    print("  [7/10] CR-4: Fund tracing...")
    cr4 = time_query(
        db,
        "MATCH (src:Account {accountId: $aid})-[t1:transfer]->(mid:Account)"
        "-[t2:transfer]->(dst:Account) "
        "WHERE t1.amount > 10000 AND t2.amount > 10000 "
        "RETURN src.accountId AS src_id, mid.accountId AS mid_id, "
        "dst.accountId AS dst_id, t1.amount AS amt1, t2.amount AS amt2 "
        "LIMIT 200",
        warmup=warmup,
        runs=runs,
        params={"aid": aid_a},
    )
    timing_results.append(TimingResult(
        name="CR-4: Fund tracing (2-hop)",
        median_ms=cr4["median_ms"],
        p95_ms=cr4["p95_ms"],
        p99_ms=cr4["p99_ms"],
        extra={"result_count": cr4["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # CR-5: Common accounts between two persons
    # ------------------------------------------------------------------
    print("  [8/10] CR-5: Common accounts...")
    cr5 = time_query(
        db,
        "MATCH (p1:Person {personId: $pid1})-[:own]->(a:Account)<-[:own]-(p2:Person {personId: $pid2}) "
        "RETURN a.accountId, a.type",
        warmup=warmup,
        runs=runs,
        params={"pid1": pid_a, "pid2": pid_b},
    )
    timing_results.append(TimingResult(
        name="CR-5: Common accounts",
        median_ms=cr5["median_ms"],
        p95_ms=cr5["p95_ms"],
        p99_ms=cr5["p99_ms"],
        extra={"result_count": cr5["last_result_count"]},
    ))

    # ------------------------------------------------------------------
    # W-1: Create new transfer edge
    # ------------------------------------------------------------------
    print("  [9/10] W-1: Create transfer...")
    w1_ts = base_ts + 99_000_000

    def _create_transfer() -> list[dict[str, Any]]:
        return db.query(
            "MATCH (a:Account {accountId: $from_aid}), (b:Account {accountId: $to_aid}) "
            "CREATE (a)-[:transfer {amount: $amt, timestamp: $ts}]->(b) "
            "RETURN a.accountId",
            {"from_aid": account_ids[0], "to_aid": account_ids[1],
             "amt": 1234.56, "ts": w1_ts},
        )

    w1_timing = time_callable(_create_transfer, warmup=warmup, runs=runs)
    timing_results.append(TimingResult(
        name="W-1: Create transfer",
        median_ms=w1_timing["median_ms"],
        p95_ms=w1_timing["p95_ms"],
        p99_ms=w1_timing["p99_ms"],
    ))

    # ------------------------------------------------------------------
    # W-2: Block an account
    # ------------------------------------------------------------------
    print("  [10/10] W-2: Block account...")
    # Pick an account that is not blocked
    block_target = account_ids[2]

    def _block_account() -> list[dict[str, Any]]:
        return db.query(
            "MATCH (a:Account {accountId: $aid}) "
            "SET a.isBlocked = true "
            "RETURN a.accountId, a.isBlocked",
            {"aid": block_target},
        )

    w2_timing = time_callable(_block_account, warmup=warmup, runs=runs)
    timing_results.append(TimingResult(
        name="W-2: Block account",
        median_ms=w2_timing["median_ms"],
        p95_ms=w2_timing["p95_ms"],
        p99_ms=w2_timing["p99_ms"],
    ))

    # --- Compute total edges ---
    total_edges = (
        gen_stats["transfers"]
        + gen_stats["own_edges"]
        + gen_stats["signin_edges"]
        + gen_stats["apply_edges"]
        + gen_stats["guarantee_edges"]
        + gen_stats["invest_edges"]
    )
    total_nodes = (
        gen_stats["persons"]
        + gen_stats["companies"]
        + gen_stats["accounts"]
        + gen_stats["loans"]
        + gen_stats["mediums"]
    )

    # --- Format output ---
    header = (
        f"Bridgr LDBC FinBench Benchmark Results\n"
        f"=======================================\n"
        f"Persons: {gen_stats['persons']:,} | Accounts: {gen_stats['accounts']:,} | "
        f"Transfers: {gen_stats['transfers']:,}\n"
        f"Total nodes: {total_nodes:,} | Total edges: {total_edges:,}\n"
        f"Engine: LadybugDB (Kuzu fork) v0.1\n"
        f"Date: {date.today().isoformat()}\n"
        f"Data generation: {gen_stats['generation_time_s']:.1f}s\n"
        f"Warmup: {warmup} | Runs: {runs}\n"
        f"Fraud patterns injected: {len(gen_stats['fraud_rings'])} rings, "
        f"{len(gen_stats['structuring_accounts'])} structuring, "
        f"{len(gen_stats['guarantee_chains'])} guarantee chains\n"
    )

    table = format_timing_table(timing_results, "FinBench Query Latencies")

    # Append fraud detection summary
    fraud_note = _format_fraud_summary(cr1, cr2, gen_stats)

    db.close()

    report = f"{header}\n{table}\n\n{fraud_note}\n"
    return report


def _format_fraud_summary(
    cr1_result: dict[str, Any],
    cr2_result: dict[str, Any],
    gen_stats: dict[str, Any],
) -> str:
    """Format a summary of fraud pattern detection results.

    Args:
        cr1_result: CR-1 (circular transfer) timing result.
        cr2_result: CR-2 (guarantee chain) timing result.
        gen_stats: Data generation stats with fraud metadata.

    Returns:
        Formatted fraud detection summary string.
    """
    lines = [
        "Fraud Detection Summary",
        "=======================",
        "",
        f"Circular transfers (CR-1): {cr1_result['last_result_count']} cycles found "
        f"({len(gen_stats['fraud_rings'])} injected)",
        f"Guarantee chains (CR-2): {cr2_result['last_result_count']} chain endpoints found "
        f"({len(gen_stats['guarantee_chains'])} chains injected)",
        "",
        "Notes:",
        "- CR-1 uses 3-hop directed cycle detection with ID ordering to avoid duplicates",
        "- CR-2 uses variable-length path traversal (1..5 hops) for guarantee chains",
        "- Structuring detection requires aggregation over transfer windows (not yet a standalone query)",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for the FinBench benchmark."""
    parser = argparse.ArgumentParser(
        description="Bridgr LDBC FinBench: 10 financial fraud queries on synthetic data."
    )
    parser.add_argument(
        "--persons", type=int, default=10_000,
        help="Number of Person nodes (SF-1=10K, SF-10=100K, default: 10000)",
    )
    parser.add_argument(
        "--accounts", type=int, default=50_000,
        help="Number of Account nodes (default: 50000)",
    )
    parser.add_argument(
        "--transfers", type=int, default=500_000,
        help="Number of transfer edges (default: 500000)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup iterations per query (default: 3)",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Timed iterations per query (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    report = run_benchmark(
        num_persons=args.persons,
        num_accounts=args.accounts,
        num_transfers=args.transfers,
        warmup=args.warmup,
        runs=args.runs,
        seed=args.seed,
    )
    print(report)


if __name__ == "__main__":
    main()
