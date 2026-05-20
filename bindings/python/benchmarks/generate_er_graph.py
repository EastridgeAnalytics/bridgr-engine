"""Generate realistic graph data for engine benchmarks.

Creates a Customer → MEMBER_OF → Household → LOCATED_IN → ZipCode graph
that represents the output of an ER pipeline. This is what HEB's graph
would look like after running Bridgr's entity resolution.

Usage:
    python benchmarks/generate_er_graph.py --customers 1000000 --output /tmp/graph_1m
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from faker import Faker

fake = Faker("en_US")
Faker.seed(42)
random.seed(42)

TX_ZIPS = [
    ("78701", "Austin", "TX"), ("78702", "Austin", "TX"), ("78703", "Austin", "TX"),
    ("78704", "Austin", "TX"), ("78705", "Austin", "TX"),
    ("78209", "San Antonio", "TX"), ("78212", "San Antonio", "TX"), ("78201", "San Antonio", "TX"),
    ("77098", "Houston", "TX"), ("77005", "Houston", "TX"), ("77002", "Houston", "TX"),
    ("77003", "Houston", "TX"), ("77004", "Houston", "TX"),
    ("75201", "Dallas", "TX"), ("75202", "Dallas", "TX"), ("75204", "Dallas", "TX"),
    ("75024", "Plano", "TX"), ("75025", "Plano", "TX"), ("75023", "Plano", "TX"),
    ("76102", "Fort Worth", "TX"), ("76107", "Fort Worth", "TX"),
    ("79901", "El Paso", "TX"), ("79902", "El Paso", "TX"),
    ("78401", "Corpus Christi", "TX"), ("78411", "Corpus Christi", "TX"),
    ("79401", "Lubbock", "TX"), ("79410", "Lubbock", "TX"),
    ("78501", "McAllen", "TX"), ("78504", "McAllen", "TX"),
]
# Pad to ~1000 zips for realistic distribution
while len(TX_ZIPS) < 1000:
    base_zip, city, state = random.choice(TX_ZIPS[:29])
    new_zip = f"{int(base_zip[:3])}{random.randint(0,99):02d}"
    TX_ZIPS.append((new_zip, city, state))


def generate(
    n_customers: int = 1_000_000,
    household_size_range: tuple[int, int] = (1, 5),
    output_dir: str | Path = "/tmp/graph_benchmark",
) -> Path:
    """Generate Parquet files for bulk graph import.

    Returns path to output directory containing:
    - customers.parquet
    - households.parquet
    - zipcodes.parquet
    - member_of.parquet (customer_id → household_id)
    - located_in.parquet (household_id → zip)
    """
    t0 = time.monotonic()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate zip codes (static reference table)
    zip_data = {"zip": [], "city": [], "state": []}
    seen_zips = set()
    for z, c, s in TX_ZIPS:
        if z not in seen_zips:
            zip_data["zip"].append(z)
            zip_data["city"].append(c)
            zip_data["state"].append(s)
            seen_zips.add(z)
    pq.write_table(pa.table(zip_data), output_path / "zipcodes.parquet")
    zip_list = zip_data["zip"]

    # Generate households (each gets a zip)
    avg_size = sum(household_size_range) / 2
    n_households = int(n_customers / avg_size)

    hh_ids, hh_zips, hh_sizes, hh_phones = [], [], [], []
    for i in range(n_households):
        hh_ids.append(f"HH{i:08d}")
        hh_zips.append(random.choice(zip_list))
        hh_sizes.append(random.randint(*household_size_range))
        hh_phones.append(fake.numerify("##########"))

    pq.write_table(pa.table({
        "household_id": hh_ids,
        "zip": hh_zips,
        "member_count": hh_sizes,
        "primary_phone": hh_phones,
    }), output_path / "households.parquet")

    # Generate customers (assigned to households)
    cust_ids, cust_names, cust_emails, cust_phones, cust_zips = [], [], [], [], []
    member_of_src, member_of_dst = [], []

    cust_idx = 0
    for hh_idx in range(n_households):
        hh_id = hh_ids[hh_idx]
        hh_zip = hh_zips[hh_idx]
        hh_phone = hh_phones[hh_idx]
        size = hh_sizes[hh_idx]
        last_name = fake.last_name()

        for _ in range(size):
            if cust_idx >= n_customers:
                break
            cid = f"C{cust_idx:08d}"
            first = fake.first_name()
            cust_ids.append(cid)
            cust_names.append(f"{first} {last_name}")
            cust_emails.append(f"{first.lower()}.{last_name.lower()}@{fake.free_email_domain()}")
            cust_phones.append(hh_phone)
            cust_zips.append(hh_zip)
            member_of_src.append(cid)
            member_of_dst.append(hh_id)
            cust_idx += 1
        if cust_idx >= n_customers:
            break

    pq.write_table(pa.table({
        "customer_id": cust_ids,
        "name": cust_names,
        "email": cust_emails,
        "phone": cust_phones,
        "zip": cust_zips,
    }), output_path / "customers.parquet")

    # Edge tables
    pq.write_table(pa.table({
        "from_id": member_of_src,
        "to_id": member_of_dst,
    }), output_path / "member_of.parquet")

    # located_in: household → zip
    pq.write_table(pa.table({
        "from_id": hh_ids[:len(set(hh_ids))],
        "to_id": hh_zips[:len(set(hh_ids))],
    }), output_path / "located_in.parquet")

    elapsed = time.monotonic() - t0
    actual_customers = len(cust_ids)
    actual_households = len(set(member_of_dst))
    print(f"Generated graph data in {elapsed:.1f}s:")
    print(f"  Customers: {actual_customers:,}")
    print(f"  Households: {actual_households:,}")
    print(f"  ZipCodes: {len(zip_list):,}")
    print(f"  MEMBER_OF edges: {len(member_of_src):,}")
    print(f"  LOCATED_IN edges: {actual_households:,}")
    print(f"  Output: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graph data for engine benchmarks")
    parser.add_argument("--customers", type=int, default=1_000_000)
    parser.add_argument("--output", type=str, default="/tmp/graph_benchmark")
    args = parser.parse_args()

    generate(args.customers, output_dir=args.output)
