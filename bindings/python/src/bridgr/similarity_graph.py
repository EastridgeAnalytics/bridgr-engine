"""Similarity graph projection — builds unipartite SIMILAR_TO edges from shared attributes.

H-E-B use case: Customer nodes connected by shared zip, phone, address_hash.
Louvain runs on this unipartite graph to detect fraud communities.

Usage:
    from bridgr.similarity_graph import SimilarityGraphBuilder

    builder = SimilarityGraphBuilder(db, node_label="Customer")
    stats = builder.build(
        attributes=["zip", "phone", "address_hash"],
        edge_label="SIMILAR_TO",
        min_shared=2,
    )
    print(f"Created {stats['edge_count']} similarity edges")

    # Or pass node_label to build() for backward compatibility:
    builder = SimilarityGraphBuilder(db)
    stats = builder.build(
        node_label="Customer",
        attributes=["zip", "phone", "address_hash"],
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any

from bridgr.database import Database

log = logging.getLogger(__name__)

__all__ = ["SimilarityGraphBuilder"]


class SimilarityGraphBuilder:
    """Builds SIMILAR_TO edges between nodes that share attribute values.

    For each attribute in the list, finds all pairs of nodes sharing the same
    value. Creates an edge for each pair with properties tracking which
    attributes matched and how many.

    Uses a blocking-then-expand approach: groups nodes by attribute value,
    then generates pairs within each group. This avoids full Cartesian
    joins and handles 1M+ nodes without OOM.
    """

    def __init__(self, db: Database, node_label: str = "Customer"):
        self._db = db
        self._default_label = node_label

    def build(
        self,
        attributes: list[str],
        *,
        min_shared: int = 1,
        edge_label: str = "SIMILAR_TO",
        batch_size: int = 10_000,
        max_bucket_size: int = 5000,
        max_pairs: int = 10_000_000,
        # Backward-compatible aliases
        node_label: str | None = None,
        min_weight: int | None = None,
    ) -> dict[str, Any]:
        """Build similarity edges.

        For each pair of nodes sharing >= min_shared attribute values,
        creates an edge with properties:
        - shared_count: int (number of shared attributes)
        - shared_attributes: str (comma-separated list of which attributes matched)

        Uses a blocking-then-expand approach: for each attribute, groups
        nodes by attribute value and generates pairs within each group.
        Buckets larger than max_bucket_size are skipped to avoid Cartesian
        explosion at scale.

        Args:
            attributes: Properties to check for shared values
                (e.g., ["phone", "address", "email"]).
            min_shared: Minimum shared attributes to create an edge.
            edge_label: Label for the similarity edges.
            batch_size: Edges per batch insert.
            max_bucket_size: Skip attribute values shared by more than
                this many nodes (prevents Cartesian explosion).
            max_pairs: Cap on total pair count to prevent OOM.
            node_label: Override the default node label set in constructor.
            min_weight: Deprecated alias for min_shared (backward compat).

        Returns:
            Dict with: edge_count, node_count, elapsed_seconds,
            plus detail fields: total_pairs, pairs_above_threshold,
            attributes_checked, buckets_skipped.
        """
        label = node_label or self._default_label
        threshold = min_weight if min_weight is not None else min_shared

        t0 = time.monotonic()
        pk_col = self._find_pk(label)

        # Ensure edge table exists with shared_count and shared_attributes
        try:
            self._db.create_edge_table(
                edge_label, label, label,
                {"shared_count": "INT64", "shared_attributes": "STRING"},
            )
        except Exception:
            pass

        # Phase 1: Blocking — for each attribute, group by value, generate pairs
        # Track both count and which attributes matched per pair
        pair_attrs: dict[tuple[str, str], list[str]] = {}
        buckets_skipped = 0
        nodes_seen: set[str] = set()

        for attr in attributes:
            log.info("Blocking on attribute: %s", attr)

            # Get distinct attribute values with their counts (blocking keys)
            bucket_rows = self._db.query(
                f"MATCH (n:{label}) WHERE n.{attr} IS NOT NULL "
                f"RETURN n.{attr} AS val, count(n) AS cnt ORDER BY cnt DESC"
            )

            for bucket in bucket_rows:
                val = bucket["val"]
                cnt = int(bucket["cnt"])
                if cnt < 2:
                    continue
                if cnt > max_bucket_size:
                    log.debug(
                        "  Skipping %s=%s (%d nodes > max_bucket_size %d)",
                        attr, val, cnt, max_bucket_size,
                    )
                    buckets_skipped += 1
                    continue

                # Expand within this bucket: get all node PKs sharing this value
                members = self._db.query(
                    f"MATCH (n:{label}) WHERE n.{attr} = $val "
                    f"RETURN n.{pk_col} AS pk ORDER BY n.{pk_col}",
                    {"val": val},
                )
                pks = [str(r["pk"]) for r in members]
                nodes_seen.update(pks)

                # Generate pairs within bucket (a < b for uniqueness)
                for i in range(len(pks)):
                    for j in range(i + 1, len(pks)):
                        pair = (pks[i], pks[j])
                        if pair not in pair_attrs:
                            pair_attrs[pair] = []
                        pair_attrs[pair].append(attr)
                if len(pair_attrs) >= max_pairs:
                    log.warning(
                        "Hit max_pairs cap (%d). Stopping pair generation.",
                        max_pairs,
                    )
                    break
            if len(pair_attrs) >= max_pairs:
                break

        # Phase 2: Filter by threshold and batch-create edges
        edges_to_create = [
            (a, b, attrs_list)
            for (a, b), attrs_list in pair_attrs.items()
            if len(attrs_list) >= threshold
        ]
        log.info(
            "Creating %d edges (from %d total pairs, min_shared=%d)",
            len(edges_to_create), len(pair_attrs), threshold,
        )

        edges_created = 0
        for i in range(0, len(edges_to_create), batch_size):
            batch = edges_to_create[i : i + batch_size]
            for a_id, b_id, attrs_list in batch:
                shared_count = len(attrs_list)
                shared_attributes = ",".join(sorted(set(attrs_list)))
                try:
                    self._db.execute(
                        f"MATCH (a:{label} {{{pk_col}: $a_id}}), "
                        f"(b:{label} {{{pk_col}: $b_id}}) "
                        f"CREATE (a)-[:{edge_label} {{shared_count: $shared_count, "
                        f"shared_attributes: $shared_attributes}}]->(b)",
                        {
                            "a_id": a_id,
                            "b_id": b_id,
                            "shared_count": shared_count,
                            "shared_attributes": shared_attributes,
                        },
                    )
                    edges_created += 1
                except Exception as e:
                    log.warning("Failed to create edge %s->%s: %s", a_id, b_id, e)

            if edges_created > 0 and edges_created % 10000 == 0:
                log.info(
                    "  Progress: %d / %d edges", edges_created, len(edges_to_create),
                )

        elapsed = time.monotonic() - t0
        stats = {
            # Spec-required fields
            "edge_count": edges_created,
            "node_count": len(nodes_seen),
            "elapsed_seconds": round(elapsed, 2),
            # Detailed fields
            "total_pairs": len(pair_attrs),
            "pairs_above_threshold": len(edges_to_create),
            "attributes_checked": len(attributes),
            "buckets_skipped": buckets_skipped,
            # Backward-compatible alias
            "edges_created": edges_created,
        }
        log.info(
            "Similarity graph: %d edges from %d pairs "
            "(%d attributes, %d buckets skipped) in %.1fs",
            edges_created, len(pair_attrs), len(attributes), buckets_skipped, elapsed,
        )
        return stats

    def _find_pk(self, node_label: str) -> str:
        """Find the primary key column for a node table."""
        try:
            rows = self._db.query(f"CALL table_info('{node_label}') RETURN *")
            for r in rows:
                if r.get("isPrimaryKey") or r.get("is_primary_key"):
                    return str(r.get("name", "id"))
            if rows:
                return str(rows[0].get("name", "id"))
        except Exception:
            pass
        return "id"
