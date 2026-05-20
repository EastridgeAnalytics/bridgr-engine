"""Real-time fraud risk scoring against an existing graph.

Scores customers by their position in the fraud graph: community size,
degree centrality, PageRank, and configurable risk thresholds.

Usage:
    from bridgr.scoring import FraudScorer

    scorer = FraudScorer(db)
    risk = scorer.score_customer("C001")
    # risk: {"customer_id": "C001", "risk_score": 0.72, "factors": [...], ...}

    tx = scorer.score_transaction("TX-001")
    # tx: {"tx_id": "TX-001", "risk_score": 0.72, "factors": [...], ...}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bridgr.database import Database

__all__ = ["FraudScorer"]

DEFAULT_RISK_CONFIG = {
    "critical": {"min_community_size": 50, "min_degree": 5},
    "high": {"min_community_size": 20, "min_degree": 3},
    "medium": {"min_community_size": 5, "min_degree": 2},
}

# Weights for composite risk_score (0-1)
_SCORE_WEIGHTS = {
    "community_size": 0.4,
    "degree": 0.3,
    "shared_attributes": 0.3,
}

# Normalisation caps (values at or above these map to 1.0 for that factor)
_SCORE_CAPS = {
    "community_size": 100,
    "degree": 10,
    "shared_attributes": 5,
}


class FraudScorer:
    """Fast fraud risk scoring using pre-computed graph features.

    Scoring factors:
    - community_size: larger community = higher risk
    - degree: more connections = higher risk
    - shared_attribute_count: more shared attributes = higher risk

    Each factor is normalised to 0-1 and combined into a composite risk_score.
    """

    def __init__(
        self,
        db: Database,
        *,
        node_label: str = "Customer",
        edge_label: str = "SIMILAR_TO",
        risk_config: dict[str, dict[str, Any]] | None = None,
    ):
        self._db = db
        self._node_label = node_label
        self._edge_label = edge_label
        self._risk_config = risk_config or DEFAULT_RISK_CONFIG
        self._pk_col = self._find_pk(node_label)

    def score_customer(self, customer_id: str) -> dict[str, Any]:
        """Score a single customer. Returns {customer_id, risk_score, factors, ...}.

        risk_score is a float in [0.0, 1.0].
        factors is a list of human-readable strings explaining the score.

        Target: <10ms.
        """
        rows = self._db.query(
            f"MATCH (c:{self._node_label} {{{self._pk_col}: $cid}}) "
            f"OPTIONAL MATCH (c)-[r:{self._edge_label}]-() "
            f"RETURN c.community_id AS community_id, "
            f"c.pagerank AS pagerank, count(r) AS degree",
            {"cid": customer_id},
        )

        if not rows:
            return self._empty_score(customer_id)

        row = rows[0]
        community_id = row.get("community_id")
        degree = int(row.get("degree", 0))
        pagerank = float(row.get("pagerank", 0.0) or 0.0)

        # Get community size if community_id exists
        community_size = 0
        if community_id is not None:
            size_rows = self._db.query(
                f"MATCH (c:{self._node_label}) "
                f"WHERE c.community_id = $comm RETURN count(c) AS size",
                {"comm": community_id},
            )
            if size_rows:
                community_size = int(size_rows[0]["size"])

        # Count shared attributes (neighbors sharing 2+ attributes)
        shared_attribute_count = self._count_shared_attributes(customer_id)

        # Compute composite risk_score (0-1)
        risk_score = self._compute_risk_score(
            community_size, degree, shared_attribute_count
        )

        # Assess risk level and factors
        risk_level, factors = self._assess_risk(
            community_size, degree, pagerank, shared_attribute_count
        )

        return {
            "customer_id": customer_id,
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "factors": factors,
            "community_id": community_id,
            "community_size": community_size,
            "degree": degree,
            "shared_attribute_count": shared_attribute_count,
            "pagerank": pagerank,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }

    def score_transaction(self, tx_id: str) -> dict[str, Any]:
        """Score a transaction by its connected entities.

        Looks up the transaction node and scores the connected customer.
        If no transaction node exists, returns a low-risk empty score.

        Target: <50ms.

        Args:
            tx_id: Transaction primary key.

        Returns:
            Dict with tx_id, risk_score (0-1), factors, and related fields.
        """
        # Try to find a connected customer via any edge type
        rows = self._db.query(
            f"MATCH (t {{id: $txid}})-[]-(c:{self._node_label}) "
            f"RETURN c.{self._pk_col} AS customer_id LIMIT 1",
            {"txid": tx_id},
        )

        if rows and rows[0].get("customer_id"):
            customer_id = str(rows[0]["customer_id"])
            score = self.score_customer(customer_id)
        else:
            score = self._empty_score("")

        score["tx_id"] = tx_id
        return score

    def batch_score(self, customer_ids: list[str]) -> list[dict[str, Any]]:
        """Batch score multiple customers. Returns list of score dicts."""
        return [self.score_customer(cid) for cid in customer_ids]

    # ------------------------------------------------------------------
    # Scoring internals
    # ------------------------------------------------------------------

    def _compute_risk_score(
        self,
        community_size: int,
        degree: int,
        shared_attribute_count: int,
    ) -> float:
        """Compute a composite risk score in [0.0, 1.0].

        Each factor is normalised to 0-1 using a cap, then weighted.
        """
        cs_norm = min(community_size / _SCORE_CAPS["community_size"], 1.0)
        deg_norm = min(degree / _SCORE_CAPS["degree"], 1.0)
        sa_norm = min(shared_attribute_count / _SCORE_CAPS["shared_attributes"], 1.0)

        score = (
            _SCORE_WEIGHTS["community_size"] * cs_norm
            + _SCORE_WEIGHTS["degree"] * deg_norm
            + _SCORE_WEIGHTS["shared_attributes"] * sa_norm
        )
        return min(max(score, 0.0), 1.0)

    def _count_shared_attributes(self, customer_id: str) -> int:
        """Count distinct neighbors connected via shared-attribute edges."""
        try:
            rows = self._db.query(
                f"MATCH (c:{self._node_label} {{{self._pk_col}: $cid}})"
                f"-[r:{self._edge_label}]-(other) "
                f"RETURN count(DISTINCT other) AS cnt",
                {"cid": customer_id},
            )
            if rows:
                return int(rows[0].get("cnt", 0))
        except Exception:
            pass
        return 0

    def _assess_risk(
        self,
        community_size: int,
        degree: int,
        pagerank: float,
        shared_attribute_count: int = 0,
    ) -> tuple[str, list[str]]:
        """Determine risk level and human-readable factor descriptions."""
        factors: list[str] = []
        if community_size > 50:
            factors.append(f"Large fraud community ({community_size} members)")
        elif community_size > 5:
            factors.append(f"Moderate fraud community ({community_size} members)")
        if degree > 5:
            factors.append(f"High connectivity ({degree} similar customers)")
        elif degree > 2:
            factors.append(f"Moderate connectivity ({degree} similar customers)")
        if pagerank > 0.01:
            factors.append(f"High centrality (PageRank {pagerank:.4f})")
        if shared_attribute_count > 3:
            factors.append(
                f"Many shared attributes ({shared_attribute_count} connected neighbors)"
            )

        for level in ("critical", "high", "medium"):
            cfg = self._risk_config.get(level, {})
            if (
                community_size >= cfg.get("min_community_size", 999)
                and degree >= cfg.get("min_degree", 999)
            ):
                return level, factors

        return "low", factors

    def _empty_score(self, customer_id: str) -> dict[str, Any]:
        """Return a zero-risk score dict."""
        return {
            "customer_id": customer_id,
            "risk_score": 0.0,
            "risk_level": "low",
            "factors": [],
            "community_id": None,
            "community_size": 0,
            "degree": 0,
            "shared_attribute_count": 0,
            "pagerank": 0.0,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }

    def _find_pk(self, label: str) -> str:
        """Discover the primary key column for a node table."""
        try:
            rows = self._db.query(f"CALL table_info('{label}') RETURN *")
            for r in rows:
                if r.get("isPrimaryKey") or r.get("is_primary_key"):
                    return str(r.get("name", "id"))
        except Exception:
            pass
        return "id"
