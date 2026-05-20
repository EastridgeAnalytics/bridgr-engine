"""Tests for the streaming pipeline: IncrementalWriter, FraudScorer,
AlgorithmTrigger, AlertEngine.

Covers E1 (IncrementalWriter), E3 (FraudScorer), E4 (AlgorithmTrigger),
E5 (AlertEngine) from the H-E-B Streaming Pipeline spec.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import bridgr
from bridgr import Database
from bridgr.streaming import IncrementalWriter
from bridgr.scoring import FraudScorer
from bridgr.triggers import AlgorithmTrigger
from bridgr.alerts import AlertEngine, AlertRule, Alert


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def db():
    """Create an in-memory DB with Customer + SIMILAR_TO schema."""
    d = bridgr.open(":memory:")
    d.create_node_table(
        "Customer",
        {
            "id": "STRING PRIMARY KEY",
            "name": "STRING",
            "zip": "STRING",
            "phone": "STRING",
            "risk_score": "DOUBLE",
            "community_id": "INT64",
            "pagerank": "DOUBLE",
            "component_id": "INT64",
            "updated_at": "STRING",
        },
    )
    d.create_edge_table(
        "SIMILAR_TO",
        "Customer",
        "Customer",
        {"weight": "INT64", "updated_at": "STRING"},
    )
    yield d
    d.close()


@pytest.fixture
def populated_db(db):
    """DB with a small fraud community for scoring/alert tests."""
    # Community 1: C001-C004 (connected cluster)
    for cid, name, zip_code, comm in [
        ("C001", "Alice", "78701", 1),
        ("C002", "Bob", "78701", 1),
        ("C003", "Charlie", "78702", 1),
        ("C004", "Diana", "78702", 1),
    ]:
        db.execute(
            "CREATE (:Customer {id: $id, name: $name, zip: $zip, "
            "community_id: $comm, pagerank: $pr})",
            {"id": cid, "name": name, "zip": zip_code, "comm": comm, "pr": 0.005},
        )

    # Isolated node
    db.execute(
        "CREATE (:Customer {id: $id, name: $name, zip: $zip})",
        {"id": "C005", "name": "Eve", "zip": "90210"},
    )

    # Edges within community
    for src, dst in [("C001", "C002"), ("C001", "C003"), ("C002", "C004"), ("C003", "C004")]:
        db.execute(
            "MATCH (a:Customer {id: $src}), (b:Customer {id: $dst}) "
            "CREATE (a)-[:SIMILAR_TO {weight: 1}]->(b)",
            {"src": src, "dst": dst},
        )
    return db


@pytest.fixture
def tx_db(populated_db):
    """DB with a Transaction node table linked to customers."""
    populated_db.create_node_table(
        "Transaction",
        {
            "id": "STRING PRIMARY KEY",
            "amount": "DOUBLE",
            "updated_at": "STRING",
        },
    )
    populated_db.create_edge_table("ORIGINATED", "Transaction", "Customer")
    populated_db.execute(
        "CREATE (:Transaction {id: $id, amount: $amt})",
        {"id": "TX-001", "amt": 500.00},
    )
    populated_db.execute(
        "MATCH (t:Transaction {id: 'TX-001'}), (c:Customer {id: 'C001'}) "
        "CREATE (t)-[:ORIGINATED]->(c)",
    )
    return populated_db


# ==================================================================
# E1: IncrementalWriter
# ==================================================================


class TestIncrementalWriterUpsertNode:
    """Test single-node upsert operations."""

    def test_upsert_node_creates(self, db):
        writer = IncrementalWriter(db)
        result = writer.upsert_node("Customer", "NEW01", {"name": "NewUser"})
        assert result["action"] == "created"
        assert result["id"] == "NEW01"

        node = db.get_node("NEW01", "Customer")
        assert node is not None
        assert node["name"] == "NewUser"

    def test_upsert_node_updates(self, db):
        writer = IncrementalWriter(db)
        writer.upsert_node("Customer", "U01", {"name": "Original"})
        result = writer.upsert_node("Customer", "U01", {"name": "Updated"})
        assert result["action"] == "updated"
        assert result["id"] == "U01"

        node = db.get_node("U01", "Customer")
        assert node["name"] == "Updated"

    def test_upsert_node_sets_updated_at(self, db):
        writer = IncrementalWriter(db)
        writer.upsert_node("Customer", "TS01", {"name": "Timestamped"})
        node = db.get_node("TS01", "Customer")
        assert node["updated_at"] is not None
        assert len(node["updated_at"]) > 10  # ISO timestamp

    def test_upsert_node_dict_form(self, db):
        """Legacy calling convention: pass full dict as second arg."""
        writer = IncrementalWriter(db)
        result = writer.upsert_node("Customer", {"id": "D01", "name": "DictForm"})
        assert result["action"] == "created"
        assert result["id"] == "D01"

    def test_upsert_node_missing_pk_raises(self, db):
        writer = IncrementalWriter(db)
        with pytest.raises(ValueError, match="Missing primary key"):
            writer.upsert_node("Customer", {"name": "NoPK"})

    def test_upsert_node_tracks_write_count(self, db):
        writer = IncrementalWriter(db)
        writer.upsert_node("Customer", "W01", {"name": "A"})
        writer.upsert_node("Customer", "W02", {"name": "B"})
        assert writer.write_counts["Customer"] == 2


class TestIncrementalWriterUpsertEdge:
    """Test single-edge upsert operations."""

    def test_upsert_edge_creates(self, populated_db):
        writer = IncrementalWriter(populated_db)
        result = writer.upsert_edge(
            "SIMILAR_TO", "C001", "C005",
            {"weight": 2},
            from_label="Customer", to_label="Customer",
        )
        assert result["action"] == "created"

    def test_upsert_edge_updates(self, populated_db):
        writer = IncrementalWriter(populated_db)
        # C001->C002 edge already exists from fixture
        result = writer.upsert_edge(
            "SIMILAR_TO", "C001", "C002",
            {"weight": 99},
            from_label="Customer", to_label="Customer",
        )
        assert result["action"] == "updated"

    def test_upsert_edge_auto_resolves_labels(self, populated_db):
        writer = IncrementalWriter(populated_db)
        # Should auto-detect Customer-Customer from SIMILAR_TO edge table
        result = writer.upsert_edge("SIMILAR_TO", "C001", "C005", {"weight": 1})
        assert result["action"] == "created"


class TestIncrementalWriterBatch:
    """Test batch upsert operations."""

    def test_batch_upsert_nodes_creates(self, db):
        writer = IncrementalWriter(db)
        nodes = [
            {"id": "B01", "name": "Batch1"},
            {"id": "B02", "name": "Batch2"},
            {"id": "B03", "name": "Batch3"},
        ]
        result = writer.batch_upsert_nodes("Customer", nodes)
        assert result["created"] == 3
        assert result["updated"] == 0
        assert result["elapsed_ms"] >= 0
        assert isinstance(result["errors"], list)

    def test_batch_upsert_nodes_mixed(self, db):
        writer = IncrementalWriter(db)
        writer.upsert_node("Customer", "M01", {"name": "Existing"})
        nodes = [
            {"id": "M01", "name": "UpdatedName"},
            {"id": "M02", "name": "NewNode"},
        ]
        result = writer.batch_upsert_nodes("Customer", nodes)
        assert result["created"] == 1
        assert result["updated"] == 1

    def test_batch_upsert_edges(self, populated_db):
        writer = IncrementalWriter(populated_db)
        edges = [
            {"from_id": "C001", "to_id": "C005", "weight": 1},
            {"from_id": "C002", "to_id": "C005", "weight": 2},
        ]
        result = writer.batch_upsert_edges(
            "SIMILAR_TO", edges,
            from_label="Customer", to_label="Customer",
        )
        assert result["created"] == 2
        assert result["updated"] == 0
        assert result["elapsed_ms"] >= 0

    def test_batch_upsert_nodes_empty_list(self, db):
        writer = IncrementalWriter(db)
        result = writer.batch_upsert_nodes("Customer", [])
        assert result["created"] == 0
        assert result["updated"] == 0


class TestIncrementalWriterTriggerIntegration:
    """Test IncrementalWriter integration with AlgorithmTrigger."""

    def test_trigger_notified_on_write(self, db):
        notified = []

        class FakeTrigger:
            def notify_write(self, count=1):
                notified.append(count)

        writer = IncrementalWriter(db, trigger=FakeTrigger())
        writer.upsert_node("Customer", "T01", {"name": "Trigger"})
        assert len(notified) == 1


# ==================================================================
# E3: FraudScorer
# ==================================================================


class TestFraudScorerCustomer:
    """Test single-customer scoring."""

    def test_score_connected_customer(self, populated_db):
        scorer = FraudScorer(populated_db)
        result = scorer.score_customer("C001")

        assert result["customer_id"] == "C001"
        assert "risk_score" in result
        assert 0.0 <= result["risk_score"] <= 1.0
        assert isinstance(result["factors"], list)
        assert "risk_level" in result
        assert "community_id" in result
        assert "degree" in result
        assert "scored_at" in result

    def test_score_isolated_customer(self, populated_db):
        scorer = FraudScorer(populated_db)
        result = scorer.score_customer("C005")

        assert result["customer_id"] == "C005"
        assert result["risk_score"] == 0.0
        assert result["degree"] == 0

    def test_score_nonexistent_customer(self, populated_db):
        scorer = FraudScorer(populated_db)
        result = scorer.score_customer("DOES_NOT_EXIST")

        assert result["customer_id"] == "DOES_NOT_EXIST"
        assert result["risk_score"] == 0.0
        assert result["risk_level"] == "low"

    def test_score_has_community_size(self, populated_db):
        scorer = FraudScorer(populated_db)
        result = scorer.score_customer("C001")
        # C001 is in community 1 with 4 members
        assert result["community_size"] == 4

    def test_higher_degree_higher_score(self, populated_db):
        """Node with more edges should have a higher risk score."""
        scorer = FraudScorer(populated_db)
        # C001 has edges to C002, C003 (degree 2)
        # C005 has no edges (degree 0)
        score_c001 = scorer.score_customer("C001")
        score_c005 = scorer.score_customer("C005")
        assert score_c001["risk_score"] > score_c005["risk_score"]


class TestFraudScorerTransaction:
    """Test transaction scoring."""

    def test_score_transaction_with_customer(self, tx_db):
        scorer = FraudScorer(tx_db)
        result = scorer.score_transaction("TX-001")

        assert result["tx_id"] == "TX-001"
        assert "risk_score" in result
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_score_transaction_no_match(self, tx_db):
        scorer = FraudScorer(tx_db)
        result = scorer.score_transaction("TX-NONEXISTENT")

        assert result["tx_id"] == "TX-NONEXISTENT"
        assert result["risk_score"] == 0.0


class TestFraudScorerBatch:
    """Test batch scoring."""

    def test_batch_score_returns_list(self, populated_db):
        scorer = FraudScorer(populated_db)
        results = scorer.batch_score(["C001", "C005"])

        assert len(results) == 2
        assert results[0]["customer_id"] == "C001"
        assert results[1]["customer_id"] == "C005"
        for r in results:
            assert 0.0 <= r["risk_score"] <= 1.0

    def test_batch_score_empty_list(self, populated_db):
        scorer = FraudScorer(populated_db)
        results = scorer.batch_score([])
        assert results == []


# ==================================================================
# E4: AlgorithmTrigger
# ==================================================================


class TestAlgorithmTriggerSimple:
    """Test simple (single-algorithm + threshold) mode."""

    def test_writes_below_threshold_no_trigger(self, populated_db):
        trigger = AlgorithmTrigger(
            populated_db, algorithm="louvain", threshold=10
        )
        result = trigger.notify_write(5)
        assert result is None
        assert trigger.writes_since_last_run == 5

    def test_writes_at_threshold_triggers(self, populated_db):
        trigger = AlgorithmTrigger(
            populated_db, algorithm="louvain", threshold=3
        )
        trigger.notify_write(2)
        assert trigger.writes_since_last_run == 2

        result = trigger.notify_write(1)
        assert result is not None
        assert result["algorithm"] == "louvain"
        assert "results" in result
        assert "count" in result
        assert "elapsed_ms" in result
        # write counter should reset
        assert trigger.writes_since_last_run == 0

    def test_force_run(self, populated_db):
        trigger = AlgorithmTrigger(
            populated_db, algorithm="louvain", threshold=1000
        )
        trigger.notify_write(5)
        result = trigger.force_run()
        assert result["algorithm"] == "louvain"
        assert isinstance(result["results"], list)
        assert trigger.writes_since_last_run == 0

    def test_callback_invoked(self, populated_db):
        callback_results = []

        def on_complete(results):
            callback_results.append(len(results))

        trigger = AlgorithmTrigger(
            populated_db,
            algorithm="louvain",
            threshold=1,
            callback=on_complete,
        )
        trigger.notify_write(1)
        assert len(callback_results) == 1
        assert callback_results[0] > 0


class TestAlgorithmTriggerAdvanced:
    """Test register-based (multi-algorithm) mode."""

    def test_register_and_check(self, populated_db):
        trigger = AlgorithmTrigger(populated_db)
        trigger.register(
            "louvain", "Customer", "SIMILAR_TO", every_n_writes=5
        )
        # Write 4 — not enough
        for _ in range(4):
            trigger.notify_write(1)
        executed = trigger.check_and_run()
        assert executed == []

        # Write 1 more — total 5, should trigger
        trigger.notify_write(1)
        executed = trigger.check_and_run()
        assert "louvain" in executed

    def test_writes_since_last_run_property(self, db):
        trigger = AlgorithmTrigger(db, algorithm="louvain", threshold=100)
        assert trigger.writes_since_last_run == 0
        trigger.notify_write(7)
        assert trigger.writes_since_last_run == 7
        trigger.notify_write(3)
        assert trigger.writes_since_last_run == 10


# ==================================================================
# E5: AlertEngine
# ==================================================================


class TestAlertEngineAddRule:
    """Test rule registration."""

    def test_add_rule_dataclass(self, db):
        engine = AlertEngine(db)
        rule = AlertRule(
            name="test_rule",
            cypher="MATCH (n:Customer) RETURN n.id LIMIT 1",
            severity="high",
        )
        engine.add_rule(rule)
        # Should not raise
        assert len(engine._rules) == 1
        assert engine._rules[0].name == "test_rule"
        assert engine._rules[0].cypher == "MATCH (n:Customer) RETURN n.id LIMIT 1"

    def test_add_rule_kwargs(self, db):
        engine = AlertEngine(db)
        engine.add_rule(
            "test_rule",
            query="MATCH (n:Customer) RETURN n.id LIMIT 1",
            severity="high",
        )
        assert engine._rules[0].name == "test_rule"

    def test_add_rule_cypher_kwarg(self, db):
        engine = AlertEngine(db)
        engine.add_rule(
            "test_rule",
            cypher="MATCH (n:Customer) RETURN n.id LIMIT 1",
        )
        assert engine._rules[0].cypher == "MATCH (n:Customer) RETURN n.id LIMIT 1"

    def test_add_rule_no_query_raises(self, db):
        engine = AlertEngine(db)
        with pytest.raises(ValueError, match="query"):
            engine.add_rule("bad_rule")


class TestAlertEngineCheckAll:
    """Test check_all() evaluation."""

    def test_check_all_fires_matching(self, populated_db):
        engine = AlertEngine(populated_db)
        engine.add_rule(AlertRule(
            name="has_customers",
            cypher="MATCH (c:Customer) RETURN c.id LIMIT 1",
            severity="low",
        ))
        results = engine.check_all()
        assert len(results) == 1
        assert results[0]["rule_name"] == "has_customers"
        assert results[0]["fired"] is True
        assert results[0]["matches"] >= 1

    def test_check_all_no_match(self, db):
        engine = AlertEngine(db)
        engine.add_rule(AlertRule(
            name="no_match",
            cypher="MATCH (c:Customer {id: 'NONEXISTENT'}) RETURN c.id",
        ))
        results = engine.check_all()
        assert len(results) == 1
        assert results[0]["fired"] is False
        assert results[0]["matches"] == 0

    def test_check_all_respects_cooldown(self, populated_db):
        engine = AlertEngine(populated_db)
        engine.add_rule(AlertRule(
            name="cooldown_test",
            cypher="MATCH (c:Customer) RETURN c.id LIMIT 1",
            cooldown_seconds=3600,
        ))
        # First check fires
        results1 = engine.check_all()
        assert results1[0]["fired"] is True

        # Second check within cooldown should not fire
        results2 = engine.check_all()
        assert results2[0]["fired"] is False

    def test_check_all_multiple_rules(self, populated_db):
        engine = AlertEngine(populated_db)
        engine.add_rule(AlertRule(
            name="rule_a",
            cypher="MATCH (c:Customer) RETURN c.id LIMIT 1",
        ))
        engine.add_rule(AlertRule(
            name="rule_b",
            cypher="MATCH (c:Customer {id: 'NOPE'}) RETURN c.id",
        ))
        results = engine.check_all()
        assert len(results) == 2
        fired_names = {r["rule_name"] for r in results if r["fired"]}
        assert "rule_a" in fired_names
        assert "rule_b" not in fired_names


class TestAlertEngineCheckRule:
    """Test check_rule() single-rule evaluation."""

    def test_check_rule_fires(self, populated_db):
        engine = AlertEngine(populated_db)
        engine.add_rule(AlertRule(
            name="single",
            cypher="MATCH (c:Customer) RETURN c.id LIMIT 1",
        ))
        result = engine.check_rule("single")
        assert result is not None
        assert result["fired"] is True

    def test_check_rule_cooldown_returns_none(self, populated_db):
        engine = AlertEngine(populated_db)
        engine.add_rule(AlertRule(
            name="cd",
            cypher="MATCH (c:Customer) RETURN c.id LIMIT 1",
            cooldown_seconds=3600,
        ))
        engine.check_rule("cd")  # fires
        result = engine.check_rule("cd")  # cooldown
        assert result is None

    def test_check_rule_unknown_raises(self, db):
        engine = AlertEngine(db)
        with pytest.raises(ValueError, match="No rule named"):
            engine.check_rule("nonexistent")


class TestAlertEngineHandler:
    """Test rule-specific and global handlers."""

    def test_rule_handler_invoked(self, populated_db):
        handler_calls = []

        def on_alert(alert_dict):
            handler_calls.append(alert_dict)

        engine = AlertEngine(populated_db)
        engine.add_rule(AlertRule(
            name="handled",
            cypher="MATCH (c:Customer) RETURN c.id LIMIT 1",
            handler=on_alert,
        ))
        engine.check_all()
        assert len(handler_calls) == 1
        assert handler_calls[0]["rule_name"] == "handled"
        assert handler_calls[0]["matches"] >= 1

    def test_global_handler_invoked(self, populated_db):
        global_calls = []

        def on_alert(alert):
            global_calls.append(alert)

        engine = AlertEngine(populated_db)
        engine.register_handler(on_alert)
        engine.add_rule(AlertRule(
            name="global_test",
            cypher="MATCH (c:Customer) RETURN c.id LIMIT 1",
        ))
        engine.check_all()
        assert len(global_calls) == 1
        assert isinstance(global_calls[0], Alert)


class TestAlertEngineLegacy:
    """Test evaluate_all() backward compatibility."""

    def test_evaluate_all_returns_alerts(self, populated_db):
        engine = AlertEngine(populated_db)
        engine.add_rule(AlertRule(
            name="legacy",
            cypher="MATCH (c:Customer) RETURN c.id LIMIT 1",
        ))
        alerts = engine.evaluate_all()
        assert len(alerts) == 1
        assert isinstance(alerts[0], Alert)
        assert alerts[0].rule_name == "legacy"
        assert len(alerts[0].rows) >= 1


# ==================================================================
# Integration: Writer + Trigger + Scorer + Alerts
# ==================================================================


class TestEndToEndPipeline:
    """Integration test: write data, trigger algorithm, score, alert."""

    def test_write_then_score(self, db):
        writer = IncrementalWriter(db)
        writer.upsert_node("Customer", "INT01", {"name": "Integration"})
        writer.upsert_node("Customer", "INT02", {"name": "Test"})
        writer.upsert_edge(
            "SIMILAR_TO", "INT01", "INT02",
            {"weight": 1},
            from_label="Customer", to_label="Customer",
        )

        scorer = FraudScorer(db)
        result = scorer.score_customer("INT01")
        assert result["customer_id"] == "INT01"
        assert result["degree"] >= 1
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_writer_with_trigger_and_alert(self, db):
        alert_log = []

        def on_alert(alert_dict):
            alert_log.append(alert_dict)

        trigger = AlgorithmTrigger(
            db, algorithm="louvain", threshold=5,
            node_label="Customer", edge_label="SIMILAR_TO",
        )
        writer = IncrementalWriter(db, trigger=trigger)

        # Add some connected nodes
        for i in range(6):
            writer.upsert_node("Customer", f"P{i:03d}", {"name": f"Person{i}"})
        for i in range(5):
            writer.upsert_edge(
                "SIMILAR_TO", f"P{i:03d}", f"P{i+1:03d}",
                from_label="Customer", to_label="Customer",
            )

        # Set up alert
        engine = AlertEngine(db)
        engine.add_rule(AlertRule(
            name="any_customers",
            cypher="MATCH (c:Customer) RETURN c.id LIMIT 1",
            handler=on_alert,
        ))
        results = engine.check_all()
        assert any(r["fired"] for r in results)


# ==================================================================
# Exports
# ==================================================================


class TestExports:
    """Verify all streaming classes are importable from bridgr."""

    def test_incremental_writer_importable(self):
        from bridgr import IncrementalWriter
        assert IncrementalWriter is not None

    def test_fraud_scorer_importable(self):
        from bridgr import FraudScorer
        assert FraudScorer is not None

    def test_algorithm_trigger_importable(self):
        from bridgr import AlgorithmTrigger
        assert AlgorithmTrigger is not None

    def test_alert_engine_importable(self):
        from bridgr import AlertEngine
        assert AlertEngine is not None

    def test_alert_rule_importable(self):
        from bridgr import AlertRule
        assert AlertRule is not None

    def test_alert_importable(self):
        from bridgr import Alert
        assert Alert is not None
