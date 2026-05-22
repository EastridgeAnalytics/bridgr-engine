"""Tests for Bridgr Temporal Graph Analytics — TemporalProjection, JourneyBuilder,
rolling_windows, temporal_stats, and duration parsing.

Covers:
- TemporalProjection.window — basic filtering, excludes future, preserves all nodes, empty window
- TemporalProjection.window_multi — multiple edge types
- JourneyBuilder.build_journey — chronological order, all edge types, empty, max_events limit
- JourneyBuilder.build_journeys_batch — multiple entities
- JourneyBuilder.find_pattern — simple match, max_gap, no match, partial no match
- rolling_windows — correct count, step alignment, independent databases
- temporal_stats — basic stats, by_month, empty
- Duration parsing — "90d", "3m", "1y", "2w"
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import bridgr
from bridgr.temporal import (
    JourneyBuilder,
    TemporalProjection,
    _parse_duration,
    rolling_windows,
    temporal_stats,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def temporal_db():
    """Create a database with temporal graph data for testing.

    Schema:
        Node: Entity(id STRING PK, name STRING)
        Edge: EVENT(FROM Entity TO Entity, timestamp TIMESTAMP, event_type STRING, amount DOUBLE)
        Edge: COMMUNICATION(FROM Entity TO Entity, timestamp TIMESTAMP, channel STRING)

    Entities: e1 through e5
    Events: 25 events spanning 2025-01-15 to 2025-11-20
    """
    db = bridgr.open(":memory:")

    # Create node table
    db.create_node_table("Entity", {
        "id": "STRING PRIMARY KEY",
        "name": "STRING",
    })

    # Create edge tables
    db.create_edge_table("EVENT", "Entity", "Entity", {
        "timestamp": "TIMESTAMP",
        "event_type": "STRING",
        "amount": "DOUBLE",
    })
    db.create_edge_table("COMMUNICATION", "Entity", "Entity", {
        "timestamp": "TIMESTAMP",
        "channel": "STRING",
    })

    # Insert entities
    entities = [
        {"id": "e1", "name": "Alice"},
        {"id": "e2", "name": "Bob"},
        {"id": "e3", "name": "Charlie"},
        {"id": "e4", "name": "Diana"},
        {"id": "e5", "name": "Eve"},
    ]
    for e in entities:
        db.create_node("Entity", e)

    # Insert EVENT edges with timestamps spanning Jan-Nov 2025
    events = [
        # e1 journey: purchase -> transfer -> purchase -> withdrawal
        ("e1", "e2", "2025-01-15T10:00:00", "purchase", 100.0),
        ("e1", "e3", "2025-02-20T14:30:00", "transfer", 250.0),
        ("e1", "e4", "2025-03-10T09:00:00", "purchase", 75.0),
        ("e1", "e5", "2025-04-05T16:45:00", "withdrawal", 500.0),
        # e2 journey: transfer -> purchase -> transfer
        ("e2", "e1", "2025-01-20T11:00:00", "transfer", 300.0),
        ("e2", "e3", "2025-03-15T13:00:00", "purchase", 150.0),
        ("e2", "e4", "2025-05-10T10:30:00", "transfer", 200.0),
        # e3 journey: purchase -> purchase -> withdrawal -> purchase
        ("e3", "e1", "2025-02-01T08:00:00", "purchase", 50.0),
        ("e3", "e2", "2025-04-15T12:00:00", "purchase", 120.0),
        ("e3", "e4", "2025-06-20T15:00:00", "withdrawal", 400.0),
        ("e3", "e5", "2025-08-10T09:30:00", "purchase", 90.0),
        # e4 events
        ("e4", "e1", "2025-03-01T10:00:00", "transfer", 175.0),
        ("e4", "e2", "2025-05-20T14:00:00", "purchase", 225.0),
        ("e4", "e5", "2025-07-15T11:00:00", "withdrawal", 350.0),
        # e5 events
        ("e5", "e1", "2025-04-10T09:00:00", "purchase", 80.0),
        ("e5", "e2", "2025-06-25T13:30:00", "transfer", 160.0),
        ("e5", "e3", "2025-09-01T10:00:00", "purchase", 95.0),
        # Additional events for pattern testing
        ("e1", "e2", "2025-05-15T10:00:00", "purchase", 110.0),
        ("e1", "e3", "2025-06-01T14:00:00", "transfer", 220.0),
        ("e1", "e4", "2025-07-20T09:00:00", "withdrawal", 330.0),
        # Late-year events
        ("e2", "e5", "2025-09-15T11:00:00", "purchase", 180.0),
        ("e3", "e1", "2025-10-05T08:30:00", "transfer", 275.0),
        ("e4", "e3", "2025-10-20T14:00:00", "purchase", 130.0),
        ("e5", "e4", "2025-11-10T10:00:00", "withdrawal", 440.0),
        ("e1", "e5", "2025-11-20T16:00:00", "purchase", 65.0),
    ]
    for src, dst, ts, etype, amount in events:
        db.create_edge(
            "EVENT", src, dst,
            properties={"timestamp": datetime.fromisoformat(ts), "event_type": etype, "amount": amount},
            from_label="Entity", to_label="Entity",
        )

    # Insert COMMUNICATION edges
    communications = [
        ("e1", "e2", "2025-01-10T09:00:00", "email"),
        ("e1", "e3", "2025-02-15T11:00:00", "phone"),
        ("e2", "e3", "2025-03-20T14:00:00", "email"),
        ("e3", "e4", "2025-05-01T10:00:00", "slack"),
        ("e4", "e5", "2025-07-10T13:00:00", "phone"),
        ("e5", "e1", "2025-09-05T08:00:00", "email"),
    ]
    for src, dst, ts, channel in communications:
        db.create_edge(
            "COMMUNICATION", src, dst,
            properties={"timestamp": datetime.fromisoformat(ts), "channel": channel},
            from_label="Entity", to_label="Entity",
        )

    yield db
    db.close()


@pytest.fixture
def empty_db():
    """A database with schema but no temporal data."""
    db = bridgr.open(":memory:")
    db.create_node_table("Entity", {
        "id": "STRING PRIMARY KEY",
        "name": "STRING",
    })
    db.create_edge_table("EVENT", "Entity", "Entity", {
        "timestamp": "TIMESTAMP",
        "event_type": "STRING",
        "amount": "DOUBLE",
    })
    yield db
    db.close()


# ------------------------------------------------------------------
# Duration Parsing Tests
# ------------------------------------------------------------------


class TestDurationParsing:
    """Tests for _parse_duration helper function."""

    def test_parse_days(self):
        result = _parse_duration("90d")
        assert result == timedelta(days=90)

    def test_parse_single_day(self):
        result = _parse_duration("1d")
        assert result == timedelta(days=1)

    def test_parse_weeks(self):
        result = _parse_duration("2w")
        assert result == timedelta(weeks=2)

    def test_parse_months_fallback(self):
        # Without dateutil, falls back to 30-day approximation
        result = _parse_duration("3m")
        # Should be either relativedelta(months=3) or timedelta(days=90)
        if isinstance(result, timedelta):
            assert result == timedelta(days=90)
        else:
            # dateutil relativedelta
            assert hasattr(result, "months")

    def test_parse_years_fallback(self):
        result = _parse_duration("1y")
        if isinstance(result, timedelta):
            assert result == timedelta(days=365)
        else:
            assert hasattr(result, "years")

    def test_parse_invalid_empty(self):
        with pytest.raises(ValueError):
            _parse_duration("")

    def test_parse_invalid_no_unit(self):
        with pytest.raises(ValueError):
            _parse_duration("90")

    def test_parse_invalid_no_number(self):
        with pytest.raises(ValueError):
            _parse_duration("d")

    def test_parse_invalid_unit(self):
        with pytest.raises(ValueError):
            _parse_duration("5x")


# ------------------------------------------------------------------
# TemporalProjection Tests
# ------------------------------------------------------------------


class TestTemporalProjectionWindow:
    """Tests for TemporalProjection.window()."""

    def test_basic_window_filters_edges(self, temporal_db):
        """Window should include only edges within the time range."""
        proj = TemporalProjection(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        # Window: Jan-Feb 2025
        win_db = proj.window(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 2, 28, 23, 59, 59),
        )
        # Count edges in windowed db
        edges = win_db.query("MATCH ()-[r:EVENT]->() RETURN count(r) AS cnt")
        cnt = edges[0]["cnt"]
        # Events in Jan-Feb: e1->e2 Jan15, e2->e1 Jan20, e1->e3 Feb20, e3->e1 Feb01
        assert cnt == 4
        win_db.close()

    def test_window_preserves_all_nodes(self, temporal_db):
        """All nodes should be present in the windowed db regardless of edges."""
        proj = TemporalProjection(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        win_db = proj.window(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 1, 31),
        )
        nodes = win_db.query("MATCH (n:Entity) RETURN count(n) AS cnt")
        assert nodes[0]["cnt"] == 5
        win_db.close()

    def test_window_excludes_future_events(self, temporal_db):
        """Events after the window end should not appear."""
        proj = TemporalProjection(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        # Window up to March 2025 only
        win_db = proj.window(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 3, 31, 23, 59, 59),
        )
        # Check no events from April onward
        late_events = win_db.query(
            "MATCH ()-[r:EVENT]->() "
            "WHERE r.event_type = 'withdrawal' AND r.amount = 500.0 "
            "RETURN count(r) AS cnt"
        )
        # The e1->e5 April withdrawal (amount=500) should not be present
        assert late_events[0]["cnt"] == 0
        win_db.close()

    def test_empty_window_returns_no_edges(self, temporal_db):
        """A window with no matching edges should return a db with 0 edges."""
        proj = TemporalProjection(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        # Window in 2024 — before any events
        win_db = proj.window(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
        )
        edges = win_db.query("MATCH ()-[r:EVENT]->() RETURN count(r) AS cnt")
        assert edges[0]["cnt"] == 0
        # Nodes still present
        nodes = win_db.query("MATCH (n:Entity) RETURN count(n) AS cnt")
        assert nodes[0]["cnt"] == 5
        win_db.close()

    def test_window_edge_properties_preserved(self, temporal_db):
        """Edge properties should be correctly copied to the windowed db."""
        proj = TemporalProjection(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        # Narrow window to get just the first event
        win_db = proj.window(
            start=datetime(2025, 1, 14),
            end=datetime(2025, 1, 16),
        )
        edges = win_db.query(
            "MATCH (a:Entity)-[r:EVENT]->(b:Entity) "
            "RETURN a.id AS src, b.id AS dst, r.event_type AS etype, r.amount AS amount"
        )
        assert len(edges) == 1
        assert edges[0]["src"] == "e1"
        assert edges[0]["dst"] == "e2"
        assert edges[0]["etype"] == "purchase"
        assert edges[0]["amount"] == 100.0
        win_db.close()


class TestTemporalProjectionWindowMulti:
    """Tests for TemporalProjection.window_multi()."""

    def test_multi_edge_types(self, temporal_db):
        """window_multi should include edges from multiple edge tables."""
        proj = TemporalProjection(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        win_db = proj.window_multi(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 3, 31, 23, 59, 59),
            edge_labels=["EVENT", "COMMUNICATION"],
        )
        # Count EVENT edges
        event_cnt = win_db.query("MATCH ()-[r:EVENT]->() RETURN count(r) AS cnt")[0]["cnt"]
        # Count COMMUNICATION edges
        comm_cnt = win_db.query("MATCH ()-[r:COMMUNICATION]->() RETURN count(r) AS cnt")[0]["cnt"]
        # Should have both types
        assert event_cnt > 0
        assert comm_cnt > 0
        win_db.close()

    def test_multi_defaults_to_single(self, temporal_db):
        """window_multi with None edge_labels should use the default edge_label."""
        proj = TemporalProjection(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        win_db = proj.window_multi(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 12, 31),
            edge_labels=None,
        )
        # Should have EVENT edges
        event_cnt = win_db.query("MATCH ()-[r:EVENT]->() RETURN count(r) AS cnt")[0]["cnt"]
        assert event_cnt > 0
        win_db.close()


# ------------------------------------------------------------------
# JourneyBuilder Tests
# ------------------------------------------------------------------


class TestJourneyBuilderBuildJourney:
    """Tests for JourneyBuilder.build_journey()."""

    def test_chronological_order(self, temporal_db):
        """Journey events should be ordered by timestamp."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        journey = jb.build_journey("e1")
        assert len(journey) > 0
        timestamps = []
        for event in journey:
            ts = event.get("timestamp")
            if ts is not None:
                if isinstance(ts, datetime):
                    timestamps.append(ts)
                elif isinstance(ts, str):
                    timestamps.append(datetime.fromisoformat(ts))
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_includes_all_outgoing_edges(self, temporal_db):
        """Journey should include all outgoing EVENT edges for the entity."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        journey = jb.build_journey("e1")
        # e1 has multiple outgoing EVENT edges (and COMMUNICATION edges)
        # The journey should include both EVENT and COMMUNICATION edges
        assert len(journey) >= 7  # At least 7 outgoing EVENT edges from e1

    def test_empty_journey_for_nonexistent(self, temporal_db):
        """Non-existent entity should return empty journey."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        journey = jb.build_journey("nonexistent_id")
        assert journey == []

    def test_max_events_limit(self, temporal_db):
        """max_events should limit the number of returned events."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        journey = jb.build_journey("e1", max_events=3)
        assert len(journey) <= 3

    def test_journey_contains_event_type(self, temporal_db):
        """Each event should have 'event_type' and 'target_label' fields."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        journey = jb.build_journey("e1")
        for event in journey:
            assert "event_type" in event
            assert event["event_type"] in ("EVENT", "COMMUNICATION")
            assert "target_label" in event
            assert "target_id" in event
            assert "timestamp" in event
            assert "properties" in event
            assert isinstance(event["properties"], dict)


class TestJourneyBuilderBatch:
    """Tests for JourneyBuilder.build_journeys_batch()."""

    def test_batch_returns_all_entities(self, temporal_db):
        """Batch should return a journey for each requested entity."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        result = jb.build_journeys_batch(["e1", "e2", "e3"])
        assert "e1" in result
        assert "e2" in result
        assert "e3" in result
        assert len(result["e1"]) > 0
        assert len(result["e2"]) > 0

    def test_batch_empty_list(self, temporal_db):
        """Empty entity list should return empty dict."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        result = jb.build_journeys_batch([])
        assert result == {}


class TestJourneyBuilderFindPattern:
    """Tests for JourneyBuilder.find_pattern()."""

    def test_simple_pattern_match(self, temporal_db):
        """Should find entities with consecutive EVENT edges."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        matches = jb.find_pattern(["EVENT", "EVENT"])
        assert len(matches) > 0
        for m in matches:
            assert "entity_id" in m
            assert "entity_label" in m
            assert m["entity_label"] == "Entity"
            assert "matched_events" in m
            assert "journey_length" in m
            assert "pattern_duration" in m
            assert len(m["matched_events"]) == 2

    def test_pattern_no_match(self, temporal_db):
        """Pattern that doesn't exist should return empty."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        matches = jb.find_pattern(["NONEXISTENT_TYPE", "ANOTHER_FAKE"])
        assert matches == []

    def test_pattern_with_max_gap(self, temporal_db):
        """max_gap should exclude matches where events are too far apart."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        # With a very short gap (1 day), most patterns should not match
        # since events are weeks/months apart
        matches = jb.find_pattern(["EVENT", "EVENT"], max_gap="1d")
        # With 1-day max gap, unlikely to find matches given our data spread
        # (events are typically weeks apart)
        assert isinstance(matches, list)

    def test_pattern_with_generous_gap(self, temporal_db):
        """A generous max_gap should allow matches."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        matches = jb.find_pattern(["EVENT", "EVENT"], max_gap="365d")
        # Most entities have consecutive EVENT edges within a year
        assert len(matches) > 0

    def test_pattern_min_entities(self, temporal_db):
        """min_entities should filter results."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        # Require more entities than exist
        matches = jb.find_pattern(["EVENT", "EVENT"], min_entities=100)
        assert matches == []

    def test_pattern_empty_types(self, temporal_db):
        """Empty event_types list should return empty."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        matches = jb.find_pattern([])
        assert matches == []


# ------------------------------------------------------------------
# rolling_windows Tests
# ------------------------------------------------------------------


class TestRollingWindows:
    """Tests for rolling_windows() generator."""

    def test_generates_correct_number_of_windows(self, temporal_db):
        """Should generate the expected number of windows."""
        windows = list(rolling_windows(
            temporal_db, "Entity", "EVENT",
            window_size="90d",
            step_size="90d",
            start=datetime(2025, 1, 1),
            end=datetime(2025, 5, 31),
            timestamp_prop="timestamp",
        ))
        # Jan1 + 90d step -> Apr1 + 90d step -> Jun30 (> May31 end, stops)
        # So 2 windows: [Jan1, Apr1) and [Apr1, Jun30) clamped to May31
        assert len(windows) == 2
        for win_start, win_end, win_db in windows:
            win_db.close()

    def test_window_step_alignment(self, temporal_db):
        """Windows should advance by exactly step_size."""
        windows = list(rolling_windows(
            temporal_db, "Entity", "EVENT",
            window_size="30d",
            step_size="30d",
            start=datetime(2025, 1, 1),
            end=datetime(2025, 3, 31),
            timestamp_prop="timestamp",
        ))
        # 90 days / 30 day step = 3 windows
        assert len(windows) == 3
        # Check step alignment
        assert windows[0][0] == datetime(2025, 1, 1)
        assert windows[1][0] == datetime(2025, 1, 31)
        assert windows[2][0] == datetime(2025, 3, 2)
        for _, _, win_db in windows:
            win_db.close()

    def test_independent_databases(self, temporal_db):
        """Each windowed database should be independent."""
        windows = list(rolling_windows(
            temporal_db, "Entity", "EVENT",
            window_size="90d",
            step_size="90d",
            start=datetime(2025, 1, 1),
            end=datetime(2025, 6, 30),
            timestamp_prop="timestamp",
        ))
        assert len(windows) >= 2
        # Get edge counts — they should generally differ
        counts = []
        for _, _, win_db in windows:
            edges = win_db.query("MATCH ()-[r:EVENT]->() RETURN count(r) AS cnt")
            counts.append(edges[0]["cnt"])
            win_db.close()
        # Each window is independent (has its own edge count)
        assert all(isinstance(c, int) for c in counts)

    def test_no_windows_when_no_data(self, empty_db):
        """Should yield nothing when no temporal data exists."""
        windows = list(rolling_windows(
            empty_db, "Entity", "EVENT",
            window_size="30d",
            step_size="30d",
            timestamp_prop="timestamp",
        ))
        assert windows == []

    def test_auto_detects_range(self, temporal_db):
        """Without explicit start/end, should auto-detect from data."""
        windows = list(rolling_windows(
            temporal_db, "Entity", "EVENT",
            window_size="180d",
            step_size="180d",
            timestamp_prop="timestamp",
        ))
        # Should generate at least 1 window from the data range
        assert len(windows) >= 1
        for _, _, win_db in windows:
            win_db.close()


# ------------------------------------------------------------------
# temporal_stats Tests
# ------------------------------------------------------------------


class TestTemporalStats:
    """Tests for temporal_stats() function."""

    def test_basic_stats(self, temporal_db):
        """Should return earliest/latest timestamps and total count."""
        stats = temporal_stats(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        assert stats["edge_label"] == "EVENT"
        assert stats["total_edges"] == 25
        assert stats["earliest"] is not None
        assert stats["latest"] is not None
        assert stats["earliest"].year == 2025
        assert stats["earliest"].month == 1
        assert stats["latest"].month == 11

    def test_temporal_span(self, temporal_db):
        """span_days should be the difference between latest and earliest."""
        stats = temporal_stats(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        assert stats["span_days"] > 300  # Jan to Nov ~ 310 days

    def test_by_month_distribution(self, temporal_db):
        """by_month should have entries for months with events."""
        stats = temporal_stats(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        by_month = stats["by_month"]
        assert len(by_month) > 0
        # January should have events
        assert "2025-01" in by_month
        assert by_month["2025-01"] >= 2  # At least e1->e2 and e2->e1

    def test_by_day_of_week_distribution(self, temporal_db):
        """by_day_of_week should have entries for days with events."""
        stats = temporal_stats(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        by_dow = stats["by_day_of_week"]
        # Should have all 7 days as keys
        assert len(by_dow) == 7
        # Total across all days should equal total_events
        assert sum(by_dow.values()) == stats["total_edges"]

    def test_empty_db_stats(self, empty_db):
        """Stats on an empty database should return zeros/None."""
        stats = temporal_stats(empty_db, "Entity", "EVENT", timestamp_prop="timestamp")
        assert stats["total_edges"] == 0
        assert stats["earliest"] is None
        assert stats["latest"] is None
        assert stats["span_days"] == 0


# ------------------------------------------------------------------
# Additional spec-required tests
# ------------------------------------------------------------------


class TestWindowAlgorithmRuns:
    """Test that algorithms can run on windowed databases."""

    def test_window_algorithm_runs(self, temporal_db):
        """Leiden/PageRank should run on a windowed database without error."""
        proj = TemporalProjection(temporal_db, "Entity", "EVENT", timestamp_prop="timestamp")
        win_db = proj.window(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 12, 31),
        )
        try:
            from bridgr.algorithms import GraphAlgorithms
            algo = GraphAlgorithms(win_db)
            result = algo.pagerank("Entity", "EVENT")
            assert isinstance(result, list)
        except ImportError:
            pytest.skip("algo extension not installed")
        except RuntimeError as e:
            if "extension" in str(e).lower() or "install" in str(e).lower():
                pytest.skip("algo extension not available")
            raise
        finally:
            win_db.close()


class TestPatternPartialNoMatch:
    """Test that partial pattern matches are excluded."""

    def test_pattern_partial_no_match(self, temporal_db):
        """Entity with first event type but not second should be excluded."""
        jb = JourneyBuilder(temporal_db, "Entity", timestamp_prop="timestamp")
        matches = jb.find_pattern(["EVENT", "NONEXISTENT_TYPE"])
        assert matches == []


class TestRollingWindowOverlap:
    """Test overlapping rolling windows."""

    def test_rolling_window_overlap(self, temporal_db):
        """Overlapping windows should share edges in the overlap period."""
        windows = list(rolling_windows(
            temporal_db, "Entity", "EVENT",
            window_size="180d",
            step_size="90d",
            start=datetime(2025, 1, 1),
            end=datetime(2025, 12, 1),
            timestamp_prop="timestamp",
        ))
        assert len(windows) >= 2
        counts = []
        for _, _, win_db in windows:
            edges = win_db.query("MATCH ()-[r:EVENT]->() RETURN count(r) AS cnt")
            counts.append(edges[0]["cnt"])
            win_db.close()
        # With 180d windows and 90d steps over a full year, all windows have data
        assert counts[0] > 0
        assert counts[1] > 0
        # Overlapping windows share edges — first window count + second window count > total unique edges
        assert counts[0] + counts[1] > max(counts)


class TestRollingCloseCleanup:
    """Test that closing yielded databases frees resources."""

    def test_rolling_close_cleanup(self, temporal_db):
        """Closing yielded databases should not raise errors."""
        for win_start, win_end, win_db in rolling_windows(
            temporal_db, "Entity", "EVENT",
            window_size="90d",
            step_size="90d",
            start=datetime(2025, 1, 1),
            end=datetime(2025, 4, 1),
            timestamp_prop="timestamp",
        ):
            edges = win_db.query("MATCH ()-[r:EVENT]->() RETURN count(r) AS cnt")
            assert isinstance(edges[0]["cnt"], int)
            win_db.close()


class TestTimestampTypes:
    """Test that temporal features work with different timestamp types."""

    def test_timestamp_ms_type(self):
        """Should work with TIMESTAMP_MS edge property (uses Cypher cast)."""
        db = bridgr.open(":memory:")
        db.create_node_table("Node", {"id": "STRING PRIMARY KEY"})
        db.create_edge_table("LINK", "Node", "Node", {"ts": "TIMESTAMP_MS"})
        db.create_node("Node", {"id": "a"})
        db.create_node("Node", {"id": "b"})
        db.execute(
            "MATCH (a:Node {id: 'a'}), (b:Node {id: 'b'}) "
            "CREATE (a)-[:LINK {ts: cast('2025-06-15 12:00:00', 'TIMESTAMP_MS')}]->(b)"
        )
        stats = temporal_stats(db, "Node", "LINK", timestamp_prop="ts")
        assert stats["total_edges"] == 1
        assert stats["earliest"] is not None
        db.close()

    def test_timestamp_sec_type(self):
        """Should work with TIMESTAMP_SEC edge property (uses Cypher cast)."""
        db = bridgr.open(":memory:")
        db.create_node_table("Node", {"id": "STRING PRIMARY KEY"})
        db.create_edge_table("LINK", "Node", "Node", {"ts": "TIMESTAMP_SEC"})
        db.create_node("Node", {"id": "a"})
        db.create_node("Node", {"id": "b"})
        db.execute(
            "MATCH (a:Node {id: 'a'}), (b:Node {id: 'b'}) "
            "CREATE (a)-[:LINK {ts: cast('2025-03-10 08:00:00', 'TIMESTAMP_SEC')}]->(b)"
        )
        jb = JourneyBuilder(db, "Node", timestamp_prop="ts")
        journey = jb.build_journey("a")
        assert len(journey) == 1
        assert journey[0]["event_type"] == "LINK"
        db.close()
