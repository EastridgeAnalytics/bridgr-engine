"""Bridgr Temporal Graph Analytics — time-windowed projections, journeys, and rolling analysis.

Provides pure-Python temporal graph analytics on top of the existing Bridgr
Database and Cypher query interface. No engine modifications required.

Usage:
    from bridgr import Database
    from bridgr.temporal import TemporalProjection, JourneyBuilder, rolling_windows, temporal_stats

    db = Database(":memory:")
    # ... populate with timestamped edges ...

    # Time-windowed projection
    proj = TemporalProjection(db, "Entity", "EVENT", timestamp_prop="timestamp")
    windowed_db = proj.window(start=datetime(2025, 1, 1), end=datetime(2025, 6, 30))

    # Journey analysis
    jb = JourneyBuilder(db, "Entity", timestamp_prop="timestamp")
    journey = jb.build_journey("entity_1")

    # Rolling windows
    for win_start, win_end, win_db in rolling_windows(db, "Entity", "EVENT", "90d", "30d"):
        # Analyze each window
        pass

    # Summary statistics
    stats = temporal_stats(db, "Entity", "EVENT", timestamp_prop="timestamp")
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Generator

from bridgr.database import Database


# ------------------------------------------------------------------
# Duration parsing
# ------------------------------------------------------------------

def _parse_duration(s: str) -> timedelta:
    """Parse a duration string into a timedelta.

    Supported formats:
        "Nd" — N days (e.g., "90d", "1d")
        "Nw" — N weeks (e.g., "2w")
        "Nm" — N months (approximated as N * 30 days)
        "Ny" — N years (approximated as N * 365 days)

    For precise month/year arithmetic, install python-dateutil.
    This function tries dateutil's relativedelta first and falls back
    to timedelta approximation.

    Args:
        s: Duration string like "90d", "3m", "1y", "2w".

    Returns:
        A timedelta representing the duration.

    Raises:
        ValueError: If the format is not recognized.
    """
    s = s.strip().lower()
    if not s:
        raise ValueError("Empty duration string")

    # Extract numeric prefix and unit suffix
    i = 0
    while i < len(s) and (s[i].isdigit() or s[i] == '.'):
        i += 1
    if i == 0 or i == len(s):
        raise ValueError(f"Invalid duration format: '{s}'. Expected format like '90d', '3m', '1y', '2w'.")

    num_str = s[:i]
    unit = s[i:]

    try:
        num = int(num_str) if '.' not in num_str else float(num_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value in duration: '{num_str}'")

    if unit == "d":
        return timedelta(days=num)
    elif unit == "w":
        return timedelta(weeks=num)
    elif unit == "m":
        # Try dateutil for precise month arithmetic
        try:
            from dateutil.relativedelta import relativedelta
            # Return a relativedelta; callers handle both types
            return relativedelta(months=int(num))  # type: ignore[return-value]
        except ImportError:
            return timedelta(days=int(num * 30))
    elif unit == "y":
        try:
            from dateutil.relativedelta import relativedelta
            return relativedelta(years=int(num))  # type: ignore[return-value]
        except ImportError:
            return timedelta(days=int(num * 365))
    else:
        raise ValueError(f"Unknown duration unit: '{unit}'. Supported: d, w, m, y.")


def _add_duration(dt: datetime, duration) -> datetime:
    """Add a duration (timedelta or relativedelta) to a datetime."""
    return dt + duration


# ------------------------------------------------------------------
# TemporalProjection
# ------------------------------------------------------------------

class TemporalProjection:
    """Create time-windowed graph projections from a source database.

    Copies all nodes and only the edges whose timestamp falls within
    the specified time window into a new in-memory database.

    Args:
        db: Source Bridgr Database.
        node_label: Label of the node table to project.
        edge_label: Label of the edge table with temporal data.
        timestamp_prop: Name of the timestamp property on edges.
    """

    def __init__(
        self,
        db: Database,
        node_label: str,
        edge_label: str,
        timestamp_prop: str = "timestamp",
    ):
        self._db = db
        self._node_label = node_label
        self._edge_label = edge_label
        self._timestamp_prop = timestamp_prop

    def window(self, start: datetime, end: datetime) -> Database:
        """Create a temporal projection with edges filtered to [start, end).

        All nodes are preserved. Only edges with timestamp >= start and
        timestamp < end are included (half-open interval).

        Args:
            start: Window start (inclusive).
            end: Window end (exclusive).

        Returns:
            A new in-memory Database containing the projected subgraph.
        """
        return self._project(start, end, [self._edge_label])

    def window_multi(
        self,
        start: datetime,
        end: datetime,
        edge_labels: list[str] | None = None,
    ) -> Database:
        """Create a temporal projection with multiple edge types.

        Args:
            start: Window start (inclusive).
            end: Window end (exclusive).
            edge_labels: List of edge labels to include. If None, uses the
                default edge_label from __init__.

        Returns:
            A new in-memory Database containing the projected subgraph.
        """
        labels = edge_labels if edge_labels else [self._edge_label]
        return self._project(start, end, labels)

    def _project(
        self, start: datetime, end: datetime, edge_labels: list[str]
    ) -> Database:
        """Internal: build the projected database."""
        temp_db = Database(":memory:")

        # Step 1: Get node table schema and create in temp db
        node_schema = self._get_node_schema(self._node_label)
        temp_db.create_node_table(self._node_label, node_schema)

        # Step 2: Copy all nodes
        self._copy_nodes(temp_db, self._node_label, node_schema)

        # Step 3: For each edge label, get schema, create table, copy filtered edges
        for edge_label in edge_labels:
            src_label, dst_label = self._get_edge_endpoints(edge_label)
            edge_schema = self._get_edge_schema(edge_label)
            temp_db.create_edge_table(
                edge_label, src_label, dst_label, edge_schema if edge_schema else None
            )
            self._copy_edges_filtered(
                temp_db, edge_label, src_label, dst_label, edge_schema, start, end
            )

        return temp_db

    def _get_node_schema(self, label: str) -> dict[str, str]:
        """Get node table schema as {col_name: cypher_type_str}."""
        rows = self._db.query(f"CALL table_info('{label}') RETURN *")
        schema: dict[str, str] = {}
        for r in rows:
            name = r.get("name", "")
            col_type = r.get("type", "STRING")
            is_pk = (
                r.get("primary key")
                or r.get("isPrimaryKey")
                or r.get("is_primary_key")
            )
            if is_pk:
                schema[name] = f"{col_type} PRIMARY KEY"
            else:
                schema[name] = col_type
        return schema

    def _get_edge_schema(self, label: str) -> dict[str, str]:
        """Get edge table property schema (excludes internal columns)."""
        rows = self._db.query(f"CALL table_info('{label}') RETURN *")
        schema: dict[str, str] = {}
        for r in rows:
            name = r.get("name", "")
            col_type = r.get("type", "STRING")
            # Skip internal edge columns (src/dst references)
            if name.lower() in ("_src", "_dst", "src", "dst"):
                continue
            schema[name] = col_type
        return schema

    def _get_edge_endpoints(self, edge_label: str) -> tuple[str, str]:
        """Get source and destination node labels for an edge table."""
        rows = self._db.query(f"CALL show_connection('{edge_label}') RETURN *")
        if rows:
            src = rows[0].get("source table name", self._node_label)
            dst = rows[0].get("destination table name", self._node_label)
            return src, dst
        return self._node_label, self._node_label

    def _copy_nodes(
        self, temp_db: Database, label: str, schema: dict[str, str]
    ) -> None:
        """Copy all nodes from source to temp database."""
        # Get property names (strip PRIMARY KEY annotation)
        prop_names = list(schema.keys())

        nodes = self._db.query(f"MATCH (n:{label}) RETURN n.*")
        for node in nodes:
            props: dict[str, Any] = {}
            for pname in prop_names:
                # query returns columns as "n.propname"
                val = node.get(f"n.{pname}", node.get(pname))
                if val is not None:
                    props[pname] = val
            if props:
                temp_db.create_node(label, props)

    def _copy_edges_filtered(
        self,
        temp_db: Database,
        edge_label: str,
        src_label: str,
        dst_label: str,
        edge_schema: dict[str, str],
        start: datetime,
        end: datetime,
    ) -> None:
        """Copy edges that fall within the time window."""
        ts_prop = self._timestamp_prop

        # Format timestamps for Cypher comparison
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%S")

        # Build property return columns
        prop_names = list(edge_schema.keys())
        prop_return = ", ".join(f"r.{p} AS {p}" for p in prop_names) if prop_names else ""

        # Query edges within the time window
        cypher = (
            f"MATCH (a:{src_label})-[r:{edge_label}]->(b:{dst_label}) "
            f"WHERE r.{ts_prop} >= timestamp('{start_str}') "
            f"AND r.{ts_prop} < timestamp('{end_str}') "
            f"RETURN a.id AS src_id, b.id AS dst_id"
        )
        if prop_return:
            cypher = (
                f"MATCH (a:{src_label})-[r:{edge_label}]->(b:{dst_label}) "
                f"WHERE r.{ts_prop} >= timestamp('{start_str}') "
                f"AND r.{ts_prop} < timestamp('{end_str}') "
                f"RETURN a.id AS src_id, b.id AS dst_id, {prop_return}"
            )

        edges = self._db.query(cypher)
        for edge in edges:
            src_id = edge["src_id"]
            dst_id = edge["dst_id"]
            props: dict[str, Any] = {}
            for pname in prop_names:
                val = edge.get(pname)
                if val is not None:
                    props[pname] = val
            temp_db.create_edge(
                edge_label,
                src_id,
                dst_id,
                properties=props if props else None,
                from_label=src_label,
                to_label=dst_label,
            )


# ------------------------------------------------------------------
# JourneyBuilder
# ------------------------------------------------------------------

class JourneyBuilder:
    """Build temporal event journeys for entities.

    Extracts ordered sequences of events (outgoing edges) for a given
    entity, enabling pattern analysis and sequence mining.

    Args:
        db: Source Bridgr Database.
        entity_label: Label of the entity node table.
        timestamp_prop: Name of the timestamp property on edges.
    """

    def __init__(
        self,
        db: Database,
        entity_label: str,
        timestamp_prop: str = "timestamp",
    ):
        self._db = db
        self._entity_label = entity_label
        self._timestamp_prop = timestamp_prop

    def build_journey(
        self, entity_id: str, max_events: int = 1000
    ) -> list[dict[str, Any]]:
        """Build a chronological event journey for a single entity.

        Retrieves all outgoing edges of the entity, ordered by timestamp.

        Args:
            entity_id: Primary key of the entity node.
            max_events: Maximum number of events to return.

        Returns:
            List of event dicts with edge properties, ordered by timestamp.
            Each dict includes: type (edge label), target_id, timestamp,
            and all other edge properties.
        """
        ts_prop = self._timestamp_prop
        cypher = (
            f"MATCH (e:{self._entity_label} {{id: $eid}})-[r]->(t) "
            f"RETURN label(r) AS edge_type, t.id AS target_id, label(t) AS target_label, r.* "
            f"ORDER BY r.{ts_prop} "
            f"LIMIT {max_events}"
        )
        rows = self._db.query(cypher, {"eid": entity_id})

        events = []
        for row in rows:
            props: dict[str, Any] = {}
            ts_val = None
            for key, val in row.items():
                if key.startswith("r.") and not key.startswith("r._"):
                    clean_key = key[2:]
                    if clean_key == ts_prop:
                        ts_val = val
                    else:
                        props[clean_key] = val
            event: dict[str, Any] = {
                "event_type": row.get("edge_type", ""),
                "target_id": row.get("target_id", ""),
                "target_label": row.get("target_label", ""),
                "timestamp": ts_val,
                "properties": props,
            }
            events.append(event)

        return events

    def build_journeys_batch(
        self, entity_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Build journeys for multiple entities.

        Args:
            entity_ids: List of entity primary keys.

        Returns:
            Dict mapping entity_id to its journey (list of events).
        """
        return {eid: self.build_journey(eid) for eid in entity_ids}

    def find_pattern(
        self,
        event_types: list[str],
        max_gap: str | None = None,
        min_entities: int = 1,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Find entities whose journeys contain a sequence of event types.

        Scans entity journeys for subsequences matching the given
        event_types in order. Optionally enforces a maximum time gap
        between consecutive events.

        Args:
            event_types: Ordered list of event type labels to match.
            max_gap: Optional maximum time gap between consecutive events
                (e.g., "7d", "1m"). If None, no gap constraint.
            min_entities: Minimum number of entities that must match.
            limit: Maximum number of entities to scan (default 1000).

        Returns:
            List of match dicts, each with entity_id, entity_label,
            matched_events, journey_length, and pattern_duration.
        """
        if not event_types:
            return []

        gap_delta = _parse_duration(max_gap) if max_gap else None

        entities = self._db.query(
            f"MATCH (e:{self._entity_label}) RETURN e.id AS id LIMIT {limit}"
        )

        matches: list[dict[str, Any]] = []

        for entity_row in entities:
            eid = entity_row["id"]
            journey = self.build_journey(eid)

            if not journey:
                continue

            matched_events = self._find_subsequence(journey, event_types, gap_delta)
            if matched_events:
                first_ts = _parse_timestamp(matched_events[0].get("timestamp"))
                last_ts = _parse_timestamp(matched_events[-1].get("timestamp"))
                if first_ts and last_ts:
                    duration = str(last_ts - first_ts)
                else:
                    duration = "unknown"

                matches.append({
                    "entity_id": eid,
                    "entity_label": self._entity_label,
                    "matched_events": matched_events,
                    "journey_length": len(journey),
                    "pattern_duration": duration,
                })

        if len(matches) < min_entities:
            return []

        return matches

    def _find_subsequence(
        self,
        journey: list[dict[str, Any]],
        event_types: list[str],
        gap_delta: timedelta | None,
    ) -> list[dict[str, Any]] | None:
        """Find the first subsequence matching event_types in the journey."""
        if not journey or not event_types:
            return None

        ts_prop = self._timestamp_prop
        pattern_idx = 0
        matched: list[dict[str, Any]] = []

        for event in journey:
            etype = event.get("event_type", "")
            if etype == event_types[pattern_idx]:
                # Check gap constraint
                if gap_delta is not None and matched:
                    prev_ts = _parse_timestamp(matched[-1].get("timestamp"))
                    curr_ts = _parse_timestamp(event.get("timestamp"))
                    if prev_ts and curr_ts:
                        gap = curr_ts - prev_ts
                        # Convert gap_delta to timedelta for comparison
                        if isinstance(gap_delta, timedelta):
                            if gap > gap_delta:
                                # Gap too large, restart
                                pattern_idx = 0
                                matched = []
                                if etype == event_types[0]:
                                    matched.append(event)
                                    pattern_idx = 1
                                continue
                        else:
                            # relativedelta: approximate comparison
                            try:
                                max_dt = prev_ts + gap_delta
                                if curr_ts > max_dt:
                                    pattern_idx = 0
                                    matched = []
                                    if etype == event_types[0]:
                                        matched.append(event)
                                        pattern_idx = 1
                                    continue
                            except (TypeError, OverflowError):
                                pass

                matched.append(event)
                pattern_idx += 1

                if pattern_idx == len(event_types):
                    return matched
            else:
                # If we haven't started matching yet, just skip
                # If we have partial match and this doesn't match, check if
                # it matches the first pattern element (restart)
                if pattern_idx > 0:
                    pattern_idx = 0
                    matched = []
                    # Check if current event starts a new match
                    if etype == event_types[0]:
                        matched.append(event)
                        pattern_idx = 1

        return None



# ------------------------------------------------------------------
# rolling_windows
# ------------------------------------------------------------------

def rolling_windows(
    db: Database,
    node_label: str,
    edge_label: str,
    window_size: str,
    step_size: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    timestamp_prop: str = "timestamp",
) -> Generator[tuple[datetime, datetime, Database], None, None]:
    """Generate rolling time-windowed database projections.

    Creates a series of temporal projections by sliding a window of
    `window_size` across the temporal range, advancing by `step_size`.

    Args:
        db: Source Bridgr Database.
        node_label: Label of the node table.
        edge_label: Label of the edge table with temporal data.
        window_size: Size of each window (e.g., "90d", "3m", "1y").
        step_size: Step between window starts (e.g., "30d", "1m").
        start: Optional explicit start datetime. If None, uses the
            earliest timestamp in the data.
        end: Optional explicit end datetime. If None, uses the latest
            timestamp in the data.
        timestamp_prop: Name of the timestamp property on edges.

    Yields:
        Tuples of (window_start, window_end, Database) for each window.
    """
    win_dur = _parse_duration(window_size)
    step_dur = _parse_duration(step_size)

    # Determine temporal range from data if not specified
    if start is None or end is None:
        stats = temporal_stats(db, node_label, edge_label, timestamp_prop=timestamp_prop)
        if start is None:
            start = stats.get("earliest")
        if end is None:
            end = stats.get("latest")

    if start is None or end is None:
        return  # No temporal data

    proj = TemporalProjection(db, node_label, edge_label, timestamp_prop=timestamp_prop)

    current_start = start
    while True:
        current_end = _add_duration(current_start, win_dur)
        if current_start > end:
            break

        # Clamp window end to data end
        effective_end = min(current_end, end) if isinstance(current_end, datetime) else current_end

        windowed_db = proj.window(current_start, effective_end)
        yield current_start, effective_end, windowed_db

        current_start = _add_duration(current_start, step_dur)


# ------------------------------------------------------------------
# temporal_stats
# ------------------------------------------------------------------

# Note: node_label is required for the MATCH pattern but was not in the original spec.
def temporal_stats(
    db: Database,
    node_label: str,
    edge_label: str,
    *,
    timestamp_prop: str = "timestamp",
) -> dict[str, Any]:
    """Compute summary statistics about temporal data in the graph.

    Args:
        db: Source Bridgr Database.
        node_label: Label of the node table.
        edge_label: Label of the edge table with temporal data.
        timestamp_prop: Name of the timestamp property on edges.

    Returns:
        Dict containing:
            - earliest: Earliest timestamp (datetime or None).
            - latest: Latest timestamp (datetime or None).
            - total_edges: Total count of temporal edges.
            - edge_label: The edge label queried.
            - by_month: Dict mapping "YYYY-MM" to count.
            - by_day_of_week: Dict mapping day name to count.
            - span_days: Number of days between earliest and latest.
    """
    result: dict[str, Any] = {
        "edge_label": edge_label,
        "total_edges": 0,
        "earliest": None,
        "latest": None,
        "span_days": 0,
        "by_month": {},
        "by_day_of_week": {},
    }

    # Get min/max/count
    try:
        rows = db.query(
            f"MATCH (:{node_label})-[r:{edge_label}]->() "
            f"RETURN min(r.{timestamp_prop}) AS min_ts, "
            f"max(r.{timestamp_prop}) AS max_ts, "
            f"count(r) AS total"
        )
    except RuntimeError:
        return result

    if not rows:
        return result

    row = rows[0]
    min_ts = _parse_timestamp(row.get("min_ts"))
    max_ts = _parse_timestamp(row.get("max_ts"))
    total = row.get("total", 0)

    result["earliest"] = min_ts
    result["latest"] = max_ts
    result["total_edges"] = total

    if min_ts and max_ts:
        result["span_days"] = (max_ts - min_ts).days

    # Get all timestamps for distribution analysis
    try:
        ts_rows = db.query(
            f"MATCH (:{node_label})-[r:{edge_label}]->() "
            f"RETURN r.{timestamp_prop} AS ts"
        )
    except RuntimeError:
        return result

    by_month: dict[str, int] = {}
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_dow: dict[str, int] = {d: 0 for d in day_names}

    for ts_row in ts_rows:
        ts = _parse_timestamp(ts_row.get("ts"))
        if ts:
            month_key = ts.strftime("%Y-%m")
            by_month[month_key] = by_month.get(month_key, 0) + 1
            by_dow[day_names[ts.weekday()]] += 1

    result["by_month"] = by_month
    result["by_day_of_week"] = by_dow

    return result


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse a timestamp value from the engine into a Python datetime.

    Handles datetime objects, ISO-8601 strings, and numeric epoch seconds.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        for fmt in (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except (OSError, OverflowError, ValueError):
            pass
    return None
