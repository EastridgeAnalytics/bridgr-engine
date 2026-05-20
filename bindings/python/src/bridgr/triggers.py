"""Triggered algorithm re-runs based on data change thresholds.

Automatically re-runs Louvain/PageRank/WCC when enough new data accumulates,
keeping fraud community assignments up to date.

Usage (simple — single algorithm with threshold):
    from bridgr.triggers import AlgorithmTrigger

    trigger = AlgorithmTrigger(db, algorithm="louvain", threshold=100)
    # ... after writes ...
    result = trigger.notify_write(5)  # returns None until threshold reached

Usage (advanced — multiple algorithms):
    trigger = AlgorithmTrigger(db)
    trigger.register("louvain", "Customer", "SIMILAR_TO", every_n_writes=1000)
    trigger.register("pagerank", "Customer", "SIMILAR_TO", every_seconds=300)
    trigger.start_background()
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bridgr.algorithms import GraphAlgorithms
from bridgr.database import Database

log = logging.getLogger(__name__)

__all__ = ["AlgorithmTrigger"]


@dataclass
class TriggerConfig:
    """Configuration for a single algorithm trigger."""

    algorithm: str
    node_label: str
    edge_label: str
    every_n_writes: int = 0
    every_seconds: int = 0
    callback: Callable[[list[dict[str, Any]]], None] | None = None
    _last_run: float = field(default_factory=time.monotonic)
    _writes_since_run: int = 0


class AlgorithmTrigger:
    """Triggers algorithm re-runs after incremental writes.

    Two usage modes:

    1. **Simple**: Pass algorithm and threshold to constructor. Call
       notify_write() after writes; it returns algorithm results when the
       threshold is reached.

    2. **Advanced**: Use register() to configure multiple triggers with
       different thresholds and time intervals, then call check_and_run()
       or start_background().
    """

    def __init__(
        self,
        db: Database,
        algorithm: str | None = None,
        threshold: int = 100,
        callback: Callable[[list[dict[str, Any]]], None] | None = None,
        *,
        node_label: str = "Customer",
        edge_label: str = "SIMILAR_TO",
    ):
        """Create an AlgorithmTrigger.

        Args:
            db: Database connection.
            algorithm: Algorithm to run when threshold reached (louvain,
                pagerank, wcc). If None, use register() to add triggers.
            threshold: Number of writes before triggering the algorithm.
            callback: Optional function called with algorithm results.
            node_label: Default node label for the simple-mode trigger.
            edge_label: Default edge label for the simple-mode trigger.
        """
        self._db = db
        self._triggers: list[TriggerConfig] = []
        self._algo = GraphAlgorithms(db)
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Simple mode: single algorithm + threshold
        self._write_count = 0
        self._simple_algorithm = algorithm
        self._simple_threshold = threshold
        self._simple_callback = callback

        if algorithm is not None:
            self.register(
                algorithm,
                node_label,
                edge_label,
                every_n_writes=threshold,
                callback=callback,
            )

    # ------------------------------------------------------------------
    # Simple API (spec E4)
    # ------------------------------------------------------------------

    def notify_write(self, count: int = 1) -> dict[str, Any] | None:
        """Called after writes. If threshold reached, runs algorithm and resets.

        In simple mode (algorithm passed to constructor), tracks a global
        write counter and triggers the algorithm when the threshold is met.

        In advanced mode (register-based), notifies all registered triggers
        and runs check_and_run().

        Args:
            count: Number of writes to record (default 1).

        Returns:
            Algorithm results dict if triggered, None otherwise.
            Result contains {algorithm, results, count, elapsed_ms}.
        """
        with self._lock:
            self._write_count += count
            # Also update per-trigger counters
            for t in self._triggers:
                t._writes_since_run += count

        if self._simple_algorithm and self._write_count >= self._simple_threshold:
            return self.force_run()

        # For non-simple triggers, check time-based ones
        executed = self._check_time_triggers()
        if executed:
            return executed[0]
        return None

    def force_run(self) -> dict[str, Any]:
        """Force an algorithm run regardless of write count.

        Returns:
            Dict with {algorithm, results, count, elapsed_ms}.
        """
        with self._lock:
            self._write_count = 0

        if self._simple_algorithm:
            # Run the simple-mode algorithm
            target = None
            for t in self._triggers:
                if t.algorithm == self._simple_algorithm:
                    target = t
                    break
            if target is None:
                raise ValueError(
                    f"No trigger registered for algorithm '{self._simple_algorithm}'"
                )
            start = time.perf_counter()
            results = self._run_algorithm(target)
            elapsed_ms = (time.perf_counter() - start) * 1000
            with self._lock:
                target._writes_since_run = 0
                target._last_run = time.monotonic()
            if self._simple_callback:
                self._simple_callback(results)
            return {
                "algorithm": self._simple_algorithm,
                "results": results,
                "count": len(results),
                "elapsed_ms": round(elapsed_ms, 2),
            }

        # No simple algorithm configured -- run all registered triggers
        all_results: dict[str, Any] = {"algorithms_run": []}
        for t in self._triggers:
            start = time.perf_counter()
            results = self._run_algorithm(t)
            elapsed = (time.perf_counter() - start) * 1000
            with self._lock:
                t._writes_since_run = 0
                t._last_run = time.monotonic()
            all_results["algorithms_run"].append({
                "algorithm": t.algorithm,
                "results": results,
                "count": len(results),
                "elapsed_ms": round(elapsed, 2),
            })
            if t.callback:
                t.callback(results)
        return all_results

    @property
    def writes_since_last_run(self) -> int:
        """Number of writes since the last algorithm run."""
        return self._write_count

    # ------------------------------------------------------------------
    # Advanced API (register + background)
    # ------------------------------------------------------------------

    def register(
        self,
        algorithm: str,
        node_label: str,
        edge_label: str,
        *,
        every_n_writes: int = 0,
        every_seconds: int = 0,
        callback: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        """Register a trigger for an algorithm.

        Args:
            algorithm: Algorithm name (louvain, pagerank, wcc).
            node_label: Node table label.
            edge_label: Edge table label.
            every_n_writes: Trigger after this many writes (0 = disabled).
            every_seconds: Trigger after this many seconds (0 = disabled).
            callback: Optional function called with results on trigger.
        """
        self._triggers.append(
            TriggerConfig(
                algorithm=algorithm,
                node_label=node_label,
                edge_label=edge_label,
                every_n_writes=every_n_writes,
                every_seconds=every_seconds,
                callback=callback,
            )
        )

    def check_and_run(self) -> list[str]:
        """Check all triggers, run algorithms that are due.

        Returns list of algorithm names that were executed.
        """
        executed: list[str] = []
        now = time.monotonic()

        with self._lock:
            for t in self._triggers:
                should_run = False
                if t.every_n_writes > 0 and t._writes_since_run >= t.every_n_writes:
                    should_run = True
                if t.every_seconds > 0 and (now - t._last_run) >= t.every_seconds:
                    should_run = True

                if should_run:
                    try:
                        results = self._run_algorithm(t)
                        t._last_run = now
                        t._writes_since_run = 0
                        executed.append(t.algorithm)
                        if t.callback:
                            t.callback(results)
                    except Exception as e:
                        log.warning("Trigger %s failed: %s", t.algorithm, e)

        return executed

    def start_background(self, check_interval: float = 5.0) -> None:
        """Start a background thread that calls check_and_run() periodically."""
        self._running = True

        def _loop() -> None:
            while self._running:
                self.check_and_run()
                time.sleep(check_interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_background(self) -> None:
        """Stop the background trigger loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_time_triggers(self) -> list[dict[str, Any]]:
        """Check only time-based triggers (for use inside notify_write)."""
        results: list[dict[str, Any]] = []
        now = time.monotonic()
        with self._lock:
            for t in self._triggers:
                if t.every_seconds > 0 and (now - t._last_run) >= t.every_seconds:
                    try:
                        algo_results = self._run_algorithm(t)
                        t._last_run = now
                        t._writes_since_run = 0
                        if t.callback:
                            t.callback(algo_results)
                        results.append({
                            "algorithm": t.algorithm,
                            "results": algo_results,
                            "count": len(algo_results),
                        })
                    except Exception as e:
                        log.warning("Trigger %s failed: %s", t.algorithm, e)
        return results

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

    def _run_algorithm(self, t: TriggerConfig) -> list[dict[str, Any]]:
        """Execute an algorithm and write results back to node properties."""
        if t.algorithm == "louvain":
            results = self._algo.louvain(t.node_label, t.edge_label)
        elif t.algorithm == "pagerank":
            results = self._algo.pagerank(t.node_label, t.edge_label)
        elif t.algorithm == "wcc":
            results = self._algo.weakly_connected_components(
                t.node_label, t.edge_label
            )
        else:
            raise ValueError(f"Unknown algorithm: {t.algorithm}")

        pk_col = self._find_pk(t.node_label)

        # Write results back to node properties
        for row in results:
            nid = row.get("node_id")
            if nid is None:
                continue
            try:
                if t.algorithm == "louvain":
                    self._db.execute(
                        f"MATCH (n:{t.node_label} {{{pk_col}: $nid}}) "
                        f"SET n.community_id = $val",
                        {"nid": nid, "val": row.get("community_id")},
                    )
                elif t.algorithm == "pagerank":
                    self._db.execute(
                        f"MATCH (n:{t.node_label} {{{pk_col}: $nid}}) "
                        f"SET n.pagerank = $val",
                        {"nid": nid, "val": row.get("score")},
                    )
                elif t.algorithm == "wcc":
                    self._db.execute(
                        f"MATCH (n:{t.node_label} {{{pk_col}: $nid}}) "
                        f"SET n.component_id = $val",
                        {"nid": nid, "val": row.get("component_id")},
                    )
            except Exception as e:
                log.debug("Writeback failed for %s: %s", nid, e)

        log.info("Trigger %s completed: %d results", t.algorithm, len(results))
        return results
