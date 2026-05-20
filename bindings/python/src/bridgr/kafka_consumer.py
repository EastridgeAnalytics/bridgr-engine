"""Kafka consumer for near real-time graph updates.

Consumes JSON messages from a Kafka topic and writes to the Bridgr graph
via IncrementalWriter. Batches writes for throughput.

Usage:
    from bridgr.kafka_consumer import BridgrKafkaConsumer

    consumer = BridgrKafkaConsumer(
        db, bootstrap_servers="localhost:9092", topic="trust-squad-signals",
        schema_map={"visitor_id": ("Customer", "customer_id"), "signal_type": "risk_signal"},
    )
    consumer.start()

Requires: pip install bridgr[streaming]  (installs confluent-kafka)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from bridgr.database import Database
from bridgr.streaming import IncrementalWriter

log = logging.getLogger(__name__)

__all__ = ["BridgrKafkaConsumer"]


class BridgrKafkaConsumer:
    """Consume Kafka messages and write to a Bridgr graph."""

    def __init__(
        self,
        db: Database,
        bootstrap_servers: str,
        topic: str,
        *,
        group_id: str = "bridgr-consumer",
        schema_map: dict[str, Any] | None = None,
        node_label: str = "Customer",
        pk_field: str = "customer_id",
        flush_interval_ms: int = 1000,
        batch_size: int = 100,
    ):
        try:
            from confluent_kafka import Consumer
        except ImportError as e:
            raise ImportError(
                "confluent-kafka required. Install with: pip install bridgr[streaming]"
            ) from e

        self._db = db
        self._writer = IncrementalWriter(db)
        self._topic = topic
        self._node_label = node_label
        self._pk_field = pk_field
        self._schema_map = schema_map or {}
        self._flush_interval = flush_interval_ms / 1000.0
        self._batch_size = batch_size
        self._running = False
        self._thread: threading.Thread | None = None

        self._consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        })

        self._stats = {
            "messages_consumed": 0,
            "nodes_written": 0,
            "edges_written": 0,
            "errors": 0,
            "last_offset": -1,
        }

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def start(self) -> None:
        """Start consuming in a background thread."""
        self._consumer.subscribe([self._topic])
        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        log.info("Kafka consumer started: topic=%s", self._topic)

    def stop(self) -> None:
        """Gracefully stop the consumer with explicit offset commit."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        try:
            self._consumer.commit(asynchronous=False)
        except Exception:
            pass
        self._consumer.close()
        log.info("Kafka consumer stopped. Stats: %s", self._stats)

    def _consume_loop(self) -> None:
        batch: list[dict[str, Any]] = []
        last_flush = time.monotonic()

        while self._running:
            msg = self._consumer.poll(timeout=0.1)
            if msg is None:
                pass
            elif msg.error():
                self._stats["errors"] += 1
                log.warning("Kafka error: %s", msg.error())
            else:
                try:
                    value = json.loads(msg.value().decode("utf-8"))
                    batch.append(value)
                    self._stats["messages_consumed"] += 1
                    self._stats["last_offset"] = msg.offset()
                except Exception:
                    self._stats["errors"] += 1

            now = time.monotonic()
            if batch and (len(batch) >= self._batch_size or now - last_flush >= self._flush_interval):
                self._flush_batch(batch)
                batch = []
                last_flush = now

        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: list[dict[str, Any]]) -> None:
        records = []
        for msg in batch:
            try:
                props = self._apply_schema_map(msg)
                if self._pk_field in props:
                    records.append(props)
                else:
                    self._send_to_dlq(msg, f"Missing PK field '{self._pk_field}'")
            except Exception as e:
                self._send_to_dlq(msg, str(e))

        if records:
            result = self._writer.batch_upsert_nodes(self._node_label, records)
            self._stats["nodes_written"] += result["created"] + result["updated"]
            for err in result.get("errors", []):
                self._stats["errors"] += 1
                log.warning("Upsert error: %s", err)

    def _apply_schema_map(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Apply schema_map to transform message fields to node properties."""
        if not self._schema_map:
            return dict(msg)
        props: dict[str, Any] = {}
        for msg_field, target in self._schema_map.items():
            if msg_field not in msg:
                continue
            if isinstance(target, str):
                props[target] = msg[msg_field]
            elif isinstance(target, tuple) and len(target) == 2:
                # (node_label, property_name) — use property_name
                props[target[1]] = msg[msg_field]
            else:
                props[msg_field] = msg[msg_field]
        # Pass through unmapped fields
        for k, v in msg.items():
            if k not in self._schema_map and k not in props:
                props[k] = v
        return props

    def _send_to_dlq(self, msg: dict[str, Any], reason: str) -> None:
        """Write failed message to dead letter queue (local file)."""
        self._stats["errors"] += 1
        dlq_entry = {"message": msg, "reason": reason, "timestamp": time.time()}
        dlq_path = getattr(self, "_dlq_path", None)
        if dlq_path is None:
            import tempfile
            self._dlq_path = os.path.join(tempfile.gettempdir(), f"bridgr_dlq_{self._topic}.jsonl")
            dlq_path = self._dlq_path
        try:
            with open(dlq_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dlq_entry, default=str) + "\n")
        except Exception:
            pass
        log.warning("DLQ: %s — %s", reason, msg.get(self._pk_field, "unknown"))
