"""Rule-based alerting on graph state changes.

Evaluates Cypher queries as alert rules. When a query returns rows, the
alert fires. Cooldown prevents alert storms.

Usage:
    from bridgr.alerts import AlertEngine, AlertRule

    engine = AlertEngine(db)
    engine.add_rule(AlertRule(
        name="large_fraud_community",
        cypher="MATCH (c:Customer) WITH c.community_id AS cid, count(*) AS sz "
               "WHERE sz > 50 RETURN cid, sz",
        severity="critical",
        cooldown_seconds=3600,
    ))
    alerts = engine.check_all()
    # [{"rule_name": "large_fraud_community", "severity": "critical",
    #   "matches": 2, "fired": True}]
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bridgr.database import Database

log = logging.getLogger(__name__)

__all__ = ["Alert", "AlertRule", "AlertEngine"]


@dataclass
class Alert:
    """A fired alert with matched rows."""

    rule_name: str
    severity: str
    description: str
    rows: list[dict[str, Any]]
    fired_at: float


@dataclass
class AlertRule:
    """An alert rule definition.

    Attributes:
        name: Unique rule identifier.
        cypher: Cypher query. If it returns rows, the alert fires.
        severity: low, medium, high, or critical.
        cooldown_seconds: Minimum seconds between fires for this rule.
        handler: Optional callback invoked with the alert dict when fired.
        description: Human-readable description of what this rule detects.
    """

    name: str
    cypher: str
    severity: str = "medium"
    cooldown_seconds: int = 3600
    handler: Callable[[dict[str, Any]], None] | None = None
    description: str = ""


class AlertEngine:
    """Rule-based alerting on graph patterns.

    Rules are Cypher queries. When a query returns rows, the corresponding
    alert fires. Cooldown prevents the same rule from firing repeatedly.
    """

    def __init__(self, db: Database):
        self._db = db
        self._rules: list[AlertRule] = []
        self._last_fired: dict[str, float] = {}
        self._handlers: list[Callable[[Alert], None]] = []

    def add_rule(
        self,
        rule_or_name: AlertRule | str,
        query: str | None = None,
        severity: str = "medium",
        *,
        cypher: str | None = None,
        description: str = "",
        cooldown_seconds: int = 3600,
        handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Register an alert rule.

        Supports two calling conventions:

            # Pass an AlertRule dataclass
            engine.add_rule(AlertRule(name="...", cypher="..."))

            # Pass keyword arguments (backward compatible)
            engine.add_rule(name="...", query="MATCH ...", severity="high")

        Args:
            rule_or_name: An AlertRule instance, or the rule name (str).
            query: Cypher query string (when passing name as str). Alias for cypher.
            severity: Alert severity level.
            cypher: Cypher query string (preferred over query).
            description: Human-readable description.
            cooldown_seconds: Minimum time between fires.
            handler: Callback invoked with alert dict when fired.
        """
        if isinstance(rule_or_name, AlertRule):
            self._rules.append(rule_or_name)
            return

        # String name — build AlertRule from kwargs
        resolved_cypher = cypher or query
        if resolved_cypher is None:
            raise ValueError("Either 'cypher' or 'query' must be provided")
        self._rules.append(
            AlertRule(
                name=rule_or_name,
                cypher=resolved_cypher,
                severity=severity,
                cooldown_seconds=cooldown_seconds,
                handler=handler,
                description=description,
            )
        )

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a global handler called for every fired alert."""
        self._handlers.append(handler)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def check_all(self) -> list[dict[str, Any]]:
        """Check all rules. Returns list of {rule_name, severity, matches, fired}.

        Respects cooldown: a rule that fired within cooldown_seconds is skipped
        and reported with fired=False.
        """
        results: list[dict[str, Any]] = []
        now = time.monotonic()

        for rule in self._rules:
            last = self._last_fired.get(rule.name, 0.0)
            if now - last < rule.cooldown_seconds:
                results.append({
                    "rule_name": rule.name,
                    "severity": rule.severity,
                    "matches": 0,
                    "fired": False,
                })
                continue

            result = self._evaluate_rule(rule, now)
            results.append(result)

        return results

    def check_rule(self, rule_name: str) -> dict[str, Any] | None:
        """Check a single rule by name.

        Returns alert dict if triggered, None if cooldown is active.
        """
        now = time.monotonic()
        for rule in self._rules:
            if rule.name == rule_name:
                last = self._last_fired.get(rule.name, 0.0)
                if now - last < rule.cooldown_seconds:
                    return None
                result = self._evaluate_rule(rule, now)
                return result if result.get("fired") else None
        raise ValueError(f"No rule named '{rule_name}'")

    def evaluate_all(self) -> list[Alert]:
        """Evaluate all rules. Returns list of fired Alert objects.

        This is the legacy API; prefer check_all() for dict-based results.
        """
        now = time.monotonic()
        fired: list[Alert] = []

        for rule in self._rules:
            last = self._last_fired.get(rule.name, 0.0)
            if now - last < rule.cooldown_seconds:
                continue
            try:
                rows = self._db.query(rule.cypher)
                if rows:
                    alert = Alert(
                        rule_name=rule.name,
                        severity=rule.severity,
                        description=rule.description,
                        rows=rows,
                        fired_at=now,
                    )
                    fired.append(alert)
                    self._last_fired[rule.name] = now
                    for handler in self._handlers:
                        try:
                            handler(alert)
                        except Exception as he:
                            log.warning(
                                "Alert handler failed for rule '%s': %s",
                                rule.name,
                                he,
                            )
                    if rule.handler:
                        try:
                            rule.handler({
                                "rule_name": rule.name,
                                "severity": rule.severity,
                                "matches": len(rows),
                                "rows": rows,
                            })
                        except Exception as he:
                            log.warning(
                                "Rule handler failed for '%s': %s",
                                rule.name,
                                he,
                            )
            except Exception as e:
                log.warning("Alert rule '%s' query failed: %s", rule.name, e)

        return fired

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_rule(
        self, rule: AlertRule, now: float
    ) -> dict[str, Any]:
        """Evaluate a single rule and return a result dict."""
        try:
            rows = self._db.query(rule.cypher)
        except Exception as e:
            log.warning("Alert rule '%s' query failed: %s", rule.name, e)
            return {
                "rule_name": rule.name,
                "severity": rule.severity,
                "matches": 0,
                "fired": False,
                "error": str(e),
            }

        fired = len(rows) > 0
        if fired:
            self._last_fired[rule.name] = now

            # Invoke rule-specific handler
            if rule.handler:
                try:
                    rule.handler({
                        "rule_name": rule.name,
                        "severity": rule.severity,
                        "matches": len(rows),
                        "rows": rows,
                    })
                except Exception as he:
                    log.warning(
                        "Rule handler failed for '%s': %s", rule.name, he
                    )

            # Invoke global handlers
            alert = Alert(
                rule_name=rule.name,
                severity=rule.severity,
                description=rule.description,
                rows=rows,
                fired_at=now,
            )
            for handler in self._handlers:
                try:
                    handler(alert)
                except Exception as he:
                    log.warning(
                        "Global handler failed for rule '%s': %s",
                        rule.name,
                        he,
                    )

        return {
            "rule_name": rule.name,
            "severity": rule.severity,
            "matches": len(rows),
            "fired": fired,
        }
