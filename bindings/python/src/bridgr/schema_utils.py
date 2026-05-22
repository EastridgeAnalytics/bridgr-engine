"""Shared schema and path utilities for Bridgr.

Internal but stable API surface — used by both the MIT engine and proprietary
connectors (bridgr_platform).  Functions here were originally private helpers
in database.py and export.py; they are extracted so downstream packages can
import them without reaching into implementation details.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "arrow_type_to_cypher",
    "arrow_schema_to_cypher",
    "find_primary_key",
    "cypher_path",
]


def arrow_type_to_cypher(arrow_type) -> str:
    """Map an Arrow data type to a Cypher type string."""
    import pyarrow as pa

    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "STRING"
    elif pa.types.is_int8(arrow_type) or pa.types.is_int16(arrow_type) or pa.types.is_int32(arrow_type):
        return "INT32"
    elif pa.types.is_int64(arrow_type):
        return "INT64"
    elif pa.types.is_float16(arrow_type) or pa.types.is_float32(arrow_type):
        return "FLOAT"
    elif pa.types.is_float64(arrow_type):
        return "DOUBLE"
    elif pa.types.is_boolean(arrow_type):
        return "BOOLEAN"
    elif pa.types.is_date(arrow_type):
        return "DATE"
    elif pa.types.is_timestamp(arrow_type):
        return "TIMESTAMP"
    elif pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return "STRING[]"
    else:
        return "STRING"


def arrow_schema_to_cypher(schema, primary_key: str | None) -> dict[str, str]:
    """Convert an Arrow schema to a Cypher property dict with PRIMARY KEY annotation."""
    props: dict[str, str] = {}
    for field in schema:
        cypher_type = arrow_type_to_cypher(field.type)
        if field.name == primary_key:
            props[field.name] = f"{cypher_type} PRIMARY KEY"
        else:
            props[field.name] = cypher_type
    return props


def find_primary_key(schema, *, label_hint: str | None = None) -> str | None:
    """Heuristic: find the best PK column from an Arrow schema."""
    import pyarrow as pa
    import re

    label_lower = label_hint.lower() if label_hint else ""
    # Split CamelCase into words for substring matching (PurchaseOrder -> purchase, order)
    label_words = [w.lower() for w in re.findall(r"[A-Z]?[a-z]+", label_hint or "")]

    candidates = []
    for field in schema:
        name_lower = field.name.lower()
        prefix = name_lower.replace("_id", "") if name_lower.endswith("_id") else ""
        # Prefer <label>_id or <label_word>_id (e.g., order_id for "PurchaseOrder")
        if label_hint and name_lower.endswith("_id") and (
            prefix == label_lower or prefix in label_words
        ):
            candidates.append((-1, field.name))
        elif name_lower == "id":
            candidates.append((0, field.name))
        elif name_lower.endswith("_id"):
            candidates.append((1, field.name))
        elif "key" in name_lower or "pk" in name_lower:
            candidates.append((2, field.name))
    if candidates:
        candidates.sort()
        return candidates[0][1]
    # Fallback: first string or int column
    for field in schema:
        if pa.types.is_string(field.type) or pa.types.is_integer(field.type):
            return field.name
    return None


def cypher_path(path: str) -> str:
    """Convert a filesystem path to a Cypher-safe string (forward slashes)."""
    return str(Path(path).resolve()).replace("\\", "/")
