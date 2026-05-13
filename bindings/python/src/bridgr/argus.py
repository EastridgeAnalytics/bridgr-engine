"""BridgrStore — drop-in replacement for Argus CaseGraph.

Stores case data in a LadybugDB .lbug file instead of .md files on disk.
Maintains the same public API so Argus route files can swap with a single import change.

Usage:
    from bridgr.argus import BridgrStore

    store = BridgrStore(case_dir)
    store.load()
    node = store.read_node("entity_abc123")
"""

from __future__ import annotations

import base64
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bridgr
from bridgr.database import Database

NODE_TYPES = ["entity", "fact", "source", "issue", "question", "authority", "tag"]
_COUNTER_LABEL = "_Counter"


def _encode_fm(fm: dict) -> str:
    return base64.b64encode(json.dumps(fm, default=str).encode()).decode()


def _decode_fm(s: str) -> dict:
    if not s:
        return {}
    return json.loads(base64.b64decode(s).decode())


class ReferentialIntegrityError(Exception):
    def __init__(self, node_id: str, refs: list[dict]):
        self.node_id = node_id
        self.refs = refs
        super().__init__(f"Node {node_id} is referenced by {len(refs)} other nodes")


class ShortNameCollisionError(Exception):
    def __init__(self, short_name: str, existing_id: str):
        self.short_name = short_name
        self.existing_id = existing_id
        super().__init__(f"Short name '{short_name}' already used by {existing_id}")


class BridgrStore:
    """Graph-backed storage for Argus case data.

    Drop-in replacement for CaseGraph. Same public API, backed by LadybugDB.
    """

    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
        self._db_path = str(self.case_dir / "bridgr.lbug")
        self._db: Database | None = None
        self._counter_lock = threading.Lock()
        self._batch_mode = False

    @property
    def db(self) -> Database:
        if self._db is None:
            raise RuntimeError("BridgrStore not loaded. Call .load() first.")
        return self._db

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Open the database and ensure schema exists."""
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self._db = bridgr.open(self._db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        existing = set()
        result = self.db.execute("CALL SHOW_TABLES() RETURN name")
        while result.has_next():
            existing.add(result.get_next()[0])

        if "Node" not in existing:
            self.db.execute("""
                CREATE NODE TABLE Node(
                    id STRING PRIMARY KEY,
                    node_type STRING,
                    frontmatter STRING,
                    body STRING
                )
            """)

        if _COUNTER_LABEL not in existing:
            self.db.execute(f"""
                CREATE NODE TABLE {_COUNTER_LABEL}(
                    id STRING PRIMARY KEY,
                    value INT64
                )
            """)

        if "EDGE" not in existing:
            self.db.execute("""
                CREATE REL TABLE EDGE(
                    FROM Node TO Node,
                    edge_type STRING,
                    field STRING
                )
            """)

        try:
            self.db.execute(
                f"MATCH (c:{_COUNTER_LABEL} {{id: 'fact_number'}}) RETURN c.value"
            )
        except RuntimeError:
            pass
        result = self.db.execute(
            f"MATCH (c:{_COUNTER_LABEL} {{id: 'fact_number'}}) RETURN c.value"
        )
        if not result.has_next():
            self.db.execute(
                f"CREATE (:{_COUNTER_LABEL} {{id: 'fact_number', value: 0}})"
            )

    def ensure_fresh(self) -> None:
        """No-op for BridgrStore — the database is always consistent."""
        pass

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def read_node(self, node_id: str) -> dict[str, Any] | None:
        """Read a node by ID. Returns {frontmatter, body, path, type} or None."""
        result = self.db.execute(
            "MATCH (n:Node {id: $id}) RETURN n.node_type, n.frontmatter, n.body",
            {"id": node_id},
        )
        if not result.has_next():
            return None
        row = result.get_next()
        node_type = row[0]
        frontmatter = _decode_fm(row[1])
        body = row[2] or ""
        return {
            "frontmatter": frontmatter,
            "body": body,
            "path": self.case_dir / node_type / f"{node_id}.md",
            "type": node_type,
        }

    def write_node(
        self, node_id: str, node_type: str, frontmatter: dict, body: str = ""
    ) -> Path:
        """Create or update a node. Returns the virtual path."""
        fm_enc = _encode_fm(frontmatter)
        frontmatter.setdefault("id", node_id)
        frontmatter.setdefault("type", node_type)

        existing = self.db.execute(
            "MATCH (n:Node {id: $id}) RETURN n.id", {"id": node_id}
        )
        if existing.has_next():
            self.db.execute(
                "MATCH (n:Node {id: $id}) SET n.node_type = $ntype, "
                "n.frontmatter = $fm, n.body = $body",
                {"id": node_id, "ntype": node_type, "fm": fm_enc, "body": body},
            )
        else:
            self.db.execute(
                "CREATE (:Node {id: $id, node_type: $ntype, frontmatter: $fm, body: $body})",
                {"id": node_id, "ntype": node_type, "fm": fm_enc, "body": body},
            )

        self._sync_edges(node_id, node_type, frontmatter)

        return self.case_dir / node_type / f"{node_id}.md"

    def delete_node(
        self, node_id: str, *, force: bool = False, cascade: bool = False
    ) -> bool:
        """Delete a node. Returns True if deleted.

        If force=False, raises ReferentialIntegrityError if other nodes reference this one.
        If cascade=True, also deletes referencing nodes.
        """
        existing = self.read_node(node_id)
        if existing is None:
            return False

        if not force:
            refs = self.get_references_to(node_id)
            if refs:
                raise ReferentialIntegrityError(node_id, refs)

        if cascade:
            refs = self.get_references_to(node_id)
            for ref in refs:
                self.delete_node(ref["node_id"], force=True)

        self.db.execute("MATCH (n:Node {id: $id}) DETACH DELETE n", {"id": node_id})
        return True

    def list_nodes(self, node_type: str) -> list[dict]:
        """List all nodes of a given type. Returns list of frontmatter dicts."""
        result = self.db.execute(
            "MATCH (n:Node {node_type: $ntype}) RETURN n.id, n.frontmatter, n.body",
            {"ntype": node_type},
        )
        nodes = []
        while result.has_next():
            row = result.get_next()
            fm = _decode_fm(row[1])
            fm["_has_body"] = bool(row[2])
            nodes.append(fm)
        return nodes

    def node_exists(self, node_id: str) -> bool:
        result = self.db.execute(
            "MATCH (n:Node {id: $id}) RETURN n.id", {"id": node_id}
        )
        return result.has_next()

    def unique_id(self, base_slug: str) -> str:
        """Generate a unique node ID, appending -2, -3, etc. on collision."""
        if not self.node_exists(base_slug):
            return base_slug
        i = 2
        while self.node_exists(f"{base_slug}-{i}"):
            i += 1
        return f"{base_slug}-{i}"

    # ------------------------------------------------------------------
    # Short name operations
    # ------------------------------------------------------------------

    def get_by_short_name(self, short_name: str) -> dict[str, Any] | None:
        """Look up an entity by its short_name field."""
        result = self.db.execute(
            "MATCH (n:Node {node_type: 'entity'}) RETURN n.id, n.frontmatter",
        )
        while result.has_next():
            row = result.get_next()
            fm = _decode_fm(row[1])
            if fm.get("short_name", "").lower() == short_name.lower():
                return self.read_node(row[0])
        return None

    def all_short_names(self) -> dict[str, str]:
        """Return {short_name: entity_id} for all entities with short names."""
        result = self.db.execute(
            "MATCH (n:Node {node_type: 'entity'}) RETURN n.id, n.frontmatter",
        )
        mapping = {}
        while result.has_next():
            row = result.get_next()
            fm = _decode_fm(row[1])
            sn = fm.get("short_name")
            if sn:
                mapping[sn] = row[0]
        return mapping

    # ------------------------------------------------------------------
    # Issue tree
    # ------------------------------------------------------------------

    def get_issue_tree(self) -> list[dict]:
        """Return all issues as a tree with nested children."""
        issues = self.list_nodes("issue")
        by_id = {i["id"]: {**i, "children": []} for i in issues}

        roots = []
        for issue in issues:
            parent_id = issue.get("parent_id")
            if parent_id and parent_id in by_id:
                by_id[parent_id]["children"].append(by_id[issue["id"]])
            else:
                roots.append(by_id[issue["id"]])
        return roots

    def get_descendants(self, issue_id: str) -> list[str]:
        """Return all descendant issue IDs recursively."""
        result = self.db.execute(
            "MATCH (p:Node {id: $id})-[:EDGE*1..10]->(c:Node) "
            "WHERE c.node_type = 'issue' "
            "RETURN DISTINCT c.id",
            {"id": issue_id},
        )
        ids = []
        while result.has_next():
            ids.append(result.get_next()[0])
        return ids

    # ------------------------------------------------------------------
    # Referential integrity
    # ------------------------------------------------------------------

    def get_references_to(self, node_id: str) -> list[dict]:
        """Return all nodes that reference this node via edges."""
        result = self.db.execute(
            "MATCH (src:Node)-[e:EDGE]->(dst:Node {id: $id}) "
            "RETURN src.id AS node_id, src.node_type AS node_type, e.field AS field",
            {"id": node_id},
        )
        refs = []
        while result.has_next():
            row = result.get_next()
            cols = result.get_column_names()
            refs.append(dict(zip(cols, row)))
        return refs

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def rename_entity(self, entity_id: str, new_name: str) -> list[str]:
        """Rename an entity and update short_name. Returns list of updated fact IDs."""
        node = self.read_node(entity_id)
        if node is None:
            return []

        fm = node["frontmatter"]
        fm["name"] = new_name
        sn = new_name.lower().replace(" ", "_")[:30]
        fm["short_name"] = sn
        self.write_node(entity_id, "entity", fm, node["body"])

        updated_facts = []
        facts = self.get_facts_about(entity_id)
        for fact in facts:
            updated_facts.append(fact["id"])
        return updated_facts

    # ------------------------------------------------------------------
    # Traversals
    # ------------------------------------------------------------------

    def get_facts_about(self, entity_id: str) -> list[dict]:
        """Get all facts that involve this entity."""
        result = self.db.execute(
            "MATCH (f:Node)-[e:EDGE {field: 'involves'}]->(ent:Node {id: $eid}) "
            "WHERE f.node_type = 'fact' "
            "RETURN f.frontmatter",
            {"eid": entity_id},
        )
        facts = []
        while result.has_next():
            fm = _decode_fm(result.get_next()[0])
            facts.append(fm)
        return facts

    def get_facts_citing(self, source_id: str) -> list[dict]:
        """Get all facts that cite this source."""
        result = self.db.execute(
            "MATCH (f:Node)-[e:EDGE {field: 'sources'}]->(s:Node {id: $sid}) "
            "WHERE f.node_type = 'fact' "
            "RETURN f.frontmatter",
            {"sid": source_id},
        )
        facts = []
        while result.has_next():
            fm = _decode_fm(result.get_next()[0])
            facts.append(fm)
        return facts

    def get_facts_bearing_on(
        self, issue_id: str, include_descendants: bool = False
    ) -> list[dict]:
        """Get all facts that bear on this issue."""
        issue_ids = [issue_id]
        if include_descendants:
            issue_ids.extend(self.get_descendants(issue_id))

        all_facts = []
        seen = set()
        for iid in issue_ids:
            result = self.db.execute(
                "MATCH (f:Node)-[e:EDGE {field: 'bears_on'}]->(i:Node {id: $iid}) "
                "WHERE f.node_type = 'fact' "
                "RETURN f.frontmatter",
                {"iid": iid},
            )
            while result.has_next():
                fm = _decode_fm(result.get_next()[0])
                if fm["id"] not in seen:
                    seen.add(fm["id"])
                    all_facts.append(fm)
        return all_facts

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def next_fact_number(self) -> int:
        """Atomically increment and return the next fact number."""
        with self._counter_lock:
            result = self.db.execute(
                f"MATCH (c:{_COUNTER_LABEL} {{id: 'fact_number'}}) RETURN c.value"
            )
            current = result.get_next()[0]
            new_val = current + 1
            self.db.execute(
                f"MATCH (c:{_COUNTER_LABEL} {{id: 'fact_number'}}) SET c.value = $val",
                {"val": new_val},
            )
            return new_val

    def set_counter(self, value: int) -> None:
        self.db.execute(
            f"MATCH (c:{_COUNTER_LABEL} {{id: 'fact_number'}}) SET c.value = $val",
            {"val": value},
        )

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------

    def begin_batch(self) -> None:
        self._batch_mode = True
        self.db.begin_transaction()

    def end_batch(self) -> None:
        self._batch_mode = False
        self.db.commit()

    # ------------------------------------------------------------------
    # Edge sync (internal)
    # ------------------------------------------------------------------

    def _sync_edges(self, node_id: str, node_type: str, frontmatter: dict) -> None:
        """Synchronize graph edges from frontmatter array fields.

        When a fact has involves: [e1, e2], create EDGE relationships
        from the fact node to each entity node.
        """
        edge_fields = {
            "involves": "involves",
            "sources": "sources",
            "bears_on": "bears_on",
            "tags": "tags",
        }

        self.db.execute(
            "MATCH (n:Node {id: $id})-[e:EDGE]->() DELETE e",
            {"id": node_id},
        )

        for field_name, edge_field in edge_fields.items():
            targets = frontmatter.get(field_name, [])
            if not isinstance(targets, list):
                continue
            for target_id in targets:
                if isinstance(target_id, dict):
                    target_id = target_id.get("id", str(target_id))
                target_id = str(target_id)
                if self.node_exists(target_id):
                    self.db.execute(
                        "MATCH (src:Node {id: $src}), (dst:Node {id: $dst}) "
                        "CREATE (src)-[:EDGE {edge_type: 'ref', field: $field}]->(dst)",
                        {"src": node_id, "dst": target_id, "field": edge_field},
                    )

        if node_type == "issue" and frontmatter.get("parent_id"):
            parent_id = frontmatter["parent_id"]
            if self.node_exists(parent_id):
                self.db.execute(
                    "MATCH (child:Node {id: $child}), (parent:Node {id: $parent}) "
                    "CREATE (parent)-[:EDGE {edge_type: 'parent_of', field: 'parent_id'}]->(child)",
                    {"child": node_id, "parent": parent_id},
                )
