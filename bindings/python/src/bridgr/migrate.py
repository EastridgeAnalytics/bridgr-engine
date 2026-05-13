"""Filesystem-to-Bridgr migration script.

Reads an existing Argus case directory (with .md files) and imports all nodes
and edges into a BridgrStore-backed .lbug database.

Usage:
    from bridgr.migrate import migrate_case
    migrate_case(Path("~/.config/Argus/cases/MyCaseName"))

Or from the command line:
    python -m bridgr.migrate /path/to/case/dir
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from bridgr.argus import BridgrStore

NODE_TYPE_DIRS = {
    "entity": "entities",
    "fact": "facts",
    "source": "sources",
    "issue": "issues",
    "question": "questions",
    "authority": "authorities",
    "tag": "tags",
    "trace": "traces",
}


def parse_md_file(path: Path) -> tuple[dict, str]:
    """Parse a YAML-frontmatter markdown file into (frontmatter_dict, body_string)."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        frontmatter = {}

    body = parts[2].strip()
    return frontmatter, body


def migrate_case(case_dir: Path, *, verbose: bool = True) -> dict[str, Any]:
    """Migrate an Argus case from filesystem (.md files) to BridgrStore (.lbug).

    Args:
        case_dir: Path to the case directory (e.g., ~/.config/Argus/cases/CaseName)
        verbose: Print progress information

    Returns:
        dict with migration stats (nodes_imported, edges_created, errors, etc.)
    """
    case_dir = Path(case_dir)
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    stats = {
        "nodes_imported": 0,
        "edges_created": 0,
        "errors": [],
        "node_types": {},
    }

    if verbose:
        print(f"Migrating case: {case_dir}")

    store = BridgrStore(case_dir)
    store.load()

    try:
        store.begin_batch()

        for node_type, dir_name in NODE_TYPE_DIRS.items():
            node_dir = case_dir / dir_name
            if not node_dir.exists():
                continue

            md_files = list(node_dir.glob("*.md"))
            if verbose:
                print(f"  {node_type}: {len(md_files)} files")

            count = 0
            for md_file in md_files:
                try:
                    frontmatter, body = parse_md_file(md_file)
                    if not frontmatter:
                        continue

                    node_id = frontmatter.get("id")
                    if not node_id:
                        node_id = md_file.stem
                        frontmatter["id"] = node_id

                    if "type" not in frontmatter:
                        frontmatter["type"] = node_type

                    store.write_node(node_id, node_type, frontmatter, body)
                    count += 1
                    stats["nodes_imported"] += 1
                except Exception as e:
                    stats["errors"].append({
                        "file": str(md_file),
                        "error": str(e),
                    })
                    if verbose:
                        print(f"    ERROR: {md_file.name}: {e}")

            stats["node_types"][node_type] = count

        store.end_batch()

        counter_file = case_dir / "counters.json"
        if counter_file.exists():
            try:
                counters = json.loads(counter_file.read_text(encoding="utf-8"))
                fact_number = counters.get("fact_number", 0)
                store.set_counter(fact_number)
                if verbose:
                    print(f"  counters: fact_number = {fact_number}")
            except Exception as e:
                stats["errors"].append({"file": str(counter_file), "error": str(e)})

    except Exception as e:
        stats["errors"].append({"phase": "import", "error": str(e)})
        if verbose:
            print(f"  FATAL ERROR during import: {e}")
        raise
    finally:
        store.close()

    if verbose:
        print(f"\nMigration complete:")
        print(f"  Nodes imported: {stats['nodes_imported']}")
        print(f"  Errors: {len(stats['errors'])}")
        for ntype, count in stats["node_types"].items():
            print(f"    {ntype}: {count}")

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m bridgr.migrate <case_directory>")
        sys.exit(1)

    case_dir = Path(sys.argv[1])
    stats = migrate_case(case_dir)
    if stats["errors"]:
        print(f"\n{len(stats['errors'])} errors occurred during migration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
