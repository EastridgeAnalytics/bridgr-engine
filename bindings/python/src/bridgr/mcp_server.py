"""Bridgr MCP Server — AI agent interface to a Bridgr graph database.

Exposes 24 tools: query, read_node, write_node, delete_node, create_edge,
search, traverse_graph, list_node_types, get_edges, create_node_table,
create_edge_table, list_schema, begin_transaction, commit_transaction,
rollback_transaction, drop_table, alter_table, run_algorithm, bulk_import,
create_vector_index, vector_search, hybrid_search, get_audit_log,
export_data. Runs as a stdio MCP server.

Usage:
    # As a module (for Claude Code MCP config):
    python -m bridgr.mcp_server --db /path/to/database.lbug

    # Programmatically:
    from bridgr.mcp_server import create_server
    server = create_server("/path/to/database.lbug")
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

import bridgr
from bridgr.database import Database
from bridgr.exceptions import (
    DuplicateNodeError,
    NodeNotFoundError,
    SchemaError,
    TransactionError,
)
from bridgr.algorithms import GraphAlgorithms
from bridgr.vector import VectorIndex
from bridgr.audit import AuditLog
from bridgr.export import DataExporter

TOOLS = [
    Tool(
        name="query",
        description="Execute a Cypher query against the graph database and return results as JSON. Use for complex queries, aggregations, and graph traversals.",
        inputSchema={
            "type": "object",
            "properties": {
                "cypher": {
                    "type": "string",
                    "description": "Cypher query to execute. Example: MATCH (e:Entity) WHERE e.name CONTAINS 'Smith' RETURN e.id, e.name",
                },
                "params": {
                    "type": "object",
                    "description": "Optional query parameters. Example: {\"name\": \"Smith\"}",
                    "default": {},
                },
            },
            "required": ["cypher"],
        },
    ),
    Tool(
        name="read_node",
        description="Read a single node by its ID. Returns all properties of the node, or null if not found.",
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The unique ID of the node to read.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional node label/type to narrow the search (e.g., 'Entity', 'Fact').",
                },
            },
            "required": ["node_id"],
        },
    ),
    Tool(
        name="write_node",
        description="Create a new node or update an existing one. Provide the label (type) and properties. If a node with the same ID exists, it will be updated (merge semantics).",
        inputSchema={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Node label/type (e.g., 'Entity', 'Fact', 'Source').",
                },
                "properties": {
                    "type": "object",
                    "description": "Node properties. Must include 'id' as the primary key. Example: {\"id\": \"e1\", \"name\": \"John Smith\", \"entity_type\": \"person\"}",
                },
            },
            "required": ["label", "properties"],
        },
    ),
    Tool(
        name="delete_node",
        description="Delete a node by its ID. Also removes all edges connected to this node.",
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The unique ID of the node to delete.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional node label to narrow the search.",
                },
            },
            "required": ["node_id"],
        },
    ),
    Tool(
        name="create_edge",
        description="Create a directed edge (relationship) between two nodes.",
        inputSchema={
            "type": "object",
            "properties": {
                "edge_type": {
                    "type": "string",
                    "description": "Edge type/label (e.g., 'INVOLVES', 'CONNECTED_TO', 'SOURCED_FROM').",
                },
                "from_id": {
                    "type": "string",
                    "description": "ID of the source node.",
                },
                "to_id": {
                    "type": "string",
                    "description": "ID of the target node.",
                },
                "from_label": {"type": "string", "description": "Label of the source node."},
                "to_label": {"type": "string", "description": "Label of the target node."},
                "properties": {
                    "type": "object",
                    "description": "Optional edge properties.",
                    "default": {},
                },
            },
            "required": ["edge_type", "from_id", "to_id"],
        },
    ),
    Tool(
        name="search",
        description="Search for nodes by keyword across all properties. Case-insensitive. Returns matching nodes with their type and properties.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword or phrase.",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of node labels to search within (e.g., ['Entity', 'Fact']).",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="traverse_graph",
        description="Traverse the graph from a starting node, following edges up to a specified depth. Returns all reachable nodes and the edges connecting them.",
        inputSchema={
            "type": "object",
            "properties": {
                "start_node_id": {
                    "type": "string",
                    "description": "ID of the node to start traversal from.",
                },
                "start_label": {
                    "type": "string",
                    "description": "Label of the starting node.",
                },
                "edge_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of edge types to follow. If empty, follows all edges.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum traversal depth (1-10). Default: 2.",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["start_node_id", "start_label"],
        },
    ),
    Tool(
        name="list_node_types",
        description="List all node table labels (types) in the database with their counts.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="get_edges",
        description="Get all edges connected to a node (both incoming and outgoing).",
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "ID of the node.",
                },
                "label": {
                    "type": "string",
                    "description": "Label of the node.",
                },
            },
            "required": ["node_id", "label"],
        },
    ),
    Tool(
        name="create_node_table",
        description=(
            "Create a new node table (node type) in the database. "
            "Exactly one property type must include 'PRIMARY KEY'. "
            "Example properties: {\"id\": \"STRING PRIMARY KEY\", \"name\": \"STRING\", \"age\": \"INT64\"}"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Node table label (e.g., 'Person', 'Organization').",
                },
                "properties": {
                    "type": "object",
                    "description": (
                        "Mapping of property name to Kùzu type string. "
                        "One property must include 'PRIMARY KEY' in its type. "
                        "Supported types: STRING, INT64, DOUBLE, BOOL, DATE, TIMESTAMP, STRING[]."
                    ),
                },
            },
            "required": ["label", "properties"],
        },
    ),
    Tool(
        name="create_edge_table",
        description=(
            "Create a new edge table (relationship type) connecting two node tables. "
            "Both endpoint node tables must already exist."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Edge table label (e.g., 'WORKS_AT', 'KNOWS').",
                },
                "from_label": {
                    "type": "string",
                    "description": "Label of the source node table.",
                },
                "to_label": {
                    "type": "string",
                    "description": "Label of the target node table.",
                },
                "properties": {
                    "type": "object",
                    "description": "Optional edge properties (name → Kùzu type). Can be empty.",
                    "default": {},
                },
            },
            "required": ["label", "from_label", "to_label"],
        },
    ),
    Tool(
        name="list_schema",
        description=(
            "List the full database schema: all node tables and edge tables "
            "with their columns, types, primary keys, row counts, and connections."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="begin_transaction",
        description="Begin a database transaction. All subsequent write operations will be batched until commit or rollback.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="commit_transaction",
        description="Commit the current transaction, making all batched changes permanent.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rollback_transaction",
        description="Roll back the current transaction, discarding all batched changes.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="drop_table",
        description=(
            "Drop a node or edge table. DESTRUCTIVE — permanently deletes the table and all its data. "
            "Requires confirm=true parameter. Without it, the operation is rejected."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Name of the table to drop.",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to confirm this destructive operation.",
                    "default": False,
                },
            },
            "required": ["label"],
        },
    ),
    Tool(
        name="alter_table",
        description="Alter a node or edge table schema (add/drop/rename column, rename table).",
        inputSchema={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Table name to alter.",
                },
                "operation": {
                    "type": "string",
                    "enum": ["add_column", "drop_column", "rename_column", "rename_table"],
                    "description": "The alter operation to perform.",
                },
                "column_name": {
                    "type": "string",
                    "description": "Column name (for add_column, drop_column).",
                },
                "column_type": {
                    "type": "string",
                    "description": "Column data type (for add_column). e.g. STRING, INT64, DOUBLE, BOOL.",
                },
                "default_value": {
                    "type": "string",
                    "description": "Default value for new column (for add_column). Optional.",
                },
                "old_name": {
                    "type": "string",
                    "description": "Current column name (for rename_column).",
                },
                "new_name": {
                    "type": "string",
                    "description": "New name (for rename_column, rename_table).",
                },
            },
            "required": ["label", "operation"],
        },
    ),
    Tool(
        name="run_algorithm",
        description="Run a graph algorithm and return results.",
        inputSchema={
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "enum": [
                        "pagerank", "wcc", "scc", "louvain", "k_core",
                        "degree_centrality", "shortest_path", "node_similarity",
                    ],
                    "description": "Algorithm to run.",
                },
                "node_label": {
                    "type": "string",
                    "description": "Node table label.",
                },
                "edge_label": {
                    "type": "string",
                    "description": "Edge table label.",
                },
                "damping": {
                    "type": "number",
                    "description": "PageRank damping factor. Default: 0.85.",
                    "default": 0.85,
                },
                "iterations": {
                    "type": "integer",
                    "description": "Max iterations (PageRank, Louvain). Default: 20.",
                    "default": 20,
                },
                "k": {
                    "type": "integer",
                    "description": "K-Core minimum degree. Default: 2.",
                    "default": 2,
                },
                "source_id": {
                    "type": "string",
                    "description": "Source node ID (shortest_path, node_similarity).",
                },
                "target_id": {
                    "type": "string",
                    "description": "Target node ID (shortest_path, node_similarity).",
                },
                "metric": {
                    "type": "string",
                    "enum": ["jaccard", "overlap"],
                    "description": "Similarity metric (node_similarity). Default: jaccard.",
                    "default": "jaccard",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Max hops (shortest_path). Default: 10.",
                    "default": 10,
                },
            },
            "required": ["algorithm", "node_label", "edge_label"],
        },
    ),
    Tool(
        name="bulk_import",
        description="Import data from a CSV or Parquet file into a table.",
        inputSchema={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Target table label.",
                },
                "path": {
                    "type": "string",
                    "description": "Path to CSV or Parquet file.",
                },
                "format": {
                    "type": "string",
                    "enum": ["csv", "parquet"],
                    "description": "File format. Default: csv.",
                    "default": "csv",
                },
            },
            "required": ["label", "path"],
        },
    ),
    Tool(
        name="create_vector_index",
        description="Create an HNSW vector similarity index on a table property.",
        inputSchema={
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Node table containing the vector column.",
                },
                "property": {
                    "type": "string",
                    "description": "Vector property name (FLOAT[N] or DOUBLE[N]).",
                },
                "index_name": {
                    "type": "string",
                    "description": "Index name. Defaults to '{table}_{property}_idx'.",
                },
                "metric": {
                    "type": "string",
                    "enum": ["cosine", "l2", "l2sq", "dotproduct"],
                    "description": "Distance metric. Default: cosine.",
                    "default": "cosine",
                },
            },
            "required": ["table", "property"],
        },
    ),
    Tool(
        name="vector_search",
        description="Find nearest neighbors by vector similarity.",
        inputSchema={
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Node table with the vector index.",
                },
                "index_name": {
                    "type": "string",
                    "description": "Name of the HNSW index.",
                },
                "query_vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Query embedding vector.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of nearest neighbors. Default: 10.",
                    "default": 10,
                },
            },
            "required": ["table", "index_name", "query_vector"],
        },
    ),
    Tool(
        name="hybrid_search",
        description="Vector search + graph traversal. Find similar nodes, then walk edges.",
        inputSchema={
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Node table with the vector index.",
                },
                "index": {
                    "type": "string",
                    "description": "Name of the HNSW index.",
                },
                "query_vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Query embedding vector.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of nearest neighbors. Default: 10.",
                    "default": 10,
                },
                "traverse_edge": {
                    "type": "string",
                    "description": "Edge type to traverse from matched nodes. If omitted, returns vector results only.",
                },
                "traverse_depth": {
                    "type": "integer",
                    "description": "Max traversal depth. Default: 1.",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["table", "index", "query_vector"],
        },
    ),
    Tool(
        name="get_audit_log",
        description="Query the append-only audit trail.",
        inputSchema={
            "type": "object",
            "properties": {
                "since": {
                    "type": "string",
                    "description": "ISO 8601 timestamp. Entries after this time.",
                },
                "until": {
                    "type": "string",
                    "description": "ISO 8601 timestamp. Entries before this time.",
                },
                "operation": {
                    "type": "string",
                    "description": "Filter by operation (create, update, delete, create_edge).",
                },
                "actor": {
                    "type": "string",
                    "description": "Filter by actor identifier.",
                },
                "target_type": {
                    "type": "string",
                    "description": "Filter by target node/edge label.",
                },
                "target_id": {
                    "type": "string",
                    "description": "Get full history of a specific entity.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max entries. Default: 100.",
                    "default": 100,
                },
            },
        },
    ),
    Tool(
        name="export_data",
        description="Export a node table to Parquet or CSV.",
        inputSchema={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Node table label to export.",
                },
                "path": {
                    "type": "string",
                    "description": "Output file path.",
                },
                "format": {
                    "type": "string",
                    "enum": ["parquet", "csv"],
                    "description": "Output format. Default: parquet.",
                    "default": "parquet",
                },
            },
            "required": ["label", "path"],
        },
    ),
]


def _serialize(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def create_server(db_path: str) -> Server:
    """Create an MCP server connected to a Bridgr database."""
    db = bridgr.open(db_path)
    server = Server("bridgr")

    @server.list_tools()
    async def handle_list_tools():
        return TOOLS

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None):
        arguments = arguments or {}
        try:
            result = _dispatch(db, name, arguments)
            return [TextContent(type="text", text=json.dumps(_serialize(result), indent=2))]
        except SchemaError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e), "code": "SCHEMA_CONFLICT"}))]
        except NodeNotFoundError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e), "code": "NOT_FOUND"}))]
        except DuplicateNodeError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e), "code": "DUPLICATE"}))]
        except TransactionError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e), "code": "TRANSACTION_ERROR"}))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e), "code": "VALIDATION_ERROR"}))]

    return server


def _dispatch(db: Database, tool_name: str, args: dict) -> Any:
    if tool_name == "query":
        cypher = args["cypher"]
        params = args.get("params", {})
        rows = db.query(cypher, params if params else None)
        return {"rows": rows, "count": len(rows)}

    elif tool_name == "read_node":
        node = db.get_node(args["node_id"], label=args.get("label"))
        if node is None:
            return {"found": False, "node": None}
        return {"found": True, "node": node}

    elif tool_name == "write_node":
        label = args["label"]
        props = args["properties"]
        node_id = props.get("id")
        if not node_id:
            return {"error": "Properties must include 'id' as primary key."}
        existing = db.get_node(node_id, label=label)
        if existing:
            update_props = {k: v for k, v in props.items() if k != "id"}
            db.update_node(node_id, update_props, label=label)
            return {"action": "updated", "node_id": node_id}
        else:
            db.create_node(label, props)
            return {"action": "created", "node_id": node_id}

    elif tool_name == "delete_node":
        node_id = args["node_id"]
        label = args.get("label")
        try:
            db.delete_node(node_id, label=label)
            return {"deleted": True, "node_id": node_id}
        except Exception as e:
            return {"deleted": False, "error": str(e)}

    elif tool_name == "create_edge":
        db.create_edge(
            args["edge_type"],
            args["from_id"],
            args["to_id"],
            properties=args.get("properties"),
            from_label=args.get("from_label"),
            to_label=args.get("to_label"),
        )
        return {"created": True, "edge_type": args["edge_type"],
                "from_id": args["from_id"], "to_id": args["to_id"]}

    elif tool_name == "search":
        results = db.search(args["query"], labels=args.get("labels"))
        return {"results": results, "count": len(results)}

    elif tool_name == "traverse_graph":
        start_id = args["start_node_id"]
        start_label = args["start_label"]
        edge_types = args.get("edge_types", [])
        max_depth = min(args.get("max_depth", 2), 10)

        if edge_types:
            edge_filter = "|".join(f":{et}" for et in edge_types)
            edge_pattern = f"[{edge_filter}*1..{max_depth}]"
        else:
            edge_pattern = f"[*1..{max_depth}]"

        rows = db.query(
            f"MATCH (start:{start_label} {{id: $start_id}})-{edge_pattern}-(reached) "
            f"RETURN DISTINCT reached.id AS id, label(reached) AS label",
            {"start_id": start_id},
        )

        return {
            "start_node": start_id,
            "max_depth": max_depth,
            "reachable_nodes": rows,
            "node_count": len(rows),
        }

    elif tool_name == "list_node_types":
        labels = db._get_node_labels()
        type_counts = []
        for lbl in labels:
            result = db.execute(f"MATCH (n:{lbl}) RETURN count(n) AS cnt")
            cnt = result.get_next()[0]
            type_counts.append({"label": lbl, "count": cnt})
        return {"types": type_counts}

    elif tool_name == "get_edges":
        edges = db.get_edges(args["node_id"], label=args.get("label"))
        return {"edges": edges, "count": len(edges)}

    elif tool_name == "create_node_table":
        label = args["label"]
        properties = args["properties"]
        try:
            db.create_node_table(label, properties)
        except Exception as e:
            return {"success": False, "error": str(e), "code": "SCHEMA_CONFLICT"}
        return {"success": True, "label": label, "properties": properties}

    elif tool_name == "create_edge_table":
        label = args["label"]
        from_label = args["from_label"]
        to_label = args["to_label"]
        properties = args.get("properties") or {}
        try:
            db.create_edge_table(label, from_label, to_label, properties or None)
        except Exception as e:
            return {"success": False, "error": str(e), "code": "SCHEMA_CONFLICT"}
        return {
            "success": True,
            "label": label,
            "from_label": from_label,
            "to_label": to_label,
        }

    elif tool_name == "begin_transaction":
        db.begin_transaction()
        return {"success": True}

    elif tool_name == "commit_transaction":
        db.commit()
        return {"success": True}

    elif tool_name == "rollback_transaction":
        db.rollback()
        return {"success": True}

    elif tool_name == "drop_table":
        label = args["label"]
        if not args.get("confirm"):
            return {
                "success": False,
                "error": f"Destructive operation: dropping table '{label}' requires confirm=true",
                "code": "CONFIRMATION_REQUIRED",
            }
        db.drop_table(label)
        return {"success": True, "label": label}

    elif tool_name == "list_schema":
        result = db.execute("CALL SHOW_TABLES() RETURN name, type")
        tables = []
        while result.has_next():
            row = result.get_next()
            tables.append({"name": row[0], "type": row[1]})

        schema: dict[str, list] = {"node_tables": [], "edge_tables": []}

        for table in tables:
            label = table["name"]
            table_type = table["type"]

            columns = []
            try:
                col_result = db.execute(f"CALL table_info('{label}') RETURN *")
                while col_result.has_next():
                    col_row = col_result.get_next()
                    col_names = col_result.get_column_names()
                    columns.append(dict(zip(col_names, col_row)))
            except RuntimeError:
                pass

            if table_type == "NODE":
                count_result = db.execute(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                count = count_result.get_next()[0]
                schema["node_tables"].append({
                    "label": label,
                    "columns": columns,
                    "count": count,
                })
            elif table_type == "REL":
                connections = []
                try:
                    conn_result = db.execute(
                        f"CALL show_connection('{label}') RETURN *"
                    )
                    while conn_result.has_next():
                        conn_row = conn_result.get_next()
                        conn_names = conn_result.get_column_names()
                        connections.append(dict(zip(conn_names, conn_row)))
                except RuntimeError:
                    pass
                schema["edge_tables"].append({
                    "label": label,
                    "columns": columns,
                    "connections": connections,
                })

        return schema

    elif tool_name == "alter_table":
        label = args["label"]
        operation = args["operation"]
        try:
            if operation == "add_column":
                cypher = f"ALTER TABLE {label} ADD {args['column_name']} {args['column_type']}"
                if args.get("default_value"):
                    cypher += f" DEFAULT '{args['default_value']}'"
            elif operation == "drop_column":
                cypher = f"ALTER TABLE {label} DROP {args['column_name']}"
            elif operation == "rename_column":
                cypher = f"ALTER TABLE {label} RENAME {args['old_name']} TO {args['new_name']}"
            elif operation == "rename_table":
                cypher = f"ALTER TABLE {label} RENAME TO {args['new_name']}"
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
            db.execute(cypher)
            return {"success": True, "label": label, "operation": operation}
        except Exception as e:
            return {"success": False, "error": str(e), "label": label}

    elif tool_name == "run_algorithm":
        algo = GraphAlgorithms(db)
        algorithm = args["algorithm"]
        node_label = args["node_label"]
        edge_label = args["edge_label"]

        try:
            if algorithm == "pagerank":
                results = algo.pagerank(
                    node_label, edge_label,
                    damping=args.get("damping", 0.85),
                    iterations=args.get("iterations", 20),
                )
            elif algorithm == "wcc":
                results = algo.weakly_connected_components(node_label, edge_label)
            elif algorithm == "scc":
                results = algo.strongly_connected_components(node_label, edge_label)
            elif algorithm == "louvain":
                results = algo.louvain(
                    node_label, edge_label,
                    max_iterations=args.get("iterations", 10),
                )
            elif algorithm == "k_core":
                results = algo.k_core(node_label, edge_label, k=args.get("k", 2))
            elif algorithm == "degree_centrality":
                results = algo.degree_centrality(node_label, edge_label)
            elif algorithm == "shortest_path":
                path = algo.shortest_path(
                    args["source_id"], args["target_id"], node_label,
                    edge_label=edge_label, max_depth=args.get("max_depth", 10),
                )
                return {
                    "algorithm": algorithm,
                    "results": path if path else [],
                    "count": 1 if path else 0,
                }
            elif algorithm == "node_similarity":
                score = algo.node_similarity(
                    args["source_id"], args["target_id"],
                    node_label, edge_label,
                    metric=args.get("metric", "jaccard"),
                )
                return {
                    "algorithm": algorithm,
                    "results": [{"score": score}],
                    "count": 1,
                }
            else:
                return {"error": f"Unknown algorithm: {algorithm}"}
            return {"algorithm": algorithm, "results": results, "count": len(results)}
        except Exception as e:
            return {"error": str(e), "algorithm": algorithm}

    elif tool_name == "bulk_import":
        exporter = DataExporter(db)
        label = args["label"]
        path = args["path"]
        fmt = args.get("format", "csv")

        try:
            if fmt == "csv":
                exporter.from_csv(label, path)
            elif fmt == "parquet":
                exporter.from_parquet(label, path)
            else:
                return {"success": False, "error": f"Unsupported format: {fmt}"}
            count = db.query(f"MATCH (n:{label}) RETURN count(n) AS cnt")[0]["cnt"]
            return {"imported": count, "label": label, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e), "label": label}

    elif tool_name == "create_vector_index":
        vi = VectorIndex(db)
        table = args["table"]
        prop = args["property"]
        index_name = args.get("index_name", f"{table}_{prop}_idx")
        metric = args.get("metric", "cosine")

        try:
            vi.create_index(table, index_name, prop, metric=metric)
            return {
                "success": True, "table": table, "property": prop,
                "index_name": index_name, "metric": metric,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    elif tool_name == "vector_search":
        vi = VectorIndex(db)
        try:
            results = vi.search(
                args["table"], args["index_name"], args["query_vector"],
                k=args.get("k", 10),
            )
            return {"results": results, "count": len(results)}
        except Exception as e:
            return {"error": str(e)}

    elif tool_name == "hybrid_search":
        vi = VectorIndex(db)
        try:
            results = vi.hybrid_search(
                args["table"], args["index"], args["query_vector"],
                k=args.get("k", 10),
                traverse_edge=args.get("traverse_edge"),
                traverse_depth=args.get("traverse_depth", 1),
            )
            return {"results": results, "count": len(results)}
        except Exception as e:
            return {"error": str(e)}

    elif tool_name == "get_audit_log":
        target_id = args.get("target_id")
        try:
            if target_id:
                if hasattr(db, "audit_log"):
                    entries = db.audit_log.get_history(target_id)
                else:
                    entries = db.query(
                        f"MATCH (a:_AuditLog) WHERE a.target_id = '{target_id}' "
                        f"RETURN a.* ORDER BY a.ts DESC"
                    )
            else:
                if hasattr(db, "audit_log"):
                    entries = db.audit_log.query(
                        since=args.get("since"),
                        until=args.get("until"),
                        operation=args.get("operation"),
                        actor=args.get("actor"),
                        target_type=args.get("target_type"),
                        limit=args.get("limit", 100),
                    )
                else:
                    where = []
                    if args.get("operation"):
                        where.append(f"a.operation = '{args['operation']}'")
                    if args.get("actor"):
                        where.append(f"a.actor = '{args['actor']}'")
                    if args.get("target_type"):
                        where.append(f"a.target_type = '{args['target_type']}'")
                    if args.get("since"):
                        where.append(f"a.ts >= '{args['since']}'")
                    if args.get("until"):
                        where.append(f"a.ts <= '{args['until']}'")
                    where_clause = " AND ".join(where)
                    if where_clause:
                        where_clause = f" WHERE {where_clause}"
                    limit = args.get("limit", 100)
                    entries = db.query(
                        f"MATCH (a:_AuditLog){where_clause} "
                        f"RETURN a.* ORDER BY a.ts DESC LIMIT {limit}"
                    )
            return {"entries": entries, "count": len(entries)}
        except Exception:
            return {"entries": [], "count": 0}

    elif tool_name == "export_data":
        exporter = DataExporter(db)
        label = args["label"]
        path = args["path"]
        fmt = args.get("format", "parquet")

        try:
            if fmt == "parquet":
                count = exporter.to_parquet(label, path)
            elif fmt == "csv":
                count = exporter.to_csv(label, path)
            else:
                return {"success": False, "error": f"Unsupported format: {fmt}"}
            return {"exported": count, "path": path, "format": fmt}
        except Exception as e:
            return {"success": False, "error": str(e)}

    else:
        return {"error": f"Unknown tool: {tool_name}"}


def main():
    parser = argparse.ArgumentParser(description="Bridgr MCP Server")
    parser.add_argument("--db", required=True, help="Path to database file (.lbug)")
    args = parser.parse_args()

    import asyncio
    from mcp.server.stdio import stdio_server

    server = create_server(args.db)

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
