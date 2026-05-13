"""Bridgr MCP Server — AI agent interface to a Bridgr graph database.

Exposes tools: query, read_node, write_node, delete_node, search,
traverse_graph, list_node_types, get_edges, create_node_table,
create_edge_table, list_schema. Runs as a stdio MCP server.

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
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

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
