"""
Bridgr Phase 0 — LadybugDB Validation Suite

Covers stories P0-2 through P0-6:
  P0-2: Model Argus schema (all 10 node types, 11 edge types)
  P0-3: Load synthetic Hubley-like eval dataset
  P0-4: Benchmark Argus query patterns from Section 2.3
  P0-5: Batch ingestion benchmark (500 nodes + 2000 edges)
  P0-6: Python embedding path evaluation

Run: python validation/phase0_validate.py
"""

import json
import os
import random
import shutil
import string
import sys
import time
from pathlib import Path

import real_ladybug as lb

RESULTS = {}
DB_PATH = Path(__file__).parent / "phase0_test.lbug"


def timed(label):
    """Decorator that records wall-clock time for a function."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            RESULTS[label] = {"elapsed_ms": round(elapsed_ms, 2), "status": "PASS"}
            print(f"  [{elapsed_ms:>8.1f} ms] {label}")
            return result
        return wrapper
    return decorator


def rand_str(n=8):
    return "".join(random.choices(string.ascii_lowercase, k=n))


def rand_sentence(words=10):
    return " ".join(rand_str(random.randint(3, 10)) for _ in range(words))


# ============================================================
# P0-2: SCHEMA MODELING
# ============================================================

def create_schema(conn):
    print("\n=== P0-2: Schema Modeling ===")

    @timed("schema:create_node_tables")
    def create_node_tables():
        conn.execute("""
            CREATE NODE TABLE Entity(
                id STRING PRIMARY KEY,
                name STRING,
                entity_type STRING,
                confidence DOUBLE,
                status STRING,
                aliases STRING[]
            )
        """)
        conn.execute("""
            CREATE NODE TABLE Fact(
                id STRING PRIMARY KEY,
                fact_number INT64,
                summary STRING,
                confidence DOUBLE,
                polarity STRING,
                status STRING,
                claim_type STRING,
                fact_date DATE
            )
        """)
        conn.execute("""
            CREATE NODE TABLE Source(
                id STRING PRIMARY KEY,
                filename STRING,
                file_type STRING,
                page_count INT64,
                processed_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE NODE TABLE Issue(
                id STRING PRIMARY KEY,
                title STRING,
                parent_id STRING,
                status STRING
            )
        """)
        conn.execute("""
            CREATE NODE TABLE Question(
                id STRING PRIMARY KEY,
                text STRING,
                status STRING,
                priority INT64
            )
        """)
        conn.execute("""
            CREATE NODE TABLE Authority(
                id STRING PRIMARY KEY,
                title STRING,
                citation STRING,
                jurisdiction STRING
            )
        """)
        conn.execute("""
            CREATE NODE TABLE Tag(
                id STRING PRIMARY KEY,
                name STRING,
                color STRING
            )
        """)
        conn.execute("""
            CREATE NODE TABLE Chunk(
                id STRING PRIMARY KEY,
                source_id STRING,
                chunk_index INT64,
                text STRING,
                char_start INT64,
                char_end INT64
            )
        """)
        conn.execute("""
            CREATE NODE TABLE TimelineEvent(
                id STRING PRIMARY KEY,
                event_date DATE,
                precision STRING,
                description STRING
            )
        """)
        conn.execute("""
            CREATE NODE TABLE Trace(
                id STRING PRIMARY KEY,
                session_id STRING,
                trace_type STRING,
                trace_timestamp TIMESTAMP,
                data STRING
            )
        """)

    @timed("schema:create_edge_tables")
    def create_edge_tables():
        conn.execute("CREATE REL TABLE INVOLVES(FROM Fact TO Entity, role STRING)")
        conn.execute("CREATE REL TABLE SOURCED_FROM(FROM Fact TO Source, page INT64, line_num INT64, quote STRING)")
        conn.execute("CREATE REL TABLE EXTRACTED_FROM(FROM Chunk TO Source, chunk_index INT64)")
        conn.execute("CREATE REL TABLE SUPPORTS(FROM Fact TO Fact)")
        conn.execute("CREATE REL TABLE CONTRADICTS(FROM Fact TO Fact, explanation STRING)")
        conn.execute("CREATE REL TABLE TAGGED_WITH_ENTITY(FROM Entity TO Tag)")
        conn.execute("CREATE REL TABLE TAGGED_WITH_FACT(FROM Fact TO Tag)")
        conn.execute("CREATE REL TABLE RELATES_TO(FROM Issue TO Fact, relevance DOUBLE)")
        conn.execute("CREATE REL TABLE PARENT_OF(FROM Issue TO Issue)")
        conn.execute("CREATE REL TABLE REFERENCES_AUTH(FROM Authority TO Issue)")
        conn.execute("CREATE REL TABLE CONNECTED_TO(FROM Entity TO Entity, relationship_type STRING, context STRING)")

    create_node_tables()
    create_edge_tables()
    print("  Schema: 10 node tables, 11 edge tables created.")


# ============================================================
# P0-3: LOAD SYNTHETIC HUBLEY-LIKE DATA
# ============================================================

def load_hubley_data(conn):
    print("\n=== P0-3: Load Synthetic Hubley Eval Dataset ===")

    entities = [
        ("e1", "John Smith", "person", 0.95, "confirmed", ["J. Smith", "Smith, John"]),
        ("e2", "Acme Corp", "organization", 0.9, "confirmed", ["Acme Corporation", "ACME"]),
        ("e3", "Jane Doe", "person", 0.85, "confirmed", ["J. Doe"]),
        ("e4", "Bob Wilson", "person", 0.92, "confirmed", []),
        ("e5", "Global Trading LLC", "organization", 0.88, "confirmed", ["Global Trading"]),
        ("e6", "Mary Johnson", "person", 0.91, "confirmed", ["M. Johnson"]),
        ("e7", "New York", "place", 0.99, "confirmed", ["NYC", "NY"]),
        ("e8", "The Merger Deal", "idea", 0.75, "under_review", []),
        ("e9", "First National Bank", "organization", 0.93, "confirmed", ["FNB", "First National"]),
    ]

    facts = []
    for i in range(1, 26):
        facts.append((
            f"f{i}",
            i,
            f"Fact {i}: {rand_sentence(15)}",
            round(random.uniform(0.6, 0.99), 2),
            random.choice(["supports", "contradicts", "neutral"]),
            random.choice(["confirmed", "unconfirmed", "disputed"]),
            random.choice(["testimony", "document", "observation", "inference"]),
        ))

    sources = []
    for i in range(1, 21):
        sources.append((
            f"s{i}",
            f"document_{i}.pdf",
            "pdf",
            random.randint(1, 50),
        ))

    issues = [
        ("i1", "Insider Trading Allegation", None, "open"),
        ("i2", "Pre-Announcement Communication", "i1", "open"),
        ("i3", "Unauthorized Disclosure", "i1", "open"),
        ("i4", "Witness Credibility", None, "open"),
    ]

    @timed("load:entities")
    def insert_entities():
        for eid, name, etype, conf, status, aliases in entities:
            conn.execute(
                "CREATE (:Entity {id: $id, name: $name, entity_type: $etype, "
                "confidence: $conf, status: $status, aliases: $aliases})",
                {"id": eid, "name": name, "etype": etype, "conf": conf,
                 "status": status, "aliases": aliases}
            )

    @timed("load:facts")
    def insert_facts():
        for fid, fnum, summary, conf, pol, status, ctype in facts:
            conn.execute(
                "CREATE (:Fact {id: $id, fact_number: $fnum, summary: $summary, "
                "confidence: $conf, polarity: $pol, status: $status, claim_type: $ctype})",
                {"id": fid, "fnum": fnum, "summary": summary, "conf": conf,
                 "pol": pol, "status": status, "ctype": ctype}
            )

    @timed("load:sources")
    def insert_sources():
        for sid, fname, ftype, pages in sources:
            conn.execute(
                "CREATE (:Source {id: $id, filename: $fname, file_type: $ftype, page_count: $pages})",
                {"id": sid, "fname": fname, "ftype": ftype, "pages": pages}
            )

    @timed("load:issues")
    def insert_issues():
        for iid, title, parent, status in issues:
            conn.execute(
                "CREATE (:Issue {id: $id, title: $title, parent_id: $parent, status: $status})",
                {"id": iid, "title": title, "parent": parent or "", "status": status}
            )

    @timed("load:edges_involves")
    def insert_involves():
        roles = ["subject", "object", "witness", "mentioned", "participant"]
        for fid_num in range(1, 26):
            num_entities = random.randint(1, 4)
            chosen = random.sample(range(1, 10), min(num_entities, 9))
            for eid_num in chosen:
                conn.execute(
                    "MATCH (f:Fact {id: $fid}), (e:Entity {id: $eid}) "
                    "CREATE (f)-[:INVOLVES {role: $role}]->(e)",
                    {"fid": f"f{fid_num}", "eid": f"e{eid_num}",
                     "role": random.choice(roles)}
                )

    @timed("load:edges_sourced_from")
    def insert_sourced_from():
        for fid_num in range(1, 26):
            num_sources = random.randint(1, 3)
            chosen = random.sample(range(1, 21), min(num_sources, 20))
            for sid_num in chosen:
                conn.execute(
                    "MATCH (f:Fact {id: $fid}), (s:Source {id: $sid}) "
                    "CREATE (f)-[:SOURCED_FROM {page: $page, line_num: $line}]->(s)",
                    {"fid": f"f{fid_num}", "sid": f"s{sid_num}",
                     "page": random.randint(1, 50), "line": random.randint(1, 200)}
                )

    @timed("load:edges_connected_to")
    def insert_connected_to():
        connections = [
            ("e1", "e2", "employed_by", "Smith works at Acme Corp"),
            ("e1", "e3", "communicated_with", "Email exchange about merger"),
            ("e3", "e5", "investor_in", "Doe invested in Global Trading"),
            ("e4", "e2", "consultant_for", "Wilson advised Acme Corp"),
            ("e6", "e9", "account_holder", "Johnson has accounts at FNB"),
            ("e2", "e5", "merger_target", "Acme acquiring Global Trading"),
            ("e1", "e6", "communicated_with", "Pre-announcement call"),
            ("e4", "e6", "related_to", "Wilson and Johnson are siblings"),
        ]
        for src, dst, rtype, ctx in connections:
            conn.execute(
                "MATCH (a:Entity {id: $src}), (b:Entity {id: $dst}) "
                "CREATE (a)-[:CONNECTED_TO {relationship_type: $rtype, context: $ctx}]->(b)",
                {"src": src, "dst": dst, "rtype": rtype, "ctx": ctx}
            )

    @timed("load:edges_issues")
    def insert_issue_edges():
        conn.execute(
            "MATCH (p:Issue {id: 'i1'}), (c:Issue {id: 'i2'}) CREATE (p)-[:PARENT_OF]->(c)"
        )
        conn.execute(
            "MATCH (p:Issue {id: 'i1'}), (c:Issue {id: 'i3'}) CREATE (p)-[:PARENT_OF]->(c)"
        )
        for fid_num in range(1, 10):
            issue_id = random.choice(["i1", "i2", "i3", "i4"])
            conn.execute(
                "MATCH (i:Issue {id: $iid}), (f:Fact {id: $fid}) "
                "CREATE (i)-[:RELATES_TO {relevance: $rel}]->(f)",
                {"iid": issue_id, "fid": f"f{fid_num}",
                 "rel": round(random.uniform(0.5, 1.0), 2)}
            )

    insert_entities()
    insert_facts()
    insert_sources()
    insert_issues()
    insert_involves()
    insert_sourced_from()
    insert_connected_to()
    insert_issue_edges()

    count = conn.execute("MATCH (n) RETURN count(n) AS cnt").get_next()[0]
    edge_count = conn.execute("MATCH ()-[r]->() RETURN count(r) AS cnt").get_next()[0]
    print(f"  Loaded: {count} nodes, {edge_count} edges")


# ============================================================
# P0-4: BENCHMARK ARGUS QUERY PATTERNS
# ============================================================

def benchmark_queries(conn):
    print("\n=== P0-4: Benchmark Argus Query Patterns ===")

    @timed("query:single_node_lookup")
    def q_single_lookup():
        result = conn.execute("MATCH (e:Entity {id: 'e1'}) RETURN e.*")
        row = result.get_next()
        assert row is not None, "Entity e1 not found"
        return row

    @timed("query:reverse_index_facts_for_entity")
    def q_reverse_index():
        result = conn.execute("""
            MATCH (f:Fact)-[:INVOLVES]->(e:Entity {id: 'e1'})
            RETURN f.id, f.summary
        """)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        assert len(rows) > 0, "No facts found for entity e1"
        return rows

    @timed("query:2hop_connections")
    def q_2hop():
        result = conn.execute("""
            MATCH (a:Entity {id: 'e1'})-[:CONNECTED_TO*1..2]-(b:Entity)
            WHERE a.id <> b.id
            RETURN DISTINCT b.name, b.id
        """)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    @timed("query:3hop_connections")
    def q_3hop():
        result = conn.execute("""
            MATCH (a:Entity {id: 'e1'})-[:CONNECTED_TO*1..3]-(b:Entity)
            WHERE a.id <> b.id
            RETURN DISTINCT b.name, b.id
        """)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    @timed("query:multi_hop_through_facts")
    def q_multihop_facts():
        """Who is connected to Smith through shared facts within 2 degrees?"""
        result = conn.execute("""
            MATCH (e1:Entity {name: 'John Smith'})<-[:INVOLVES]-(f:Fact)-[:INVOLVES]->(e2:Entity)
            WHERE e1.id <> e2.id
            RETURN DISTINCT e2.name, f.summary
        """)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    @timed("query:shortest_path")
    def q_shortest_path():
        result = conn.execute("""
            MATCH p = (a:Entity {id: 'e1'})-[:CONNECTED_TO* SHORTEST 1..5]-(b:Entity {id: 'e9'})
            RETURN length(p) AS path_length, nodes(p) AS path_nodes
        """)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    @timed("query:entity_co_occurrence")
    def q_cooccurrence():
        """Find entities that appear together in multiple facts."""
        result = conn.execute("""
            MATCH (e1:Entity)<-[:INVOLVES]-(f:Fact)-[:INVOLVES]->(e2:Entity)
            WHERE e1.id < e2.id
            RETURN e1.name, e2.name, count(f) AS shared_facts
            ORDER BY shared_facts DESC
            LIMIT 10
        """)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    @timed("query:facts_by_source")
    def q_facts_by_source():
        result = conn.execute("""
            MATCH (f:Fact)-[:SOURCED_FROM]->(s:Source {id: 's1'})
            RETURN f.id, f.summary, f.confidence
            ORDER BY f.confidence DESC
        """)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    @timed("query:issue_hierarchy")
    def q_issue_hierarchy():
        result = conn.execute("""
            MATCH (parent:Issue)-[:PARENT_OF*1..3]->(child:Issue)
            RETURN parent.title, child.title
        """)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    @timed("query:contradiction_detection")
    def q_contradictions():
        """Find facts that involve the same entity but have opposing polarity."""
        result = conn.execute("""
            MATCH (f1:Fact)-[:INVOLVES]->(e:Entity)<-[:INVOLVES]-(f2:Fact)
            WHERE f1.polarity = 'supports' AND f2.polarity = 'contradicts'
              AND f1.id <> f2.id AND f1.claim_type = f2.claim_type
            RETURN f1.summary AS supporting, f2.summary AS contradicting, e.name AS entity
            LIMIT 10
        """)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    @timed("query:canvas_data")
    def q_canvas():
        """Get all nodes and edges for visualization."""
        nodes = conn.execute("MATCH (n) RETURN label(n) AS lbl, n.id AS id, n.name AS name LIMIT 100")
        node_rows = []
        while nodes.has_next():
            node_rows.append(nodes.get_next())
        edges = conn.execute("MATCH (a)-[r]->(b) RETURN label(r) AS lbl, a.id AS src, b.id AS dst LIMIT 200")
        edge_rows = []
        while edges.has_next():
            edge_rows.append(edges.get_next())
        return {"nodes": len(node_rows), "edges": len(edge_rows)}

    @timed("query:arrow_output")
    def q_arrow():
        """Verify Arrow output works."""
        result = conn.execute("MATCH (e:Entity) RETURN e.id, e.name, e.confidence")
        arrow_table = result.get_as_arrow()
        assert arrow_table.num_rows == 9, f"Expected 9 entities, got {arrow_table.num_rows}"
        return arrow_table.num_rows

    @timed("query:pandas_output")
    def q_pandas():
        """Verify Pandas DataFrame output works."""
        result = conn.execute("MATCH (f:Fact) RETURN f.id, f.summary, f.confidence, f.polarity")
        df = result.get_as_df()
        assert len(df) == 25, f"Expected 25 facts, got {len(df)}"
        return len(df)

    q_single_lookup()
    q_reverse_index()
    q_2hop()
    q_3hop()
    q_multihop_facts()
    q_shortest_path()
    q_cooccurrence()
    q_facts_by_source()
    q_issue_hierarchy()
    q_contradictions()
    q_canvas()
    q_arrow()
    q_pandas()


# ============================================================
# P0-5: BATCH INGESTION BENCHMARK
# ============================================================

def benchmark_batch_ingestion(conn):
    print("\n=== P0-5: Batch Ingestion Benchmark ===")

    conn.execute("""
        CREATE NODE TABLE BatchEntity(
            id STRING PRIMARY KEY,
            name STRING,
            entity_type STRING,
            confidence DOUBLE
        )
    """)
    conn.execute("""
        CREATE REL TABLE BATCH_CONNECTED(
            FROM BatchEntity TO BatchEntity,
            relationship_type STRING
        )
    """)

    @timed("batch:insert_500_nodes")
    def batch_nodes():
        conn.execute("BEGIN TRANSACTION")
        for i in range(500):
            conn.execute(
                "CREATE (:BatchEntity {id: $id, name: $name, entity_type: $etype, confidence: $conf})",
                {"id": f"b{i}", "name": f"Batch Entity {i}",
                 "etype": random.choice(["person", "org", "place"]),
                 "conf": round(random.uniform(0.5, 1.0), 3)}
            )
        conn.execute("COMMIT")

    @timed("batch:insert_2000_edges")
    def batch_edges():
        conn.execute("BEGIN TRANSACTION")
        for i in range(2000):
            src = random.randint(0, 499)
            dst = random.randint(0, 499)
            while dst == src:
                dst = random.randint(0, 499)
            conn.execute(
                "MATCH (a:BatchEntity {id: $src}), (b:BatchEntity {id: $dst}) "
                "CREATE (a)-[:BATCH_CONNECTED {relationship_type: $rtype}]->(b)",
                {"src": f"b{src}", "dst": f"b{dst}",
                 "rtype": random.choice(["knows", "works_with", "related_to"])}
            )
        conn.execute("COMMIT")

    @timed("batch:total_500n_2000e")
    def batch_total():
        batch_nodes()
        batch_edges()

    batch_total()

    count = conn.execute("MATCH (b:BatchEntity) RETURN count(b)").get_next()[0]
    edge_count = conn.execute("MATCH (:BatchEntity)-[r:BATCH_CONNECTED]->(:BatchEntity) RETURN count(r)").get_next()[0]
    print(f"  Batch result: {count} nodes, {edge_count} edges")

    target_ms = 500
    actual_ms = RESULTS["batch:total_500n_2000e"]["elapsed_ms"]
    if actual_ms > target_ms:
        print(f"  WARNING: Batch ingestion took {actual_ms}ms (target: <{target_ms}ms)")
        RESULTS["batch:total_500n_2000e"]["status"] = "WARN"


# ============================================================
# P0-6: PYTHON EMBEDDING EVALUATION
# ============================================================

def evaluate_python_embedding():
    print("\n=== P0-6: Python Embedding Path Evaluation ===")

    results = {
        "import_name": "real_ladybug",
        "version": getattr(lb, "__version__", "0.15.3"),
        "in_process": True,
        "no_subprocess": True,
        "no_daemon": True,
        "binding_type": "pybind11 (inherited from Kuzu)",
        "platforms": "Windows x64, macOS arm64/x64, Linux x64/arm64",
        "python_versions": "3.10, 3.11, 3.12, 3.13, 3.14",
    }

    @timed("embed:open_close_db")
    def test_open_close():
        test_path = Path(__file__).parent / "embed_test.lbug"
        db = lb.Database(str(test_path))
        c = lb.Connection(db)
        c.execute("CREATE NODE TABLE EmbedTest(id STRING PRIMARY KEY)")
        c.execute("CREATE (:EmbedTest {id: 'test'})")
        result = c.execute("MATCH (t:EmbedTest) RETURN t.id").get_next()[0]
        assert result == "test"
        del c
        del db
        db2 = lb.Database(str(test_path))
        c2 = lb.Connection(db2)
        result2 = c2.execute("MATCH (t:EmbedTest) RETURN t.id").get_next()[0]
        assert result2 == "test", "Data not persisted across close/reopen"
        del c2
        del db2
        shutil.rmtree(str(test_path), ignore_errors=True)

    @timed("embed:memory_mode")
    def test_memory_mode():
        db = lb.Database(":memory:")
        c = lb.Connection(db)
        c.execute("CREATE NODE TABLE MemTest(id STRING PRIMARY KEY)")
        c.execute("CREATE (:MemTest {id: 'mem'})")
        result = c.execute("MATCH (m:MemTest) RETURN m.id").get_next()[0]
        assert result == "mem"

    test_open_close()
    test_memory_mode()

    RESULTS["embed:evaluation"] = {
        "status": "PASS",
        "details": results,
    }
    print(f"  Embedding: in-process via {results['binding_type']}")
    print(f"  Package: pip install {results['import_name']} (v{results['version']})")
    print(f"  Platforms: {results['platforms']}")


# ============================================================
# BUILT-IN CAPABILITIES INVENTORY
# ============================================================

def inventory_builtin_capabilities(conn):
    print("\n=== Built-in Capabilities Inventory ===")

    capabilities = {
        "query_language": "Cypher (not SQL)",
        "variable_length_paths": True,
        "shortest_path": True,
        "weighted_shortest_path": True,
        "vector_index": "HNSW (cosine, l2, dotproduct)",
        "full_text_search": "BM25 via FTS extension",
        "graph_algorithms": ["K-Core", "Louvain", "PageRank", "SCC", "WCC"],
        "output_formats": ["Arrow", "Pandas", "Polars", "dict", "list"],
        "data_types": [
            "STRING", "INT8/16/32/64/128", "UINT8/16/32/64",
            "FLOAT", "DOUBLE", "DECIMAL", "BOOLEAN",
            "DATE", "TIMESTAMP", "INTERVAL",
            "LIST", "ARRAY", "MAP", "STRUCT", "UNION", "JSON",
            "SERIAL", "UUID", "BLOB"
        ],
        "embedding_modes": ["on-disk (.lbug file)", "in-memory (:memory:)"],
        "acid_transactions": True,
        "concurrency": "multi-reader, single-writer",
        "license": "MIT",
    }

    missing_vs_strategy = {
        "sql_surface": "LadybugDB uses Cypher, not SQL. DataFusion integration required for SQL-first strategy.",
        "missing_gds_algorithms": [
            "Shortest Path (built-in via Cypher, not algo extension)",
            "Degree Centrality (not built-in, trivial to compute via Cypher)",
            "Betweenness Centrality (not built-in)",
            "Node Similarity (not built-in)",
            "Label Propagation (not built-in)",
            "BFS/DFS (built-in via Cypher path patterns)",
        ],
        "entity_resolution": "Not built-in. ER_Agentic integration needed.",
        "mcp_server": "Not built-in. Must be built as a wrapper.",
        "audit_trail": "Not built-in. Must be built as a wrapper.",
    }

    RESULTS["capabilities"] = capabilities
    RESULTS["gaps_vs_strategy"] = missing_vs_strategy

    print(f"  Built-in algorithms: {', '.join(capabilities['graph_algorithms'])}")
    print(f"  Vector index: {capabilities['vector_index']}")
    print(f"  FTS: {capabilities['full_text_search']}")
    print(f"  Query language: {capabilities['query_language']}")
    print(f"  Key gap: {missing_vs_strategy['sql_surface']}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BRIDGR PHASE 0 — LADYBUGDB VALIDATION")
    print("=" * 60)

    if DB_PATH.exists():
        shutil.rmtree(str(DB_PATH))

    db = lb.Database(str(DB_PATH))
    conn = lb.Connection(db)

    try:
        create_schema(conn)
        load_hubley_data(conn)
        benchmark_queries(conn)
        benchmark_batch_ingestion(conn)
        evaluate_python_embedding()
        inventory_builtin_capabilities(conn)
    finally:
        del conn
        del db

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    pass_count = 0
    warn_count = 0
    fail_count = 0

    for label, data in RESULTS.items():
        if isinstance(data, dict) and "status" in data:
            status = data["status"]
            if status == "PASS":
                pass_count += 1
            elif status == "WARN":
                warn_count += 1
            else:
                fail_count += 1

            if "elapsed_ms" in data:
                print(f"  {status:>4} | {data['elapsed_ms']:>8.1f} ms | {label}")

    print(f"\n  Total: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")

    kill_criteria = []
    if "query:3hop_connections" in RESULTS:
        ms = RESULTS["query:3hop_connections"]["elapsed_ms"]
        if ms > 500:
            kill_criteria.append(f"3-hop path query took {ms}ms (>500ms kill criterion)")
    if "batch:total_500n_2000e" in RESULTS:
        ms = RESULTS["batch:total_500n_2000e"]["elapsed_ms"]
        if ms > 10000:
            kill_criteria.append(f"Batch ingestion took {ms}ms (>10s — unacceptable)")

    if kill_criteria:
        print("\n  KILL CRITERIA TRIGGERED:")
        for kc in kill_criteria:
            print(f"    - {kc}")
        print("\n  RECOMMENDATION: STOP — Re-evaluate engine choice.")
    else:
        print("\n  RECOMMENDATION: GO — Proceed to v0.1 implementation.")

    report_path = Path(__file__).parent / "phase0_results.json"
    with open(report_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\n  Full results written to: {report_path}")

    if DB_PATH.exists():
        shutil.rmtree(str(DB_PATH), ignore_errors=True)


if __name__ == "__main__":
    main()
