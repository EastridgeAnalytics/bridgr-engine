// In-engine GVE-Leiden F1 gate.
// Loads the fragmented ER match graph (nodes.csv / edges.csv from er_gate_gen.py),
// projects it, runs CALL LEIDEN (now backed by GVE-Leiden), and exports the
// per-record community assignment for pairwise-F1 scoring against ground truth.
CREATE NODE TABLE Record(id INT64 PRIMARY KEY);
CREATE REL TABLE Match(FROM Record TO Record);
COPY Record FROM '/root/work/nodes.csv' (header=true);
COPY Match FROM '/root/work/edges.csv' (header=true);
CALL PROJECT_GRAPH('G', ['Record'], ['Match']);
COPY (CALL LEIDEN('G') RETURN node.id AS record_id, community_id AS community) TO '/root/work/membership_engine.csv' (header=true);
