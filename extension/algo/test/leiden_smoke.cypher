// Smoke test: the "Basic" graph from leiden.test (two triangles {0,1,2},{5,6,7}
// bridged through 3,4,8,9). A correct Leiden must NOT merge everything into one
// community. Validates the GVE-Leiden integration on a trivially-checkable graph.
CREATE NODE TABLE Node(id INT64 PRIMARY KEY);
CREATE REL TABLE Edge(FROM Node to Node);
CREATE (u0:Node {id: 0}), (u1:Node {id: 1}), (u2:Node {id: 2}), (u3:Node {id: 3}),
       (u4:Node {id: 4}), (u5:Node {id: 5}), (u6:Node {id: 6}), (u7:Node {id: 7}),
       (u8:Node {id: 8}), (u9:Node {id: 9}),
       (u0)-[:Edge]->(u1), (u0)-[:Edge]->(u2), (u1)-[:Edge]->(u2),
       (u2)-[:Edge]->(u3), (u3)-[:Edge]->(u4), (u5)-[:Edge]->(u6),
       (u5)-[:Edge]->(u7), (u6)-[:Edge]->(u7), (u7)-[:Edge]->(u8),
       (u8)-[:Edge]->(u9), (u2)-[:Edge]->(u5), (u4)-[:Edge]->(u9);
CALL PROJECT_GRAPH('Graph', ['Node'], ['Edge']);
CALL LEIDEN('Graph') WITH community_id, min(node.id) AS commId, count(*) AS nodeCount, list_sort(collect(node.id)) AS nodeIds RETURN commId, nodeCount, nodeIds ORDER BY commId;
