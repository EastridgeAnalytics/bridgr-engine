"""Bridgr exception hierarchy."""


class BridgrError(Exception):
    pass


class NodeNotFoundError(BridgrError):
    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(f"Node not found: {node_id}")


class EdgeNotFoundError(BridgrError):
    def __init__(self, edge_id: str):
        self.edge_id = edge_id
        super().__init__(f"Edge not found: {edge_id}")


class DuplicateNodeError(BridgrError):
    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(f"Node already exists: {node_id}")


class TransactionError(BridgrError):
    pass


class SchemaError(BridgrError):
    pass
