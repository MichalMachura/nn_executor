from dataclasses import dataclass


@dataclass
class Connection:
    src_node: int = None
    src_node_out_idx: int = None
    dst_node: int = None
    dst_node_in_idx: int = 0
