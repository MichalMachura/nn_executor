from typing import List, Tuple, Union
from nn_executor.model_description import ModelDescription
# from nn_executor.static.models import StaticPruner
from nn_executor.connection import Connection


class StaticAnalyzer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def find_src_for_dst(connections: List[Tuple[int, int, int, int]],
                         dst_idx: int,
                         dst_in_idx: int = 0) -> Union[int, None]:
        for conn in connections:
            conn = Connection(conn)
            if conn.dst_node == dst_idx and conn.dst_node_in_idx == dst_in_idx:
                return conn.src_node, conn.src_node_out_idx
        return None, None

    @staticmethod
    def get_branches(desc: ModelDescription) -> List[List[Tuple]]:
        # assumption: nodes generates have single output !!!

        degrees = [0] * len(desc.layers_indices)
        inputs_sources = [[]] * len(desc.layers_indices)

        for conn in desc.connections:
            conn = Connection(conn)
            # degrees[conn.src_node_idx] += 1
            degrees[conn.dst_node_idx] += 1
            inputs_sources[conn.dst_node_idx].append((conn.src_node, conn.dst_node_in_idx))

        splitting_nodes = [i for i, degree in enumerate(degrees) if degree > 2]

        branches = []
        for node in splitting_nodes:
            for prev_node, node_in_idx in inputs_sources[node]:
                iter_node = prev_node
                branch = [(node, node_in_idx), (iter_node, 0)]

                while iter_node is not None \
                        and degrees[iter_node] != 2:  # node exist and is not branched connection
                    iter_node, _ = StaticAnalyzer.find_src_for_dst(desc.connections, iter_node)
                    branch.append((iter_node, 0))

                # order from src to dst
                branch = branch[::-1]
                branches.append(branch)

        return branches

    def analyze(self, desc: ModelDescription) -> ModelDescription:

        return ModelDescription()
