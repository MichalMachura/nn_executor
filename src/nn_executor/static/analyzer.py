from typing import List, Tuple, Union
from nn_executor.connection import Connection
from nn_executor.model_description import ModelDescription
# from nn_executor.static.models import StaticPruner


class StaticAnalyzer:
    def __init__(self, md:ModelDescription) -> None:
        self.md = md.copy()
        self.new_md = None

    @staticmethod
    def find_src_for_dst(connections: List[Tuple[int, int, int, int]],
                         dst_idx: int,
                         dst_in_idx: int = 0) -> Union[int, None]:
        for conn in connections:
            conn = Connection(*conn)
            if conn.dst_node == dst_idx and conn.dst_node_in_idx == dst_in_idx:
                return conn.src_node, conn.src_node_out_idx
        return None, None

    def branch_edges_channels(self, branch: List[int]):

        return

    @staticmethod
    def get_branches(desc: ModelDescription) -> List[List[Tuple]]:
        # assumption: nodes generates single output !!!

        degrees = [0] + [0 for _ in desc.layers_indices] + [2+len(desc.outputs)]
        inputs_sources = [[]] + [[] for _ in desc.layers_indices] + [[d[:2] for d in desc.outputs]]

        for conn in desc.connections:
            conn = Connection(*conn)
            degrees[conn.src_node] += 1
            degrees[conn.dst_node] += 1
            inputs_sources[conn.dst_node].append((conn.src_node,
                                                  conn.dst_node_in_idx))

        # find splitting nodes - branches begins or ends.
        splitting_nodes = [i for i, degree in enumerate(degrees) if degree > 2]
        splitting_nodes.sort()

        branches = []
        for node in splitting_nodes:
            for prev_node, node_in_idx in inputs_sources[node]:
                iter_node = prev_node
                branch = [(node, node_in_idx)]

                while iter_node is not None \
                        and degrees[iter_node] < 3:  # node exist and is not branched connection
                    branch.append((iter_node, 0))
                    iter_node, _ = StaticAnalyzer.find_src_for_dst(desc.connections, iter_node)

                if iter_node is not None:
                    branch.append((iter_node, 0))

                # order from src to dst
                branch = branch[::-1]
                branches.append(branch)

        return branches

    def analyze(self) -> ModelDescription:
        branches = self.get_branches(self.md)

        return ModelDescription()

