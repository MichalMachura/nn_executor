from dataclasses import dataclass, field
from typing import List, Tuple, Union
from nn_executor.model_description import ModelDescription
from nn_executor import models as mm
from nn_executor.graph_analyzer import GraphDescription, BranchDescription, NodeDescription, NodeConnectedBranches
# from nn_executor.static.models import StaticPruner


@dataclass
class Residue:
    branches: List[BranchDescription] = field(default_factory=list)

    src_node: NodeDescription = None
    dst_node: NodeDescription = None


class StaticAnalyzer:
    def __init__(self, md: ModelDescription) -> None:
        self.md = md.copy()
        self.new_md = None

    def find_residues(self, desc: GraphDescription):
        residues: List[Residue] = []
        adder_nodes = desc.get_nodes_with_module_type(mm.Add)
        nodes_connected_branches = desc.splitting_nodes_with_branches()

        for adder in adder_nodes:
            connected_branches: NodeConnectedBranches = nodes_connected_branches[adder.node_idx]
            finished_branches = connected_branches.finishes_branches
            branches_sources = connected_branches.finished_branches_sources
            # there is more sources than 1
            if len(branches_sources) > 1:
                src_node = branches_sources[0]
                # if all branches have the same source and dst -> this is residual connection
                if sum(node == src_node for node in branches_sources) == len(branches_sources):
                    residue = Residue(branches=finished_branches, src_node=src_node, dst_node=adder)
                    residues.append(residue)

        return residues

    def analyze(self) -> ModelDescription:
        gd = GraphDescription.make_from_model_description(self.md)
        residues = self.find_residues(gd)

        return

