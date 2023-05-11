from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type
from nn_executor import models as mm
from nn_executor.model_description import ModelDescription
from torch import nn


_TAB = ' ' * 4


@dataclass
class NodeDescription:
    node_idx: int = None
    node_module: nn.Module = None

    is_output_node: bool = False
    is_input_node: bool = False

    _src_nodes_dict: Dict[int, Tuple['NodeDescription', int]] = field(default_factory=dict)
    """{this node input idx : (src node, src node's output idx)}"""
    _dst_nodes_dict: Dict[int, List[Tuple['NodeDescription', int]]] = field(default_factory=dict)
    """{this node output idx : [(dst node, dst node's input idx),]}"""

    @property
    def degree(self) -> int:
        degree = len(self.src_nodes) + len(self.dst_nodes) \
            + self.is_output_node * 2 + self.is_input_node * 2
        return degree

    @property
    def src_nodes(self) -> List['NodeDescription']:
        nodes = [node for (node, src_output_idx) in self._src_nodes_dict.values()]
        return nodes

    @property
    def dst_nodes(self) -> List['NodeDescription']:
        nodes = []
        for output_dst_nodes in self._dst_nodes_dict.values():
            for (node, dst_input_idx) in output_dst_nodes:
                nodes.append(node)
        return nodes

    def add_src(self, input_idx: int, src_node: 'NodeDescription', src_output_idx: int):
        self._src_nodes_dict[input_idx] = (src_node, src_output_idx)
        # sort to keep order of items
        self._src_nodes_dict = dict(sorted(self._src_nodes_dict.items()))

    def add_dst(self, output_idx: int, dst_node: 'NodeDescription', dst_input_idx: int):
        # TODO dict items should be lists
        self._dst_nodes_dict[output_idx] = self._dst_nodes_dict.get(output_idx, []) + [(dst_node, dst_input_idx)]
        # sort to keep order of items
        self._dst_nodes_dict = dict(sorted(self._dst_nodes_dict.items()))

    def __str__(self, indent: str = '') -> str:
        v = ["Node:{",
             _TAB + f"node_idx: {self.node_idx},",
             _TAB + f"module: {self.node_module},",
             _TAB + f"is_input_node: {self.is_input_node},",
             _TAB + f"is_output_node: {self.is_output_node},",
             _TAB + f"sources: {[node.node_idx for node in self.src_nodes]},",
             _TAB + f"destinations: {[node.node_idx for node in self.dst_nodes]}",
             "}"
             ]
        s = '\n'.join(v)

        s = [indent + l for l in s.splitlines()]
        s = '\n'.join(s)
        return s


@dataclass
class BranchDescription:
    branch_idx: int = None
    nodes: List[NodeDescription] = field(default_factory=list)

    @property
    def src_node(self) -> NodeDescription:
         return self.nodes[0]

    @property
    def dst_node(self) -> NodeDescription:
         return self.nodes[-1]

    def __len__(self) -> int:
        return len(self.nodes)

    def __str__(self, indent: str = '') -> str:
        v = ["Branch:{",
             _TAB + f"branch_idx: {self.branch_idx},",
             _TAB + "Nodes: {",
             ',\n'.join([node.__str__(indent=_TAB*2) for node in self.nodes]),
             _TAB + "}"]
        s = ("\n").join(v)

        s = [indent + l for l in s.splitlines()]
        s = '\n'.join(s)
        return s


@dataclass
class NodeConnectedBranches:
    node: NodeDescription = None
    finishes_branches: List[BranchDescription] = field(default_factory=list)
    generates_branches: List[BranchDescription] = field(default_factory=list)

    @property
    def finished_branches_sources(self) -> List[NodeDescription]:
        return [b.src_node for b in self.finishes_branches]

    @property
    def generated_branches_sources(self) -> List[NodeDescription]:
        return [b.dst_node for b in self.generates_branches]


@dataclass
class GraphDescription:
    branches: List[BranchDescription] = field(default_factory=list)
    splitting_nodes: List[NodeDescription] = field(default_factory=list)
    nodes: List[NodeDescription] = field(default_factory=list)

    def __str__(self, indent: str = ''):
        v = ["Graph:{",
             _TAB + "Nodes: {",
            ',\n'.join([node.__str__(indent=_TAB*2) for node in self.nodes]) + ',',
            _TAB + "},",
             _TAB + "Splitting_nodes: {",
            ',\n'.join([node.__str__(indent=_TAB*2) for node in self.splitting_nodes]) + ',',
            _TAB + "},",
             _TAB + "Branches: {",
            ',\n'.join([branch.__str__(indent=_TAB*2) for branch in self.branches]),
            _TAB + "}",
            ]
        s = '\n'.join(v)
        s = [indent + l for l in s.splitlines()]
        s = '\n'.join(s)
        return s

    def __make_nodes(self, md: ModelDescription):
        nodes = []
        for node_idx, unique_layer_idx in enumerate(md.layers_indices):
            node = NodeDescription(node_idx=node_idx+1, node_module=md.unique_layers[unique_layer_idx])
            nodes.append(node)

        input_node = NodeDescription(0, mm.Identity(), is_input_node=True)
        output_node = NodeDescription(len(nodes)+1, mm.Identity(), is_output_node=True)

        self.nodes = [input_node] + nodes + [output_node]

    def __connect_nodes(self, md: ModelDescription):
        OUTPUT_NODE_IDX = len(md.layers_indices) + 1
        output_connections = [(src_idx, out_idx, OUTPUT_NODE_IDX, dst_input_idx)
                                for src_idx, out_idx, dst_input_idx in md.outputs]
        for (src_idx, src_output_idx, dst_idx, dst_input_idx) in md.connections + output_connections:
            src_node = self.nodes[src_idx]
            dst_node = self.nodes[dst_idx]
            src_node.add_dst(src_output_idx, dst_node, dst_input_idx)
            dst_node.add_src(dst_input_idx, src_node, src_output_idx)

    def __find_splitting_nodes(self):
        splitting_nodes = [node for node in self.nodes if node.degree > 2]
        self.splitting_nodes = splitting_nodes

    def __make_branches(self):
        branches = []
        for splitting_node in self.splitting_nodes:
            # iterate over node sources
            for prev_node in splitting_node.src_nodes:
                branch_nodes = [splitting_node,]
                iter_node = prev_node
                # walk through the nodes path, till next splitting node
                while iter_node not in self.splitting_nodes:
                    # node exist and is not branched connection
                    branch_nodes.append(iter_node)
                    iter_node = iter_node.src_nodes[0]
                # add splitting node of branch src
                branch_nodes.append(iter_node)

                branch = BranchDescription(branch_idx=len(branches), # next free idx
                                           nodes=branch_nodes[::-1]) # order from src to dst
                branches.append(branch)

        self.branches = branches

    @staticmethod
    def make_from_model_description(md: ModelDescription) -> 'GraphDescription':
        # assumption: nodes generates single output !!!
        gd = GraphDescription()
        # create nodes
        gd.__make_nodes(md)
        # connect
        gd.__connect_nodes(md)
        # find nodes, which are the crossing of
        gd.__find_splitting_nodes()
        # find branches
        gd.__make_branches()
        return gd

    def get_nodes_with_module_type(self, module_type: Type[nn.Module]) -> List[NodeDescription]:
        # nodes = [node for node in self.nodes if isinstance(node.node_module, module_type)]
        nodes = []
        for node in self.nodes:
            if isinstance(node.node_module, module_type):
                nodes.append(node)
        return nodes

    def splitting_nodes_with_branches(self) -> Dict[int, NodeConnectedBranches]:
        nodes_connected_branches: Dict[int, NodeConnectedBranches] = {}
        for node in self.splitting_nodes:
            node_connected = NodeConnectedBranches(node)
            for branch in self.branches:
                if branch.src_node is node:
                    node_connected.generates_branches.append(branch)
                if branch.dst_node is node:
                    node_connected.finishes_branches.append(branch)

            nodes_connected_branches[node.node_idx] = node_connected

        return nodes_connected_branches