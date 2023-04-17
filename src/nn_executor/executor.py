from typing import Any, Dict, List, Tuple
import torch
from torch import nn
from nn_executor import models


def get_from_idx(x, idx: int):

    if type(x) in [tuple, list]:
        return x[idx]

    if idx != 0:
        raise RuntimeError("Idx != 0 for non container")

    return x


class Node:

    def __init__(self,
                 layer: nn.Module,
                 degree: Tuple[int, int] = (1, 1),
                 node_idx: int = 0) -> None:
        self.layer = layer
        self.node_idx = node_idx
        # [(dst_input_idx, src_output_idx, dst_node)]
        self.outputs: List[Tuple[int, int, 'Node']] = []
        self.degree = degree
        self.ready_inputs_cntr = 0
        self.inputs_values = [None for i in range(degree[0])]

    def set_input(self, idx: int, x: torch.Tensor) -> 'Node':
        if idx >= len(self.inputs_values):
            raise RuntimeError(
                "idx =", idx, "is over last available idx =", len(self.inputs_values) - 1)

        # set proper input
        self.inputs_values[idx] = x
        # increase cntr
        self.ready_inputs_cntr += 1

        if self.is_active:
            return self

        return None

    @property
    def is_active(self):
        return self.ready_inputs_cntr == len(self.inputs_values)

    def add_src(self, src_output_idx: int, src: 'Node', dst_input_idx: int):
        # resize input buffer to hold all inputs
        if dst_input_idx >= len(self.inputs_values):
            raise RuntimeError(
                f"dst_input_idx = {dst_input_idx} is out of range for input of size {self.degree[0]}")

        # add this node to it's src
        src.outputs.append((dst_input_idx, src_output_idx, self))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layer={self.layer}, degree={self.degree}, node_idx={self.node_idx})"

    def __call__(self) -> List['Node']:
        # basic layer forward
        output = self.layer(*self.inputs_values)
        # free inputs buffers
        self.ready_inputs_cntr = 0
        self.inputs_values = [None for i in self.inputs_values]

        activated = []
        for dst_input_idx, src_output_idx, dst_node in self.outputs:
            # choose one of results
            result_to_dst = get_from_idx(output, src_output_idx)
            # propagate to dst node
            active_node = dst_node.set_input(dst_input_idx, result_to_dst)
            # if node is activated
            if active_node is not None:
                # add to list
                activated.append(active_node)

        # return new activated nodes
        return activated


class InputNode(Node):
    def __init__(self, layer: nn.Module, num_of_inputs: int = 1) -> None:
        super().__init__(layer, (num_of_inputs, num_of_inputs), 0)
        self.inputs_values = [None for i in range(num_of_inputs)]


class OutputNode(Node):
    def __init__(self,
                 layer: nn.Module,
                 num_of_outputs: int,
                 node_idx: int) -> None:
        super().__init__(layer, (num_of_outputs, num_of_outputs), node_idx)

    def set_input(self, idx: int, x) -> 'Node':
        _ = super().set_input(idx, x)
        # always return None -- prevent execution as basic node
        # return None

    def __call__(self):
        # get buffered values
        values = self.inputs_values
        # reset buffers
        self.inputs_values = [None for i in self.inputs_values]
        self.ready_inputs_cntr = 0

        if len(values) == 0:
            return None

        if len(values) == 1:
            return values[0]

        return values


class Executor(nn.Module):

    def __init__(self,
                 model_description: Dict[str, Any],
                 ) -> None:
        super().__init__()
        layers_indices: List[int] = model_description['layers_indices'].copy()
        layers_in_out_channels: List[Tuple[List[int], List[int]]
                                     ] = model_description['layers_in_out_channels'].copy()
        unique_layers: List[nn.Module] = model_description['unique_layers'].copy()
        connections: List[Tuple[int, int, int, int]
                          ] = model_description['connections'].copy()
        outputs: List[Tuple[int, int, int]
                      ] = model_description['outputs'].copy()

        # get num of inputs and outputs of nodes
        layers_degree = [(len(in_out_ch[0]), len(in_out_ch[1]))
                         for in_out_ch in layers_in_out_channels]
        # instances of layers stay the same, only list is different
        self.model_description = model_description.copy()
        # connections list is modified by some of the following functions
        self.update_connections(layers_indices=layers_indices,
                                connections=connections,
                                outputs=outputs)
        self.nodes: List[Node] = self.create_nodes(layers_indices=layers_indices,
                                                   layers_degree=layers_degree,
                                                   unique_layers=unique_layers,
                                                   connections=connections,
                                                   outputs=outputs)
        self.register_layers(unique_layers=unique_layers)
        self.connect(connections=connections)

    def get_state(self):
        return self.model_description.copy()

    def register_layers(self,
                        unique_layers: List[nn.Module]):
        for i, L in enumerate(unique_layers):
            cls = L.__class__.__name__
            self.add_module(f"layer_{i:03}_type_{cls}", L)

    def update_connections(self,
                           layers_indices: List[int],
                           connections: List[Tuple[int, int, int, int]],
                           outputs: List[Tuple[int, int, int]]):
        dst = len(layers_indices) + 1
        for src, src_out_idx, dst_in_idx in outputs:
            link = (src, src_out_idx, dst, dst_in_idx)
            connections.append(link)

    def get_number_of_inputs(self,
                             connections: List[Tuple[int, int, int, int]]):
        noi = 0
        for src, src_out_idx, dst, dst_in_idx in connections:
            if src == 0:
                noi = max(noi, src_out_idx)

        return noi + 1

    def create_nodes(self,
                     layers_indices: List[int],
                     layers_degree: List[Tuple[int, int]],
                     unique_layers: List[nn.Module],
                     connections: List[Tuple[int, int, int, int]],
                     outputs: List[Tuple[int, int, int]],
                     ):
        noi = self.get_number_of_inputs(connections)
        noo = len(outputs)

        nodes = []
        for i, (layer_idx, degree) in enumerate(zip(layers_indices, layers_degree)):
            L = unique_layers[layer_idx]
            nodes.append(Node(L, degree=degree, node_idx=i + 1))

        inL, outL = models.Identity(), models.Identity()

        nodes = [InputNode(inL, noi),
                 *nodes,
                 OutputNode(outL, noo, len(nodes) + 1)]

        return nodes

    def connect(self,
                connections: List[Tuple[int, int, int, int]]):
        for link in connections:
            src, src_out_idx, dst, dst_in_idx = link
            dst_node: Node = self.nodes[dst]
            src_node: Node = self.nodes[src]
            dst_node.add_src(src_out_idx, src_node, dst_in_idx)

    def forward(self, *args):
        for i, x in enumerate(args):
            _ = self.nodes[0].set_input(i, x)

        if not self.nodes[0].is_active:
            raise RuntimeError("First node has not been activated.")

        active_layers = [n for n in self.nodes if n.is_active]
        while len(active_layers):
            L = active_layers.pop(0)
            newly_activated = L()
            active_layers.extend(newly_activated)

        outputs = self.nodes[-1]()

        return outputs
