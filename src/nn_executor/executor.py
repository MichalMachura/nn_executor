from typing import Any, Dict, List, Tuple
from torch import nn
from nn_executor import models
from nn_executor.executor_nodes import Node, InputNode, OutputNode
from nn_executor.executor_nodes import DATA_TYPE


class ExecutorException(Exception):
    def __init__(self, node: Node, e: Exception) -> None:
        super().__init__()
        self.e = e
        self.node = node

    def __str__(self) -> str:
        s = str(self.e)
        s += f"{repr(self.L)};\n"
        s += f"Inputs shapes: {[v.shape for v in self.node.inputs_values]}\n"

        for dst_input_idx, src_output_idx, dst_node in self.node.outputs:
            s += f"dst_input_idx: {dst_input_idx}, src_output_idx: {src_output_idx}, dst_node: {repr(src_output_idx)} \n"

        return s

    def __repr__(self) -> str:
        return self.__str__()


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

    def forward(self, *args) -> List[DATA_TYPE]:
        for i, x in enumerate(args):
            _ = self.nodes[0].set_input(i, x)

        if not self.nodes[0].is_active:
            raise RuntimeError("First node has not been activated.")

        active_layers = [n for n in self.nodes if n.is_active]
        while len(active_layers):
            L = active_layers.pop(0)
            try:
                newly_activated = L()
            except RuntimeError as e:
                raise ExecutorException(L, e)
            except Exception as e:
                raise ExecutorException(L, e)

            active_layers.extend(newly_activated)

        outputs = self.nodes[-1]()

        return outputs
