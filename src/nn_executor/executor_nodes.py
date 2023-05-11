from typing import List, Literal, Tuple
import torch
from torch import nn


DATA_TYPE = torch.Tensor


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
        self.layer: nn.Module = layer
        self.node_idx: int = node_idx
        # [(dst_input_idx, src_output_idx, dst_node)]
        self.outputs: List[Tuple[int, int, 'Node']] = []
        self.degree: int = degree
        self.ready_inputs_cntr: int = 0
        self.inputs_values: List[DATA_TYPE] = [None for i in range(degree[0])]

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


class _Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        if len(args) == 0:
            return None
        if len(args) == 1:
            return args[0]
        return args


class InputNode(Node):
    def __init__(self, num_of_inputs: int = 1) -> None:
        super().__init__(_Identity(), (num_of_inputs, num_of_inputs), 0)
        self.inputs_values = [None for i in range(num_of_inputs)]


class OutputNode(Node):
    def __init__(self,
                 num_of_outputs: int,
                 node_idx: int) -> None:
        super().__init__(_Identity(), (num_of_outputs, num_of_outputs), node_idx)

    @property
    def is_active(self) -> Literal[False]:
        return False

    def set_input(self, idx: int, x) -> 'Node':
        _ = super().set_input(idx, x)
        # always return None -- prevent execution as basic node
        return None

    def __call__(self) -> List[DATA_TYPE]:
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
