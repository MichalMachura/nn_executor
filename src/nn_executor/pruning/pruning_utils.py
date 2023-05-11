import torch
from typing import Iterable, Tuple
from nn_executor import utils


def on_propagate_backward(mask: torch.Tensor,  # bool
                          src_node_idx: int,
                          src_node_output_idx: int,
                          dst_node_idx: int,
                          dst_node_input_idx: int
                          ) -> None:
    utils.log_print("on_propagate_backward:",
                    mask,
                    (src_node_idx, src_node_output_idx),
                    (dst_node_idx, dst_node_input_idx))
    pass


def on_propagate_forward(mask_mul: Tuple[torch.Tensor, torch.Tensor],  # bool, float32
                         src_node_idx: int,
                         src_node_output_idx: int,
                         dst_node_idx: int,
                         dst_node_input_idx: int
                         ) -> None:
    utils.log_print("on_propagate_forward:",
                    mask_mul,
                    (src_node_idx, src_node_output_idx),
                    (dst_node_idx, dst_node_input_idx))
    pass


def on_available(node_idx: int):
    utils.log_print("on_available:", node_idx)
    pass


CONNECTION = Tuple[int, int, int, int]
CONNECTION_ANCHOR = Tuple[int, int]  # Node idx, input/output idx


def are_same(L1: Iterable, *L_other: Iterable) -> bool:
    if len(L_other) == 0:
        raise RuntimeError("To compare containers, there are needed at least 2 iterators.")

    lengths = [len(l) == len(L1) for l in L_other]

    if sum(lengths) != len(lengths):
        return False

    for l in L_other:
        for item, item_1 in zip(l, L1):
            if item != item_1:
                return False
    return True


def sigmoid(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    x = torch.exp(-a * x + b)
    return 1 / (1 + x)
