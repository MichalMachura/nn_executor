from dataclasses import dataclass, field, fields
from typing import List, Dict, Tuple, Any
import torch


@dataclass
class ModelDescription:
    layers_indices: List[int] = field(default_factory=list)
    """Indices of unique modules: position is index of node, value is index of unique module."""
    unique_layers: List[torch.nn.Module] = field(default_factory=list)
    """List of unique torch modules."""
    layers_in_out_channels: List[List[int]] = field(default_factory=list)
    """Two element list:
    - first is list of number of channels for each input
    - second is list of number of channels for each output"""
    connections: List[Tuple[int, int, int, int]] = field(default_factory=list)
    """[(src_node_idx, src_node_out_idx, dst_node_idx, dst_node_in_idx),]"""
    outputs: List[List[int]] = field(default_factory=list)
    """[(output_node_src_idx, output_node_src_out_idx),]"""

    @staticmethod
    def make_from_dict(d: Dict[str, Any]):
        return ModelDescription(**d)

    def copy(self) -> 'ModelDescription':
        return ModelDescription(**dict(self))

    def __getitem__(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"ModelDescription has no attr named {name}")

    def __setitem__(self, name: str, value: Any):
        if hasattr(self, name):
            return setattr(self, name, value)
        raise AttributeError(f"ModelDescription has no attr named {name}")

    # mapping to dict
    def __iter__(self):
        return iter(fields(self))

    def keys(self):
        keys = [f.name for f in fields(self)]
        return keys

    def values(self):
        values = [self[f.name] for f in fields(self)]
        return values

    def items(self):
        items = [(f.name, self[f.name]) for f in fields(self)]
        return items
