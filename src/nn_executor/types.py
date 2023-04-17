from dataclasses import dataclass
from typing import Tuple, Union
import torch


@dataclass
class ModifierType:
    mask: torch.Tensor = None
    multiplier: torch.Tensor = None
    bias: torch.Tensor = None

    def __iter__(self):
        return iter((self.mask, self.multiplier, self.bias))


# TO UPDATE: types based on ModifierType ()

FORWARD_TYPE = Tuple[torch.Tensor,  # mask
                     Union[torch.Tensor,  # backwarded multiplier if cannot be applied on previous node
                           None],
                     Union[torch.Tensor,  # bias, if node input is deleted
                           None],
                     ]  # (bool mask, mul or None, bias or None)


BACKWARD_TYPE = Tuple[torch.Tensor,  # mask
                      torch.Tensor,  # multiplier
                      ]  # (bool mask, multiplier)
