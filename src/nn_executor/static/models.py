from typing import Tuple
from nn_executor.models import PrunerBase
import torch


class StaticPruner(PrunerBase):
    def __init__(self, mask: torch.Tensor = None) -> None:
        super().__init__()
        self._cntr: int = 0
        self._mask: torch.Tensor = mask

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    @mask.getter
    def _(self, mask: torch.Tensor):
        self._mask = mask

    def pruning_mask_and_multiplier(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask
        multiplier = torch.ones_like(self.mask) * mask
        return mask, multiplier
