from typing import Tuple
import torch
from nn_executor.models import DIFFERENTIATE_TENSOR
from .pruner_base import PrunerBase
from .pruning_utils import sigmoid


class Pruner(PrunerBase):
    def __init__(
        self,
        ch: int,
        prunable: bool = False,
        activated: bool = True,
        threshold: float = 0.75,
        num_of_appearances: int = 1,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.threshold = threshold
        self.activated = activated
        self.prunable = prunable
        self.num_of_appearances = num_of_appearances
        self.pruner_weight = torch.nn.Parameter(torch.ones((1, self.ch, 1, 1),
                                                           dtype=torch.float32) * 1.0000008344650269,
                                                requires_grad=activated)
        self.init_ones()
        if activated:
            self.adjustment_mode()

    def extra_repr(self) -> str:
        s = f"ch={self.ch}, prunable={self.prunable}, activated={self.activated}, " \
            + "threshold={self.threshold}, num_of_appearances={self.num_of_appearances}"
        return s

    def pruning_mask_and_multiplier(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.prunable and self.activated:
                mask = self.pruner_weight > self.threshold
            else:
                mask = torch.ones_like(self.pruner_weight, dtype=torch.bool)

            if self.activated:
                multipliers = self.forward(torch.ones(1, dtype=torch.float32))
            else:
                multipliers = torch.ones_like(
                    self.pruner_weight, dtype=torch.float32)

        return mask.flatten(), multipliers.flatten()

    def adjustment_mode(self):
        with torch.no_grad():
            device = self.pruner_weight.device
            self.pruner_weight[:] = (1 - 0.01 * torch.rand((1, self.ch, 1, 1),
                                                           dtype=torch.float32,
                                                           device=device))
        self.pruner_weight.requires_grad = True

    def init_ones(self):
        with torch.no_grad():
            self.pruner_weight[:] = 1.0000008344650269

    def forward(self, x):
        if not self.activated:
            # cloning for differentiate input and output tensor
            return x.clone() if DIFFERENTIATE_TENSOR else x

        s = sigmoid(self.pruner_weight, 100, 86)
        # x = x * (s * self.pruner_weight**2)
        m = s * self.pruner_weight
        with torch.no_grad():
            mask = (self.pruner_weight > self.threshold).to(torch.float32)
        # hard off
        m = mask * m
        # select channels by multiplication
        x = x * m

        return x


class AllOrNothingPruner(Pruner):

    def forward(self, x):
        if not self.activated:
            # cloning for differentiate input and output tensor
            return x.clone() if DIFFERENTIATE_TENSOR else x

        mean = self.pruner_weight.mean()
        s = sigmoid(mean, 100, 86)
        m = s * self.pruner_weight

        with torch.no_grad():
            mask = (mean > self.threshold).to(torch.float32)
            mask = torch.ones_like(self.pruner_weight) * mask

        # hard off
        m = mask * m
        # select channels by multiplication
        x = x * m

        return x

    def pruning_mask_and_multiplier(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.prunable and self.activated:
                mask = self.pruner_weight.mean() > self.threshold
                mask = torch.ones_like(
                    self.pruner_weight, dtype=torch.bool) * mask
            else:
                mask = torch.ones_like(self.pruner_weight, dtype=torch.bool)

            if self.activated:
                multipliers = self.forward(torch.ones(1, dtype=torch.float32))
            else:
                multipliers = torch.ones_like(
                    self.pruner_weight, dtype=torch.float32)

        return mask.flatten(), multipliers.flatten()
