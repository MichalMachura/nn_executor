from typing import Tuple
import abc
import torch


class PrunerBase(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def pruning_mask_and_multiplier(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method returns bool mask (False means to channel to remove)
        and multiplier of channels.
        """
        NotImplementedError()
