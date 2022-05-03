from typing import List, Tuple, Union
import torch
import torch.nn as nn
from nn_executor import models


class Modifier:
    def __init__(self) -> None:
        pass
    
    def clone(self, module:nn.Module) -> nn.Module:
        raise NotImplementedError("You must implement this method in any subclasses!")
    
    def forward(self, 
                module:nn.Module, 
                masks:List[torch.Tensor]
                ) -> Union[Tuple[nn.Module,
                                 List[torch.Tensor]], None]:
        return module,masks
        
    def backward(self, 
                module:nn.Module, 
                mask_multipliers:List[Tuple[torch.Tensor,
                                            torch.Tensor]]
                ) -> Union[Tuple[nn.Module,
                                 List[Tuple[torch.Tensor,
                                            torch.Tensor]]], None]:
        return module, mask_multipliers
        