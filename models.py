from turtle import forward
from typing import List, Tuple
import torch
from torch import nn
from torch import functional as F


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *args):
        return args


class Add(nn.Module):
    def __init__(self, num:int) -> None:
        super().__init__()
        self.num = num
    
    def extra_repr(self) -> str:
        return 'num=%i'%self.num
    
    def forward(self, *args):
        x = args[0]
        
        for a in args[1:]:
            x = torch.add(x, a)
        
        # update num of add args
        self.num = len(args)
        
        return x


class Sub(nn.Module):
    def __init__(self, num) -> None:
        super().__init__()
        self.num = num
    
    def extra_repr(self) -> str:
        return 'num=%i'%self.num
        
    def forward(self, *args):
        x = args[0]
        
        for a in args[1:]:
            x = torch.sub(x, a)
        
        # update num of add args
        self.num = len(args)
        
        return x


class Cat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *x:torch.Tensor):
        return torch.cat(x, dim=1)
        

class Pruner(nn.Module):
    ACTIVATE = False
    
    def __init__(self,ch) -> None:
        super().__init__()
        self.ch = ch
    
class ResidualPrunner(Pruner):
    def __init__(self, ch, dependent_pruners=[]) -> None:
        super().__init__(ch)
        self.dependent_pruners = dependent_pruners
        
        
        