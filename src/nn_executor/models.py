from typing import List, Tuple
import torch
from torch import nn


class Identity(nn.Module):

    DIFFERENTIATE_TENSOR = False

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *args):
        if Identity.DIFFERENTIATE_TENSOR:
            args = (a.clone() for a in args)
        
        if len(args) == 0:
            return None
            
        elif len(args) == 1:
            return args[0]
        
        else:
            return args


class Add(nn.Module):
    def __init__(self, num:int=1) -> None:
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


class Constant(nn.Module):

    def __init__(self, t:torch.Tensor) -> None:
        super().__init__()
        self.t = torch.nn.Parameter(t,requires_grad=False)

    def forward(self):
        return self.t
    
    def extra_repr(self) -> str:
        return 't='+str(self.t)


class Variable(nn.Module):

    def __init__(self, t:torch.Tensor) -> None:
        super().__init__()
        self.t = torch.nn.Parameter(t,requires_grad=True)

    def forward(self):
        return self.t

    def extra_repr(self) -> str:
        return 't='+str(self.t)


class Sub(nn.Module):
    def __init__(self, num=1) -> None:
        super().__init__()
        self.num = num
    
    def extra_repr(self) -> str:
        return 'num=%i'%self.num
        
    def forward(self, *args):
        x = args[0]
        
        for a in args[1:]:
            x = torch.sub(x, a)
        
        # update num of sum args
        self.num = len(args)
        
        return x


class Mul(nn.Module):
    def __init__(self, num=1) -> None:
        super().__init__()
        self.num = num
    
    def extra_repr(self) -> str:
        return 'num=%i'%self.num
        
    def forward(self, *args):
        x = args[0]
        
        if len(args) > 1:
            x = torch.mul(x, args)
        
        # update num of mul args
        self.num = len(args)
        
        return x


class Cat(nn.Module):
    def __init__(self, dim=1) -> None:
        super().__init__()
        self.dim = 1
        self.input_shapes = []
        self.output_shape = None
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}"
    
    def forward(self, *x:torch.Tensor):
        self.input_shapes = [ t.shape for t in x]

        x = torch.cat(x, dim=self.dim)
        self.output_shape = x.shape
        
        return x


def sigmoid(x:torch.Tensor,
            a:torch.Tensor,
            b:torch.Tensor
            ):
    x = torch.exp(a*x-b)
    return x / (1+x)


class Pruner(nn.Module):
    
    ACTIVATED = False
    
    def __init__(self, ch) -> None:
        super().__init__()
        self.ch = ch
        self.weights = None
        self.mode = 'none'
        self.parent_pruner:List[ResidualPruner] = []
        self.prunable = False
    
    def get_mask(self,threashold=0.8):
        with torch.no_grad():
            mask = self.weights > threashold
            multipliers = self(1).flatten()
            
        pass
    
    def set_parent(self, p:'ResidualPruner'):
        self.parent_pruner = [p]
        
    def adjustment_mode(self,device=torch.device('cpu')):
        self.weights = torch.nn.Parameter(1-torch.rand((1,self.ch,1,1), 
                                                       dtype=torch.float32,
                                                       device=device)*0.01, 
                                          requires_grad=False)
        self.weights.requires_grad = True
        self.mode = 'train'

    def forward(self,x):
        if not Pruner.ACTIVATED and self.mode == 'none':
            return x
        
        if self.mode == 'train':
            s = sigmoid(self.weights, 
                        torch.tensor(100.0, dtype=torch.float32),
                        torch.tensor(86.0, dtype=torch.float32))
            x = x * (s * self.weights**2)
        
        return x


class ResidualPruner(Pruner):

    def __init__(self, ch, *dependent_prunners:Pruner) -> None:
        super().__init__(ch)
        self.dependent_pruners:List[Pruner] = []
        
        self.add_child_prunner(*dependent_prunners)
        
    @property
    def num_of_blocks(self):
        return len(self.dependent_pruners)
    
    def add_child_prunner(self, *ps:Pruner):
        for p in ps:
            self.dependent_pruners.append(p)
            p.set_parent(self)


class Parallel(nn.Module):
    def __init__(self,
                 merger:nn.Module, 
                 branches:List[nn.Module]
                 ) -> None:
        super().__init__()
        self.merger = merger
        self.branches = branches
        
        for i,b in enumerate(branches):
            self.add_module('Branch_'+str(i),b)
        
    def forward(self, x):
        results = (B(x) for B in self.branches)
        
        return self.merger(*results)
        