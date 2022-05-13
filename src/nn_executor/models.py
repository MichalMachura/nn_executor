from turtle import forward, position
from typing import List, OrderedDict, Tuple
import torch
from torch import nn


DIFFERENTIATE_TENSOR = False


class Identity(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *args):
        if DIFFERENTIATE_TENSOR:
            args = (a.clone() for a in args)
        
        if len(args) == 0:
            return None
            
        elif len(args) == 1:
            return args[0]
        
        else:
            return args


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


class Elementwise(nn.Module):
    def __init__(self, num, op) -> None:
        super().__init__()
        self.op = op
        self.num = num

    def extra_repr(self) -> str:
        return 'num=%i'%self.num
        
    def forward(self, *args):
        x = args[0]
        for a in args[1:]:
            x = self.op(x, a)
        
        # update num of args
        self.num = len(args)
        
        return x

class Add(Elementwise):
    def __init__(self, num:int=1) -> None:
        super().__init__(num,torch.add)

class Sub(Elementwise):
    def __init__(self, num:int=1) -> None:
        super().__init__(num,torch.sub)

class Mul(Elementwise):
    def __init__(self, num:int=1) -> None:
        super().__init__(num,torch.mul)


class WithBatchNorm2d(nn.Module):
    def __init__(self, 
                 module:nn.Module,
                 ch:int
                 ) -> None:
        super().__init__()
        self.module:nn.Module = module
        self.bn = nn.BatchNorm2d(ch)
        
    def extra_repr(self) -> str:
        s = f'module={self.module}, ch={self.bn.num_features}'
        return s
    
    def forward(self, *args):
        y = self.module(*args)
        y = self.bn(y)
        return y


class MultiWithConst(nn.Module):
    def __init__(self, 
                 module:nn.Module) -> None:
        super().__init__()
        self.module:nn.Module = module
        self.params = nn.ModuleList() # Constants or Variables modules
        self.positions = nn.ParameterList()
        
    def extra_repr(self) -> str:
        s = f'module={self.module}, ch={self.bn.num_features}'
        return s
    
    def add(self,
            pos:int,
            module:nn.Module,
            ):
        with torch.no_grad():
            self.params.append(module)
            self.positions.append(pos)
    
    def forward(self, *args):
        args_iter = iter(args)
        joined_args = []
        
        pos_param = list(zip(self.params,self.positions))
        pos_param = sorted(pos_param,key=lambda x : x[0])
        
        for pos, param in pos_param:
            # fill with input args
            while len(joined_args) != pos:
                input_ = next(args_iter)
                joined_args.append(input_)
                
            # add parameter to list of inputs
            joined_args.append(param())
        
        # forward by module
        y = self.module(*joined_args)
        
        return y
        
        

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


class ChannelsLogger(torch.nn.Module):
    def __init__(self, 
                 channels:List[int]=[]) -> None:
        super().__init__()
        self.channels:List[int] = channels
    
    def extra_repr(self) -> str:
        return f"channels={self.channels}"

    def forward(self, *x:torch.Tensor):
        # update channels info
        self.channels = [ t.shape[1] for t in x]        
        return x


class OutputLayer(ChannelsLogger):
    def __init__(self, channels:List[int]=[]) -> None:
        super().__init__(channels)
        

class InputLayer(ChannelsLogger):
    def __init__(self, channels:List[int]=[]) -> None:
        super().__init__(channels)


def sigmoid(x:torch.Tensor,
            a:torch.Tensor,
            b:torch.Tensor
            ):
    x = torch.exp(a*x-b)
    return x / (1+x)


class Pruner(nn.Module):
        
    def __init__(self, 
                 ch, 
                 prunable=False,
                 activated=True
                 ) -> None:
        super().__init__()
        self.ch = ch
        self.activated = activated
        self.prunable = prunable
        self.weights = torch.nn.Parameter(torch.ones((1,self.ch,1,1), 
                                                       dtype=torch.float32)*1.00376, 
                                          requires_grad=activated)
        self.parent_pruner:List[ResidualPruner] = []
        
        if activated:
            self.adjustment_mode()
    
    def state_dict(self, *args, **kw):
        sd = super().state_dict(*args, **kw)
        
        if not self.activated:
            sd.pop(args[0]+'weights')
    
        return sd
    
    def extra_repr(self) -> str:
        s = ''
        s += f"ch={self.ch}, prunable={self.prunable}, activated={self.activated}"
        return s
    
    def get_mask(self) -> Tuple[torch.Tensor,torch.Tensor]:
        threshold=0.8
        
        with torch.no_grad():
            
            if self.prunable and self.activated:
                mask = self.weights > threshold
            else:
                mask = torch.ones_like(self.weights,dtype=torch.bool)
            
            if self.activated:
                multipliers = self(1)
            else:
                multipliers = torch.ones_like(self.weights,dtype=torch.float32)
            
            return mask.flatten(), multipliers.flatten()
        
    def set_parent(self, p:'ResidualPruner'):
        self.parent_pruner = [p]
        self.prunable = False
        
    def adjustment_mode(self):
        with torch.no_grad():
            device = self.weights.device
            self.weights[:] = 1-torch.rand((1,self.ch,1,1),dtype=torch.float32,device=device)*0.01

        self.weights.requires_grad = True
    
    def init_ones(self):
        with torch.no_grad():
            self.weights[:] = 1.00376

    
    def forward(self,x):
    
        if not self.activated:
            return x
        else:
            s = sigmoid(self.weights, 
                        torch.tensor(100.0, dtype=torch.float32),
                        torch.tensor(86.0, dtype=torch.float32))
            x = x * (s * self.weights**2)
        
        return x


class ResidualPruner(Pruner):
    def __init__(self, 
                 ch, 
                 *dependent_prunners:Pruner, 
                 mode='none', 
                 prunable=False) -> None:
        super().__init__(ch,mode,prunable)
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
        