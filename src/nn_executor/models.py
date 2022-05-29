from typing import List, Tuple
import torch
from torch import nn


DIFFERENTIATE_TENSOR = False


class Identity(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *args):
        if DIFFERENTIATE_TENSOR:
            args = [a.clone() for a in args]
        
        if len(args) == 0:
            return None
            
        elif len(args) == 1:
            return args[0]
        
        else:
            return args
            

class Upsample(nn.Upsample):
    def __init__(self, 
                 size = None, 
                 scale_factor = None, 
                 mode: str = 'nearest', 
                 align_corners:bool = None) -> None:
        super().__init__(size, scale_factor, mode, align_corners)

    def extra_repr(self) -> str:
        s = f"size={self.size}, scale_factor={self.scale_factor}, mode='{self.mode}', align_corners={self.align_corners}"
        return s

class CONSTANTS:

    BATCH_DIM_CAT_VAR_CONST = 1

class Constant(nn.Module):

    def __init__(self, t:torch.Tensor) -> None:
        super().__init__()
        self.t = torch.nn.Parameter(t,requires_grad=False)

    def __repr__(self,):
        return f'Constant(t=torch.zeros({tuple(self.t.shape)},dtype={self.t.dtype}))'
        
    def forward(self):
        return torch.cat([self.t]*CONSTANTS.BATCH_DIM_CAT_VAR_CONST,dim=0)
    
    def extra_repr(self) -> str:
        return 't='+str(self.t)


class Variable(Constant):

    def __init__(self, t:torch.Tensor) -> None:
        super().__init__(t)
        self.t.requires_grad = True
        
    def __repr__(self,):
        return f'Variable(t=torch.zeros({tuple(self.t.shape)},dtype={self.t.dtype}))'


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
        
    def __repr__(self) -> str:
        s = f'WithBatchNorm2d(module={self.module}, ch={self.bn.num_features})'
        return s
    
    def forward(self, *args):
        y = self.module(*args)
        y = self.bn(y)
        return y


class MultiWithConst(nn.Module):
    def __init__(self, 
                 module:nn.Module,
                 modules:List[Tuple[torch.nn.Module,int]]=[]
                 ) -> None:
        super().__init__()
        self.module:nn.Module = module
        self.params_pos:List[Tuple[torch.nn.Module,int]] = [] # Constants or Variables modules
        for mp in modules:
            self.add(mp[1], mp[0])
        
    def __repr__(self):
        L = [(m,p) for m,p in self.params_pos]
        s = f'MultiWithConst(module={self.module}, modules={L})'
        
        return s
        
    def add(self,
            pos:int,
            module:nn.Module,
            ):
        with torch.no_grad():
            self.add_module(f"const_input_at_pos_{pos}",module)
            self.params_pos.append((module,pos))
            
    def forward(self, *args):
        args_iter = iter(args)
        joined_args = []
        
        params_pos = sorted(self.params_pos,key=lambda x : x[1])
        
        for param,pos in params_pos:
            # fill with input args
            while len(joined_args) != pos:
                input_ = next(args_iter)
                joined_args.append(input_)
                
            # add parameter to list of inputs
            joined_args.append(param())
        
        DST_LEN = len(args)+len(params_pos)
        while len(joined_args) != DST_LEN:
            input_ = next(args_iter)
            joined_args.append(input_)
        
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
    x = torch.exp(-a*x+b)
    return 1 / (1+x)
    # x = torch.exp(a*x-b)
    # return x / (1+x)


class Pruner(nn.Module):
        
    def __init__(self, 
                 ch:int, 
                 prunable:bool=False,
                 activated:bool=True,
                 threshold:float=0.75,
                 num_of_appearances:int=1
                 ) -> None:
        super().__init__()
        self.ch = ch
        self.threshold = threshold
        self.activated = activated
        self.prunable = prunable
        self.num_of_appearances = num_of_appearances
        self.pruner_weight = torch.nn.Parameter(torch.ones((1,self.ch,1,1), 
                                                     dtype=torch.float32)*1.0000008344650269, 
                                                requires_grad=activated)
        self.init_ones()
        if activated:
            self.adjustment_mode()
    
    def extra_repr(self) -> str:
        s = ''
        s += f"ch={self.ch}, prunable={self.prunable}, activated={self.activated}, threshold={self.threshold}, num_of_appearances={self.num_of_appearances}"
        return s
    
    def get_mask(self) -> Tuple[torch.Tensor,torch.Tensor]:
        with torch.no_grad():    
            if self.prunable and self.activated:
                mask = self.pruner_weight > self.threshold
            else:
                mask = torch.ones_like(self.pruner_weight,dtype=torch.bool)
            
            if self.activated:
                multipliers = self.forward(torch.ones(1,dtype=torch.float32))
            else:
                multipliers = torch.ones_like(self.pruner_weight,dtype=torch.float32)
            
        return mask.flatten(), multipliers.flatten()
        
    def adjustment_mode(self):
        with torch.no_grad():
            device = self.pruner_weight.device
            self.pruner_weight[:] = 1-torch.rand((1,self.ch,1,1),dtype=torch.float32,device=device)*0.01
        self.pruner_weight.requires_grad = True
    
    def init_ones(self):
        with torch.no_grad():
            self.pruner_weight[:] = 1.0000008344650269

    def forward(self,x):
        if not self.activated:
            # cloning for differentiate input and output tensor
            return x.clone() if DIFFERENTIATE_TENSOR else x
        else:
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
    def __init__(self, 
                 ch: int, 
                 prunable: bool = False, 
                 activated: bool = True, 
                 threshold: float = 0.75, 
                 num_of_appearances: int = 1
                 ) -> None:
        super().__init__(ch, prunable, activated, threshold, num_of_appearances)
    
    def forward(self,x):
        if not self.activated:
            # cloning for differentiate input and output tensor
            return x.clone() if DIFFERENTIATE_TENSOR else x
        else:
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

    def get_mask(self) -> Tuple[torch.Tensor,torch.Tensor]:
        with torch.no_grad():    
            if self.prunable and self.activated:
                mask = self.pruner_weight.mean() > self.threshold
                mask = torch.ones_like(self.pruner_weight,dtype=torch.bool)*mask
            else:
                mask = torch.ones_like(self.pruner_weight,dtype=torch.bool)
            
            if self.activated:
                multipliers = self.forward(torch.ones(1,dtype=torch.float32))
            else:
                multipliers = torch.ones_like(self.pruner_weight,dtype=torch.float32)
            
        return mask.flatten(), multipliers.flatten()


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
        results = [B(x) for B in self.branches]
        
        return self.merger(*results)


# if __name__ == "__main__":
    
#     # find weight which gives multiplier equal to 1.0
#     def fcn(p):
#         return p*sigmoid(p,100,86)
#     w = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
#     lr = 0.0001
#     E = torch.inf
#     for i in range(100):
#         # loss computing
#         v = fcn(w)
#         e = torch.abs(1-v)
#         # get grad
#         e.backward()
#         g = w.grad
#         # update
#         with torch.no_grad():
#             w -= g*lr
#         lr *= 1.2 if e < E else 1/2
#         E = e.item()
#         print(f"W: {w.item()} value: {v.item()} E: {E} g: {g.item()} lr: {lr}")
#         w.grad *= 0

    
    