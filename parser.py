import torch
from torch import nn
import models


def get_module_params(m:torch.nn.Module):
    
    return 

class Parser:
    SUPPORTED_MODULES = [
                        nn.Conv2d,
                        nn.BatchNorm2d,
                        nn.ReLU,
                        nn.Mish,
                        nn.LeakyReLU,
                        nn.Upsample,
                        models.Cat,
                        models.Sub,
                        models.Add,
                        models.Identity,
                        ]

    def __init__(self, supported_modules=[]) -> None:
        self.layers = []
        self.out_tensors = [] # (t,src,out_idx)
        self.connections = []
        
    def __call__(self, mod, inps, outps):
        self.layers.append(mod)
        dst = len(self.layers)
        
        z = 1
        # for each input
        for i, inp in enumerate(inps):
            # find src of previously saved output tensors
            for t,src, out_idx in self.out_tensors[::-1]:
                if inp.data_ptr() == t.data_ptr():
                    self.connections.append((src,out_idx, dst))
                    break
        
        for i, t in enumerate(outps):
            # prevent saving the same tensor
            if t.data_ptr() not in [tt[0].data_ptr() for tt in self.out_tensors]:
                self.out_tensors.append((t,dst,i))
        z = 0
    
    def parse_module(self, module:nn.Module, *inputs:torch.Tensor):
        hook = lambda *x: self(*x)
        
        # set hooks
        for n, m in module.named_children():
            for mm in m.modules():
                mm.register_forward_hook(hook)
        
        self.add_inputs(*inputs)
        outs = module(*inputs)
        
        # connect outputs with their sources
        self.__call__(module, outs, ())
        
    def add_inputs(self, *args):
        for i,t in enumerate(args):
            self.out_tensors.append((t,0,i))
            
    def __repr__(self) -> str:
        s = "Layers:\n"
        for L in self.layers:
            s += str(L)+'\n'
        s += 'Connections:\n'
        for c in self.connections:
            s += str(c)+'\n'
        s += 'Outs:\n'
        for o in self.out_tensors:
            s += str(id(o[0]))+', '
            s += str(o[0].shape)+', '
            s += str(o[1])+', '
            s += str(o[2])+'\n'
        
        return s
