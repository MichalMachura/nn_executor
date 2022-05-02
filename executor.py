from typing import List, Tuple
import torch
from torch import nn
from torch import functional as F
import json
from models import Add, Sub, Identity
import parser

def layer_from_config(config):
    return


def parse_config(config):
    return


def get_from_idx(x, idx:int):
    
    if type(x) in [tuple, list]:
        return x[idx]
    
    if idx != 0:
        raise RuntimeError("Idx != 0 for non container")    
    
    return x


class Node:
    
    def __init__(self, 
                 layer:nn.Module) -> None:
        self.layer = layer
        self.outputs:List[Tuple[int, int,'Node']] = []
        self.inputs_values = []
        self.ready_inputs_cntr = 0
    
    def  set_input(self, idx:int, x:torch.Tensor) -> 'Node':
        if idx >= len(self.inputs_values):
            raise RuntimeError("idx =",idx,"is over last available idx =", len(self.inputs_values)-1)
        
        # set proper input
        self.inputs_values[idx] = x
        # increase cntr
        self.ready_inputs_cntr += 1
        
        if self.is_active():
            return self
        else:
            return None
    
    def is_active(self):
        return self.ready_inputs_cntr == len(self.inputs_values)
    
    def add_src(self, src_output_idx:int, src:'Node'):
        src_idx = len(self.inputs_values)
        # add this node to it's src 
        src.outputs.append((src_idx,src_output_idx,self))
        # add empty place
        self.inputs_values.append(None)
    
    def __call__(self) -> List['Node']:
        # basic layer forward
        output = self.layer(*self.inputs_values)
        
        # free inputs buffers
        self.inputs_values = [None for i in self.inputs_values]
        
        activated = []
        for src_idx, output_idx, dst_node in self.outputs:
            # choose one of results
            result_to_dst = get_from_idx(output,output_idx)
            # propagate to dst node
            active_node = dst_node.set_input(src_idx, result_to_dst)
            # if node is activated
            if active_node is not None:
                # add to list
                activated.append(active_node)
        
        # return new activated nodes
        return activated


class InputNode(Node):
    def __init__(self, layer: nn.Module, num_of_inputs:int=1) -> None:
        super().__init__(layer)
        self.inputs_values = [None for i in range(num_of_inputs)]


class OutputNode(Node):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__(layer)
    
    def set_input(self, idx: int, x) -> 'Node':
        is_activate =  super().set_input(idx, x)
        # always return None -- prevent execution as basic node
        return None
    
    def __call__(self):
        # get buffers values
        vals = self.inputs_values
        # reset buffers
        self.inputs_values = [None for i in self.inputs_values]
        
        return vals


class Executor(nn.Module):
    
    def __init__(self, 
                 layers:List[nn.Module], 
                 connections:List[Tuple[int,int,int]],
                 outputs:List[Tuple[int,int]]) -> None:
        super().__init__()
        self.update_connections(layers, connections, outputs)
        self.nodes:List[Node] = self.create_nodes(layers,connections)
        self.register_layers(layers)
        self.connect(connections)
        
    def register_layers(self,layers):
        for i, L in enumerate(layers):
            self.add_module(str(i),L)
    
    def update_connections(self,layers, connections, outputs):
        dst_idx = len(layers)+1
        for src, out_idx in outputs:
            link = (src, out_idx, dst_idx)
            connections.append(link)
    
    def get_number_of_inputs(self,connections):
        noi = 0
        for src,out_idx,dst in connections:
            if src == 0:
                noi = max(noi,out_idx)
        return noi+1
        
    def create_nodes(self, layers:List[nn.Module], connections):
        noi = self.get_number_of_inputs(connections)
        nodes =  [Node(L) for L in layers]
        
        inL, outL = Identity(),Identity()
        # layers.insert(0,inL)
        # layers.append(outL)
        
        nodes = [InputNode(inL,noi),
                 *nodes,
                 OutputNode(outL)]
        return nodes
    
    def connect(self, connections):
        for link in connections:
            src, out_idx, dst = link
            dst_node:Node = self.nodes[dst]
            src_node:Node = self.nodes[src]
            dst_node.add_src(out_idx,src_node)
    
    def forward(self, *args):
        for i,x in enumerate(args):
            a = self.nodes[0].set_input(i,x)
        
        if not self.nodes[0].is_active():
            raise RuntimeError("First node has not been activated.")
        
        active_layers = [self.nodes[0]]
        
        while len(active_layers):
            L = active_layers.pop(0)
            newly_activated = L()
            active_layers.extend(newly_activated)
        
        outputs = self.nodes[-1]()
        
        return outputs
    
    def save(self,path):
        for ch in self.children():
            print(ch)


# class MOD(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.l1 = nn.Conv2d(3,5,3,padding=(1,1))
#         self.l2 = nn.MaxPool2d(2,2)
#         #
#         self.l3 = nn.Conv2d(5,6,3,padding=(1,1))
#         self.l4 = nn.Conv2d(6,5,3,padding=(1,1))
#         self.l5 = nn.ReLU()
#         #
#         self.l6 = nn.Conv2d(4,5,3,padding=(1,1))
#         self.l7 = nn.MaxPool2d(2,2)
#         self.l8 = Add(3)
#         #
#         self.l9 = nn.Conv2d(5,10,3,padding=(1,1))
#         self.l10 = nn.Conv2d(10,7,3,padding=(1,1))
#         self.l11 = nn.MaxPool2d(2,2)

#     def forward(self, x0, x1):
#         b0 = self.l1(x0)
#         b0 = self.l2(b0)
#         b1 = self.l3(b0)
#         b1 = self.l4(b1)
#         b1 = self.l5(b1)
#         b2 = self.l6(x1)
#         b2 = self.l7(b2)
#         a0 = self.l8(b0, b1, b2)
#         b3 = self.l9(a0)
#         b3 = self.l10(b3)
#         b3 = self.l11(b3)
        
#         return x0,x1,b0,b1,b2,a0,b3


if __name__ == '__main__':
    # from brevitas import nn as qnn
    layers = [
        nn.Conv2d(3,5,3,padding=(1,1)),
        nn.MaxPool2d(2,2),
        # res
        nn.Conv2d(5,6,3,padding=(1,1)),
        nn.Conv2d(6,5,3,padding=(1,1)),
        # qnn.QuantConv2d(6,5,3,padding=(1,1)),
        nn.ReLU(),
        # brach 2
        nn.Conv2d(4,5,3,padding=(1,1)),
        nn.MaxPool2d(2,2),
        Add(3),
        nn.Conv2d(5,10,3,padding=(1,1)),
        nn.Conv2d(10,7,3,padding=(1,1)),
        nn.MaxPool2d(2,2),
        nn.Conv2d(7,10,3,padding=(1,1)),
        nn.ReLU(),#r
        nn.Conv2d(10,10,3,padding=(1,1),groups=10),
        nn.ReLU(),#r
        nn.Conv2d(10,10,3,padding=(1,1)),
        nn.ReLU(),#r
        Sub(3),
        nn.Conv2d(10,3,3,padding=(1,1)),
    ]   
    connections = [
    (0,0,1),
    (1,0,2),#r
    (2,0,3),
    (3,0,4),
    (4,0,5),
    #b2    
    (0,1,6),
    (6,0,7),
    #res
    (2,0,8),
    (5,0,8),
    (7,0,8),#add
    
    (8,0,9),
    (9,0,10),
    (10,0,11),
    (11,0,12),
    (12,0,13),#r
    (13,0,14),
    (14,0,15),#r
    (15,0,16),
    (16,0,17),#r
    # sub
    (13,0,18),
    (15,0,18),
    # (17,0,18),
    
    (18,0,19),
    ]
    outputs = [
        (8,0),
        (19,0),
        (0,0),
        # (0,1)
    ]
    # cntr = 0
    
    # def length(x):
    #     if type(x) in [list, tuple]:
    #         return len(x)
        
    #     return 1
    
    # def hook(mod, inp, outp):
    #     global cntr
    #     print(cntr,":",mod.__class__,id(mod),length(inp),length(outp))
    #     cntr += 1
    
    h = parser.Parser()
    
    # hook = lambda *x: h(*x)
    
    exec = Executor(layers, connections,outputs).eval()
    # exec = MOD()
    
    # for n, m in list(exec.named_modules()):
    #     print(n,m)
    #     m.register_forward_hook(hook)
    
    t0 = torch.rand(1,3,64,64)
    t1 = torch.rand(1,4,64,64)
    
    h.parse_module(exec,t0,t1)
    
    # h.add_inputs(t0,t1)
    
    # result = exec(t0,t1)
    
    # for r in result:
    #     print(r.shape)
    
    # exec.save('path.pt')    
    
    print(h)