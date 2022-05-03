from typing import List, Tuple
import torch
from torch import nn

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
    
    def set_input(self, idx:int, x:torch.Tensor) -> 'Node':
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
                 layers_indeces:List[int],
                 layers_map:List[nn.Module],
                 connections:List[Tuple[int,int,int,int]],
                 outputs:List[Tuple[int,int]]) -> None:
        super().__init__()
        
        self.layers_indeces:List[int] = layers_indeces.copy()
        self.layers_map:List[nn.Module] = layers_map.copy()
        self.connections:List[Tuple[int,int,int,int]] = connections.copy()
        self.outputs:List[Tuple[int,int]] = outputs.copy()
        
        connections = connections.copy() # this list is modified by some of the following functions
        self.update_connections(layers_indeces=layers_indeces, 
                                layers_map=layers_map, 
                                connections=connections, 
                                outputs=outputs)
        self.nodes:List[Node] = self.create_nodes(layers_indeces=layers_indeces, 
                                                  layers_map=layers_map,
                                                  connections=connections)
        self.register_layers(layers_map=layers_map)
        self.connect(connections=connections)
        
    def register_layers(self,
                        layers_map:List[nn.Module]):
        for i, L in enumerate(layers_map):
            cls = L.__class__.__name__
            self.add_module("layer_{:03}_type_{}".format(i,cls),L)
    
    def update_connections(self, 
                           layers_indeces:List[int], 
                           layers_map:List[nn.Module], 
                           connections:List[Tuple[int,int,int,int]], 
                           outputs:List[Tuple[int,int]]):
        dst = len(layers_indeces)+1
        for dst_in_idx, (src, src_out_idx) in enumerate(outputs):
            link = (src, src_out_idx, dst, dst_in_idx)
            connections.append(link)
    
    def get_number_of_inputs(self,
                             connections:List[Tuple[int,int,int,int]]):
        noi = 0
        for src, src_out_idx, dst, dst_in_idx in connections:
            if src == 0:
                noi = max(noi,src_out_idx)
        
        return noi+1
    
    def create_nodes(self, 
                     layers_indeces:List[int], 
                     layers_map:List[nn.Module], 
                     connections:List[Tuple[int,int,int,int]]):
        noi = self.get_number_of_inputs(connections)
        
        nodes =  []
        for layer_idx in layers_indeces:
            L = layers_map[layer_idx]
            nodes.append(Node(L))
        
        inL, outL = Identity(),Identity()
        
        nodes = [InputNode(inL,noi),
                 *nodes,
                 OutputNode(outL)]
        return nodes
    
    def connect(self, 
                connections:List[Tuple[int,int,int,int]]):
        for link in connections:
            src, src_out_idx, dst, dst_in_idx = link
            dst_node:Node = self.nodes[dst]
            src_node:Node = self.nodes[src]
            dst_node.add_src(src_out_idx,src_node)
    
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


if __name__ == '__main__':
    from nn_executor.models import Add, Sub, Identity
    from nn_executor import parser

    # from brevitas import nn as qnn
    layers_map = [
        nn.Conv2d(3,5,3,padding=(1,1)),
        nn.MaxPool2d(2,2),
        # res
        nn.Conv2d(5,5,3,padding=(1,1)),
        # nn.Conv2d(5,5,3,padding=(1,1)),
        nn.ReLU(),
        # brach 2
        nn.Conv2d(4,5,3,padding=(1,1)),
        nn.MaxPool2d(2,2),
        Add(3),
        nn.Conv2d(5,10,3,padding=(1,1)),
        nn.Conv2d(10,7,3,padding=(1,1)),
        nn.MaxPool2d(2,2),
        nn.Conv2d(7,10,3,padding=(1,1)),
        # nn.ReLU(),#r
        nn.Conv2d(10,10,3,padding=(1,1),groups=10),
        # nn.ReLU(),#r
        nn.Conv2d(10,10,3,padding=(1,1)),
        # nn.ReLU(),#r
        Sub(3),
        nn.Conv2d(10,3,3,padding=(1,1)),
    ] 
    # layers_indeces = [ i for i in range(len(layers_map))]
    # layers_indeces=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    layers_indeces = [0,1,2,2,3,4,5,6,7,8,9,10,3,11,3,12,3,13,14]
    connections = [
    (0,0,1,0),
    (1,0,2,0),#r
    (2,0,3,0),
    (3,0,4,0),
    (4,0,5,0),
    #b2    
    (0,1,6,0),
    (6,0,7,0),
    #res
    (2,0,8,0),
    (5,0,8,1),
    (7,0,8,2),#add
    
    (8,0,9,0),
    (9,0,10,0),
    (10,0,11,0),
    (11,0,12,0),
    (12,0,13,0),#r
    (13,0,14,0),
    (14,0,15,0),#r
    (15,0,16,0),
    (16,0,17,0),#r
    # sub
    (13,0,18,0),
    (15,0,18,1),
    # (17,0,18),
    
    (18,0,19,0),
    ]
    outputs = [
        (8,0),
        (19,0),
        (0,0),
        # (0,1)
    ]
    h = parser.Parser()
    
    exec = Executor(layers_indeces, 
                    layers_map, 
                    connections,
                    outputs).eval()
    
    t0 = torch.rand(1,3,64,64)
    t1 = torch.rand(1,4,64,64)
    
    h.parse_module(exec,t0,t1)
    
    print(h)