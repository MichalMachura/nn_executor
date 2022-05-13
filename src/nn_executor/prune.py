from matplotlib import container
import torch
from torch import nn
from typing import Any, Dict, List, Tuple, Type, Union
from nn_executor import modifiers, models
import queue


def on_propagate_backward(mask:torch.Tensor, # bool
                          src_node_idx:int,
                          src_node_output_idx:int,
                          dst_node_idx:int,
                          dst_node_input_idx:int
                          )->None:
    print("on_propagate_backward:",
          mask,
          (src_node_idx,src_node_output_idx), 
          (dst_node_idx,dst_node_input_idx))
    pass


def on_propagate_forward(mask_mul:Tuple[torch.Tensor,torch.Tensor], # bool, float32
                         src_node_idx:int,
                         src_node_output_idx:int,
                         dst_node_idx:int,
                         dst_node_input_idx:int
                         )->None:
    print("on_propagate_forward:",
          mask_mul,
          (src_node_idx,src_node_output_idx), 
          (dst_node_idx,dst_node_input_idx))
    pass


def on_available(node_idx:int):
    print("on_available:", node_idx)
    pass


def from_mask(L:List[Any], mask:List[bool])->List[Any]:
    return [item for (item, available) in zip(L,mask) if available]


def to_mask(L:List[Any], mask:List[bool], replace=None)->List[Any]:
    len_L = len(L)
    available_pos = sum(mask)
    if len_L != available_pos:
        raise RuntimeError(f"L has different {len(L)} number of elements than available in mask {available_pos}")
    
    new_L = []
    cntr = 0
    for available in mask:
        # set item on position
        if available:
            new_L.append(L[cntr])
            cntr += 1
        # fill position
        else:
            new_L.append(replace)
    
    return new_L

CONNECTION = Tuple[int,int,int,int]
CONNECTION_ANCHOR = Tuple[int,int] # Node idx, input/output idx 

class BidirectionalNode:
    def __init__(self, 
                 module:torch.nn.Module,
                 node_idx:int,
                 inputs_channels:List[int] = [],
                 outputs_channels:List[int] = [],
                 connections:List[Tuple[int,int,int,int]] = [],
                 ) -> None:
        self.module:torch.nn.Module = module
        self.node_idx:int = node_idx
        self.inputs_channels:List[int] = []
        self.outputs_channels:List[int] = []
        self.connections:List[Tuple[int,int,int,int]] = []
        
        self.forward_buffers:List[modifiers.FORWARD_TYPE] = [] # mask of input tensors
        self.modifier_memory:List[modifiers.BACKWARD_TYPE] = [] # mask,mul for each output after processing by modifier
        self.backward_buffers:List[List[modifiers.BACKWARD_TYPE]] = [] # mask,mul of each output from each dst node 
        
        self.dst_nodes:List[List[CONNECTION_ANCHOR]] = [] # for each output: for each (dst_idx, dst_in/out_idx) 
        self.src_nodes:List[CONNECTION_ANCHOR] = [] # [in1,in2,in3]: in1 = (src_node_idx, src_node_out_idx)
        
        self.inputs_availability:List[bool] = []
        self.outputs_availability:List[bool] = []
        self.outputs_dst_availability:List[List[bool]] = []
        
        self.forward_buffers_ready_mask:List[bool] = []
                                        # for inputs: is assigned
        self.backward_buffers_ready_mask:List[List[bool]] = []
                                        # for outputs, for (dst_node,dst_node_in): is assigned? 
        
        self.apply_connections(inputs_channels,outputs_channels,connections)
        
        self.on_forward_available:Type[on_available] = on_available
        self.on_backward_available:Type[on_available] = on_available
        
        self.on_propagate_forward:Type[on_propagate_forward] = on_propagate_forward
        self.on_propagate_backward:Type[on_propagate_backward] = on_propagate_backward
        
        self.on_forward_done = lambda x: None
        self.on_backward_done = lambda x: None
        
    def apply_connections(self,
                          inputs_channels:List[int],
                          outputs_channels:List[int],
                          connections:List[Tuple[int,int,int,int]],
                          ):
        self.forward_buffers = [None for ch_in in inputs_channels]
        self.backward_buffers = [[] for ch_out in outputs_channels]
        self.modifier_memory = [[] for ch_out in outputs_channels]
        self.forward_buffers_ready_mask = [False for ch_in in inputs_channels]
        self.backward_buffers_ready_mask = [[] for ch_out in outputs_channels]
        self.dst_nodes = [[] for ch_out in outputs_channels]
        self.src_nodes = [None for ch_in in inputs_channels]
        
        self.inputs_channels = inputs_channels.copy()
        self.outputs_channels = outputs_channels.copy()
        self.connections = []
        
        for src_node_idx, src_node_out_idx, \
            dst_node_idx, dst_node_in_idx in connections:
            # this node is is input for another
            if src_node_idx == self.node_idx:
                self.dst_nodes[src_node_out_idx].append((dst_node_idx, dst_node_in_idx))
                self.backward_buffers[src_node_out_idx].append(None)
                self.backward_buffers_ready_mask[src_node_out_idx].append(False)
                
                self.connections.append((src_node_idx,src_node_out_idx,dst_node_idx,dst_node_in_idx))
            # another node is input for this node
            elif dst_node_idx == self.node_idx:
                self.src_nodes[dst_node_in_idx] = (src_node_idx, src_node_out_idx)
                self.connections.append((src_node_idx,src_node_out_idx,dst_node_idx,dst_node_in_idx))
        
        self.inputs_availability = [True for i in range(len(self.inputs_channels))]
        self.outputs_availability = [True for i in range(len(self.outputs_channels))]
        self.outputs_dst_availability = [[True]*len(out_buffs) for out_buffs in self.backward_buffers]
    
    @property
    def forward_ready(self):
        # check only for available inputs
        forward_buffers = from_mask(self.forward_buffers_ready_mask,
                                    self.inputs_availability)
        
        is_ready = sum(forward_buffers) == len(forward_buffers)
        return is_ready
        
    @property
    def backward_ready(self):
        # check only for available outputs
        backward_buffers = from_mask(self.backward_buffers_ready_mask,
                                     self.outputs_availability)
        
        out_ready = [sum(out) == len(out) for out in backward_buffers]
        is_ready = sum(out_ready) == len(out_ready)
        return is_ready
        
    def set_from_forward(self, mask, dst_in_idx):
        if dst_in_idx < len(self.forward_buffers):
            self.forward_buffers[dst_in_idx] = mask
            self.forward_buffers_ready_mask[dst_in_idx] = True
        else:
            raise RuntimeError(f"{dst_in_idx} out of range for inputs size {len(self.forward_buffers)} in from_forward")
        
        if self.forward_ready and self.on_forward_available:
            self.on_forward_available(self.node_idx)
        
    def set_from_backward(self, mask_mul, src_out_idx, # src and dst are for forward pass
                      dst_node_idx, dst_node_in_idx): # so src is this node
        
        if src_out_idx < len(self.backward_buffers):
            outs = self.backward_buffers[src_out_idx]
            outs_read_mask = self.backward_buffers_ready_mask[src_out_idx]
            dst_nodes = self.dst_nodes[src_out_idx]
            
            found = False
            for i,(node_idx,port_idx) in enumerate(dst_nodes):
                if node_idx == dst_node_idx and port_idx == dst_node_in_idx:
                    outs[i] = mask_mul
                    outs_read_mask[i] = True
                    found = True
                    break
                    
            # connection was not found
            if not found:
                raise RuntimeError(f"Connection unavailable {(self.node_idx,src_out_idx,dst_node_idx,dst_node_in_idx)}")
        else:
            raise RuntimeError(f"{src_out_idx} out of range for outputs size {len(self.backward_buffers)}")
        
        if self.backward_ready and self.on_backward_available:
            self.on_backward_available(self.node_idx)
    
    @property
    def num_of_input(self):
        return sum(self.inputs_availability)
    
    @property
    def num_of_output(self):
        return sum(self.outputs_availability)
    
    def from_mask(self, L, interface='in'):
        return from_mask(L,self.inputs_availability if interface == 'in' else self.outputs_availability)
    
    def from_out_dst_masks(self, L):
        masked_L = self.from_mask(L,'out')
        masked_dst_masks = self.from_mask(self.outputs_dst_availability,'out')
        
        return [from_mask(oL,o_dst) for oL,o_dst in zip(masked_L,masked_dst_masks)]
    
    def to_out_dst_masks(self,L,replace=None):
        new_L = []
        remasked_L = self.to_mask(L,None,'out')
        
        for idx, sub_L in enumerate(remasked_L):
            if sub_L:
                new_sub_L = to_mask(sub_L,self.outputs_dst_availability[idx],replace=replace)
            else:
                new_sub_L = [replace for e in self.outputs_dst_availability[idx]]
            
            new_L.append(new_sub_L)
            
        return new_L
        
    def to_mask(self, L, replace=None, interface='in'):
        return to_mask(L,self.inputs_availability if interface == 'in' else self.outputs_availability, replace=replace)
    
    def reset_forward_buffers_ready_mask(self):
        for idx in range(len(self.forward_buffers_ready_mask)):
            self.forward_buffers_ready_mask[idx] = False
    
    def reset_backward_buffers_ready_mask(self):
        for buff in self.backward_buffers_ready_mask:
            for idx in range(len(buff)):
                buff[idx] = False
    
    def forward(self, 
                mods:Dict[str,modifiers.Modifier]):
        
        in_masks = self.from_mask(self.forward_buffers,'in')
        
        # empty node -> zeroed output masks
        if self.module is None:
            new_module = None
            out_masks = [torch.zeros(ch,dtype=torch.bool) \
                         for ch in self.from_mask(self.outputs_channels,'out')]
        else:
            cls = self.module.__class__.__name__
            if cls not in mods:
                raise RuntimeError(f"{cls} has not delivered appropriate modifier in node:{self.node_idx}")
            # make forward
            new_module, out_masks= mods[cls].forward(self.module,in_masks)
        
        available_inputs = [mask.sum().item() > 0 for mask in in_masks]
        available_outputs = [mask.sum().item() > 0 for mask in out_masks]
        # if node is deleted
        if new_module is None:
            # needed backward for previous nodes -- if this node is the only one dst 
            for in_idx, available in enumerate(self.to_mask(available_inputs,False,'in')):
                # only if input has been available
                if available:
                    # call 
                    src,src_out = self.src_nodes[in_idx]
                    dst,dst_in = self.node_idx, in_idx
                    mask = torch.zeros(self.inputs_channels[in_idx],dtype=torch.bool)
                    mul = torch.ones_like(mask,dtype=torch.float32)
                    # propagate
                    self.on_propagate_backward((mask,mul),src,src_out,dst,dst_in)
        
        out_masks = self.to_mask(out_masks,torch.tensor([],dtype=torch.bool),'out')
        # forwarding output mask to next nodes
        for out_idx, available in enumerate(self.outputs_availability):
            # only for non preremoved outputs
            if available:
                # this node and current out as connection source
                src,src_out = self.node_idx, out_idx
                # list of destiny connections
                dst_connections = self.dst_nodes[out_idx]
                
                # for each destination is needed propagation
                for (dst, dst_in), dst_available in zip(dst_connections, self.outputs_dst_availability[out_idx]):
                    if dst_available:
                        mask = out_masks[out_idx]
                        # propagate
                        self.on_propagate_forward(mask,src,src_out,dst,dst_in)
        
        self.reset_forward_buffers_ready_mask()
        
        # update availabilities
        self.inputs_availability = self.to_mask(available_inputs,False,'in')
        self.outputs_availability = self.to_mask(available_outputs,False,'out')
        # zeroing outputs to dst nodes if given output is available 
        outputs_dst_availability = []
        for (out_a,dst_a) in zip(self.outputs_availability, self.outputs_dst_availability):
            if out_a:
                outputs_dst_availability.append(dst_a)
            else:
                outputs_dst_availability.append([False for out_dst in dst_a])
        self.outputs_dst_availability = outputs_dst_availability
        
        # update module
        self.module = new_module
        # update num of channels
        new_input_channel = [mask.sum().item() for mask in self.to_mask(in_masks,torch.tensor([],dtype=torch.bool),'in')]
        new_output_channel = [mask.sum().item() for mask in out_masks]
        self.inputs_channels = new_input_channel
        self.outputs_channels = new_output_channel
        
        self.on_forward_done(self.node_idx)  
        return
        
    def backward(self, 
                mods:Dict[str,modifiers.Modifier]):
        
        raw_out_mask_muls = self.from_out_dst_masks(self.backward_buffers)
        out_mask_muls = []
        
        # merge all backwarded masks and muls into single pairs for each node output
        for out_idx, single_out_mask_muls in enumerate(raw_out_mask_muls):
            if len(single_out_mask_muls):
                masks = [m for (m,mul) in single_out_mask_muls]
                muls = [mul for (m,mul) in single_out_mask_muls]
                mask = torch.logical_or(masks[0],*masks)
                mul = torch.mul(torch.ones_like(muls[0]),*muls)
                out_mask_muls.append((mask,mul))
                
            else:
                RuntimeError(f"Empty backwarded mask and multipliers for node:{self.node_idx} out:{out_idx}")
        
        # empty node -> zeroed input masks and ones muls
        if self.module is None:
            new_module = None
            in_mask_muls = [(torch.zeros(ch,dtype=torch.bool),torch.ones(ch,dtype=torch.bool)) \
                                   for ch in self.from_mask(self.inputs_channels,'in')]
            
        else:
            cls = self.module.__class__.__name__
            if cls not in mods:
                raise RuntimeError(f"{cls} has not delivered appropriate modifier!")
            # make backward
            new_module, in_mask_muls= mods[cls].backward(self.module,out_mask_muls)
        
        available_inputs = [mask.sum().item() > 0 for mask, mul in in_mask_muls]
        available_outputs = [mask.sum().item() > 0 for mask, mul in out_mask_muls]
        
        # if node is deleted
        if new_module is None:
            # forwarding output mask to next nodes
            for out_idx, available in enumerate(self.to_mask(available_outputs,False,'out')):
                # only for non preremoved outputs
                if available:
                    # this node and current out as connection source
                    src,src_out = self.node_idx, out_idx
                    # list of destiny connections
                    dst_connections = self.dst_nodes[out_idx]
                    
                    single_out_mask_muls = to_mask(raw_out_mask_muls,self.outputs_dst_availability[out_idx],None)
                    
                    # for each destination is needed propagation
                    for (dst, dst_in), dst_available, mask_mul in zip(dst_connections, 
                                                                      self.outputs_dst_availability[out_idx],
                                                                      single_out_mask_muls):
                        # only for available dst 
                        if dst_available:
                            mask, mul = mask_mul
                            # propagate only when mask was non zeroed fully
                            if mask_mul[0].sum().item() > 0:
                                self.on_propagate_forward(mask,src,src_out,dst,dst_in)
        
        # backward for previous nodes 
        remasked_in_mask_muls = self.to_mask(in_mask_muls,(torch.tensor([],dtype=torch.bool),None),'in')
        for in_idx, (available, mask_mul) in enumerate(zip(self.inputs_availability,
                                                            remasked_in_mask_muls)):
            # only for available inputs
            if available:
                # call 
                src,src_out = self.src_nodes[in_idx]
                dst,dst_in = self.node_idx, in_idx
                # propagate
                self.on_propagate_backward(mask_mul,src,src_out,dst,dst_in)
    
        self.reset_backward_buffers_ready_mask()
        
        # update availabilities
        self.inputs_availability = self.to_mask(available_inputs,False,'in')
        self.outputs_availability = self.to_mask(available_outputs,False,'out')
        # zeroing outputs to dst nodes if given output is available 
        outputs_dst_availability = []
        for (out_a,dst_avs,dst_mask_muls) in zip(self.outputs_availability, 
                                                 self.outputs_dst_availability,
                                                 self.to_mask(raw_out_mask_muls,[],'out')
                                                 ):
            # deactivate all output dst
            if not out_a:
                outputs_dst_availability.append([False for out_dst in dst_a])
            else:
                new_dst_av = []
                for dst_a, mask_mul in zip(dst_avs, # current availability mask
                                           to_mask(dst_mask_muls, # backwarded mask_muls from dst
                                                   dst_avs, # masked
                                                   None)): 
                    if not dst_a:
                        new_dst_av.append(False)
                    else:
                        # dst is available, if mask is non empty
                        non_empty = mask_mul[0].sum().item() > 0
                        new_dst_av.append(non_empty)
                
                # new availability mask for output 
                outputs_dst_availability.append(new_dst_av)
                
        self.outputs_dst_availability = outputs_dst_availability
        # update module
        self.module = new_module
        # update num of channels
        new_input_channel = [mask.sum().item() for mask,mul in remasked_in_mask_muls]
        new_output_channel = [mask.sum().item() for mask,mul in out_mask_muls]
        self.inputs_channels = new_input_channel
        self.outputs_channels = new_output_channel
        
        self.on_backward_done(self.node_idx)
        
        # if node has no inputs -> forward
        if len(self.inputs_channels) == 0:
            self.forward(mods)
            
        return


class Callable:
    def __init__(self, container, method) -> None:
        super().__init__()
        self.container = container
        self.method = method
    
    def __call__(self, *args) -> Any:
        return self.method(self.container,tuple(args)) 

class Scissors:
    
    def __init__(self,
                 model_description:Dict[str,Any],
                 nodes_modifiers:Dict[str,modifiers.Modifier], # {'class_name':Modifier object}
                 ) -> None:
        # extract necessary description
        self.unique_layers:List[nn.Module] = model_description['unique_layers'].copy()
        self.layers_indices:List[int] = model_description['layers_indices'].copy()
        self.layers_in_out_channels:List[Tuple[List[int],List[int]]] = model_description['layers_in_out_channels'].copy()
        self.connections:List[Tuple[int,int,int,int]] = model_description['connections'].copy()
        self.outputs:List[Tuple[int,int,int]] = model_description['outputs'].copy()
        self.inputs_channels:List[int] = model_description['inputs_channels'].copy()
        self.outputs_channels:List[int] = model_description['outputs_channels'].copy()
        # modules assigned to nodes
        self.layers_modules:List[nn.Module] = [self.unique_layers[idx] for idx in self.layers_indices]
        # save modifiers dict
        self.nodes_modifiers:Dict[str,modifiers.Modifier] = nodes_modifiers
        
        # structure of network
        self.nodes:List[BidirectionalNode] = []
        # create input and output nodes
        self.__add_in_out_modules()
        
    def __add_in_out_modules(self):
        # create input module and its description
        input_module = models.InputLayer(self.inputs_channels.copy())
        self.layers_modules.insert(0,input_module)
        self.layers_in_out_channels.insert(0,([],self.inputs_channels))
        # create output module and its description
        output_module = models.OutputLayer(self.outputs_channels.copy())
        self.layers_modules.append(output_module)
        self.layers_in_out_channels.append((self.outputs_channels,[]))
        # add output module connections
        out_idx = len(self.layers_modules)-1
        out_connections = [(src_idx,src_out_idx,
                            out_idx,dst_in_idx) for src_idx,src_out_idx,dst_in_idx in self.outputs]
        self.connections.extend(out_connections)
        
        # create modifiers
        in_modifier = modifiers.InputLayerModifier()
        out_modifier = modifiers.OutputLayerModifier()
        self.nodes_modifiers[input_module.__class__.__name__] = in_modifier
        self.nodes_modifiers[output_module.__class__.__name__] = out_modifier
    
    def __create_nodes(self):
        self.nodes = []
        # for each layer create node
        for node_idx, module in enumerate(self.layers_modules):
            ch_in,ch_out = self.layers_in_out_channels[node_idx]
            node = BidirectionalNode(module,node_idx,ch_in,ch_out,self.connections)
            self.nodes.append(node)
    
    def __prune(self):
        # create nodes for model description
        self.__create_nodes()
        # queues for execution of forward and backward passes
        forward_queue = queue.Queue(len(self.layers_modules))
        backward_queue = queue.Queue(len(self.layers_modules))
        forward_propagation_queue = queue.Queue(len(self.connections))
        backward_propagation_queue = queue.Queue(len(self.connections))
        
        forward_put = lambda x: forward_queue.put(x)
        backward_put = lambda x: backward_queue.put(x)
        forward_propagation_put = Callable(forward_propagation_queue,queue.Queue.put_nowait)
        backward_propagation_put = Callable(backward_propagation_queue,queue.Queue.put_nowait)
        
        # add output node to backward queue  
        backward_put(len(self.nodes)-1)
        
        # set nodes callbacks
        for node in self.nodes:
            node.on_backward_available = backward_put
            node.on_forward_available = forward_put
            node.on_propagate_backward = backward_propagation_put
            node.on_propagate_forward = forward_propagation_put
        
        # turn backward pass into forward in input node
        self.nodes[0].on_backward_done = forward_put
        
        cntr = 0
        while not forward_queue.empty()\
                or not backward_queue.empty()\
                or not forward_propagation_queue.empty()\
                or not backward_propagation_queue.empty():
            cntr += 0
            # propagate backward
            while not backward_propagation_queue.empty():
                mask_mul,\
                src_node_idx,src_out_idx,\
                dst_node_idx,dst_in_idx = backward_propagation_queue.get_nowait()
                # set mask_mul in src node backward buffers
                self.nodes[src_node_idx].set_from_backward(mask_mul,src_out_idx,dst_node_idx,dst_in_idx)
                
            # propagate forward
            while not forward_propagation_queue.empty():
                mask,\
                src_node_idx,src_out_idx,\
                dst_node_idx,dst_in_idx = forward_propagation_queue.get_nowait()
                # set mask_mul in src node backward buffers
                self.nodes[dst_node_idx].set_from_forward(mask,dst_in_idx)
                
            # backward nodes modification
            while not backward_queue.empty():
                node_idx = backward_queue.get_nowait()
                node = self.nodes[node_idx]
                node.backward(self.nodes_modifiers)
                
            # forward nodes modification
            while not forward_queue.empty():
                node_idx = forward_queue.get_nowait()
                node = self.nodes[node_idx]
                node.forward(self.nodes_modifiers)
            
            print(cntr)
            cntr += 1
            
        print("Complete")
        # EXTRACT NEW MODEL DESCRIPTION
        # reindexation of nodes and connection,
        # removing unused nodes and connections 
        
    def __call__(self):
        return self.__prune()
        
