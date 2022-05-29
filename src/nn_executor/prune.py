import torch
from torch import nn
from typing import Any, Dict, Iterable, List, Tuple, Type
from nn_executor import modifiers, models, parser, utils, executor
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
    

CONNECTION = Tuple[int,int,int,int]
CONNECTION_ANCHOR = Tuple[int,int] # Node idx, input/output idx 

def are_same(L1:Iterable,*L_other:Iterable) -> bool:
    if len(L_other) == 0:
        raise RuntimeError("To compare contaiers are needed at least 2 iterators.")
    
    lengths = [len(l) == len(L1) for l in L_other]
    
    if sum(lengths) != len(lengths):
        return False
    
    for l in L_other:
        for item, item_1 in zip(l, L1): 
            if item != item_1:
                return False
    return True
    

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
        
        self.on_forward_available:Type[on_available] = on_available
        self.on_backward_available:Type[on_available] = on_available
        
        self.on_propagate_forward:Type[on_propagate_forward] = on_propagate_forward
        self.on_propagate_backward:Type[on_propagate_backward] = on_propagate_backward
        
        self.on_forward_done = lambda x: None
        self.on_backward_done = lambda x: None
        
        self.apply_connections(inputs_channels,outputs_channels,connections)
    
    @property
    def description(self):
        sd = {}
        sd['node_idx'] = self.node_idx
        sd['module'] = self.module
        
        sd['input_channels'] = self.inputs_channels.copy()
        sd['input_mask'] = self.inputs_availability.copy()
        sd['num_of_input'] = sum(self.inputs_availability)
        
        sd['output_channels'] = self.outputs_channels.copy()
        sd['output_mask'] = self.outputs_availability.copy()
        sd['num_of_output'] = sum(self.outputs_availability)
        
        sd['connections_as_src'] = []
        sd['connections_as_dst'] = []
        
        if self.module is not None:
            # as dst
            for dst_in_idx,(src_idx, src_out_idx) in enumerate(self.src_nodes):
                # if connection is not removed
                if self.inputs_availability[dst_in_idx]:
                    sd['connections_as_dst'].append((src_idx,src_out_idx,
                                                     self.node_idx,dst_in_idx))
        
            # as src
            for (src_out_idx, 
                 (DSTs, 
                  DSTs_availability)) in enumerate(zip(self.dst_nodes,
                                                       self.outputs_dst_availability)):
                # if this output is not removed
                if self.outputs_availability[src_out_idx]:
                    # for each dst
                    for (available,
                         (dst_idx,dst_in_idx)) in zip(DSTs_availability,
                                                      DSTs):
                        if available:
                            sd['connections_as_src'].append((self.node_idx,src_out_idx,
                                                             dst_idx,dst_in_idx))
        
        return sd
    
    def apply_connections(self,
                          inputs_channels:List[int],
                          outputs_channels:List[int],
                          connections:List[Tuple[int,int,int,int]],
                          ):
        self.forward_buffers = [None for ch_in in inputs_channels]
        self.backward_buffers = [[] for ch_out in outputs_channels]
        self.modifier_memory = [None for ch_out in outputs_channels]
        self.forward_buffers_ready_mask = [False for ch_in in inputs_channels]
        self.backward_buffers_ready_mask = [[] for ch_out in outputs_channels]
        self.dst_nodes = [[] for ch_out in outputs_channels]
        self.src_nodes = [None for ch_in in inputs_channels]
        
        self.inputs_channels = inputs_channels.copy()
        self.outputs_channels = outputs_channels.copy()
        
        for src_node_idx, src_node_out_idx, \
            dst_node_idx, dst_node_in_idx in connections:
            # this node is is input for another
            if src_node_idx == self.node_idx:
                self.dst_nodes[src_node_out_idx].append((dst_node_idx, dst_node_in_idx))
                self.backward_buffers[src_node_out_idx].append(None)
                self.backward_buffers_ready_mask[src_node_out_idx].append(False)
                
            # another node is input for this node
            elif dst_node_idx == self.node_idx:
                self.src_nodes[dst_node_in_idx] = (src_node_idx, src_node_out_idx)
        
        # set availabilities for nodes' inputs and outputs interfaces
        self.inputs_availability = [self.src_nodes[i] is not None for i in range(len(self.inputs_channels))]
        self.outputs_availability = [len(self.dst_nodes[i]) > 0 for i in range(len(self.outputs_channels))]
        self.outputs_dst_availability = [[True]*len(out_buffs) for out_buffs in self.backward_buffers]
        
    
    @property
    def forward_ready(self):
        # sum inputs with ready flag and check if all are True
        is_ready = sum(self.forward_buffers_ready_mask) == len(self.forward_buffers_ready_mask)
        return is_ready
        
    @property
    def backward_ready(self):
        # sum outputs (and dst)  with ready flag and check if all are True
        backward_buffers = self.backward_buffers_ready_mask
        out_ready = [sum(out) == len(out) for out in backward_buffers]
        is_ready = sum(out_ready) == len(out_ready)
        return is_ready
        
    def take_from_forward(self, 
                         mask_mul_bias:modifiers.FORWARD_TYPE, 
                         dst_in_idx:int):
        if 0 <= dst_in_idx < len(self.forward_buffers):
            self.forward_buffers[dst_in_idx] = mask_mul_bias
            self.forward_buffers_ready_mask[dst_in_idx] = True
        else:
            raise RuntimeError(f"{dst_in_idx} out of range for inputs size {len(self.forward_buffers)} in from_forward")
        
        if self.forward_ready and self.on_forward_available:
            self.on_forward_available(self.node_idx)
        
    def take_from_backward(self, 
                          mask_mul:modifiers.BACKWARD_TYPE, 
                          src_out_idx:int, # src and dst are for forward notation
                          dst_node_idx:int, 
                          dst_node_in_idx:int): # so src is this node
        
        if 0 <= src_out_idx < len(self.backward_buffers):
            # get buffers for src_out_idx-output of this node
            outs = self.backward_buffers[src_out_idx]
            outs_read_mask = self.backward_buffers_ready_mask[src_out_idx]
            dst_nodes = self.dst_nodes[src_out_idx]
            
            # try to find position in list for given dst connection
            found = False
            for i,(node_idx,port_idx) in enumerate(dst_nodes):
                # if node is found
                if node_idx == dst_node_idx \
                        and port_idx == dst_node_in_idx:
                    # assign mask mul and set flags
                    outs[i] = mask_mul
                    outs_read_mask[i] = True
                    found = True
                    break
                    
            # connection was not found
            if not found:
                raise RuntimeError(f"Connection unavailable {(self.node_idx,src_out_idx,dst_node_idx,dst_node_in_idx)}")
        else:
            raise RuntimeError(f"{src_out_idx} out of range for outputs size {len(self.backward_buffers)}")
        
        # when buffers are completed
        if self.backward_ready and self.on_backward_available:
            self.on_backward_available(self.node_idx)
    
    def __clear_buffers(self):
        self.forward_buffers = [None for fb in self.forward_buffers]
        self.backward_buffers = [[None for dst_b in bb] for bb in self.backward_buffers]
        self.modifier_memory = [None for ch_out in self.modifier_memory]
        self.forward_buffers_ready_mask = [False for fbrm in self.forward_buffers_ready_mask]
        self.backward_buffers_ready_mask = [[False for dst_r in bbrm] for bbrm in self.backward_buffers_ready_mask]
        
    @property
    def num_of_input(self):
        return sum(self.inputs_availability)
    
    @property
    def num_of_output(self):
        return sum(self.outputs_availability)
    
    def reset_forward_buffers_ready_mask(self):
        for idx in range(len(self.forward_buffers_ready_mask)):
            self.forward_buffers_ready_mask[idx] = False
    
    def reset_backward_buffers_ready_mask(self):
        for buff in self.backward_buffers_ready_mask:
            for idx in range(len(buff)):
                buff[idx] = False
    
    def forward(self, 
                mods:Dict[str,modifiers.Modifier]
                ):
        # empty node -> zeroed output masks
        if self.module is None:
            new_module = None
            out_mask_mul_bias = [(torch.zeros(ch,dtype=torch.bool), None, None) \
                                                    for ch in self.outputs_channels]
        else:
            # get class
            cls = self.module.__class__
            if cls not in mods:
                raise RuntimeError(f"{cls} has not delivered appropriate modifier for node:{self.node_idx}")
            # make forward
            new_module, out_mask_mul_bias = mods[cls].forward(self.module,
                                                              self.forward_buffers,
                                                              self.modifier_memory)
        
        outputs_availabilities = []
        outputs_dst_availabilities = []
        new_output_channels = []
        # forward propagation
        for out_idx, ((MASK,MUL,BIAS),
                      OUTPUT_MASK_MULS) in enumerate(zip(out_mask_mul_bias,
                                                         self.backward_buffers)):
            new_output_channels.append(MASK.sum().item() if BIAS is None else 0)
            # connection src
            src_idx, src_out_idx = self.node_idx, out_idx
            # availability for this node
            outputs_availabilities.append(MASK.sum().item() > 0 and BIAS is None)
            # init availability for each dst
            outputs_dst_availabilities.append([False for N in self.dst_nodes[out_idx]])
            # iterate over all destinies
            for i,((backwared_mask,_),
                   (dst_idx,dst_in_idx)) in enumerate(zip(OUTPUT_MASK_MULS, 
                                                          self.dst_nodes[out_idx])):
                # mask both masks
                mask = torch.logical_and(backwared_mask,MASK)
                # num of channe;s
                ch = mask.sum().item()
                # determinate availability
                outputs_dst_availabilities[out_idx][i]= ch > 0 and BIAS is None
                # forward
                self.on_propagate_forward((mask,MUL,BIAS),
                                          src_idx,src_out_idx,
                                          dst_idx,dst_in_idx)
                
        new_input_channels = [m[0].sum().item() if m[2] is None else 0 for m in self.forward_buffers]
        inputs_availabilities = [ch > 0 for ch in new_input_channels]
        
        # prevention of again running forward
        self.reset_forward_buffers_ready_mask()
        # update module
        self.module = new_module
        # update num of channels
        self.inputs_channels = new_input_channels
        self.outputs_channels = new_output_channels
        # update availabilities
        self.inputs_availability = inputs_availabilities
        self.outputs_availability = outputs_availabilities
        self.outputs_dst_availability = outputs_dst_availabilities
    
        # raise event of done
        if self.on_forward_done is not None:
            self.on_forward_done(self.node_idx)  
        
        self.__clear_buffers()
        
    def backward(self, 
                mods:Dict[str,modifiers.Modifier]
                ):
        out_mask_muls:List[modifiers.BACKWARD_TYPE] = []        
        # merge all backwarded masks and muls into single pairs for each node output
        for out_idx, single_out_mask_muls in enumerate(self.backward_buffers):
            # if there is sth backwarded
            if len(single_out_mask_muls):
                masks = [m for (m,mul) in single_out_mask_muls]
                muls = [mul for (m,mul) in single_out_mask_muls]
                # mask = torch.logical_or(masks[0],*masks)
                mask = utils.between_all(torch.logical_or,masks)
                mul = utils.between_all(torch.mul,[torch.ones_like(muls[0]),*muls])
                out_mask_muls.append((mask,mul))
            
            else:
                mask =torch.zeros(self.outputs_channels[out_idx],dtype=torch.bool)
                mul = torch.ones_like(mask,dtype=torch.float32)
                out_mask_muls.append((mask,mul))
                Warning(f"Empty backwarded mask and multipliers for node:{self.node_idx} out:{out_idx}")
        
        # empty node -> zeroed input masks and ones muls
        if self.module is None:
            new_module = None
            in_mask_muls = [(torch.zeros(ch,dtype=torch.bool),torch.ones(ch,dtype=torch.bool)) \
                                   for ch in self.inputs_channels]
            out_mask_muls = [(torch.zeros(ch,dtype=torch.bool), None) \
                                   for ch in self.outputs_channels]
        else:
            # get class
            cls = self.module.__class__
            if cls not in mods:
                raise RuntimeError(f"{cls} has not delivered appropriate modifier!")
            # make backward
            new_module, in_mask_muls, out_mask_muls = mods[cls].backward(self.module,out_mask_muls)
        
        # backward for previous nodes 
        for in_idx, in_mask_mul in enumerate(in_mask_muls):
            # call 
            src, src_out = self.src_nodes[in_idx]
            dst, dst_in = self.node_idx, in_idx
            # propagate
            self.on_propagate_backward(in_mask_mul,src,src_out,dst,dst_in)
        
        # store modifier output
        self.modifier_memory = out_mask_muls
        # reset output buffers ready masks
        self.reset_backward_buffers_ready_mask()
        # update module
        self.module = new_module
        # raise event of done
        if self.on_backward_done:
            self.on_backward_done(self.node_idx)
        # if node has no inputs -> activate forward
        if len(self.inputs_channels) == 0 and self.on_forward_available:
            # raise event for forward execution
            self.on_forward_available(self.node_idx)


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
        self.nodes_modifiers[input_module.__class__] = in_modifier
        self.nodes_modifiers[output_module.__class__] = out_modifier
    
    def __create_nodes(self):
        self.nodes = []
        # for each layer create node
        for node_idx, module in enumerate(self.layers_modules):
            ch_in,ch_out = self.layers_in_out_channels[node_idx]
            node = BidirectionalNode(module,node_idx,ch_in,ch_out,self.connections)
            self.nodes.append(node)
    
    def __prune(self, 
                t:torch.Tensor,
                dst_modules:List[Type]):
        # create nodes for model description
        self.__create_nodes()
        # queues for execution of forward and backward passes
        forward_queue = queue.Queue(len(self.layers_modules))
        backward_queue = queue.Queue(len(self.layers_modules))
        forward_propagation_queue = queue.Queue(len(self.connections))
        backward_propagation_queue = queue.Queue(len(self.connections))
        
        forward_put = lambda x: forward_queue.put_nowait(x)
        backward_put = lambda x: backward_queue.put_nowait(x)
        forward_propagation_put = lambda *x: forward_propagation_queue.put_nowait(x)
        backward_propagation_put = lambda *x: backward_propagation_queue.put_nowait(x)
        
        # add output node to backward queue  
        for node in self.nodes:
            if node.backward_ready:
                backward_put(node.node_idx)
        
        # set nodes callbacks
        for node in self.nodes:
            node.on_backward_available = backward_put
            node.on_forward_available = forward_put
            node.on_propagate_backward = backward_propagation_put
            node.on_propagate_forward = forward_propagation_put
            
        while not forward_queue.empty()\
                or not backward_queue.empty()\
                or not forward_propagation_queue.empty()\
                or not backward_propagation_queue.empty():
            # propagate backward
            while not backward_propagation_queue.empty():
                mask_mul,\
                src_node_idx,src_out_idx,\
                dst_node_idx,dst_in_idx = backward_propagation_queue.get_nowait()
                # set mask_mul in src node backward buffers
                self.nodes[src_node_idx].take_from_backward(mask_mul,src_out_idx,dst_node_idx,dst_in_idx)
                
            # propagate forward
            while not forward_propagation_queue.empty():
                mask_mul_bias,\
                src_node_idx,src_out_idx,\
                dst_node_idx,dst_in_idx = forward_propagation_queue.get_nowait()
                # set mask_mul in src node backward buffers
                self.nodes[dst_node_idx].take_from_forward(mask_mul_bias,dst_in_idx)
                
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
                        
        # EXTRACT NEW MODEL DESCRIPTION
        nodes_to_dst_connections:List[CONNECTION] = []
        src_to_nodes_connections:List[CONNECTION] = []
        nodes_inputs_masks:List[List[bool]] = []
        nodes_outputs_masks:List[List[bool]] = []
        nodes_inputs_channels:List[List[int]] = []
        nodes_outputs_channels:List[List[int]] = []
        nodes_modules:List[torch.nn.Module|None] = []
        nodes_indices:List[int] = []
        for node in self.nodes:
            nd = node.description
            nodes_to_dst_connections.extend(nd['connections_as_src'])
            src_to_nodes_connections.extend(nd['connections_as_dst'])
            nodes_inputs_masks.append(nd['input_mask'])
            nodes_outputs_masks.append(nd['output_mask'])
            nodes_inputs_channels.append(nd['input_channels'])
            nodes_outputs_channels.append(nd['output_channels'])
            nodes_indices.append(nd['node_idx'])
            nodes_modules.append(nd['module'])
        
        # removing unused nodes and connections 
        connections = nodes_to_dst_connections.copy()
        # reindexation of nodes and connection,
        reindexation_maps = self.__determine_reindexation_maps(nodes_inputs_masks, 
                                                               nodes_outputs_masks, 
                                                               nodes_modules)
        nodes_indices_map, nodes_inputs_map, nodes_outputs_map = reindexation_maps
        
        # REINDEXATE NODES, CONNECTIONS, MODULES, CHANNELS
        (CONNECTIONS, 
         UNIQUE_LAYERS, 
         LAYERS_INDICES, 
         LAYERS_IN_OUT_CH, 
         INPUT_CHANNELS, 
         OUTPUT_CHANNELS,
         OUTPUTS) = self.__reindexate(nodes_inputs_masks, 
                                        nodes_outputs_masks, 
                                        nodes_inputs_channels, 
                                        nodes_outputs_channels, 
                                        nodes_modules, 
                                        connections, 
                                        nodes_indices_map, 
                                        nodes_inputs_map, 
                                        nodes_outputs_map)
        # merge into dict
        model_description ={
                            'layers_indices':LAYERS_INDICES,
                            'layers_in_out_channels':LAYERS_IN_OUT_CH,
                            'unique_layers':UNIQUE_LAYERS,
                            'connections':CONNECTIONS,
                            'outputs':OUTPUTS,
                            'inputs_channels':INPUT_CHANNELS,
                            'outputs_channels':OUTPUT_CHANNELS,
                            }

        net = executor.Executor(model_description).eval()
        
        p = parser.Parser(dst_modules)
        model_description = p.parse_module(net,t)
        
        return model_description

    def __reindexate(self, 
                     nodes_inputs_masks, 
                     nodes_outputs_masks, 
                     nodes_inputs_channels, 
                     nodes_outputs_channels, 
                     nodes_modules, 
                     connections, 
                     nodes_indices_map, 
                     nodes_inputs_map, 
                     nodes_outputs_map):
        OUTPUTS = []
        OUTPUT_IDX = len(nodes_modules)-1 # index of output layer
        reindexed_connections = []
        for src_idx,src_out_idx,dst_idx,dst_in_idx in connections:
            src_out_map = nodes_outputs_map[src_idx]
            dst_in_map = nodes_inputs_map[dst_idx]
            C = (nodes_indices_map[src_idx],
                 src_out_map[src_out_idx],
                 nodes_indices_map[dst_idx],
                 dst_in_map[dst_in_idx]
                 )
            # for non output layer add as connection 
            if dst_idx != OUTPUT_IDX:
                reindexed_connections.append(C)
            # for output layer add as output src 
            else:
                OUTPUTS.append(C[:2]+C[-1:])
        
        reindexed_modules = []
        reindexed_indices = []
        reindexed_in_out_channels = []
        for (module,
             in_map,
             in_channels,
             out_map, 
             out_channels)in zip(nodes_modules,
                                  nodes_inputs_masks,
                                  nodes_inputs_channels,
                                  nodes_outputs_masks,
                                  nodes_outputs_channels):
            if module is None:
                continue
            
            reindexed_modules.append(module)
            reindexed_indices.append(len(reindexed_indices))
            reindexed_in_out_channels.append(([ch for v,ch in zip(in_map,in_channels) if v],
                                              [ch for v,ch in zip(out_map,out_channels) if v]))
        
        CONNECTIONS = reindexed_connections
        UNIQUE_LAYERS = reindexed_modules[1:-1]
        LAYERS_INDICES = [i-1 for i in reindexed_indices[1:-1]]
        LAYERS_IN_OUT_CH = [ch_in_out for ch_in_out in reindexed_in_out_channels[1:-1]]
        INPUT_CHANNELS = reindexed_in_out_channels[0][1]
        OUTPUT_CHANNELS = reindexed_in_out_channels[-1][0]
        
        return CONNECTIONS,UNIQUE_LAYERS,LAYERS_INDICES,LAYERS_IN_OUT_CH,INPUT_CHANNELS,OUTPUT_CHANNELS,OUTPUTS

    def __determine_reindexation_maps(self, nodes_inputs_masks, nodes_outputs_masks, nodes_modules):
        nodes_indices_map = []
        nodes_inputs_map = []
        nodes_outputs_map = []
        node_cntr = 0
        for node_module, in_mask, out_mask in zip(nodes_modules,
                                                  nodes_inputs_masks,
                                                  nodes_outputs_masks):
            # node removed
            if node_module is None:
                nodes_indices_map.append(None)
                nodes_inputs_map.append(None)
                nodes_outputs_map.append(None)
                continue

            # node appears
            # new index of module
            nodes_indices_map.append(node_cntr)
            node_cntr += 1
            
            # map inputs
            node_in_mask_map = []
            in_cntr = 0
            for v in in_mask:
                node_in_mask_map.append(in_cntr if v else None)
                in_cntr = in_cntr+1 if v else in_cntr
            
            # map outputs
            node_out_mask_map = []
            out_cntr = 0
            for v in out_mask:
                node_out_mask_map.append(out_cntr if v else None)
                out_cntr = out_cntr+1 if v else out_cntr
            
            # append maps
            nodes_inputs_map.append(node_in_mask_map)
            nodes_outputs_map.append(node_out_mask_map)
        
        return nodes_indices_map,nodes_inputs_map,nodes_outputs_map
    
    def __call__(self, 
                t:torch.Tensor,
                dst_modules:List[Type]):
        return self.__prune(t, dst_modules)
        
