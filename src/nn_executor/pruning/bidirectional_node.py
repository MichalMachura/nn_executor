import torch
from typing import Dict, List, Tuple, Type
from nn_executor import modifiers,utils
from .pruning_utils import CONNECTION_ANCHOR, on_available, on_propagate_backward, on_propagate_forward


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
            for i,((backwarded_mask,_),
                   (dst_idx,dst_in_idx)) in enumerate(zip(OUTPUT_MASK_MULS,
                                                          self.dst_nodes[out_idx])):
                # mask both masks
                mask = torch.logical_and(backwarded_mask,MASK)
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
