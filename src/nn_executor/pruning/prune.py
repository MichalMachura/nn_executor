import torch
from torch import nn
from typing import Any, Dict, List, Tuple, Type
from nn_executor import modifiers, models, parser, executor
import queue
from .pruning_utils import CONNECTION
from .bidirectional_node import BidirectionalNode


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

        self.__add_output_nodes_to_backward_queue(backward_put)

        self.__set_nodes_callbacks(forward_put, backward_put,
                                   forward_propagation_put, backward_propagation_put)

        while not forward_queue.empty()\
                or not backward_queue.empty()\
                or not forward_propagation_queue.empty()\
                or not backward_propagation_queue.empty():

            # propagation of active nodes
            self.__propagate_backward(backward_propagation_queue)
            self.__propagate_forward(forward_propagation_queue)

            # nodes modification
            self.__backward_nodes_modification(backward_queue)
            self.__forward_nodes_modification(forward_queue)

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
        model_description = p.parse_module(net, t)

        return model_description

    def __propagate_forward(self, forward_propagation_queue):
        while not forward_propagation_queue.empty():
            mask_mul_bias,\
                src_node_idx,src_out_idx,\
                dst_node_idx,dst_in_idx = forward_propagation_queue.get_nowait()
                # set mask_mul in src node backward buffers
            self.nodes[dst_node_idx].take_from_forward(mask_mul_bias,dst_in_idx)

    def __propagate_backward(self, backward_propagation_queue):
        while not backward_propagation_queue.empty():
            mask_mul,\
                src_node_idx,src_out_idx,\
                dst_node_idx,dst_in_idx = backward_propagation_queue.get_nowait()
                # set mask_mul in src node backward buffers
            self.nodes[src_node_idx].take_from_backward(mask_mul,src_out_idx,dst_node_idx,dst_in_idx)

    def __set_nodes_callbacks(self, forward_put, backward_put, forward_propagation_put, backward_propagation_put):
        for node in self.nodes:
            node.on_backward_available = backward_put
            node.on_forward_available = forward_put
            node.on_propagate_backward = backward_propagation_put
            node.on_propagate_forward = forward_propagation_put

    def __add_output_nodes_to_backward_queue(self, backward_put):
        for node in self.nodes:
            if node.backward_ready:
                backward_put(node.node_idx)

    def __forward_nodes_modification(self, forward_queue):
        while not forward_queue.empty():
            node_idx = forward_queue.get_nowait()
            node = self.nodes[node_idx]
            node.forward(self.nodes_modifiers)

    def __backward_nodes_modification(self, backward_queue):
        while not backward_queue.empty():
            node_idx = backward_queue.get_nowait()
            node = self.nodes[node_idx]
            node.backward(self.nodes_modifiers)

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

