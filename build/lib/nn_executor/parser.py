from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn
from nn_executor import models, utils


def backward_connections(dst_node_idx:int,
                         node_inputs:Tuple[torch.Tensor,...],
                         saved_outputs:List[Tuple[torch.Tensor,int,int]],
                        ):
    connections = []
    # for each input
    for input_idx, input_t in enumerate(node_inputs):
        # find src of previously saved output tensors
        for t,src_node_idx, output_idx in saved_outputs[::-1]:
            if input_t.data_ptr() == t.data_ptr():
                connections.append((src_node_idx,output_idx, 
                                    dst_node_idx,input_idx))
                break
    
    return connections


def save_output_tensors(src_node_idx:int,
                        node_outputs:Tuple[torch.Tensor,...],
                        saved_outputs:List[Tuple[torch.Tensor,int,int]],
                        ):
    # for each output generated by node
    for output_idx, t in enumerate(node_outputs):
        # prevent saving the same tensor == save first occurrence
        if t.data_ptr() not in [tt[0].data_ptr() for tt in saved_outputs]:
            saved_outputs.append((t,src_node_idx,output_idx))


SUPPORTED_MODULES = [
                    nn.Conv2d,
                    nn.MaxPool2d,
                    nn.BatchNorm2d,
                    nn.ReLU,
                    nn.LeakyReLU,
                    models.Upsample,
                    models.Constant,
                    models.Variable,
                    models.Cat,
                    models.Sub,
                    models.Add,
                    models.Mul,
                    models.Pruner,
                    # Modules like Identity should not be included or should clone input tensors to be describable
                    models.Identity, 
                    ]


class Parser:

    def __init__(self, 
                 supported_modules:List=SUPPORTED_MODULES 
                 ) -> None:
        """
        """     
        self.supported_modules = supported_modules
        self.layers_indices:List[int] = [] # list of indices of layers -> one module can be used many times in network
        self.unique_layers:List[nn.Module] = [] # list of unique modules
        self.outputs:List[Tuple[int,int,int]] = [] # list of output of net (src,src_out_idx)
        self.inputs_channels:List[int] = []
        self.outputs_channels:List[int] = []
        
        self.layers:List[nn.Module] = [] # list of modules
        self.layers_in_out_channels:List[Tuple[List[int],
                                               List[int]]] = [] # list of tuples of in/out tensors' num of channels
        self.out_tensors:List[Tuple[torch.Tensor,int,int]] = [] # (t,src,out_idx)
        self.connections:List[Tuple[int,int,int,int]] = [] # (src,src_out_idx,dst,dst_inp_idx)
        
    def __call__(self, mod:nn.Module, 
                 input_tensors:Tuple[torch.Tensor,...], 
                 output_tensors:Tuple[torch.Tensor,...]):
        if output_tensors is None:
            output_tensors = []
        elif type(output_tensors) is torch.Tensor:
            output_tensors = [output_tensors]
                        
        if mod.__class__ not in self.supported_modules:
            # warning(mod.__class__.__name__+" is unsupported!\nThis could impact on correctness of parsing.")
            return
    
        self.layers.append(mod)
        node_idx = len(self.layers)
        
        # number of in/out ch
        node_in_ch = [t.shape[1] for t in input_tensors]
        node_out_ch = [t.shape[1] for t in output_tensors]
        self.layers_in_out_channels.append((node_in_ch,node_out_ch))
        
        # find source nodes        
        new_connections = backward_connections(dst_node_idx=node_idx,
                                               node_inputs=input_tensors,
                                               saved_outputs=self.out_tensors
                                               )
        # save new connections
        self.connections.extend(new_connections)
        
        save_output_tensors(src_node_idx=node_idx,
                            node_outputs=output_tensors,
                            saved_outputs=self.out_tensors
                            )
        
    def parse_module(self, 
                     model:nn.Module, 
                     *inputs:torch.Tensor) -> Dict[str,Any]:
        # set callbacks for layers
        hooks = self.__set_hooks(model)
        # parse by forward pass
        self.__add_inputs(*inputs)
        with utils.EvalMode(model):
            outputs = model(*inputs)
        # remove hooks
        self.__remove_hooks(hooks)
        
        # pack tensor into list
        if outputs is None:
            outputs = []
        elif type(outputs) is torch.Tensor:
            outputs = [outputs]
        
        # connect outputs with their sources
        self.__find_output_connetions(outputs)
        # map layers nodes to the unique layers
        self.__extract_unique_layers()
        # get channels for inputs and outputs
        self.__in_out_channels(inputs,outputs)
        # get results
        sd = self.__model_description()
        # remove stored data
        self.__clear()
        
        return sd
    
    def __set_hooks(self, 
                    model:torch.nn.Module):
        hook = lambda *x: self(*x)
        hooks = []
        # set hooks for all modules without main
        for n, m in model.named_children():
            for mm in m.modules():
                h = mm.register_forward_hook(hook)
                hooks.append(h)
        
        return hooks
        
    def __remove_hooks(self, hooks):
        # remove all hooks 
        for h in hooks:
            h.remove()
    
    def __in_out_channels(self, inputs, outputs):
        self.inputs_channels = [x.shape[1] for x in inputs]
        self.outputs_channels = [x.shape[1] for x in outputs]
    
    def __find_output_connetions(self, 
                                 outputs):
        # connect outputs with their sources
        out_src = backward_connections(dst_node_idx=-1,
                                       node_inputs=outputs,
                                       saved_outputs=self.out_tensors
                                       )
        self.outputs = [(s,si,di) for s,si,d,di in out_src]
        
    def __extract_unique_layers(self):
        self.unique_layers = []
        self.layers_indices = []
        for layer_node_idx, L in enumerate(self.layers):
            # is unique?
            if L not in self.unique_layers:
                self.unique_layers.append(L)
            
            # find unique layer idx
            for idx, unique_layer in enumerate(self.unique_layers):
                # is the same layer?
                if L == unique_layer:
                    self.layers_indices.append(idx)
                    break
    
    def __clear(self):
        self.layers_indices = []
        self.unique_layers = []
        self.outputs = []
        self.inputs_channels = []
        self.outputs_channels = []
        self.layers = []
        self.layers_in_out_channels = []
        self.out_tensors = []
        self.connections = []
        
    def __add_inputs(self, *args):
        # for zero node
        for i,t in enumerate(args):
            self.out_tensors.append((t,0,i))
    
    def __model_description(self) -> Dict:
        sd = {
             'layers_indices':self.layers_indices,
             'layers_in_out_channels':self.layers_in_out_channels,
             'unique_layers':self.unique_layers,
             'connections':self.connections,
             'outputs':self.outputs,
             'inputs_channels':self.inputs_channels,
             'outputs_channels':self.outputs_channels,
             }
        
        return sd
    
    def __repr__(self) -> str:
        s = "Layers:\n"
        for L in self.layers:
            # s += str(id(L)) + '\t' + str(L)+'\n'
            s += str(L)+'\n'
            
        s += "Unique layers:\n"
        for L in self.unique_layers:
            # s += str(id(L)) + '\t' + str(L)+'\n'
            s += str(L)+'\n'
            
        s += "Layers indices:\n"
        s += str(self.layers_indices)+'\n'
            
        s += 'Connections [(src,src_out_idx,dst,dst_in_idx)]:\n'
        for c in self.connections:
            s += str(c)+'\n'
            
        s += 'Output source layers [(src,src_out_idx,output_idx)]:\n'
        for c in self.outputs:
            s += str(c)+'\n'
            
        s += 'Outputs of nodes [(id,shape,src,src_out_idx)]:\n'
        for o in self.out_tensors:
            s += str(id(o[0]))+', '
            s += str(tuple(o[0].shape))+', '
            s += str(o[1])+', '
            s += str(o[2])+'\n'
        
        return s