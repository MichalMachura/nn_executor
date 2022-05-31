from typing import Any, Dict, List, Tuple, Union, Type
from warnings import warn
import torch
from torch import nn
import json
from nn_executor import models


def between_all(op, L:List):
    result = L[0]
    for element in L[1:]:
        result = op(result,element)
    return result


def get_number_of_params(model):
    p = 0

    for param in model.parameters():
        size = param.size()
        tmp = 1
        for i in range(len(size)):
            tmp *= size[i] 
        tmp = tmp if len(size) else 0
        p += tmp

    return p
    
    
class DifferentiateTensors:
    def __init__(self, differentiable:bool=True) -> None:
        self.differentiable:bool = differentiable
        self.buffered:bool = models.DIFFERENTIATE_TENSOR
    
    def __enter__(self, *args):
        self.buffered = models.DIFFERENTIATE_TENSOR
        models.DIFFERENTIATE_TENSOR = self.differentiable 
    
    def __exit__(self, *args):
        global DIFFERENTIATE_TENSOR
        DIFFERENTIATE_TENSOR = self.buffered

class TrainingMode:
    def __init__(self,model:torch.nn.Module, train:bool=True) -> None:
        self.train:bool = train
        self.model:torch.nn.Module = model
        self.buffered_mode:bool = model.training
    
    def __enter__(self, *args):
        self.buffered_mode = self.model.training
        self.model.train(self.train) 
    
    def __exit__(self, *args):
        self.model.train(self.buffered_mode)


class EvalMode(TrainingMode):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model, False)        
        
        
def save(filepaths:Union[str,Tuple[str,str]],
         model_description:Dict[str,Any],
         ):
    """
    Save state of executor.

    :param filepaths: str - save all into one file 
            or tuple of strings - save model as json to first, 
            and state dict to second pkl
    
    :param model_description: dict with parsing results
    """
    model_description = model_description.copy()
    unique_layers:List[nn.Module] = model_description.pop('unique_layers')
    
    layers_indices:List[int] = model_description.pop('layers_indices')
    layers_in_out_channels = model_description.pop('layers_in_out_channels')
    
    # assing node indices for each unique layer
    unique_layers_nodes = [[] for u in unique_layers]
    for node_idx, u_idx in enumerate(layers_indices):
        unique_layers_nodes[u_idx].append(node_idx+1)
    
    # assign in/out channels to unique rather than every node
    unique_layers_in_out = [([],[]) for q in unique_layers]
    for idx, in_out_ch in zip(layers_indices, 
                              layers_in_out_channels):
        unique_layers_in_out[idx] = in_out_ch
    
    unique_layers_recreators = [[nodes,str(L),in_out_ch,L.__class__.__module__] \
                                for L,in_out_ch,nodes in zip(unique_layers, 
                                                       unique_layers_in_out,
                                                       unique_layers_nodes)]
    unique_layers_state_dicts = [L.state_dict() for L in unique_layers]
    model_description['layers'] = unique_layers_recreators
    model_description['layers_state_dicts'] = unique_layers_state_dicts
    
    if type(filepaths) is str:
        torch.save(model_description,filepaths)
        
    if type(filepaths) is tuple:
        # take off state dicts from dict to prevent saving in json 
        state_dicts = model_description.pop('layers_state_dicts')
        # and save it in *.pkl
        torch.save(state_dicts,filepaths[1])
        
        # rest of description save as json
        with open(filepaths[0],'w') as f:
            json.dump(model_description, f, sort_keys=False, indent=4)
        
        


def import_module_of_class(variable:str,
                           module:str,
                           cmd:str) -> str:
    return f'import torch, {module}\n{variable} = {module}.{cmd}'


def split_module_class(module_class_call:str):
    parenthesis_beg = module_class_call.find('(')
    module_class = module_class_call[:parenthesis_beg]
    split_pos = module_class.rfind('.')
    # extract
    module_name = module_class_call[:split_pos]
    layer_class = module_class_call[split_pos+1:]
    
    return module_name, layer_class


def check_format(input_list, pattern):
    if len(input_list) != len(pattern):
        return False
        
    for data, ref_type in zip(input_list, pattern):
        if type(data) is ref_type \
                or ref_type is list and type(data) is tuple: # treat tuple as list
            continue
        
        else:
            return False
    
    return True


def first_available(indices:List[int]):
    idx = 0
    while idx in indices:
        idx += 1
    
    return idx


def indices_availability(new_indices:List[int], 
                         indices_in_use:List[int]):
    for idx in new_indices:
        if idx in indices_in_use:
            return False
    return True


def find_src_of_dst(connections, dst_idx, dst_in_idx):
    
    for (user_src_idx, user_src_out_idx, user_dst_idx, user_dst_in_idx) in connections:
        if user_dst_idx == dst_idx and user_dst_in_idx == dst_in_idx:
            return (user_src_idx, user_src_out_idx, user_dst_idx, user_dst_in_idx)
    
    return None



def load(file_paths:Union[str,Tuple[str,str]],
         map_location=None,
         strict=True,
         command_transformer=import_module_of_class)-> Tuple[
                                                Dict[str,Any],
                                                Dict[str,Any]
                                                ]:
    if type(file_paths) is str:
        model_description:Dict = torch.load(file_paths, map_location=map_location)
    else:
        f = open(file_paths[0],'r')
        model_description:Dict = json.load(f)
        f.close()
    
        if len(file_paths) > 1:
            try:
                model_description['layers_state_dicts'] = torch.load(file_paths[1], map_location=map_location)
            except:
                print("Problem with opening file", file_paths[1])
                print("State dicts not loaded!!!")
                pass
    
    layers_description = model_description.pop('layers')
    layers_state_dicts = model_description.pop('layers_state_dicts',
                                                [{} for i in layers_description])
    
    # extract descriptions
    nodes_indices, layers, nodes_channels, modules = [],[],[],[]
    
    # init connections if not defined 
    model_description['connections'] = model_description.get('connections',[])
    
    # for auto channels deduction use only first input
    prev_ch = model_description['inputs_channels'][0] 
    # input layer index
    indices_in_use = [0]
    for layer_desc in layers_description: 
        free_idx = first_available(indices_in_use)
        
        # non list -> list with one element  
        # module.layer
        if type(layer_desc) is not list:
            layer_desc = [layer_desc]
        
        # parsing depends on format
        # [module.layer]
        if check_format(layer_desc, [str]):
            node_indices = [free_idx]
            channels = [[prev_ch],[prev_ch]]
            # split module and class constructor
            module_name, layer_class = split_module_class(layer_desc[0])

        # [module.layer, ch_out]
        elif check_format(layer_desc, [str, int]):
            node_indices = [free_idx]
            channels = [[prev_ch],[layer_desc[1]]]
            # split module and class constructor
            module_name, layer_class = split_module_class(layer_desc[0])
        
        # [module.layer, list_of_channels_list]
        elif check_format(layer_desc, [str, list])\
                and check_format(layer_desc[1], [list, list]):
            node_indices = [free_idx]
            channels = [L.copy() for L in layer_desc[1]]
            # split module and class constructor
            module_name, layer_class = split_module_class(layer_desc[0])
        
        # [indices_list, module.layer]
        elif check_format(layer_desc, [list, str]):
            node_indices = layer_desc[0].copy()
            channels = [[prev_ch],[prev_ch]]
            # split module and class constructor
            module_name, layer_class = split_module_class(layer_desc[1])
        
        # [indices_list, module.layer, ch_out]
        elif check_format(layer_desc, [list, str, int]):
            node_indices = layer_desc[0].copy()
            channels = [[prev_ch],[layer_desc[2]]]
            # split module and class constructor
            module_name, layer_class = split_module_class(layer_desc[1])
        
        # [indices_list, module.layer, list_of_channels_lists]
        elif check_format(layer_desc, [list, str, list]) \
                and check_format(layer_desc[2],[list, list]):
            node_indices = layer_desc[0].copy()
            channels = [L.copy() for L in layer_desc[2]]
            # split module and class constructor
            module_name, layer_class = split_module_class(layer_desc[1])
        
        # [module.layer, list_of_channels_lists, connections]
        elif check_format(layer_desc, [str, list, list]) \
                and check_format(layer_desc[1],[list, list]):
            node_indices = [free_idx]
            channels = [L.copy() for L in layer_desc[1]]
            # split module and class constructor
            module_name, layer_class = split_module_class(layer_desc[0])
            
            # relative connections
            for dst_in_idx, (rel_src_idx, rel_src_out_idx) in enumerate(layer_desc[2]):
                model_description['connections'].append((rel_src_idx, rel_src_out_idx, 
                                                         free_idx, dst_in_idx))
            
        
        # basic / automatic format pattern
        # [indices_list, layer, list_of_channels_lists, module]
        elif check_format(layer_desc, [list, str, list, str]) \
                and check_format(layer_desc[2],[list, list]):
            node_indices = layer_desc[0].copy()
            layer_class = layer_desc[1]
            channels = [L.copy() for L in layer_desc[2]]
            # split module and class constructor
            module_name = layer_desc[3]
        
        # wrong format
        else:
            raise RuntimeError(str(layer_desc),"is not proper format of layer description")
        
        # check proposed indices
        if not indices_availability(node_indices, indices_in_use):
            raise RuntimeError("Proposed indices", node_indices, " are not proper for:", layer_desc)
        
        nodes_indices.append(node_indices)
        nodes_channels.append(channels)
        layers.append(layer_class)
        modules.append(module_name)
        
        # get first output' channels
        prev_ch = channels[1][0]
        # store used new indices
        indices_in_use.extend(node_indices) 
        
    # recreate torch modules
    layers_nn_modules = []
    lcl, glb = {}, globals()
    for class_call, module_name in zip(layers, modules):
        # run dynamic python
        cmd = command_transformer('L',module_name,class_call)
        exec(cmd, glb, lcl)
        L:nn.Module = lcl['L']
        layers_nn_modules.append(L)
    
    # load state dicts
    for L, state_dict in zip(layers_nn_modules, layers_state_dicts):
        L.load_state_dict(state_dict, strict)
    
    # recreate layers indices
    indices = []
    for u_idx, nodes in enumerate(nodes_indices):
        for node_idx in nodes:
            indices.append((node_idx-1, u_idx))
    
    layers_indices = sorted(indices,key=lambda x:x[0])
    layers_indices = [idx[1] for idx in layers_indices]
    
    # get in/out channels for each node
    layers_in_out_channels = [nodes_channels[idx] for idx in layers_indices]
    
    model_description['unique_layers'] = layers_nn_modules
    model_description['layers_in_out_channels'] = layers_in_out_channels
    model_description['layers_indices'] = layers_indices
    
    
    
    # relative connections
    connections = []
    for src_idx, src_out_idx, dst_idx, dst_in_idx in model_description['connections']:
        src_idx_computed = dst_idx + src_idx
        # negative idx mean: use src_idx'th layer before dst_idx
        if src_idx < 0 and src_idx_computed >= 0:
            connections.append((src_idx_computed, src_out_idx, dst_idx, dst_in_idx))
        else:
            connections.append((src_idx, src_out_idx, dst_idx, dst_in_idx))
    
    
    # auto connection
    auto_connections = []
    if model_description.get("auto_connect", None):
        # create list of connections
        input_layer_ch_in_out = [[], model_description.get('inputs_channels',[0])]
        extended_in_out_ch = [input_layer_ch_in_out,*layers_in_out_channels]
                
        # for each layer
        for dst_idx, (channels_of_inputs, _) in enumerate(extended_in_out_ch):
            # for each of node inputs 
            for dst_in_idx, ch_in in enumerate(channels_of_inputs):
                # if user defined connection not exist for this input
                connection_established = find_src_of_dst(connections,dst_idx,dst_in_idx) is not None

                # for previous layers in order from the closest
                for non_pos_src_idx, (_,channels_of_outputs) in enumerate(extended_in_out_ch[:dst_idx][::-1]):
                    # previous iteration has found connection -> check next input
                    if connection_established:
                        break
                    # to true src layer index
                    src_idx = dst_idx-1-non_pos_src_idx
                    # for each src output
                    for src_out_idx, ch_out in enumerate(channels_of_outputs):
                        # criterion can be only channels -- whole shape is unknown and can depends on input shape
                        if ch_out == ch_in:
                            auto_connections.append((src_idx, src_out_idx, dst_idx, dst_in_idx))
                            # prevent of checking next previous layers and next outputs
                            connection_established = True
                            break
                
                if not connection_established:
                    # check non established connections in user defined connections
                    warn(f"Connection for dst={(dst_idx,dst_in_idx)} is not established.")
        
        
        # extend user connections 
        connections.extend(auto_connections)
        connections = sorted(connections, key=lambda x: 10*x[2]+x[3])
    
    # update connections
    model_description['connections'] = connections
            
    
    return model_description
    
    # return {'layers_indices':layers_indices,
    #         'unique_layers':unique_layers,
    #         'layers_in_out_channels':layers_in_out_channels,
    #         'connections':connections,
    #         'outputs':outputs}
    
    
