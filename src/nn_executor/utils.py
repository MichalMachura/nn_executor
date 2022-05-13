from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn
import json


# def save(filepaths:Union[str,Tuple[str,str]],
#          layers_indices:List[int],
#          layers_in_out_channels:List[Tuple[List[int],List[int]]],
#          unique_layers:List[nn.Module],
#          connections:List[Tuple[int,int,int,int]],
#          outputs:List[Tuple[int,int,int]], 
#          **kwargs):
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
    model_description['unique_layers_recreators'] = unique_layers_recreators
    model_description['unique_layers_state_dicts'] = unique_layers_state_dicts
    
    if type(filepaths) is str:
        torch.save(model_description,filepaths)
        
    if type(filepaths) is tuple:
        # take off state dicts from dict to prevent saving in json 
        state_dicts = model_description.pop('unique_layers_state_dicts')
        # and save it in *.pkl
        torch.save(state_dicts,filepaths[1])
        
        # rest of description save as json
        with open(filepaths[0],'w') as f:
            json.dump(model_description, f, sort_keys=False, indent=4)
        
        


def import_module_of_class(variable:str,
                           module:str,
                           cmd:str) -> str:
    return f'import {module}\n{variable} = {module}.{cmd}'


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
    
        if len(file_paths) > 1 or strict:
            model_description['unique_layers_state_dicts'] = torch.load(file_paths[1], map_location=map_location)
    
    unique_layers_recreators = model_description.pop('unique_layers_recreators')
    unique_layers_state_dicts = model_description.pop('unique_layers_state_dicts',
                                                      [{} for i in unique_layers_recreators])
    
    # recreate torch modules
    unique_layers = []
    lcl = {}
    glb = globals()
    for (node_idx, command, 
         in_out_ch, module), state_dict in zip(unique_layers_recreators, 
                                                        unique_layers_state_dicts):
        # run dynamic python
        cmd = command_transformer('L',module,command)
        exec(cmd, glb, lcl)
        L:nn.Module = lcl['L']
        L.load_state_dict(state_dict, strict)
        unique_layers.append(L)
    
    # recreate layers indices
    unique_layers_nodes = [u[0] for u in unique_layers_recreators]
    indices = []
    for u_idx, nodes in enumerate(unique_layers_nodes):
        for node_idx in nodes:
            indices.append((node_idx-1, u_idx))
    
    layers_indices = sorted(indices,key=lambda x:x[0])
    layers_indices = [idx[1] for idx in indices]
    
    # get in/out channels for each node
    unique_in_out_ch = [u[2] for u in unique_layers_recreators]
    layers_in_out_channels = [unique_in_out_ch[idx] for idx in layers_indices]
    
    model_description['unique_layers'] = unique_layers
    model_description['layers_in_out_channels'] = layers_in_out_channels
    model_description['layers_indices'] = layers_indices
    
    return model_description
    
    # return {'layers_indices':layers_indices,
    #         'unique_layers':unique_layers,
    #         'layers_in_out_channels':layers_in_out_channels,
    #         'connections':connections,
    #         'outputs':outputs}
    
    
