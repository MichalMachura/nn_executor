from typing import Any, Dict, List, Tuple
import torch
from torch import nn


def save(filepath:str,
         layers_indeces:List[int],
         layers_map:List[nn.Module],
         connections:List[Tuple[int,int,int,int]],
         outputs:List[Tuple[int,int]], 
         **kwargs):
    layers_map_recreators = [(L.__class__.__module__+'.'+str(L),L.state_dict()) for L in layers_map]
    sd = {
         'layers_indeces':layers_indeces,
         'layers_map_recreators':layers_map_recreators,
         'connections':connections,
         'outputs':outputs,
         **kwargs
         }
    torch.save(sd,filepath)


def load(file_path,
         map_location=None,
         command_transformer=lambda x:x)-> Tuple[
                                                List[int],
                                                List[nn.Module],
                                                List[Tuple[int,
                                                        int,
                                                        int,
                                                        int]],
                                                List[Tuple[int,int]],
                                                Dict[str,Any]
                                                ]:
    sd:Dict = torch.load(file_path, map_location=map_location)
    
    layers_indeces = sd.pop('layers_indeces')
    layers_map_recreators = sd.pop('layers_map_recreators')
    connections = sd.pop('connections')
    outputs = sd.pop('outputs')
    kwargs = sd
    
    # recreate torch modules
    layers_map = []
    for command, state_dict in layers_map_recreators:
        # run dynamic python
        L:nn.Module
        cmd = 'L = '+command_transformer(command)
        exec(cmd)
        L.load_state_dict(state_dict)
        layers_map.append(L)
    
    return layers_indeces, layers_map, connections,outputs, kwargs
    
    
