# NN-Executor

Python package for:
* storing and defining models,
* parsing existing model,
* pruning,
* and training

with PyTorch.


Model execution is based on the graph.\
None-layer forward  is called when all needed inputs are ready: \
passed directly to network input or comes from other nodes.\
Execution is finished, when all nodes are executed.


## Define model in json
```
{
  "auto_connect": true,
  "connections":[
      [0,0,9,0] // define source for second branch
      // [src_node,src_output_idx, dst_node, dst_input_idx]
      // node '0' is input of network
  ],
  "outputs": [
    // [src_node,src_output_idx, network_output_idx]
    [-1, 0, 0], // negative indexing is also available
    [ 4, 0, 1] 
    
  ],
    "inputs_channels": [3],
    "outputs_channels": [25],
    "layers": [
      ["torch.nn.Conv2d(3,64,3,padding=1)", 64],
      ["torch.nn.BatchNorm2d(64)"],
      ["torch.nn.ReLU()"],
      ["torch.nn.MaxPool2d(kernel_size=2,stride=2)"],
      
      ["torch.nn.Conv2d(64,128,3,padding=1)", 128],
      ["torch.nn.BatchNorm2d(128)"],
      ["torch.nn.ReLU()"],
      ["torch.nn.MaxPool2d(kernel_size=2,stride=2)"],
      
      # second branch
      ["torch.nn.Conv2d(3,128,7, padding=3)", 128],
      ["torch.nn.MaxPool2d(kernel_size=4,stride=4)"],
      
      // there are allowed multi input and multi output module
      // some of them are predefined
      
      ["nn_executor.models.Add(num=2)", 
       [[128,128], [128]], // inputs channels, outputs channel(s)
       [[-3,0], [-1,0]] ], // connection to first and second input
      
      ["torch.nn.Conv2d(128,256,3,padding=1)", 128],
      ["torch.nn.BatchNorm2d(128)"],
      ["torch.nn.ReLU()"],
      ["torch.nn.MaxPool2d(kernel_size=2,stride=2)"],
      
      ["torch.nn.Conv2d(128,15,1,padding=0)", 15],
      
      
      // use own modules
      ["own_module.YOLO(anchors=[[3,4]])", [[15],[15]], [[-1,0],
                                                         [0,0] // <- to get input size
                                                         ]],
    ]
  }
```

## Load network
```
import torch
from nn_exec import utils as nn_exe_utils

device = torch.device('cuda')
model_arch = 'arch.json'
model_state = 'state.pth'
model_description = nn_exe_utils.load((model_arch,
                                        model_state), 
                                        map_location=device,
                                        strict=False)
net = executor.Executor(model_description)

t = torch.rand(3,480,640, dtype=torch.float32)

output = net(t)

```
## Documentation
For now there is no documentation!\
Code's description is also not completed!