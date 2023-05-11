import os
from typing import Any, List, Tuple
import torch
from torch import nn
from nn_executor import utils
from nn_executor.executor import Executor
from nn_executor.model_description import ModelDescription
import nn_executor.models as mm
from nn_executor.parser import Parser, SUPPORTED_MODULES
from nn_executor.utils import DifferentiateTensors


TEST_DIR = os.path.dirname(__file__)


class ExampleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p1 = mm.Parallel(mm.Add(2),
                              [
            nn.Sequential(
                nn.Conv2d(3, 5, 3, padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(5, 6, 3, padding=(1, 1)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(3, 10, 5, padding=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(10, 6, 3, padding=(1, 1)),
                nn.ReLU(),
            ),
        ])
        self.p2 = mm.Parallel(mm.Cat(dim=1),
                              [
            nn.Sequential(
                mm.Identity()
            ),
            nn.Sequential(
                nn.Conv2d(6, 10, 5, padding=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(10, 1, 3, padding=(1, 1)),
                nn.ReLU(),
            ),
        ])

        self.last = nn.Conv2d(7, 22, 7, stride=(2, 2))
        self.repeated = nn.Conv2d(22, 22, 3, padding=(1, 1))
        self.pruner = mm.Pruner(22, True, True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        p1 = self.p1(x1)
        p2 = self.p2(p1)
        last = self.last(p2)
        r = self.repeated(last)
        r = self.repeated(r)
        p = self.pruner(r)
        return p, x2

def get_example_model_2():
    model = nn.Sequential(
                        mm.Identity(),
                        mm.Parallel(mm.Cat(1),
                                    nn.Sequential(
                                                nn.Conv2d(5, 5, 1),
                                                mm.ResBranch(5, 5, 1, 3),
                                                mm.Parallel(mm.Cat(1),
                                                            mm.ResBranch(5, 5, 1, 3),
                                                            mm.ResBranch(5, 5, 1, 2)
                                                            ),
                                                nn.Conv2d(10, 5, 1),
                                                nn.Conv2d(5, 5, 1),
                                                ),
                                    nn.Conv2d(5,1,1),
                                    mm.Identity(),
                                    ),
                        mm.Identity(),
                        )
    return model

def get_example_description_1() -> ModelDescription:
    unique_layers = [
        nn.Conv2d(3, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(5, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Conv2d(3, 10, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(10, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        mm.Add(num=2),
        mm.Identity(),
        nn.Conv2d(6, 10, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        nn.ReLU(),
        nn.Conv2d(10, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        mm.Cat(dim=1),
        nn.Conv2d(7, 22, kernel_size=(7, 7), stride=(2, 2)),
        nn.Conv2d(22, 22, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    ]
    layers_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18]
    layers_in_out_channels = [([3], [5]),     ([5], [5]),  ([5], [5]),  ([5], [6]),
                              ([6], [6]),     ([3], [10]), ([10], [10]),([10], [10]),
                              ([10], [6]),    ([6], [6]),  ([6,6], [6]),([6], [6]),
                              ([6], [10]),    ([10], [10]),([10], [1]), ([1], [1]),
                              ([6, 1], [7]),  ([7], [22]), ([22], [22]),([22], [22]),
                              ]
    connections = [
        (0, 0, 1, 0),
        (1, 0, 2, 0),
        (2, 0, 3, 0),
        (3, 0, 4, 0),
        (4, 0, 5, 0),
        (0, 0, 6, 0),
        (6, 0, 7, 0),
        (7, 0, 8, 0),
        (8, 0, 9, 0),
        (9, 0, 10, 0),
        (5, 0, 11, 0),
        (10, 0, 11, 1),
        (11, 0, 12, 0),
        (11, 0, 13, 0),
        (13, 0, 14, 0),
        (14, 0, 15, 0),
        (15, 0, 16, 0),
        (11, 0, 17, 0),
        (16, 0, 17, 1),
        (17, 0, 18, 0),
        (18, 0, 19, 0),
        (19, 0, 20, 0),
    ]
    outputs = [(20, 0, 0), (0, 1, 1)]
    md = {
        'layers_indices': layers_indices,
        'unique_layers': unique_layers,
        'layers_in_out_channels': layers_in_out_channels,
        'connections': connections,
        'outputs': outputs,
        'inputs_channels': [3, 3],
        'outputs_channels': [22, 3],
    }

    return ModelDescription(**md)


def pruner(mask: List[int] = None,
           prunable=True,
           activated=True) -> mm.Pruner:
    if mask is None:
        mask = []

    p = mm.Pruner(len(mask), prunable, activated)
    with torch.no_grad():
        p.init_ones()
        p.pruner_weight[:] *= torch.tensor(mask).reshape(p.pruner_weight.shape)

    return p

def reparse_model_description(model_description: ModelDescription,
                              inputs_shapes: List[Tuple[int, int, int, int]],
                              differentiate_tensors: bool,
                              supported_modules: List[Any] = SUPPORTED_MODULES,
                              ) -> ModelDescription:
    model = Executor(model_description)
    md = get_model_description(model, inputs_shapes, differentiate_tensors, supported_modules, reparse=False)

    # simulate writing to file
    storage_format = utils.model_description_to_storage_format(md)
    md = utils.storage_format_to_model_description(storage_format)

    return md


def get_model_description(model,
                          inputs_shapes: List[Tuple[int, int, int, int]],
                          differentiate_tensors: bool = False,
                          supported_modules: List[Any] = SUPPORTED_MODULES,
                          reparse: bool = True) -> ModelDescription:
    with DifferentiateTensors(differentiate_tensors):
        p = Parser(supported_modules)
        inputs = [torch.rand(sh) for sh in inputs_shapes]
        md = p.parse_module(model, *inputs)

    if reparse:
        rmd = reparse_model_description(md, inputs_shapes, differentiate_tensors, supported_modules)
        md = rmd

    return md
