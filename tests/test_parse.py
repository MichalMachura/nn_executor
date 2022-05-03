from turtle import forward
from unittest import TestCase


class TestParser(TestCase):
    
    def test_parse(self):
        import torch
        import torch.nn as nn
        from nn_executor import models as mm
        from nn_executor import parser
        # from nn_executor import executor

        class ExampleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p1 = mm.Parallel(mm.Add(2),
                                 [
                                 nn.Sequential(
                                              nn.Conv2d(3,5,3,padding=(1,1)),
                                              nn.ReLU(),
                                              nn.MaxPool2d(2,2),
                                              nn.Conv2d(5,6,3,padding=(1,1)),
                                              nn.ReLU(),
                                              ),
                                 nn.Sequential(
                                              nn.Conv2d(3,10,5,padding=(2,2)),
                                              nn.ReLU(),
                                              nn.MaxPool2d(2,2),
                                              nn.Conv2d(10,6,3,padding=(1,1)),
                                              nn.ReLU(),
                                              ),
                                  ])
                self.p2 = mm.Parallel(mm.Cat(dim=1),
                                 [
                                 nn.Sequential(
                                              mm.Identity()
                                              ),
                                 nn.Sequential(
                                              nn.Conv2d(6,10,5,padding=(2,2)),
                                              nn.ReLU(),
                                              nn.Conv2d(10,1,3,padding=(1,1)),
                                              nn.ReLU(),
                                              ),
                                  ])
                
                self.last = nn.Conv2d(7,22,7)
                
            def forward(self,x1,x2):
                p1 = self.p1(x1)
                p2 = self.p2(p1)
                last = self.last(p2)
                
                return last, x2
        
        net = ExampleModel()
        # example inputs
        t0 = torch.rand(1,3,64,64)
        t1 = torch.rand(1,4,64,64)
                
        h = parser.Parser()
        
        h.parse_module(net,t0,t1)
        
        print(h)