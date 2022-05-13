from unittest import TestCase
from nn_executor import parser
from nn_executor import models as mm
from shared import ExampleModel
import torch


class TestParser(TestCase):
    
    def test_parse(self):        
        net = ExampleModel()
        # example inputs
        t0 = torch.rand(1,3,64,64)
        t1 = torch.rand(1,4,64,64)
        
        h = parser.Parser()
        
        model_desc = h.parse_module(net,t0,t1)
        
        print(model_desc)
        
    
    