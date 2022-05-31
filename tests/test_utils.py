from unittest import TestCase
import torch
import torch.nn as nn
from nn_executor import models as mm
from nn_executor import utils, executor
import shared


class TestSaveLoad(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
    
    def test_with_example_model_pth(self):
        
        model_description = shared.get_model_description_1()
        # test save
        utils.save('tests/tmp/test_tmp.pth',model_description)
        # test load
        md = utils.load('tests/tmp/test_tmp.pth',torch.device('cpu'))
        
        with torch.no_grad():
            t1 = torch.rand((1,3,64,64))
            t2 = torch.rand((1,3,3,3))
            
            e1 = executor.Executor(model_description).eval()
            e2 = executor.Executor(md).eval()
            
            v1 = e1(t1,t2)
            v2 = e2(t1,t2)
            d1 = (v1[0]-v2[0]).abs().sum()
            d2 = (v1[1]-v2[1]).abs().sum()
            
            self.assertAlmostEquals(d1,0.0,3,"Results of models before and after save/load to file are different.")
            self.assertAlmostEquals(d2,0.0,3,"Results of models before and after save/load to file are different.")
        
    
    def test_with_example_model_json(self):
        
        model_description = shared.get_model_description_1()
        # test save
        utils.save(('tests/tmp/test_json.json','tests/tmp/test_json.pth'),model_description)
        # test load
        md = utils.load(('tests/tmp/test_json.json','tests/tmp/test_json.pth'),torch.device('cpu'),True)
        
        with torch.no_grad():
            t1 = torch.rand((1,3,64,64))
            t2 = torch.rand((1,3,3,3))
            
            e1 = executor.Executor(model_description).eval()
            e2 = executor.Executor(md).eval()
            
            v1 = e1(t1,t2)
            v2 = e2(t1,t2)
            d1 = (v1[0]-v2[0]).abs().sum()
            d2 = (v1[1]-v2[1]).abs().sum()
            
            self.assertAlmostEquals(d1,0.0,3,"Results of models before and after save/load to file are different.")
            self.assertAlmostEquals(d2,0.0,3,"Results of models before and after save/load to file are different.")
        
    
    def test_layers_description_formats(self):
        
        model_description = shared.get_model_description_1()
        # test load
        md = utils.load(('tests/tmp/test_layers_description_formats.json',),torch.device('cpu'),strict=False)
        utils.save(('tests/tmp/test_layers_description_formats_save.json','tests/tmp/test_layers_description_formats_save.pth'),md)
        
        with torch.no_grad():
            t1 = torch.rand((1,3,64,64))
            t2 = torch.rand((1,3,3,3))
            
            e1 = executor.Executor(model_description).eval()
            e2 = executor.Executor(md).eval()
            # use the same weights -- checked is only topology
            e2.load_state_dict(e1.state_dict())
            
            v1 = e1(t1,t2)
            v2 = e2(t1,t2)
            d1 = (v1[0]-v2[0]).abs().sum()
            d2 = (v1[1]-v2[1]).abs().sum()
            
            self.assertAlmostEquals(d1,0.0,3,"Results of models before and after save/load to file are different.")
            self.assertAlmostEquals(d2,0.0,3,"Results of models before and after save/load to file are different.")
        
        
        