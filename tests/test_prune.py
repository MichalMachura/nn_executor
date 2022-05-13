import unittest
import shared
from nn_executor import modifiers, models, prune, parser, utils
import torch


class TestScissors(unittest.TestCase):
    
    def test_run(self):
        L = [
             torch.nn.Conv2d(3,5,3, padding=1),
             torch.nn.BatchNorm2d(5),
             torch.nn.ReLU(),
             shared.pruner([1,0,1,1,0]),
             torch.nn.MaxPool2d(2,2),
             
             torch.nn.Conv2d(5,8,7, padding=3),
             torch.nn.BatchNorm2d(8),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(2,2),
             shared.pruner([1,1,1,1,0,0,0,0]),
             
             torch.nn.Conv2d(8,2,3, padding=1),
             torch.nn.Conv2d(2,4,3, padding=1),
             shared.pruner([1,1,0,1],False,True),
             ]
        model = torch.nn.Sequential(*L)
        
        p = parser.Parser()
        # t = torch.rand((1,3,2*2**2,1*2**2),dtype=torch.float32)
        t = torch.rand((1,3,20*2**2,10*2**2),dtype=torch.float32)
        model_description = p.parse_module(model,t)
        
        utils.save(('tmp.json','tmp.pth'),model_description)
        
        print('Parsed model:')
        print(model_description)
        
        modifiers_map = {
                        torch.nn.Conv2d.__name__:modifiers.Conv2dModifier(),
                        torch.nn.MaxPool2d.__name__:modifiers.MaxPool2dModifier(),
                        torch.nn.ReLU.__name__:modifiers.ReLUModifier(),
                        torch.nn.BatchNorm2d.__name__:modifiers.BatchNorm2dModifier(),
                        models.Pruner.__name__:modifiers.PrunerModifier(True),
                        models.Identity.__name__:modifiers.IdentityModifier(),
                         }
        
        sc = prune.Scissors(model_description,modifiers_map)
        
        sc()
        
        

if __name__ == '__main__':
    
    t = TestScissors()
    t.test_run()