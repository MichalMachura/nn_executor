import unittest
import shared
from nn_executor import modifiers, models, prune, parser, utils
import torch


class TestScissors(unittest.TestCase):
    
    def test_run(self):
        R1 = torch.nn.Sequential(
            shared.pruner([1,1,1,1,0]),
            models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5,3,3, padding=1),
                                shared.pruner([1,1,0]),
                                torch.nn.Conv2d(3,5,3, padding=1),
                                shared.pruner([1,1,0,1,0],False),
                            )
                            ]),
            shared.pruner([1,1,1,1,0]),
            models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5,3,3, padding=1),
                                shared.pruner([0,0,0]),
                                torch.nn.Conv2d(3,5,3, padding=1),
                                shared.pruner([1,0,0,1,0],False),
                            )
                            ]),
            shared.pruner([1,1,1,1,0]),
            models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5,8,3, padding=1),
                                shared.pruner([1,1,1,0,0,0,1,1]),
                                torch.nn.Conv2d(8,5,3, padding=1),
                                shared.pruner([0,1,0,1,0],False),
                            )
                            ]),
            shared.pruner([1,1,1,1,0]),
        )
        R2 = torch.nn.Sequential(
            shared.pruner([0,0,0,0,0]),
            models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5,3,3, padding=1),
                                shared.pruner([1,1,0]),
                                torch.nn.Conv2d(3,5,3, padding=1),
                                shared.pruner([1,1,0,1,0],False),
                            )
                            ]),
            shared.pruner([0,0,0,0,0]),
            models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5,3,3, padding=1),
                                shared.pruner([0,0,0]),
                                torch.nn.Conv2d(3,5,3, padding=1),
                                shared.pruner([1,0,0,1,0],False),
                            )
                            ]),
            shared.pruner([0,0,0,0,0]),
            models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5,8,3, padding=1),
                                shared.pruner([1,1,1,0,0,0,1,1]),
                                torch.nn.Conv2d(8,5,3, padding=1),
                                shared.pruner([0,1,0,1,0],False),
                            )
                            ]),
            shared.pruner([0,0,0,0,0]),
        )
    
        L = [
             torch.nn.Conv2d(3,5,3, padding=1),
             torch.nn.BatchNorm2d(5),
             torch.nn.ReLU(),
             shared.pruner([1,0,1,1,0]),
             torch.nn.MaxPool2d(2,2),
             
             # x3 cat
             models.Parallel(models.Cat(1),
                             [
                             models.Identity(),
                             torch.nn.Sequential(
                                    torch.nn.Conv2d(5,6,3, padding=1),
                                    shared.pruner([1,1,1,0,0,0]),
                                    torch.nn.Conv2d(6,6,3, padding=1),
                                    shared.pruner([1,1,0,0,1,0]),
                                    shared.pruner([1,1,1,0,1,0]),
                                    torch.nn.Conv2d(6,2,3, padding=1),
                                    shared.pruner([1,0]),
                                    ),
                             torch.nn.Sequential(
                                    torch.nn.Conv2d(5,6,9, padding=4),
                                    shared.pruner([0,0,0,0,0,0]),
                                    torch.nn.Conv2d(6,2,3, padding=1),
                                    ),
                             torch.nn.Sequential(
                                    torch.nn.Conv2d(5,6,3, padding=1),
                                    shared.pruner([1,1,1,1,1,1]),
                                    torch.nn.Conv2d(6,5,3, padding=1),
                                    shared.pruner([1,0,1,0,1]),
                                    ),
                             torch.nn.Sequential(
                                    torch.nn.Conv2d(5,6,3, padding=1),
                                    shared.pruner([1,1,0,1,1,1]),
                                    torch.nn.Conv2d(6,3,3, padding=1),
                                    shared.pruner([1,0,0]),
                                    ),
                             ]),
             
             shared.pruner([1,1,1,1,0,
                            0,1,
                            1,1,
                            1,1,1,0,1,
                            0,1,0]),
             torch.nn.Conv2d(17,8,7, padding=3),
             torch.nn.BatchNorm2d(8),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(2,2),
             shared.pruner([1,1,1,1,0,0,0,0]),
             
             torch.nn.Conv2d(8,5,3, padding=1),
             models.Parallel(models.Cat(1),[R1,R2]),
             torch.nn.Conv2d(10,5,3, padding=1),
             torch.nn.Conv2d(5,2,3, padding=1),
             torch.nn.Conv2d(2,4,3, padding=1),
             shared.pruner([1,1,0,1],False,True),
             ]
        model = torch.nn.Sequential(*L)
        
        p = parser.Parser()
        # t = torch.rand((1,3,2*2**2,1*2**2),dtype=torch.float32)
        t = torch.rand((1,3,20*2**2,10*2**2),dtype=torch.float32)
        
        with utils.DifferentiateTensors(False):
            model_description = p.parse_module(model,t)
        
        utils.save(('tmp.json','tmp.pth'),model_description)
        
        modifiers_map = {
                        torch.nn.Conv2d:modifiers.Conv2dModifier(),
                        torch.nn.MaxPool2d:modifiers.MaxPool2dModifier(),
                        torch.nn.ReLU:modifiers.ReLUModifier(),
                        torch.nn.BatchNorm2d:modifiers.BatchNorm2dModifier(),
                        models.Pruner:modifiers.PrunerModifier(True),
                        models.Identity:modifiers.IdentityModifier(),
                        models.Cat:modifiers.CatModifier(),
                        models.Add:modifiers.AddModifier(),
                        }
        
        sc = prune.Scissors(model_description,modifiers_map)
        with utils.DifferentiateTensors(False):
            model_description = sc(t,parser.SUPPORTED_MODULES[:-1])
        
        utils.save(('tmp2.json','tmp2.pth'),model_description)

if __name__ == '__main__':
    
    t = TestScissors()
    t.test_run()