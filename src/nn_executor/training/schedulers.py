from typing import Dict
import numpy as np
import torch


class BaseScheduler:

    def __init__(self):
        pass

    def step(self, optimizer:torch.optim.Optimizer, config:Dict, loss:float, epoch:int):
        pass


class LossDependentScheduler(BaseScheduler):

    def __init__(self, mul=2.0, div=4, init_loss=np.inf, init_lr=1, lr_min=1e-5, lr_max=1):
        super().__init__()
        self.init_lr = init_lr
        self.init_loss = init_loss
        self.mul = mul
        self.div = 1/div
        self.lr_min = lr_min
        self.lr_max = lr_max

    # def calc_lr(self, config, loss):
    #     lr = config.get('scheduler_lr',self.init_lr)
    #     prev_loss = config.get('scheduler_prev_loss',self.init_loss)
        
    #     # decide of lr change 
    #     lr *= self.mul if loss < prev_loss else self.div
    #     lr = max(self.lr_min, min(lr, self.lr_max))
        
    #     # update config
    #     config['scheduler_lr'] = lr
    #     config['scheduler_prev_loss'] = loss
        
    #     return lr
        
        
    def step(self, 
             optimizer:torch.optim.Optimizer, 
             config:Dict, 
             loss:float, 
             epoch:int):
        # calculate lr
        # lr = self.calc_lr(config, loss)
        lr_hist = config.get('scheduler_lr_history',[])
        prev_loss = config.get('scheduler_prev_loss',self.init_loss)
        
        current_lr = []
        # set optimizer's lr
        for p in optimizer.param_groups:
            if 'lr' in p.keys():
                lr = p['lr']
                # decide of new value of lr
                lr *= self.mul if loss < prev_loss else self.div
                lr = max(self.lr_min, min(lr, self.lr_max))
                # update
                p['lr'] = lr
                # store
                current_lr.append(lr)
        
        # store info about lr for each all groups
        lr_hist.insert(epoch,current_lr)
        config['scheduler_lr_history'] = lr_hist
        # store loss value
        config['scheduler_prev_loss'] = loss
        

class CosineScheduler(schedulers.BaseScheduler):
    
    def __init__(self, iter_max:int, lr_min=1e-8, lr_max=1e-2):
        super().__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.iter_max = iter_max
        
    def step(self, 
             optimizer:torch.optim.Optimizer, 
             config:Dict, 
             loss:float, 
             epoch:int):
        
        e = epoch if epoch < self.iter_max else self.iter_max
        lr_current = self.lr_min + 0.5*(self.lr_max-self.lr_min) *(1+torch.cos(torch.tensor(torch.pi* e/self.iter_max), dtype=torch.float32))

        # set optimizer's lr
        for p in optimizer.param_groups:
            if 'lr' in p.keys():
                p['lr'] = lr_current
                
        # store info about lr for each all groups
        H = config.get('scheduler_lr_history', [])
        H.insert(epoch,lr_current)
        config['scheduler_lr_history'] = H
        

