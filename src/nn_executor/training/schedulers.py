from typing import Dict
import numpy as np
import torch


class BaseScheduler:

    def __init__(self):
        pass

    def step(self, optimizer:torch.optim.Optimizer, config:Dict, loss:float, epoch:int):
        pass


class LossDependentScheduler(BaseScheduler):

    def __init__(self, mul=2.0, div=4, init_loss=-np.inf, init_lr=1, lr_min=1e-5, lr_max=1):
        super().__init__()
        self.init_lr = init_lr
        self.init_loss = init_loss
        self.mul = mul
        self.div = 1/div
        self.lr_min = lr_min
        self.lr_max = lr_max

    def calc_lr(self, config, loss):
        lr = config.get('scheduler_lr',self.init_lr)
        prev_loss = config.get('scheduler_prev_loss',self.init_loss)
        
        # decide of lr change 
        lr *= self.mul if loss < prev_loss else self.div
        lr = max(self.lr_min, min(lr, self.lr_max))
        
        # update config
        config['scheduler_lr'] = lr
        config['scheduler_prev_loss'] = loss
        
        return lr
        
        
    def step(self, 
             optimizer:torch.optim.Optimizer, 
             config:Dict, 
             loss:float, 
             epoch:int):
        # calculate lr
        lr = self.calc_lr(config, loss)

        # set optimizer's lr
        for p in optimizer.param_groups:
            if 'lr' in p.keys():
                p['lr'] = lr




