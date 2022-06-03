import torch, os, json
from nn_executor import executor, utils as nn_exe_utils
import numpy as np
from .utils import plot_history, log_print


class Saver:
    def __init__(self, 
                 config_path:str,
                 model_state_path:str,
                 opt_state_path:str
                 ) -> None:
        self.config_path = config_path
        self.model_state_path = model_state_path
        self.opt_state_path = opt_state_path

    def __call__(self, 
                 config, model:torch.nn.Module, opt:torch.optim.Optimizer):
        with open(self.config_path, 'w') as cfg_f:
            json.dump(config,cfg_f)

        torch.save(model.state_dict(), self.model_state_path)
        torch.save(opt.state_dict(), self.opt_state_path)


class ExecutorSaver:
    def __init__(self, 
                 config_path:str,
                 model_arch_path:str,
                 model_state_path:str,
                 opt_state_path:str
                 ) -> None:
        self.config_path = config_path
        self.model_state_path = model_state_path
        self.model_arch_path = model_arch_path
        self.opt_state_path = opt_state_path

    def __call__(self, 
                 config, 
                 model:torch.nn.Module,
                 opt:torch.optim.Optimizer):
        
        # save config
        with open(self.config_path, 'w') as cfg_f:
            json.dump(config,cfg_f)
        
        # save model arch and state
        model:executor.Executor = model
        description = model.get_state()
        nn_exe_utils.save((self.model_arch_path, self.model_state_path),
                            description)
        # save opt state
        torch.save(opt.state_dict(), self.opt_state_path)


class BaseCallback:
    """
    Base of callbacks - each callback method takes 
    as argument dictionary with config of model and training process
    """
    def __init__(self, device='cpu'):
        self.early_stop = False
    
    def on_fit_begin(self, config, model:torch.nn.Module, opt:torch.optim.Optimizer): pass
    def on_epoch_begin(self, config, model:torch.nn.Module, opt:torch.optim.Optimizer): pass
    
    def on_training_begin(self, config, model:torch.nn.Module, opt:torch.optim.Optimizer): pass
    def on_training_end(self, config, model:torch.nn.Module, opt:torch.optim.Optimizer): pass
    
    def on_validation_begin(self, config, model:torch.nn.Module, opt:torch.optim.Optimizer): pass
    def on_validation_end(self, config, model:torch.nn.Module, opt:torch.optim.Optimizer): pass
    
    def on_epoch_end(self, config, model:torch.nn.Module, opt:torch.optim.Optimizer): pass
    def on_fit_end(self, config, model:torch.nn.Module, opt:torch.optim.Optimizer): pass
    
    def check_stop(self) -> bool:
        return self.early_stop
    
    def reset(self):
        self.early_stop = False


class ProxyCallback(BaseCallback):

    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    def on_fit_begin(self, config, model:torch.nn.Module, opt:torch.optim.Optimizer): 
        if hasattr(self.obj, 'on_fit_begin') \
           and callable(self.obj.on_fit_begin):
            self.obj.on_fit_begin(config, model, opt)

    def on_epoch_begin(self, config, model, opt): 
        if hasattr(self.obj, 'on_epoch_begin') \
           and callable(self.obj.on_epoch_begin):
            self.obj.on_epoch_begin(config, model, opt)

    def on_training_begin(self, config, model, opt):
        if hasattr(self.obj, 'on_training_begin') \
           and callable(self.obj.on_training_begin):
            self.obj.on_training_begin(config, model, opt)                

    def on_training_end(self, config, model, opt):
        if hasattr(self.obj, 'on_training_end') \
           and callable(self.obj.on_training_end):
            self.obj.on_training_end(config, model, opt)                    

    def on_validation_begin(self, config, model, opt):
        if hasattr(self.obj, 'on_validation_begin') \
           and callable(self.obj.on_validation_begin):
            self.obj.on_validation_begin(config, model, opt)                

    def on_validation_end(self, config, model, opt):
        if hasattr(self.obj, 'on_validation_end') \
           and callable(self.obj.on_validation_end):
            self.obj.on_validation_end(config, model, opt)                

    def on_epoch_end(self, config, model, opt):
        if hasattr(self.obj, 'on_epoch_end') \
           and callable(self.obj.on_epoch_end):
            self.obj.on_epoch_end(config, model, opt)                

    def on_fit_end(self, config, model, opt):
        if hasattr(self.obj, 'on_fit_end') \
           and callable(self.obj.on_fit_end):
            self.obj.on_fit_end(config, model, opt)                


class ListCallback(BaseCallback):
    
    def __init__(self, callbacks=[]):
        super().__init__()
        self.callbacks = callbacks
    
    def on_fit_begin(self, config, model, opt):
        for clb in self.callbacks:
            clb.on_fit_begin(config, model, opt)
    
    def on_epoch_begin(self, config, model, opt): 
        for clb in self.callbacks:
            clb.on_epoch_begin(config, model, opt)
    
    def on_training_begin(self, config, model, opt):
        for clb in self.callbacks:
            clb.on_training_begin(config, model, opt)
    
    def on_training_end(self, config, model, opt):
        for clb in self.callbacks:
            clb.on_training_end(config, model, opt)
    
    def on_validation_begin(self, config, model, opt):
        for clb in self.callbacks:
            clb.on_validation_begin(config, model, opt)
    
    def on_validation_end(self, config, model, opt):
        for clb in self.callbacks:
            clb.on_validation_end(config, model, opt)
    
    def on_epoch_end(self, config, model, opt):
        for clb in self.callbacks:
            clb.on_epoch_end(config, model, opt)
    
    def on_fit_end(self, config, model, opt):
        for clb in self.callbacks:
            clb.on_fit_end(config, model, opt)
    
    def reset(self):
        for clb in self.callbacks:
            clb.reset()
    
    def check_stop(self) -> bool:
        
        for clb in self.callbacks:
            if clb.check_stop():
                return True
        
        return False


class ProxyListCallback(ListCallback):

    def __init__(self, obj_list):
        obj_list = [ProxyCallback(obj) for obj in obj_list]
        super().__init__(obj_list)


class Checkpoint(BaseCallback):
    
    def __init__(self, saver):
        super().__init__()
        self.saver = saver
    
    def on_epoch_end(self, config, model, opt):
        self.saver(config, model, opt)
        log_print("Checkpoint saved !")


class SaveBest(BaseCallback):
    def __init__(self, saver, monitored=[], multipliers=1):
        super().__init__()
        self.saver:Saver = saver
        self.monitored = monitored
        
        # if lists lengths are different
        if type(multipliers) is list and len(multipliers) != len(monitored):
            raise Exception('Different numer of monitored vars and multipliers')
            
        elif multipliers is not None and type(multipliers) is not list: # is number
            multipliers = [multipliers]*len(monitored)
        
        self.multipliers = np.array(multipliers, dtype=np.float32)
        
    def on_epoch_end(self, config, model, opt):
        hist = config.get('history', None)
        
        if hist is None:
            return
        
        is_better = True
        for n, m in zip(self.monitored, self.multipliers):
            vals = hist[n]
            if len(vals) < 2:
                is_better = True
                continue
            
            vals = np.array(vals)*m
            best_idx = np.argmax(vals)
            
            if best_idx != vals.flatten().shape[0]-1:
                is_better = False
                break
        
        if not is_better:
            return
        
        # save info about best epoch
        config['better_'+('_'.join(self.monitored))] = config.get('epoch', 0)
        # save best state
        self.saver(config, model, opt)
        log_print('Best ' + (' and '.join(self.monitored)), 'saved !')


class PlotHistory(BaseCallback):

    def __init__(self, period=10):
        super().__init__()
        self.period = period

    def on_fit_begin(self, config, model, opt):
        epoch = config.get('epoch',0)
        if epoch > 0:
            plot_history(config['history'].copy())

    def on_fit_end(self, config, model, opt):
        epoch = config.get('epoch',0)
        if epoch > 0:
            plot_history(config['history'].copy())

    def on_epoch_end(self, config, model, opt):
        epoch = config.get('epoch',0) + 1

        if (epoch % self.period) == 0:
            plot_history(config['history'].copy())




