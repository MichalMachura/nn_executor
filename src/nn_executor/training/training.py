from typing import Dict, List
import time
from . import utils
from . import callbacks as clbs
from . import criterions as crits
from . import metrics as mets
from . import preprocessing as prep
from . import schedulers as schs
from .utils import seconds_to_hmsms
import torch


def compute_metrics(metrics, y_predict, labels):
    metrics_values = {}
    
    for k, f in metrics.items():
        metrics_values[k] = f(y_predict, labels)
        
    return metrics_values


def mult_dict_scalar(dict_, scalar):
    return {k:scalar*v for k,v in dict_.items()}


def dict_values_from_items(dict_values):
    return {k:v.item() if isinstance(v,torch.Tensor) else v for k,v in dict_values.items()}


def dict_add(d1, d2):
    d3 = {}
    
    for k in d2.keys():
        d3[k] = d1[k] + d2[k]
        
    return d3
    

def mean_dict(d1, d2, mul1, mul2):
    d1_ = mult_dict_scalar(d1, mul1)
    d2_ = mult_dict_scalar(d2, mul2)
    
    d_out = mult_dict_scalar(dict_add(d1_, d2_), 1.0/(mul1+mul2))
    
    return d_out


def mean_loss(loss1, loss2, mul1, mul2):
    loss1 = loss1 * mul1
    loss2 = loss2 * mul2
    
    loss_out =(loss1+loss2)/(mul1+mul2)
    
    return loss_out


def dict_to_str(d):
    s = ' '
    for k,v in d.items():
        s += k +' = '+str(v)+' '
    return s


def keys_to_str_format(keys, notation='.5f'):
    s = ''
    for k in keys:
        s += str(k)+'={:'+notation+'} '
    
    return s


def now():
    return time.ctime(time.time())


class NetTrainer:
    def __init__(self, 
                 model, 
                 config, 
                 criterion, 
                 optimizer, 
                 metrics={}, 
                 callbacks=[],
                 scheduler=schs.BaseScheduler()
                 ) -> None:
        self.config:Dict = config
        self.model:torch.nn.Module = model
        self.metrics:Dict[str,mets.BaseMetric] = metrics
        self.optimizer:torch.optim.Optimizer = optimizer
        self.callbacks:List[clbs.BaseCallback] = callbacks
        self.criterion:crits.BaseCriterion = criterion
        self.scheduler:schs.BaseScheduler = scheduler
    
    @property
    def epoch(self):
        e = self.config.get('epoch',0)
        return e

    @epoch.setter
    def epoch(self,v):
        self.config['epoch'] = v
        
    @epoch.deleter
    def epoch(self):
        self.config.pop('epoch',None)
    
    @property
    def history(self):
        H = self.config.get('history',{})
        return H

    @history.setter
    def history(self,v):
        self.config['history'] = v
        
    @history.deleter
    def history(self):
        self.config.pop('history',None)

    def set_history(self, idx, **kwargs):
        history:Dict = self.history
        for k,v in kwargs.items():
            k_hist = history.get(k,[])
            k_hist.extend([None]*(1+idx-len(k_hist)))
            # insert value
            k_hist[idx] = v
            # update history for k-key
            history[k] = k_hist
        
        self.history = history

    def fit(self, 
              train_generator:prep.BaseGenerator,
              val_generator:prep.BaseGenerator=None,
              scheduler_every_batch=-1):
        # get model device
        dev = next(self.model.parameters()).device
        # get scheduler object
        scheduler = self.scheduler
        # copy list of original callbacks
        clbks = self.callbacks.copy()
        # add all callbacks
        clbks.append(train_generator)
        clbks.append(clbs.ProxyCallback(self.criterion))
        clbks.append(clbs.ProxyCallback(scheduler))
        clbks.append(clbs.ProxyListCallback(self.metrics.values()))
        # create proxy callback to call all callback
        callback = clbs.ListCallback(clbks)
        # bar post text schemes
        bar_len = 20
        train_txt = 'loss={:.5f} '+keys_to_str_format(self.metrics.keys())+'time={} now={}'
        val_txt = 'loss={:.5f} '+keys_to_str_format(self.metrics.keys()) \
                  +'| val_loss={:.5f} '+keys_to_str_format(['val_'+k for k in self.metrics.keys()])+'time={} now={}'
                  
        # get from config
        update_period = self.config['update_period']
        epochs = self.config['max_epochs']
        
        #  fit begin event
        callback.on_fit_begin(self.config, self.model, self.optimizer)
        
        # start from the trainer's epoch state
        for epoch in range(self.epoch, epochs):
            pre_text = f'Epoch {epoch}/{epochs}'
            # display bar
            utils.bar(0,len(train_generator), length=bar_len,pre_text=pre_text, post_text='start')
            # reset loss and metrics
            running_loss = 0.0
            running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)
            # set training mode
            self.model.train()
            # epoch begin event
            callback.on_epoch_begin(self.config, self.model, self.optimizer)
            # training begin event
            callback.on_training_begin(self.config, self.model, self.optimizer)
            # time measurement
            t0 = time.time()
            # reset gradient
            self.optimizer.zero_grad()
            batches_time = 0.0
            
            for i in range(len(train_generator)):
                # get data and labels
                # inputs, labels = train_generator[i]
                inputs, labels = train_generator.__getitem__(i,debug=True)
                batch_size = inputs.shape[0]
                # get outputs
                outputs = self.model(inputs)
                # get sum of output losses 
                loss:torch.Tensor = self.criterion(outputs, labels) / update_period
                # backward propagation
                loss.backward()
                
                # calculate metrics
                metrics_dict = compute_metrics(self.metrics, outputs, labels)
                # compute mean loss and metrics
                running_loss = mean_loss(running_loss, loss.item()*update_period, i*train_generator.batch_size, 1.0*batch_size)
                running_metrics = mean_dict(running_metrics, metrics_dict, i*train_generator.batch_size, 1.0*batch_size)
                
                # update based on batch number/idx
                if ((i+1) % update_period) == 0:
                    # update model params
                    self.optimizer.step()
                    # reset gradient
                    self.optimizer.zero_grad()
                    # when frequent scheduler step
                    if scheduler_every_batch > 0 and ((i+1) % scheduler_every_batch) == 0:
                        scheduler.step(self.optimizer, self.config, running_loss, epoch)
                    
                # exec. time 
                batches_time = time.time() - t0
                # display bar
                utils.bar(i,
                          len(train_generator),
                          pre_text=pre_text,
                          length=bar_len,
                          post_text=train_txt.format(running_loss, 
                                                     *tuple(running_metrics.values()), 
                                                     seconds_to_hmsms(batches_time),
                                                     now()))
        
            # update model params for last uncompleted period
            self.optimizer.step()
            self.optimizer.zero_grad(True)
            # update optimizer lr
            scheduler.step(self.optimizer, self.config, running_loss, epoch)
            
            # release memory
            torch.cuda.empty_cache()
            # training time
            train_time = time.time() - t0

            # insert loss and metrics into history
            self.set_history(epoch, loss=running_loss,**dict_values_from_items(running_metrics))
            
            # training end event
            callback.on_training_end(self.config, self.model, self.optimizer)
            
            # display finish epoch bar
            utils.bar(len(train_generator),
                      len(train_generator), 
                      pre_text=pre_text, 
                      length=bar_len,
                      post_text=train_txt.format(running_loss, 
                                                 *tuple(running_metrics.values()), 
                                                 seconds_to_hmsms(train_time),
                                                 now()),
                      end='' if val_generator is not None else '\n')
            
            # validation
            if val_generator is not None:
                # validation begin event
                callback.on_validation_begin(self.config, self.model, self.optimizer)
                
                val_loss, val_metrics = self.score(val_generator, quiet=True)
                # reformat val_metrics keys 
                val_metrics = {'val_'+k : v for k,v in val_metrics.items()}
                
                # release memory
                torch.cuda.empty_cache()
                
                train_and_val_time = time.time() - t0
                # insert loss and metrics into history
                self.set_history(epoch, val_loss=val_loss, **val_metrics)
                
                # validation end event
                callback.on_validation_end(self.config, self.model, self.optimizer)
                
                # display finish epoch barval_metrics
                utils.bar(len(train_generator),
                          len(train_generator), 
                          pre_text=pre_text, 
                          length=bar_len,
                          post_text=val_txt.format(running_loss, 
                                                   *tuple(running_metrics.values()), 
                                                   val_loss, 
                                                   *tuple(val_metrics.values()), 
                                                   seconds_to_hmsms(train_and_val_time),
                                                   now()),
                          end='\n')
            
            # update epoch
            self.epoch = epoch + 1
            
            # epoch end event
            callback.on_epoch_end(self.config, self.model, self.optimizer)
            
            if callback.check_stop():
                break
        
        # fit end event
        callback.on_fit_end(self.config, self.model, self.optimizer)
        
        utils.log_print('Training finished')
        
        return self.history.copy()
    
    def score(self, generator, quiet=False):
        """
        :param self: -
        :param generator: generator with data of type X, y
        :param quiet: flag, if false, progress is printed, else not 
        :return mean loss, mean metrics:
        """
        # set evaluation mode
        self.model = self.model.eval()
        # visual params
        bar_len = 20
        score_txt = 'loss={:.5f} '+keys_to_str_format(self.metrics.keys())+'time={} now={}'
        # display bar
        if not quiet:
            utils.bar(0,len(generator), length=bar_len, pre_text='Score', post_text='start')
        
        # init loss and metrics
        running_loss = 0.0
        running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)

        # score begin event
        if hasattr(generator, 'on_score_begin'):
            generator.on_score_begin()
        
        # time measurement
        t0 = time.time()
        with torch.no_grad():
            for i in range(len(generator)):
                # get data and labels
                # inputs, labels = generator[i]
                inputs, labels = generator.__getitem__(i,debug=True)
                # number of samples
                batch_size = inputs.shape[0]
                # get outputs
                outputs = self.model(inputs)
                # get sum of output losses 
                loss = self.criterion(outputs, labels)
                # calculate metrics
                metrics_dict = compute_metrics(self.metrics, outputs, labels)
                # compute mean loss and metrics
                running_loss = mean_loss(running_loss, loss.item(), i*generator.batch_size, 1.0*batch_size)
                running_metrics = mean_dict(running_metrics, metrics_dict, i*generator.batch_size, 1.0*batch_size)
                running_metrics = dict_values_from_items(running_metrics)
                
                # exec. time
                dt = time.time() - t0
                # display bar
                if not quiet:
                    utils.bar(i,
                              len(generator), 
                              pre_text='Score', 
                              length=bar_len,
                              post_text=score_txt.format(running_loss, 
                                                         *tuple(running_metrics.values()), 
                                                         seconds_to_hmsms(dt),
                                                         now()))
        
        # release memory
        torch.cuda.empty_cache()
        # exec. time
        dt = time.time() - t0
        # display finsh epoch bar
        if not quiet:
            utils.bar(len(generator), 
                        len(generator), 
                        pre_text='Score', 
                        length=bar_len,
                        post_text=score_txt.format(running_loss, 
                                                   *tuple(running_metrics.values()), 
                                                   seconds_to_hmsms(dt),
                                                   now()),
                        end='\n')
        
        # score end event
        if hasattr(generator, 'on_score_end'):
            generator.on_score_end()
            
        return running_loss, running_metrics
