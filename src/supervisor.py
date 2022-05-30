from bdb import set_trace
from cmath import log
from sqlite3 import Timestamp
from statistics import mean
from network import *
import copy
from dataset import cifar_dataset
from log import Logger
from utils import calculate_accuracy, \
    make_uncertainty_loss_function, \
    make_crossentropy_loss_function, \
    calculate_uncertainty, calculate_mean, \
    calculate_entropy
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import pcifar10
import numpy as np
import time
import os
from loss import LOSS
from keep_metric import KEEP_METRIC

class BranchyNet(nn.Module):
    def __init__(self, network, cfg):
        super(BranchyNet, self).__init__()
        module_list = nn.ModuleList()
        # MAKE_MODEL
        self.models = []
        for module in network:
            if isinstance(module, Branch):
                model = Model(copy.deepcopy(module_list), module)
                self.models.append(model)
            else:
                module_list.extend([module])
        self.main_model = Model(module_list, copy.deepcopy(network[-1]))
        self.update_params()
        
        log_config = cfg.get('log')
        log_file = log_config.get('log_file', None)
        if log_file is None:
            log_file = time.strftime("%Y%m%d_%H%M%S", time.localtime())+'.log'
        log_file = os.path.join('workdir', log_file)
        self.LOGGER = Logger(log_file)

        lr = cfg.get('lr', 0.001)
        weigth_decay = cfg.get('decay', 1e-5)
        self.main_optimizer = optim.Adam(self.main_model.parameters(), lr=lr, weight_decay=weigth_decay)
        self.optimizers = [optim.Adam(model.parameters(), lr=lr, weight_decay=weigth_decay) for model in self.models]
     
        # loss = cfg.get('loss_function', None)
        # assert loss is not None, "please specify loss function name"
        # loss_name = loss.get('name')
        # loss_args = loss.get('args')
        # self.loss_function = LOSS_REGISTRY.create(loss_name, **loss_args)
        train_cfg = cfg.get('train')

        loss_cfg = train_cfg.get('loss')
        loss_name = loss_cfg.get('name')
        loss_args = loss_cfg.get('args', {})
        self.loss_function = LOSS.build(loss_name, **loss_args)
        self.LOSS_TYPE = loss_name

        self.calculate_accurary = calculate_accuracy     
        
        # self.calculate_keep_metric = calculate_uncertainty if train_cfg.get('uncertainty', True) \
        #     else calculate_entropy
        kmetric_cfg = train_cfg.get('keep_metric')
        name = kmetric_cfg.get('name')
        args = kmetric_cfg.get('args', {})
        self.calculate_keep_metric = KEEP_METRIC.build(name, **args)
        self.THRESHOLD = kmetric_cfg.get('threshold')
        self.KEEP_METRIC = name

        self.FORWARD_MAIN = train_cfg.get('forward_main', True)
        self.MODEL_WEIGHT = train_cfg.get('model_weight', [1, 0.7, 0.4])
        self.NUM_EPOCH = train_cfg.get('num_epoch')
        self.VAL_PER_EPOCH = train_cfg.get('val_frequency')
        assert self.NUM_EPOCH >= self.VAL_PER_EPOCH, 'number of train epoch must be larger than validation frequency'

       

    def set_device(self, device):
        self.DEVICE = device
        self.main_model.to(device)
        [model.to(device) for model in self.models]
    
    def train_main(self, x, y):
        '''
        return total loss, mean acc
        '''
        self.main_model.train()
        self.main_optimizer.zero_grad()
        out = self.main_model(x)
        loss = self.loss_function(out, y)
        accuracy = self.calculate_accurary(out, y, mean=True)
        loss.backward()
        self.main_optimizer.step()
        return {'loss':loss.item(), 'acc':accuracy.item()}
    
    def add_grad_to_main(self, model):
        # import pdb; pdb.set_trace()
        length = len(model.body)
        target = self.main_model.body[:length]
        for m1, m2 in zip(model.body.parameters(), target.parameters()):
            if m1.grad is not None:
                m2.grad += m1.grad
        
    def update_params(self):
        for model in self.models[:-1]:
            for i, module in enumerate(model.body):
                module.load_state_dict(self.main_model.body[i].state_dict())
        self.models[-1].load_state_dict(self.main_model.state_dict())
    
    def train_all(self, x, y):
        '''
        Return
        - total loss 
        - mean accuracy
        Describe:
        Model still incur loss for all remaining x
        But accuracy calculate for only instance that model is sure of
        '''
        # ZERO GRADS
        self.main_optimizer.zero_grad()
        [optimizer.zero_grad() for optimizer in self.optimizers]
        
        # FORWARD MAIN and Model
        if self.FORWARD_MAIN:
            out = self.main_model(x)
            main_loss = self.loss_function(out, y)
            main_loss.backward()
        
        remaining_x = x
        remaining_y = y
        num_model = len(self.models)
        total_loss = 0
        total_accuracy = 0
        for i, model in enumerate(self.models):
            # import pdb; pdb.set_trace()
            out = model(remaining_x)
            
            loss = self.loss_function(out, remaining_y)
            total_loss += loss * self.MODEL_WEIGHT[i]
            
            if i == num_model - 1:
                accuracy = self.calculate_accurary(out, remaining_y, mean=False)
                total_accuracy += accuracy
            
            keep_metrics = self.calculate_keep_metric(out)
            keep_idx = torch.zeros((keep_metrics.shape[0]), dtype = torch.bool)
            keep_idx[keep_metrics < self.THRESHOLD] = True
            # calculate accuracy for input that model confident of
            keep_out = out[keep_idx]
            keep_y = remaining_y[keep_idx]
            accuracy = self.calculate_accurary(keep_out, keep_y, mean = False)
            total_accuracy += accuracy
            num_remain = keep_metrics.shape[0] - torch.sum(keep_idx)
            if num_remain > 0:
                remaining_x = remaining_x[~keep_idx]
                remaining_y = remaining_y[~keep_idx]
            else:
                break
        
        # calculate gradient
        total_loss.backward()
        
        # send gradients to main model
        models = self.models[:-1] if self.FORWARD_MAIN else self.models
        [self.add_grad_to_main(model) for model in models]
        
        # add gradient to parameter
        # question why need to update small model parameters
        # answer because small model have their own branch
        self.main_optimizer.step()
        [optimizer.step() for optimizer in self.optimizers]
        self.update_params()
        
        mean_acc = total_accuracy / x.shape[0]
        return {'loss': total_loss.item(), 'acc':mean_acc.item()}
    
    def val_main(self, x, y):
        '''
        return true accuracy
        '''
        self.main_model.eval()
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            out = self.main_model(x)
            loss = self.loss_function(out, y)
            accuracy = self.calculate_accurary(out, y, mean=True)
            
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            return {'loss':loss.item(), 'acc':accuracy.item(), 'elapsed_time':elapsed_time}
    
    def val_all(self, x, y):
        '''
        return mean accuracy
        '''
        
        remaining_x = x
        remaining_y = y
        
        num_model = len(self.models)
        # TODO: add entropy metrics

        
        total_loss = 0
        total_acc = 0
        [model.eval() for model in self.models]


        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for i, model in enumerate(self.models):
                out = model(remaining_x)
                loss = self.loss_function(out, remaining_y)
                total_loss += loss
                if i == num_model - 1:
                    acc = self.calculate_accurary(out, remaining_y, mean = False)
                    total_acc += acc
                    break
                
                # calculate index that this branch is confident of
                keep_metrics = self.calculate_keep_metric(out)
                keep_idx = torch.zeros((keep_metrics.shape[0]), dtype = torch.bool)
                keep_idx[keep_metrics < self.THRESHOLD] = True
                # 
                keep_out = out[keep_idx]
                keep_y = remaining_y[keep_idx]
                accuracy = self.calculate_accurary(keep_out, keep_y, mean = False)
                total_acc += accuracy
                # omit confident instancen
                num_remain = keep_metrics.shape[0] - torch.sum(keep_idx)
                if num_remain > 0:
                    remaining_x = remaining_x[~keep_idx]
                    remaining_y = remaining_y[~keep_idx]
                else:
                    break
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
        
        mean_acc = total_acc/x.shape[0]
        res = {'loss':total_loss.item(),
        'acc':mean_acc.item(),
        'elapsed_time': elapsed_time}
        return res
    
    def val(self, loader, train_main):
        val_func = self.val_main if train_main else self.val_all
        last_val_losses = []
        last_val_accuracies = []
        elapsed_times = []
        
        for x, y in loader:
            x = x.to(self.DEVICE)
            y = y.to(self.DEVICE)
            res = val_func(x, y)
            last_val_losses.append(res['loss'])
            last_val_accuracies.append(res['acc'])
            elapsed_times.append(res['elapsed_time'])
        return {'loss':last_val_accuracies, 'acc':last_val_accuracies, 'elapsed_time':elapsed_times}
    
    def log(self, log):
        new_log = {}
        for k, v in log.items():
            if isinstance(v, list):
                v = calculate_mean(v)
                new_log[k] = v
            else:
                new_log[k] = v
        # import pdb; pdb.set_trace()
        self.LOGGER(new_log)


    def train(self, loader, train_main):
        '''
        return loss, accuracy np array
        assume every batch same size
        '''
        self.log({'LOSS_TYPE':self.LOSS_TYPE})
        self.log({'TRAIN_MODE':('MAIN' if train_main else 'ALL')})
        train_func = self.train_main if train_main else self.train_all
        for epoch in range(self.NUM_EPOCH):
            last_train_losses = []
            last_train_accuracies = []
            log_info = {'epoch':epoch}
            for x, y in loader['train']:
                x = x.to(self.DEVICE)
                y = y.to(self.DEVICE)
                res = train_func(x, y)
                last_train_losses.append(res['loss'])
                last_train_accuracies.append(res['acc'])
            
            log_info['train_loss'] = last_train_losses
            log_info['train_accuracy'] = last_train_accuracies   
                      
            if (epoch + 1) % self.VAL_PER_EPOCH == 0:
                res = self.val(loader['val'], train_main)
                last_val_losses = res['loss']
                last_val_accuracies = res['acc']
                elapsed_times = res['elapsed_time']
                log_info['val_loss'] = res['loss']
                log_info['val_accuracy'] = res['acc']
            
            self.log(log_info)
        
        mean_batch_val_time = torch.mean(torch.FloatTensor(elapsed_times)).item()
        self.log({'Mean evaluation time':mean_batch_val_time})

        res = {'train_loss':last_train_losses,
        'train_acc':last_train_accuracies,
        'val_loss':last_val_losses,
        'val_acc':last_val_accuracies,
        'mean_val_time': mean_batch_val_time}
        
        return res
    
    def test(self, loader):
        return 
            

if __name__ == '__main__':
    x = get_network()
    cfg = {}
    model = BranchyNet(x, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.set_device(device)
    
    x_train, y_train, x_test, y_test = pcifar10.get_data()
    train_size = int(x_train.shape[0] * 0.7)
    x_train, y_train, x_val, y_val = x_train[:train_size], y_train[:train_size], x_train[train_size:], y_train[train_size:]
    lst = [(x_train, y_train), (x_val, y_val), (x_test, y_test)]
    ds = [cifar_dataset(x, y) for x, y in lst]
    from torch.utils.data import DataLoader
    # dl = {}
    # dl['train'] = DataLoader(ds[0], batch_size=32, shuffle=True, drop_last=True)
    # dl['val'] = DataLoader(ds[1], batch_size=32, shuffle=False, drop_last=True)
    # dl['test'] = DataLoader(ds[2], batch_size=32, shuffle= False, drop_last = True)
    x, y = ds[0][0]
    print(x.shape)
    print(y.shape)

    res = model.train(dl, 1, 1, train_main=True)
    model.forward_main = False
    res = model.train(dl, 1, 1, train_main=False)
    with open('result.txt', 'w') as f:
        pass
    for k, v in res.items():
        write_file(k, calculate_mean(v))


    # model.train_main(x, y)
    # train_loader = DataLoader(data_train, batch_size=32, shuffle=True, droplast=True)

    # x = nn.Conv2d(1, 1, 1, 1)
    # y = nn.Conv2d(1, 1, 1, 1)
    # z = nn.ModuleList((x, y))
    # print(len(z))
    # for name, layer in y.named_modules():
    #     print(name)

