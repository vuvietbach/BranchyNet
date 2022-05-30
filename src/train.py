import argparse
from cProfile import run
from re import T
import time

from yaml import parse
import yaml
from dataset import cifar_dataset, get_dataloader
from network import get_network
from supervisor import BranchyNet
import torch
import numpy as np
import random
import wandb

from torch.utils.data import DataLoader
def debug_dl():
    x = np.ones((2, 3, 32, 32))
    y = np.ones((2))
    x1 = np.ones((2, 3, 32, 32))
    y1 = np.ones((2))
    ds1 = cifar_dataset(x, y)
    ds2 = cifar_dataset(x1, y1)
    dl = {}
    dl['train'] = DataLoader(ds1, batch_size=2)
    dl['val'] = DataLoader(ds2, batch_size=2)
    return dl

def edit_config(cfg, arg):
    cfg['model']['train']['keep_metric']['name'] = arg.keep_metric
    cfg['model']['train']['keep_metric']['threshold'] = arg.threshold
    cfg['model']['train']['loss']['name'] = arg.loss

def main():
    
    start = time.time()
    with open('cfg/branchynet.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    parser = parse_arg()
    arg = parser.parse_args()
    edit_config(cfg, arg)

    run = wandb_init(cfg)
    #Prepare dataloader
    dl = get_dataloader()
    
    model_cfg = cfg.get('model')
    network = get_network()
    model = BranchyNet(network, model_cfg)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.set_device(DEVICE)
    # dl = debug_dl()
    res = model.train(dl, train_main=True)
    res = model.train(dl, train_main=False)
    for val_acc in res['val_acc']:
        run.log({'loss':val_acc, 'val_acc':val_acc})
    end = time.time()
    print(f"Time elapsed:{(end-start)/60} minutes")

def debug():
    dl = get_dataloader()
    for x, y in dl['train']:
        print(y.shape)
        break

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=5)
    parser.add_argument('--keep_metric', type=str, default='entropy')
    parser.add_argument('--loss', type=str, default='cross_entropy')
    return parser

def seed():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

def wandb_init(cfg):
    run = wandb.init(entity = 'aiotlab', 
    project = 'BranchyNet', 
    group = 'BranchyNet',
    config = cfg)
    return run

if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    seed()
    main()
    # train main
        

