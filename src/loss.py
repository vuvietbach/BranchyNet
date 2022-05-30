import torch
import torch.nn.functional as F
from registry import Registry
LOSS = Registry('Loss')

@LOSS.register_module('uncertainty')
def make_uncertainty_loss_function(global_step=1, annealing_step=1):
    def calculate_loss(out, y):
        evidence = F.relu(out)
        alpha = evidence + 1
        loss = torch.mean(mse_loss(y, alpha, global_step, annealing_step))
        return loss
    return calculate_loss

@LOSS.register_module('cross_entropy')
def make_crossentropy_loss_function():
    def calculate_loss(out, y):
        out = F.softmax(out, dim=1)
        loss = F.binary_cross_entropy(out, y)
        return loss
    return calculate_loss

def KL(alpha):
    beta = torch.ones((1, alpha.shape[1]), device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    kl = torch.sum((alpha - beta)*(dg1-dg0), dim=1,keepdim=True) + lnB + lnB_uni
    return kl

def mse_loss(p, alpha, global_step, annealing_step):
    '''
    TODO: analyzing global step, annealing step
    ''' 
    S = torch.sum(alpha, dim=1, keepdim=True) 
    E = alpha - 1
    m = alpha / S
    
    A = torch.sum((p-m)**2, dim=1, keepdim=True) 
    B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), dim=1, keepdim=True) 
    # this should be differnet
    annealing_coef = 1.0
    
    alp = E*(1-p) + 1 
    C =  annealing_coef * KL(alp)
    return (A + B) + C
