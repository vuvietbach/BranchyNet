import torch
import torch.nn.functional as F

def make_uncertainty_loss_function(global_step, annealing_step):
    def calculate_loss(out, y):
        evidence = F.relu(out)
        alpha = evidence + 1
        loss = torch.mean(mse_loss(y, alpha, global_step, annealing_step))
        return loss
    return calculate_loss

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

def calculate_accuracy(logits, y, mean):
    if y.shape[0] == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    truth = torch.argmax(y, dim=1)
    match = pred == truth
    match = torch.reshape(match.to(torch.float32), (-1, ))
    return torch.mean(match) if mean == True else torch.sum(match)

def calculate_uncertainty(logits):
    evidence = F.relu(logits)
    alpha = evidence + 1
    u = logits.shape[1] / torch.sum(alpha, dim=1, keepdim=True)
    return u.flatten()

def calculate_entropy(logits):
    probs = F.softmax(logits, dim=1)
    tmp = torch.log(probs)
    tmp = probs * tmp
    tmp = torch.sum(tmp, dim=1)
    return tmp

def calculate_mean(lst):
    num_item = len(lst)
    res = 0
    for item in lst:
        res += item
    res /= num_item
    if hasattr(res, 'item'):
        res = res.item()
    return res
