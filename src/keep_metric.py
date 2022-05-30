import torch
import torch.nn.functional as F
from registry import Registry
KEEP_METRIC = Registry('keep_metric')

@KEEP_METRIC.register_module('uncertainty')
def make_calculate_uncertainty():
    def calculate_uncertainty(logits):
        evidence = F.relu(logits)
        alpha = evidence + 1
        u = logits.shape[1] / torch.sum(alpha, dim=1, keepdim=True)
        return u.flatten()
    return calculate_uncertainty

@KEEP_METRIC.register_module('entropy')
def make_calculate_entropy():
    def calculate_entropy(logits):
        probs = F.softmax(logits, dim=1)
        tmp = - torch.log(probs)
        tmp = probs * tmp
        tmp = torch.sum(tmp, dim=1)
        return tmp
    return calculate_entropy