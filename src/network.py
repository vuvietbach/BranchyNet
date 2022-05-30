import torch.nn as nn
import torch
import torch.optim as optim
class Branch(nn.Module):
    def __init__(self, branch, weight = 1.):
        super(Branch, self).__init__()
        self.branch = nn.ModuleList(branch)
        self.weight = weight
    def forward(self, x):
        for module in self.branch:
            x = module(x)
        return x

class Model(nn.Module):
    def __init__(self, body, head):
        super(Model, self).__init__()
        self.body = body
        self.head = head
    def forward(self, x):
        for module in self.body:
            x = module(x)

        return self.head(x)
        
class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_size, out_size, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_size)
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        # if out.shape[1] == 32:
        #     import pdb; pdb.set_trace()
        if x.shape != out.shape:
            n, c, row, col = x.shape
            pad_c = out.shape[1] - c
            p = torch.zeros((n, pad_c, row, col), dtype = torch.float32, device = x.device)
            x = torch.cat((x, p), dim = 1)
            if x.shape[2:] != out.shape[2:]:
                x = nn.functional.avg_pool2d(x, 1, 2)
        return nn.functional.relu(x + out)


def norm():
    return nn.Sequential(
        nn.ReLU(),
        nn.LocalResponseNorm(3, 5e-5, 0.75)
    )
def get_network():
    conv = lambda n: nn.Sequential(nn.Conv2d(n, 32, 3, stride = 1, padding = 1), nn.ReLU())
    network = [
        nn.Conv2d(3, 16, 3, 1, 0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
    ]
    '''
    FIRST BRANCH
    '''
    network += [Branch([
        nn.Conv2d(16, 64, 5, 1, 2),
        norm(),
        conv(64),
        conv(32),
        nn.Flatten(),
        nn.Linear(28800, 10)
    ])]

    num_res_block = 18
    for i in range(num_res_block):
        network += [ResBlock(16, 16)]
    '''
    SECOND
    '''
    network += [Branch([
        ResBlock(16, 16),
        nn.Flatten(),
        nn.Linear(14400, 10)
    ])]
    for i in range(num_res_block):
        network += [ResBlock(32 if i > 0 else 16, 32,
                                1 if i > 0 else 2)]
    for i in range(num_res_block):
        network += [ResBlock(64 if i > 0 else 32, 64,
                                1 if i > 0 else 2)]
    '''
    THIRD
    '''
    network += [
        nn.AvgPool2d(6, 1),
        Branch([nn.Flatten(), 
            nn.Linear(576, 10)])
    ]
    return network
