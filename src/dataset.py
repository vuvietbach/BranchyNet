from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pcifar10

class cifar_dataset(Dataset):
  def __init__(self, x, y):
    super(cifar_dataset, self).__init__()
    self.x = x
    self.y = y
  def __len__(self):
    return self.x.shape[0]
  def __getitem__(self, idx):
    y = torch.zeros((10), dtype = torch.float32)
    y[int(self.y[idx])] = 1
    return torch.FloatTensor(self.x[idx]), y

def get_dataloader():
  x_train, y_train, x_test, y_test = pcifar10.get_data()
  x = np.concatenate((x_train, x_test), axis=0)
  y = np.concatenate((y_train, y_test), axis=0)
  total_size = x.shape[0]
  train_size = int(total_size * 0.6)
  val_size = int(total_size * 0.2)
  test_size = total_size - train_size - val_size
  cur_ptr = 0
  dl = {}
  for split, size in zip(['train', 'val', 'test'], [train_size, val_size, test_size]):
    x_t = x[cur_ptr:cur_ptr+size]
    y_t = y[cur_ptr:cur_ptr+size]
    cur_ptr += size
    ds = cifar_dataset(x_t, y_t)
    dl[split] = DataLoader(ds, batch_size=16, drop_last=True)
  return dl