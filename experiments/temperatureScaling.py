import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torch.optim as optim    

class myDataset(tdata.Dataset):
    def __init__(self, X, labels):
        self.labels = torch.tensor(labels)
        self.X = torch.tensor(X)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        Xs = self.X[idx,:]
        return Xs, label

### Code for this function taken from https://github.com/aangelopoulos/conformal_classification
### Uses temperature scaling to normalize the probabilities output by a neural network
def torch_ts(X,Y, max_iters=1000, lr=0.1, epsilon=0.0001):
    dat = myDataset(X,Y)

    calib_loader = tdata.DataLoader(dat, batch_size=128, shuffle=True, pin_memory=True)
    nll_criterion = nn.CrossEntropyLoss()

    T = nn.Parameter(torch.Tensor([1.3]))

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 
