import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class PTLogreg(nn.Module):
    def __init__(self, D, C):
        super(PTLogreg, self).__init__()
        """Arguments:
           - D: dimensions of each datapoint 
           - C: number of classes
        """
    
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        self.W = nn.Parameter(torch.randn(D, C, dtype=torch.float32))
        self.b = nn.Parameter(torch.randn(C, dtype=torch.float32))
    def forward(self, X):
        """
        Arguments:
            - X: model inputs [NxD], type: torch.Tensor
        Output:
            - y_pred: model prediction [NxC]
        """
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        # ...
        #print("TORCH MAX", torch.max(X, 1))
        #Xmax_dim1, _ = torch.max(X, dim=1, keepdim=True)
        #X = X - Xmax_dim1 # prevencija overflowa
        return torch.softmax(torch.mm(X, self.W) + self.b, dim=1)

    def get_loss(self, y_pred, Yoh_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.exp, torch.sum
        #   pripaziti na numerički preljev i podljev
        # ...
        N = y_pred.shape[0]
        Yoh_indices = torch.argmax(Yoh_, dim=1)
        #print(y_pred, torch.log(y_pred[np.arange(0, N), Yoh_indices]))
        return -torch.mean(torch.log(y_pred[np.arange(0, N), Yoh_indices]))
        

def train(model, X, Yoh_, param_niter=1000, param_delta=0.5, param_lambda=0):
    """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Yoh_: ground truth [NxC], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
    """
    X = torch.from_numpy(X).float()
    Yoh_ = torch.from_numpy(Yoh_).float()

    # inicijalizacija optimizatora
    
    optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)
    # petlja učenja
    for epoch in range(int(param_niter)):
        # ispisujte gubitak tijekom učenja
        # forward prop
        y_pred = model.forward(X)
        # backward prop
        loss = model.get_loss(y_pred, Yoh_)
        loss.backward()
        # korak optimizacije
        optimizer.step()
        # postavljanje gradijenata na nula
        optimizer.zero_grad()
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}/{int(param_niter)}, Loss: {loss}')
        

def eval(model, X):
    """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    X = torch.Tensor(X)
    y_pred = model.forward(X)
    y_pred = torch.Tensor.detach(y_pred)
    return torch.Tensor.numpy(y_pred)

def pt_logreg_decfun(model, X):
    def classify(X):
        return np.argmax(eval(model, X), axis=1)
    return classify