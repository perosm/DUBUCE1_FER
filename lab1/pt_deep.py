import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PTDeep(nn.Module):
    def __init__(self, layer_dims, activation_func):
        """Arguments:
           - layer_dims: num of neurons per layer 
           - activation_func: type of activation func
        """
        super(PTDeep, self).__init__()
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        self.weights = nn.ParameterList([])
        self.biases = nn.ParameterList([])
        for i in range(1, len(layer_dims)):
            self.weights.append(nn.Parameter(torch.randn(layer_dims[i-1], layer_dims[i], dtype=torch.float32)))
            self.biases.append(nn.Parameter(torch.randn(layer_dims[i], dtype=torch.float32)))
                
        self.activation_funcs = activation_func
        
    def forward(self, X):
        """
        Arguments:
            - X: model inputs [NxD], type: torch.Tensor
        Output:
            - y_pred: model prediction [NxC]
        """
        for i in range(len(self.weights) - 1):
            X = torch.mm(X, self.weights[i]) + self.biases[i]
            X = self.activation_funcs(X)
        
        X = torch.mm(X, self.weights[len(self.weights)-1]) + self.biases[len(self.weights)-1]
        
        return torch.softmax(X, dim=1)

    def get_loss(self, y_pred, Yoh_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.exp, torch.sum
        #   pripaziti na numerički preljev i podljev
        # ...
        N = y_pred.shape[0]
        Yoh_indices = torch.argmax(Yoh_, dim=1)
        #print(y_pred, torch.log(y_pred[np.arange(0, N), Yoh_indices]))
        return -torch.mean(torch.log(y_pred[np.arange(0, N), Yoh_indices]))
    
    
def train(model, X, Yoh_, param_niter=1000, param_delta=0.5, param_lambda=0, every_n_epochs=100):
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
        if epoch % every_n_epochs == 0:
            print(f'Epoch: {epoch}/{int(param_niter)}, Loss: {loss}')
            
def eval(model, X):
    """Arguments:
     - model: type: PTDeep
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

def count_params(model):
    cnt = 0
    total_params = 0
    print("Model", model.__class__.__name__)
    for name, param in model.named_parameters():
        print(name)
        print(param)
        if cnt % 2 == 0:
            total_params += (param.shape[0] * param.shape[1])
        else:
            total_params += param.shape[0]
        cnt += 1
    print("Total parameters:", total_params)

def pt_deep_decfun(model, X):
    def classify(X):
        return np.argmax(eval(model, X), axis=1)
    return classify
