import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def linreg_pytorch(a, b, X, Y):
    ## Definicija računskog grafa
    # optimizacijski postupak: gradijentni spust
    optimizer = optim.SGD([a, b], lr=0.01)
    N = X.shape[0]
    
    for epoch in range(100):
        # afin regresijski model
        Y_ = a*X + b

        diff = (Y-Y_)

        # kvadratni gubitak
        loss = torch.sum(diff**2) / N # neovisni o broju podataka

        # računanje gradijenata
        loss.backward()
        
        # korak optimizacije
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f'Epoch: {epoch}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')
            print(f'Epoch: {epoch}, gradients for a: {a.grad}, and b:{b.grad}')
            
        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()
    return a, b


def linreg_pytorch_gradients_by_hand(a, b, X, Y):
    ## Definicija računskog grafa
    # podaci i parametri, inicijalizacija parametara
    # optimizacijski postupak: gradijentni spust
    lr = 0.01
    N = X.shape[0]
    
    for epoch in range(100):
        # afin regresijski model
        Y_ = a*X + b

        diff = (Y-Y_)

        # kvadratni gubitak
        loss = torch.sum(diff**2) / N # neovisni o broju podataka
        
        # računanje gradijenata
        dY_ = - 2 / N * diff
        da = torch.sum(dY_ * X)
        db = torch.sum(dY_ * 1)
        
        # korak optimizacije
        with torch.no_grad():
            a -= lr * da
            b -= lr * db
        
        
        if epoch % 50 == 0:
            print(f'Epoch: {epoch}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')
            print(f'Epoch: {epoch}, gradients for a: {da}, and b:{db}')
    return a, b
