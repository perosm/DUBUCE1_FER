import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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
            self.weights.append(nn.Parameter(torch.zeros((layer_dims[i-1], layer_dims[i]), dtype=torch.float32).to(device)))
            self.biases.append(nn.Parameter(torch.zeros((1, layer_dims[i]), dtype=torch.float32, device=device).to(device)))
        self.activation_funcs = activation_func

    def forward(self, X):
        X = X.to(device)
        """
        Arguments:
            - X: model inputs [NxD], type: torch.Tensor
        Output:
            - y_pred: model prediction [NxC]
        """

        for i in range(len(self.weights) - 1):
            X = torch.mm(X, self.weights[i]) + self.biases[i]
            X = self.activation_funcs(X)

        X = torch.mm(X, self.weights[len(self.weights)-1]) + self.biases[len(self.biases)-1]

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


def train(model, X, Yoh_, param_niter=1000, param_delta=0.5, param_lambda=0.01, every_n_epochs=100, which_optim='SGD'):
    """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Yoh_: ground truth [NxC], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
    """
    # inicijalizacija optimizatora
    model.to(device)
    X.to(device)
    Yoh_.to(device)

    optimizer = None
    if which_optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)
    else:
        optimizer = optim.Adam(model.parameters(), lr=param_delta, weight_decay=param_lambda)
    losses = []
    # petlja učenja
    for epoch in range(int(param_niter)):
        # ispisujte gubitak tijekom učenja
        # forward prop
        y_pred = model.forward(X)
        # backward prop
        loss = model.get_loss(y_pred, Yoh_)
        for w in model.weights:
          loss += param_lambda * torch.norm(w)
        loss.backward()
        # korak optimizacije
        optimizer.step()
        # postavljanje gradijenata na nula
        optimizer.zero_grad()
        if epoch % every_n_epochs == 0:
            print(f'Epoch: {epoch}/{int(param_niter)}, Loss: {loss}')

        losses.append(loss)

    return losses

def eval(model, X):
    """Arguments:
     - model: type: PTDeep
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    X = torch.Tensor(X).to(device)
    y_pred_tensor = model.forward(X)
    y_pred_numpy = y_pred_tensor.cpu().detach().numpy()
    return y_pred_numpy

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


def plot_mnist_last_layer_weights(weights):
    weights_mnist_last_layer = weights[-1].reshape(28, 28, 10)
    cmap = plt.get_cmap('gray')
    fig, axis = plt.subplots(2, 5)
    row, col = 0, 0
    for c in range(weights_mnist_last_layer.shape[2]):
        axis[row, col].imshow(weights_mnist_last_layer[:, :, c].cpu().detach().numpy())
        col += 1
        if col % 5 == 0:
            row += 1
            col = 0

def plot_loss_history(loss, epochs):
    plt.plot(np.arange(0, epochs + 1), loss.cpu().detach().to_numpy())


def pt_deep_decfun(model, X):
    def classify(X):
        return np.argmax(eval(model, X), axis=1)
    return classify

def model_performance(X, Y_, Yoh_, model, which):
    # evaluacija na podatcima
    y_pred = eval(model, X)
    # ispiši performansu (preciznost i odziv po razredima)
    acuraccy, precision, recall = data.eval_perf_multi(np.argmax(y_pred, axis=1), Y_)
    print(f'------------------------------------------------------------{which}------------------------------------------------------------')
    print(f'Acuraccy: {acuraccy}')
    print(f'Precision: {precision}')
    print(f'Recall: \n{recall}')

    return acuraccy, precision, recall


def early_stopping(x_train_split, y_train_split, x_val_split, y_val_split):
  y_train_split_oh = data.class_to_onehot(y_train_split)
  y_val_split_oh = data.class_to_onehot(y_val_split)

  model = PTDeep([784, 10], torch.relu)
  model.to(device)

  x_train_split = x_train_split.reshape(x_train_split.shape[0], 28 * 28)
  x_val_split = x_val_split.reshape(x_val_split.shape[0], 28 * 28)

  x_train_split.to(device)
  y_train_split.to(device)
  x_val_split.to(device)
  y_val_split.to(device)

  param_delta, param_lambda = 0.1, 1e-3
  optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)
  losses = []

  every_n_epochs = 500
  improvement_threshold = 20
  last_improvement_cnt = 0
  best_val_acuraccy = 0
  best_model = None
  param_niter = 5000
  epoch_stopped = 0

  # petlja učenja
  for epoch in range(int(param_niter)):
    # ispisujte gubitak tijekom učenja
    # forward prop
    y_train_pred = model.forward(x_train_split)
    #print(y_train_pred.dtype)
    # backward prop
    loss = model.get_loss(y_train_pred, torch.from_numpy(y_train_split_oh).to(device))
    for w in model.weights:
      loss += param_lambda * torch.norm(w)
    losses.append(loss)
    loss.backward()
    # korak optimizacije
    optimizer.step()
    # postavljanje gradijenata na nula
    optimizer.zero_grad()
    if epoch % every_n_epochs == 0:
        print(f'Epoch: {epoch}/{int(param_niter)}, Loss: {loss}')

    y_val_pred = eval(model, x_val_split) # u numpy-u
    val_acuraccy, _, _ = data.eval_perf_multi(np.argmax(y_val_pred, axis=1), y_val_split)

    if val_acuraccy > best_val_acuraccy:
      best_val_acuraccy = val_acuraccy
      best_model = model
      last_improvement_cnt = 0
    else:
      last_improvement_cnt += 1

    if last_improvement_cnt > improvement_threshold:
      epoch_stopped = epoch
      break

  return best_model, best_val_acuraccy, epoch_stopped


def train_mb(X, Yoh_, batch_size=1000, param_niter=500, param_delta=0.1, param_lambda=1e-3, every_n_epochs=50, which_optim='SGD'):
  model = PTDeep([784, 10], torch.relu).to(device)

  X = X.reshape(X.shape[0], 28 * 28)

  optimizer = None
  if which_optim == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)
  elif which_optim == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

  num_batches = X.shape[0] // batch_size
  losses = []


  for epoch in range(int(param_niter)):
    epoch_loss = 0
    # mijesanje indeksa
    indices = torch.randperm(X.shape[0])
    X_shuffled = X[indices]
    Yoh_shuffled = Yoh_[indices]

    for mb in range(0, X.shape[0], batch_size):
      # minibatchevi podataka
      X_minibatch = X_shuffled[mb:mb+batch_size].to(device)
      y_minibatch = Yoh_shuffled[mb:mb+batch_size].to(device)
      #X_minibatch.to(device)
      #y_minibatch.to(device)
      # forward prop
      y_pred_minibatch = model.forward(X_minibatch)
      # backward prop
      loss = model.get_loss(y_pred_minibatch, y_minibatch)

      for w in model.weights:
        loss += param_lambda * torch.norm(w)

      print("mb", mb, "epoch", epoch, "loss", loss)
      # korak optimizacije
      loss.backward()

      # postavljanje gradijenata na nulu
      optimizer.zero_grad()
      epoch_loss += loss

    epoch_loss = epoch_loss / num_batches
    if epoch % 10 == 0:
      print(f'Epoch: {epoch}/{int(param_niter)}, Loss: {epoch_loss}')
    losses.append(epoch_loss)

  return losses

def adam_and_lr_optimizer(x_train, y_train, x_test, y_test, param_delta, lr_scheduler_flag, param_niter):
  y_train_oh = data.class_to_onehot(y_train)
  y_test_oh = data.class_to_onehot(y_test)

  model = PTDeep([784, 10], torch.relu)
  model.to(device)

  x_train = x_train.reshape(x_train.shape[0], 28 * 28)
  x_test = x_test.reshape(x_test.shape[0], 28 * 28)

  x_train.to(device)
  y_train.to(device)
  x_test.to(device)
  y_test.to(device)

  param_lambda = 0 # 1e-3
  optimizer = optim.Adam(model.parameters(), lr=param_delta)
  scheduler = None

  if lr_scheduler_flag is True:
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1-1e-4)
  losses = []

  every_n_epochs = 100

  # petlja učenja
  for epoch in range(int(param_niter)):
    # ispisujte gubitak tijekom učenja
    # forward prop
    y_train_pred = model.forward(x_train)
    #print(y_train_pred.dtype)
    # backward prop
    loss = model.get_loss(y_train_pred, torch.from_numpy(y_train_oh).to(device))
    for w in model.weights:
      loss += param_lambda * torch.norm(w)
    losses.append(loss)
    loss.backward()
    # korak optimizacije
    optimizer.step()
    # postavljanje gradijenata na nula
    optimizer.zero_grad()
    if lr_scheduler_flag is True:
      scheduler.step()
    if epoch % every_n_epochs == 0:
        print(f'Epoch: {epoch}/{int(param_niter)}, Loss: {loss}')

  return losses


def train_linear_rbf_SVM_MNIST(x_train, y_train):
  X = x_train.reshape(x_train.shape[0], 28 * 28)

  linear_model =  SVC(C=1, kernel='linear', decision_function_shape='ovo').fit(X, y_train)
  rbf_model = SVC(C=1, kernel='rbf', decision_function_shape='ovo').fit(X, y_train)


  return linear_model, rbf_model

