import os
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import skimage as ski
import skimage.io
from pathlib import Path
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out2'

#DATA_DIR = 'datasets/MNIST'
#SAVE_DIR_WEIGHTS = 'task_3_weights'

def load_preprocess_and_create_MNIST_dataloaders():
  # transforms to use
  transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (1,)) #, (0.3081,))
      ])

  # loading train and val data
  ds_train = MNIST(DATA_DIR, train=True,
                   download=True,
                   transform=transform)

  # loading test data
  ds_test = MNIST(DATA_DIR,
                  train=False,
                  transform=transform)

  # splitting into train and val
  ds_train, ds_val = torch.utils.data.random_split(ds_train, [55000, 5000])

  # data loaders for train, val and test
  train_loader_MNIST = DataLoader(ds_train, batch_size=50, shuffle=True)
  val_loader_MNIST = DataLoader(ds_val, batch_size=50, shuffle=True)
  test_loader_MNIST = DataLoader(ds_test, batch_size=50, shuffle=True)

  return train_loader_MNIST, val_loader_MNIST, test_loader_MNIST


class ConvolutionalModel(nn.Module):
  # Arhitektura
  # conv -> max pool -> relu -> conv -> max pool -> relu -> FC -> relu -> fc -> softmax
  def __init__(self, in_channels, out_channels, class_count):
    super().__init__()
    # sloj 1
    #in: batch_size x 1 x 28 x 28 (N=batch_size, C=1, H=28, W=28)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True)
    #out: batch_size x 16 x 28 x 28
    self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    #out: batch_size x 16 x 14 x 14
    self.relu1 = nn.ReLU()
    #out: batch_size x 16 x 14 x 14
    # sloj 2
    in_channels2, out_channels2 = 16, 32
    self.conv2 = nn.Conv2d(in_channels2, out_channels2, kernel_size=5, stride=1, padding=2, bias=True)
    #out: batch_size x 32 x 14 x 14
    self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    #out: batch_size x 32 x 7 x 7
    self.relu2 = nn.ReLU()
    # out: batch_size x 32 x 7 x 7
    # sloj 3
    fc1_width = 32 * 7 * 7
    self.fc3 = nn.Linear(fc1_width, 512, bias=True)
    self.relu3 = nn.ReLU()
    self.fc_logits = nn.Linear(512, class_count, bias=True)

    # parametri su već inicijalizirani pozivima Conv2d i Linear
    # ali možemo ih drugačije inicijalizirati
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    # sloj 1
    h = self.conv1(x)
    h = self.max_pool1(h)
    h = self.relu1(h)  # može i h.relu() ili nn.functional.relu(h)
    # sloj 2
    h = self.conv2(h)
    h = self.max_pool2(h)
    h = self.relu2(h)
    # sloj 3
    h = h.view(h.shape[0], -1)
    h = self.fc3(h)
    h = self.relu3(h)
    logits = self.fc_logits(h)
    return logits

def loss_function(logits, y):
  logits_shifted = logits - torch.max(logits, dim=1, keepdim=True)[0] + 1e-13
  first_part = torch.log(torch.sum(torch.exp(logits_shifted), dim=1))
  second_part = torch.sum((logits_shifted) * F.one_hot(y, 10), dim=1)
  
  return torch.mean(first_part - second_part)


def evaluate(name, model, val_loader, config):
  #model.eval()
  print(f'\nRunning evaluation: {name}')
  loss_avg = 0
  cnt_correct = 0

  for batch_idx, (x, y) in enumerate(val_loader):
    x = x.to(device)
    y = y.to(device)
    # forward pass
    logits = model.forward(x.to(device))
    y_pred = torch.argmax(logits, dim=1)
    cnt_correct += (y == y_pred).sum()
    # lce = torch.nn.CrossEntropyLoss()
    loss_val = loss_function(logits, y) # lce(logits, y)
    loss_avg += loss_val

  valid_acc = cnt_correct / (len(val_loader) * config['batch_size'])
  loss_avg /= len(val_loader)

  print(f'{name} acuraccy = {valid_acc}')
  print(f'{name} avg loss = {loss_avg}')

  return loss_avg, valid_acc

def train(model, train_loader, val_loader, config):
  print_every_n = config['print_every_n']
  lr_policy = config['lr_policy']
  batch_size = config['batch_size']
  max_epochs = config['max_epochs']
  save_dir = config['save_dir']
  weight_decay = config['weight_decay']
  optimizer = torch.optim.Adam(model.parameters(), lr=lr_policy[1], weight_decay=weight_decay) # za prvu epohu

  valid_losses, valid_accs = [], []
  train_losses, train_accs = [], []

  model.train()

  for epoch in range(1, max_epochs+1):
    epoch_loss = 0
    cnt_correct = 0  
    
    if epoch in lr_policy and epoch != 1:
      optimizer = torch.optim.Adam(model.parameters(), lr=lr_policy[epoch], weight_decay=weight_decay)
    
    for batch_idx, (x, y) in enumerate(train_loader):
      x = x.to(device)
      y = y.to(device)
      # zero grad
      optimizer.zero_grad()
      # forward pass
      logits = model.forward(x)
      y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
      cnt_correct += np.sum(y_pred == y.detach().cpu().numpy())
      # backward pass
      lce = torch.nn.CrossEntropyLoss()
      loss = lce(logits, y) # loss_function(logits, y)
      #print(f'kako kad je batch={batch_idx}, loss={loss.item()}')
      loss.backward()
      # update params
      optimizer.step()
      with torch.no_grad():
        epoch_loss += loss

      if batch_idx % 5 == 0:
        print(f'Epoch: {epoch}/{max_epochs}, Step: {batch_idx*batch_size}/{len(train_loader)*batch_size}, Batch loss: {loss}')
      
      if batch_idx > 0 and batch_idx % 100 == 0:
        draw_conv_filters(epoch, (batch_idx+1)*batch_size, model.conv1, 'task_3_weights')
      
      if batch_idx > 0 and batch_idx % 50 == 0:
        print(f'Train acuraccy = {cnt_correct / ((batch_idx+1)*batch_size) * 100}')

    if epoch % print_every_n == 0:
      print(f'Epoch: {epoch}/{max_epochs}, Loss: {epoch_loss / len(train_loader)}')
      print(f'Train acuraccy = {cnt_correct / (len(train_loader)*batch_size) * 100}')
    valid_loss, valid_acc = evaluate("Validation", model, val_loader, config)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)
    train_losses.append(epoch_loss / len(train_loader))
    train_accs.append(cnt_correct / ((batch_idx+1)*batch_size))
  
  directory = 'task_3_losses_accs'
  write_into_file(directory, 'valid_losses', valid_losses)
  write_into_file(directory, 'valid_accs', valid_accs)
  write_into_file(directory, 'train_losses', train_losses)
  write_into_file(directory, 'train_accs', train_accs)

def write_into_file(directory, filename, content):
  filepath = f'{directory}/{filename}.txt'
  with open(filepath, 'w') as f:
    for elem in content:
      f.write(str(elem.item()) + '\n')

def draw_conv_filters(epoch, step, layer, save_dir, C=1):  # input_channel is 1, C=1
  w = layer.weight.detach().cpu().numpy()
  num_filters = w.shape[0] # out_channels, in_channels/groups, kernel_size[0], kernel_size[1]
  k = int(w.shape[2])
  w = w.reshape(num_filters, C, k, k)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  #for i in range(C):
  for i in range(1):
    img = np.zeros([height, width])
    for j in range(num_filters):
      r = int(j / cols) * (k + border)
      c = int(j % cols) * (k + border)
      img[r:r+k,c:c+k] = w[j,i]
    filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % ('conv1', epoch, step, i)
    img = (img * 255).astype(np.uint8) # dodano
    ski.io.imsave(os.path.join(save_dir, filename), img)

def plot_conv_filters(title, w):
  w = w.detach().cpu().numpy()
  num_kernels, C, H, W = w.shape
  plt_cnt = 1
  print(title)
  for k in range(num_kernels):
    plt.subplot(2, num_kernels // 2, plt_cnt)
    plt.imshow((w[k, 0] * 255.0).astype(np.uint8))
    plt.axis('off')
    plt_cnt += 1

def read_file_return_as_list(directory, filename):
  filepath = f'{directory}/{filename}.txt'
  f = open(filepath, 'r')
  return [float(line.strip()) for line in f.readlines()]

def plot_data(data, label):
  plt.plot(np.arange(0, len(data)), data, label=label)
