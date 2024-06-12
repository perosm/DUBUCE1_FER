import os
import pickle
import numpy as np
import skimage as ski
import skimage.io
import math
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import skimage as ski
import skimage.io
from torch.utils.data import TensorDataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConvolutionalModelCIFAR10(nn.Module):
  # conv(16,5) -> relu() -> pool(3,2) ->
  # conv(32,5) -> relu() -> pool(3,2) ->
  # fc(256) -> relu() ->
  # fc(128) -> relu() ->
  # fc(10)
  def __init__(self, in_channels1=3, out_channels1=16, kernel_size1=5, class_count=10):
    super().__init__()
    # sloj 1
    # in: batch_size x 3 x 32 x 32 (N=batch_size, C=3, H=32, W=32)
    self.conv1 = nn.Conv2d(in_channels1, out_channels1, kernel_size=kernel_size1, stride=1, padding=2, bias=True)
    # out: batch_size x 16 x 32 x 32
    self.relu1 = nn.ReLU()
    # out: batch_size x 16 x 32 x 32
    self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    # out: batch_size x 16 x 15 x 15

    # sloj 2
    out_channels2 = 32
    self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=kernel_size1, stride=1, padding=2, bias=True)
    # out: batch_size x 32 x 15 x 15
    self.relu2 = nn.ReLU()
    # out: batch_size x 32 x 15 x 15
    self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    # out: batch_size x 32 x 7 x 7

    # sloj 3
    linear_in3 = 32 * 7 * 7
    self.fc3 = nn.Linear(in_features=linear_in3, out_features=256, bias=True)
    self.relu3 = nn.ReLU()

    # sloj 4
    self.fc4 = nn.Linear(in_features=256, out_features=128, bias=True)
    self.relu4 = nn.ReLU()

    # sloj 5
    self.fc_logits = nn.Linear(in_features=128, out_features=class_count, bias=True)

    # parametri su već inicijalizirani pozivima Conv2d i Linear
    # ali možemo ih drugačije inicijalizirati


    self.reset_parameters() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.xavier_normal_(m.weight)
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
    h = h.contiguous().view(h.shape[0], -1) # https://stackoverflow.com/questions/66750391/runtimeerror-view-size-is-not-compatible-with-input-tensors-size-and-stride-a
    h = self.fc3(h)
    h = self.relu3(h)

    # sloj 4
    h = self.fc4(h)
    h = self.relu4(h)

    # sloj 5
    logits = self.fc_logits(h)
    return logits


  def get_loss(self, logits, y, loss_fn_name):
    if loss_fn_name == 'lce':
      return F.cross_entropy(logits, y)
    else:
      return multiclass_hinge_loss(logits, y)


def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y


def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict


def preprocess_data():
  DATA_DIR = 'datasets/CIFAR10/cifar-10-batches-py/'

  img_height = 32
  img_width = 32
  num_channels = 3
  num_classes = 10

  train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
  train_y = []
  for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
  train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
  train_y = np.array(train_y, dtype=np.int32)

  subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
  test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
  test_y = np.array(subset['labels'], dtype=np.int32)

  valid_size = 5000
  train_x, train_y = shuffle_data(train_x, train_y)
  valid_x = train_x[:valid_size, ...]
  valid_y = train_y[:valid_size, ...]
  train_x = train_x[valid_size:, ...]
  train_y = train_y[valid_size:, ...]
  data_mean = train_x.mean((0, 1, 2))
  data_std = train_x.std((0, 1, 2))

  train_x = (train_x - data_mean) / data_std
  valid_x = (valid_x - data_mean) / data_std
  test_x = (test_x - data_mean) / data_std

  train_x = train_x.transpose(0, 3, 1, 2)
  valid_x = valid_x.transpose(0, 3, 1, 2)
  test_x = test_x.transpose(0, 3, 1, 2)

  return train_x, train_y, valid_x, valid_y, test_x, test_y


def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2, 3, 1, 0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  img = (img * 255).astype(np.uint8)
  ski.io.imsave(os.path.join(save_dir, filename), img)


def plot_conv_filters(title, weights):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2, 3, 1, 0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  print(title)
  plt.imshow((img * 255.0).astype(np.uint8))
  plt.axis('off')


def plot_training_progress(save_dir, data, loss_fn_name):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss' if loss_fn_name == 'lce' else 'Multiclass hinge loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)


def class_to_onehot(y):
  Yoh = np.zeros((y.shape[0],max(y)+1))
  Yoh[np.arange(len(y)),y] = 1
  return Yoh


def eval_perf_multi(Y_, Y):
  pr = []
  n = max(Y_)+1
  M = np.bincount(n * Y_ + Y, minlength=n*n).reshape(n, n)
  for i in range(n):
    tp_i = M[i,i]
    fn_i = np.sum(M[i,:]) - tp_i
    fp_i = np.sum(M[:,i]) - tp_i
    tn_i = np.sum(M) - fp_i - fn_i - tp_i
    recall_i = tp_i / (tp_i + fn_i)
    precision_i = tp_i / (tp_i + fp_i)
    pr.append( (recall_i, precision_i) )

  accuracy = np.trace(M)/np.sum(M)
  return accuracy, pr, M


def evaluate(model, x, y, loss_fn_name='lce'):
  with torch.no_grad():
    # loss
    logits = model.forward(x)
    loss = model.get_loss(logits, y, loss_fn_name)
    # točnost, preciznost, matricu zabune
    y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
    acuraccy, precision, M = eval_perf_multi(y.detach().cpu(), y_pred)

  return loss, acuraccy


def train(model, train_x, train_labels, valid_x, valid_labels, num_epochs, n_batch, bsz, optimizer, lr_scheduler, SAVE_DIR='task_4_plot_data', loss_fn_name='lce'):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  plot_data = {}
  plot_data['train_loss'] = []
  plot_data['valid_loss'] = []
  plot_data['train_acc'] = []
  plot_data['valid_acc'] = []
  plot_data['lr'] = []

  train_x_torch = torch.tensor(train_x).to(device)
  train_y_torch = torch.tensor(train_labels, dtype=torch.long).to(device)
  valid_x_torch = torch.tensor(valid_x).to(device)
  valid_y_torch = torch.tensor(valid_labels, dtype=torch.long).to(device)


  train_ds = TensorDataset(train_x_torch, train_y_torch)
  dataloader = DataLoader(train_ds, batch_size=bsz, shuffle=True)

  for epoch in range(num_epochs):
    for batch, (batch_x, batch_y) in enumerate(dataloader):
      model.train()
      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)

      logits = model.forward(batch_x)
      loss = model.get_loss(logits, batch_y, loss_fn_name)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if batch%100 == 0:
        print("epoch: {}, step: {}/{}, batch_loss: {}".format(epoch, batch, n_batch, loss))

      if batch%200 == 0:
        draw_conv_filters(epoch, batch, model.conv1.weight.detach().cpu().numpy(), SAVE_DIR)

    train_loss, train_acc = evaluate(model, train_x_torch, train_y_torch, loss_fn_name)
    val_loss, val_acc = evaluate(model, valid_x_torch, valid_y_torch, loss_fn_name)
    print(f'Epoch: {epoch}/{num_epochs}')
    print(f'Train loss: {train_loss}')
    print(f'Train acc: {train_acc}')
    print(f'Val loss: {val_loss}')
    print(f'Val acc: {val_acc}')

    plot_data['train_loss'] += [train_loss.detach().cpu().numpy()]
    plot_data['valid_loss'] += [val_loss.detach().cpu().numpy()]
    plot_data['train_acc'] += [train_acc]
    plot_data['valid_acc'] += [val_acc]
    plot_data['lr'] += [lr_scheduler.get_last_lr()]
    lr_scheduler.step()

  plot_training_progress(SAVE_DIR, plot_data, loss_fn_name)

  return plot_data


def overfit_NN(model, train_x, train_labels, num_epochs, n_batch, bsz, optimizer, lr_scheduler, SAVE_DIR='task_4_plot_data', loss_fn_name='lce'):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  train_x_torch = torch.tensor(train_x).to(device)
  train_y_torch = torch.tensor(train_labels, dtype=torch.long).to(device)

  train_ds = TensorDataset(train_x_torch, train_y_torch)
  dataloader = DataLoader(train_ds, batch_size=bsz, shuffle=True)

  for epoch in range(num_epochs):
    for batch, (batch_x, batch_y) in enumerate(dataloader):
      model.train()
      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)

      logits = model.forward(batch_x)
      loss = model.get_loss(logits, batch_y, loss_fn_name)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    train_loss, train_acc = evaluate(model, train_x_torch, train_y_torch, loss_fn_name)
    if epoch % 20 == 0:
      print(f'Epoch: {epoch}/{num_epochs}')
      print(f'Train loss: {train_loss}')
      print(f'Train acc: {train_acc}')


def draw_image(img, mean, std):
  img = img.transpose(1, 2, 0)
  img *= std
  img += mean
  #img = img.astype(np.uint8)
  ski.io.imshow(img)
  ski.io.show()


def show_k_missclassified_pictures_with_biggest_loss(model, x, y, mean, std, k=20):
  x = torch.tensor(x).to(device)
  y = torch.tensor(y, dtype=torch.long).to(device)
  with torch.no_grad():
    logits = model.forward(x)
    #mean = torch.mean(x, dim=(0, 2, 3))
    #std = torch.std(x, dim=(0, 2, 3))
  loss_per_example = F.cross_entropy(logits, y, reduction='none')
  top_20 = torch.topk(loss_per_example, k) # vraca k najvecih losseva i njihovih indeksa
  top_20_biggest_loss_x = x[top_20[1]]
  top_20_biggest_loss_correct_class = y[top_20[1]]
  top_20_biggest_loss_predicted_class_3, _ = torch.topk(F.softmax(logits[top_20[1]], dim=1), k=3, dim=1)

  for i in range(k):
    print(f'Correct class: {top_20_biggest_loss_correct_class[i]}')
    print(f'Top 3 largest predicted probabilities: {top_20_biggest_loss_predicted_class_3[i]}')
    draw_image(top_20_biggest_loss_x[i].detach().cpu().numpy(), mean.cpu().numpy(), std.cpu().numpy())



def compare_loss_functions_graphs(data1, data2, loss_fn_name1, loss_fn_name2):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color1 = 'm'
  val_color1 = 'c'
  train_color2 = 'r'
  val_color2 = 'b'

  num_points = len(data1['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss vs Multiclass hinge loss')
  ax1.plot(x_data, data1['train_loss'], marker='o', color=train_color1,
           linewidth=linewidth, linestyle='-', label='train lce')
  ax1.plot(x_data, data2['train_loss'], marker='o', color=train_color2,
           linewidth=linewidth, linestyle='-', label='train mchl')
  
  ax1.plot(x_data, data1['valid_loss'], marker='o', color=val_color1,
           linewidth=linewidth, linestyle='-', label='validation lce')
  ax1.plot(x_data, data2['valid_loss'], marker='o', color=val_color2,
           linewidth=linewidth, linestyle='-', label='validation mchl')

  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data1['train_acc'], marker='o', color=train_color1,
           linewidth=linewidth, linestyle='-', label='train lce')
  ax2.plot(x_data, data2['train_acc'], marker='o', color=train_color2,
           linewidth=linewidth, linestyle='-', label='train mchl')
  
  ax2.plot(x_data, data1['valid_acc'], marker='o', color=val_color1,
           linewidth=linewidth, linestyle='-', label='validation lce')
  ax2.plot(x_data, data2['valid_acc'], marker='o', color=val_color2,
           linewidth=linewidth, linestyle='-', label='validation mchl')
  
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data1['lr'], marker='o', color=train_color1,
           linewidth=linewidth, linestyle='-', label='learning_rate lce')
  ax3.plot(x_data, data2['lr'], marker='o', color=train_color2,
           linewidth=linewidth, linestyle='-', label='learning_rate mchl')
  ax3.legend(loc='upper left', fontsize=legend_size)



def multiclass_hinge_loss(logits: torch.Tensor, target: torch.Tensor, delta=1.):
    '''
        Args:
            logits: torch.Tensor with shape (B, C), where B is batch size, and C is number of classes.
            target: torch.LongTensor with shape (B, ) representing ground truth labels.
            delta: Hyperparameter.
        Returns:
            Loss as scalar torch.Tensor.
    '''
    target_oh = F.one_hot(target, 10)
    mask = target_oh > 0
    logits_correct_class = torch.masked_select(logits, mask)[:, None]
    zeros = torch.zeros(logits.shape).to(device)
    return torch.mean(torch.sum(torch.max(zeros, logits - logits_correct_class + torch.tensor(delta).to(device)[None, None]), dim=1))


