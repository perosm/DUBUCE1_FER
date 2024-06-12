import torch
import torch.nn as nn
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import *


TRAIN_PATH = 'sst_train_raw.csv'
VAL_PATH = 'sst_valid_raw.csv'
TEST_PATH = 'sst_test_raw.csv'

@dataclass
class Instance:
  df = None

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    return self.df.iloc[index, 0].split(), self.df.iloc[index, 1]


class Vocab:
  """
  vokabular se izgrađuje samo na train skupu podataka
  Jednom izgrađeni vokabular na train skupu postavljate kao 
  vokabular testnog i validacijskog skupa podataka
  smatra najkorektniji u analizi teksta jer kroz izgradnju vokabulara na 
  testnom i validacijskom skupu imamo curenje informacija u treniranje modela
  """
  itos = {} # index-to-string
  stoi = {} # string-to-index

  def __init__(self, frequencies, max_size, min_freq, text_vocab=True):
    if text_vocab:
      self.itos = {0: "<PAD>", 1: "<UNK>"}
      self.stoi = {"<PAD>": 0, "<UNK>": 1}

      cnt = len(self.stoi)
      for key, value in sorted(frequencies.items(), key=lambda item: item[1], reverse=True):
        if max_size == -1:
          if value > min_freq:
            self.stoi[key] = cnt
            self.itos[cnt] = key
        elif len(self.stoi) > max_size:
          break
        cnt += 1
    else:
      self.itos = {0: "positive", 1: "negative"}
      self.stoi = {"positive": 0, "negative": 1}

  def encode(self, sequence):
    encoded_sequence = []

    if isinstance(sequence, str):
      return torch.tensor(self.stoi[sequence.strip()])

    for word in sequence:
      token_for_word = self.stoi.get(word)
      if token_for_word is None:
        encoded_sequence.append(self.stoi.get('<UNK>')) #JEL OVO OK?
      else:
        encoded_sequence.append(token_for_word)
    
    return torch.tensor(encoded_sequence)


class NLPDataset(Dataset):
  
  instances = Instance()
  text_vocab = None
  label_vocab = None
  def __init__(self, csv_file):
    self.instances.df = pd.read_csv(csv_file, header=None)

  def __len__(self):
    return len(self.instances)

  def __getitem__(self, index):
    instance_text, instance_label = self.instances[index]
    return self.text_vocab.encode(instance_text), self.label_vocab.encode(instance_label)



def count_frequency(path):
  df = pd.read_csv(path)
  frequencies = {}
  for row in range(len(df)):
    for word in df.iloc[row, 0].split():
      if word not in frequencies.keys():
        frequencies[word] = 1
      else:
        frequencies[word] += 1

  return frequencies

def generate_word_embeddings(vocab: Vocab, which='normal', filename=None):
  embeddings = dict(zip(vocab.stoi.keys(), torch.normal(0, 1, size=(len(vocab.stoi), 300))))
  embeddings['<PAD>'] = torch.zeros(300)
  if which == 'file':
    with open(filename, 'r') as file:
      lines = file.readlines()
      for line in lines:
        tmp = line.split()
        if tmp[0] in embeddings.keys():
          embeddings[tmp[0]] = torch.tensor([float(num) for num in tmp[1:]])

  return torch.stack(list(embeddings.values()))


def pad_collate_fn(batch, pad_index=0):
  """
  Arguments:
    Batch:
      list of Instances returned by `Dataset.__getitem__`.
  Returns:
    A tensor representing the input batch.
  """
  texts, labels = zip(*batch) # Assuming the instance is in tuple-like form
  lengths = torch.tensor([len(text) for text in texts]) # Needed for later

  return pad_sequence([text for text in texts], batch_first=True, padding_value=pad_index), labels, lengths

