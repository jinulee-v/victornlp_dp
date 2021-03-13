"""
Script for training the dependency parser.
"""

import os, sys
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import *

from .preprocessing.corpus import VictorNLPDataset, preprocessor_DependencyParsing

from .victornlp_utils.embedding.bert_embeddings import *
from .victornlp_utils.embedding.dict_embeddings import *

from .model.model import *
from .model.loss import *
from .model.parse import *
from .tools.analyze import *

# FIXME replace this hard-coded vars to cmd line args.
config_path = 'victornlp_dp/config_DependencyParsing.json'
language = 'Korean'
parser_model = 'DeepBiaffineParser'

def main():
  """
  Training routine.
  """
  torch.cuda.empty_cache()
  
  # Load configuration file
  config = None
  with open(config_path) as config_file:
    config = json.load(config_file)
  assert config
  
  language_config = config['language'][language]
  embedding_config = config['embedding']
  parser_config = config['parser'][parser_model]
  train_config = config['train']
  
  # Load corpus
  train_path = language_config['corpus']['train']
  test_path = language_config['corpus']['test']
  labels_path = language_config['corpus']['labels']
  train_dev_ratio = language_config['corpus']['train_dev_ratio']
  
  train_dataset, dev_dataset, test_dataset, type_label = None, None, None, None
  with open(train_path) as train_corpus_file:
    train_dataset = VictorNLPDataset(json.load(train_corpus_file), [preprocessor_DependencyParsing])
  with open(test_path) as test_corpus_file:
    test_dataset = VictorNLPDataset(json.load(test_corpus_file),  [preprocessor_DependencyParsing])
  with open(labels_path) as type_label_file:
    type_label = json.load(type_label_file)
  
  # Split train/dev datasets
  if train_dev_ratio < 1.:
    split = random_split(train_dataset, [int(len(train_dataset) * train_dev_ratio), len(train_dataset) - int(len(train_dataset)*train_dev_ratio)])
    train_dataset = split[0]
    dev_dataset = split[1]
  else:
    dev_dataset = VictorNLPDataset({})
  
  # Prepare DataLoader instances
  train_loader = DataLoader(train_dataset, train_config['batch_size'], shuffle=True, collate_fn=VictorNLPDataset.collate_fn)
  if dev_dataset:
    dev_loader = DataLoader(dev_dataset, train_config['batch_size'], shuffle=False, collate_fn=VictorNLPDataset.collate_fn)
  test_loader = DataLoader(test_dataset, train_config['batch_size'], shuffle=False, collate_fn=VictorNLPDataset.collate_fn)
  
  # Create parser module
  device = torch.device(train_config['device'])
  embeddings = [globals()[embedding_type](embedding_config[embedding_type]).to(device) for embedding_type in language_config['embedding']]
  parser = globals()[parser_model](embeddings, type_label, parser_config)
  parser = parser.to(device)
  
  # Backpropagation settings
  loss_fn = globals()[train_config['loss_fn']]
  optimizer = globals()[train_config['optimizer']](parser.parameters(), train_config['learning_rate'])
  parse_fn = globals()[train_config['parse_fn']]
  
  # Training
  for epoch in range(1, train_config['epoch']+1):
    print('-'*40)
    print('Epoch:', epoch)
    
    print()
    print('Train')
    parser.train()
    
    iter = tqdm(train_loader)
    for i, batch in enumerate(iter):
      optimizer.zero_grad()
      loss = loss_fn(parser, batch)
      loss.backward()
      optimizer.step()
    
    # Early stopping
    if dev_dataset:
      print()
      print('Early stopping')
      
      parser.eval()
      loss = 0
      cnt = 0
      for batch in tqdm(dev_loader):
        cnt += len(batch)
        loss += float(loss_fn(parser, batch)) * len(batch)
      print('Dev loss:', loss/cnt)
    
    
    # Accuracy
    print()
    print('Accuracy')
    
    parser.eval()
    for batch in tqdm(test_loader): 
      # Call by reference modifies the original batch
      parse_fn(parser, batch, language_config['parse']) 
    
    print(accuracy(test_dataset))
    print('-'*40)
    print()

  
if __name__ == '__main__':
  # Change working directory to project root
  if os.getcwd().endswith('victornlp_dp'):
    os.chdir('..') 
  main()
