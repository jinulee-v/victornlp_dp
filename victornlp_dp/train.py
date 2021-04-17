"""
@module train

Script for training the dependency parser.
"""

import os, sys
import json
from tqdm import tqdm
import logging
from datetime import datetime
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import *

from .victornlp_utils.corpora.dataset import *

from .victornlp_utils.embedding.bert_embeddings import *
from .victornlp_utils.embedding.dict_embeddings import *

from .victornlp_utils.utils.early_stopping import EarlyStopping

from .model.model import *
from .model.loss import *
from .model.parse import *
from .tools.analyze import *

def argparse_cmd_args() :
  parser = argparse.ArgumentParser(description='Train the depedency parser model.')
  parser.add_argument('config_file', type=str, nargs='?', default='victornlp_dp/config_DependencyParsing.json')
  parser.add_argument('--model', choices=[fn for fn in globals().keys() if fn.endswith('Parser')], help='parser model. Choose parser name from default config file.')
  parser.add_argument('--language', type=str, help='language. Choose language name from default config file.')
  parser.add_argument('--epoch', type=int, help='training epochs')
  parser.add_argument('--batch_size', type=int, help='batch size for training')
  parser.add_argument('--loss_fn', choices=[fn for fn in globals().keys() if fn.startswith('loss_')], help='loss functions')
  parser.add_argument('--parse_fn', choices=[fn for fn in globals().keys() if fn.startswith('parse_')], help='parse functions')
  parser.add_argument('--optimizer', type=str, help='optimizer. Choose class name from torch.optim')
  parser.add_argument('--learning_rate', type=float, help='learning rate')
  parser.add_argument('--device', type=str, help='device. Follows the torch.device format')
  
  args = parser.parse_args()
  
  return args

def main():
  """
  Training routine.
  """
  args = argparse_cmd_args()

  # Load configuration file
  config = None
  config_path = args.config_file
  with open(config_path) as config_file:
    config = json.load(config_file)
  assert config
  
  train_config = config['train'] if 'train' in config else {}
  for arg in vars(args):
    if getattr(args, arg):
      train_config[arg] = getattr(args, arg)
  language_config = config['language'][train_config['language']]
  embedding_config = {name:conf for name, conf in config['embedding'].items() if name in language_config['embedding']}
  parser_config = config['parser'][train_config['model']]
  

  # Set frequent constant variables
  language = train_config['language']
  parser_model = train_config['model']

  now = datetime.now().strftime(u'%Y%m%d %H:%M:%S')
  title = train_config['title'] if 'title' in train_config else now + ' ' + language + ' ' + parser_model
  train_config['title'] = title

  # Extract only required features for clarity
  config = {
    'language': {
      language: language_config
    },
    'embedding': embedding_config,
    'parser': {
      parser_model: parser_config
    },
    'train': train_config
  }

  # Directory for logging, config & model storage
  os.makedirs('models/' + title)

  formatter = logging.Formatter(fmt="%(asctime)s %(message)s")

  fileHandler = logging.FileHandler(filename='models/{}/train_{}.log'.format(title, now), encoding='utf-8')
  fileHandler.setFormatter(formatter)
  streamHandler = logging.StreamHandler()
  streamHandler.setFormatter(formatter)
  logger = logging.getLogger()
  logger.addHandler(fileHandler)
  logger.addHandler(streamHandler)
  logger.setLevel(logging.INFO)

  logger.info(title)

  logger.info('\n' + json.dumps(config, indent=4))
  with open('models/' + title + '/config.json', 'w', encoding='UTF-8') as f:
    json.dump(config, f, indent=4)
  
  # Load corpus
  logger.info('Preparing data...')
  train_path = language_config['corpus']['train']
  dev_path = language_config['corpus']['dev'] if 'dev' in language_config['corpus'] else None
  test_path = language_config['corpus']['test']
  labels_path = language_config['corpus']['labels']
  train_dev_ratio = language_config['corpus']['train_dev_ratio'] if 'train_dev_ratio' in language_config['corpus'] else None
  
  train_dataset, dev_dataset, test_dataset, type_label = None, None, None, None
  with open(train_path) as train_corpus_file:
    train_dataset = VictorNLPDataset(json.load(train_corpus_file), [preprocessor_WordCount, preprocessor_DependencyParsing])
  with open(test_path) as test_corpus_file:
    test_dataset = VictorNLPDataset(json.load(test_corpus_file),  [preprocessor_WordCount, preprocessor_DependencyParsing])
  with open(labels_path) as type_label_file:
    type_label = json.load(type_label_file)['dp_labels']

  # Split dev datasets
  if dev_path:
    with open(dev_path) as dev_corpus_file:
      dev_dataset = VictorNLPDataset(json.load(dev_corpus_file), [preprocessor_WordCount, preprocessor_DependencyParsing])
  else:
    if train_dev_ratio and train_dev_ratio < 1.:
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
  logger.info('done\n')
  
  # Create parser module
  logger.info('Preparing models and optimizers...')
  device = torch.device(train_config['device'])
  embeddings = [globals()[embedding_type](embedding_config[embedding_type]).to(device) for embedding_type in language_config['embedding']]
  parser = globals()[parser_model](embeddings, type_label, parser_config)
  parser = parser.to(device)
  
  # Backpropagation settings
  loss_fn = globals()[train_config['loss_fn']]
  optimizer = globals()[train_config['optimizer']](parser.parameters(), train_config['learning_rate'])
  parse_fn = globals()[train_config['parse_fn']]

  # Early Stopping settings
  if dev_dataset:
    es_config = train_config['early_stopping']
    early_stopper = EarlyStopping(es_config['patience'], es_config['eps'], es_config['maximize'])
  logger.info('done\n')
  
  # Training
  for epoch in range(1, train_config['epoch']+1):
    logger.info('-'*40)
    logger.info('Epoch: %d', epoch)
    
    logger.info('')
    logger.info('Train')
    parser.train()
    
    iter = tqdm(train_loader)
    for i, batch in enumerate(iter):
      optimizer.zero_grad()
      loss = loss_fn(parser, batch)
      loss.backward()
      optimizer.step()
    
    # Validation
    if dev_dataset:
      logger.info('')
      logger.info('Validation')
      
      with torch.no_grad():
        parser.eval()
        loss = 0
        cnt = 0
        for batch in tqdm(dev_loader):
          cnt += len(batch)
          loss += float(loss_fn(parser, batch)) * len(batch)
        logger.info('Dev loss: %f', loss/cnt)
        if early_stopper(epoch, loss/cnt, parser, 'models/' + title + '/model.pt'):
          break
    
    # Accuracy
    logger.info('')
    logger.info('Accuracy')
    
    with torch.no_grad():
      parser.eval()
      for batch in tqdm(test_loader): 
        # Call by reference modifies the original batch
        parse_fn(parser, batch, language_config['parse']) 
      
      logger.info(analyze_accuracy(test_dataset))
      logger.info('-'*40)
      logger.info('')
  
  logger.info('Training completed.')
  logger.info('Check {} for logs, configurations, and the trained model file.'.format('models/' + title))

  
if __name__ == '__main__':
  main()
