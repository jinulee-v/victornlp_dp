"""
@module parse

Supports evaluation, interactive&batched parsing.

Input data file must strictly follow VictorNLP format.

if 'pos' in input.keys():
  # Golden PoS tag information is given
  # (its format may vary among languages.)
else:
  # Perform language-specific PoS tagging(defined in victornlp_utils.pos_tagger)

if 'dependency' in input.keys():
  # Golden dependency tree is given
  # Perform evaluation/analysis (specified by -a, --analyze)

Exceptionally, for stdin inputs, it only requires raw texts, a sentence per line.
PoS tags are automatically generated and no evaluation is performed.
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
from .victornlp_utils.pos_tagger.pos_tagger import *

from .victornlp_utils.utils.early_stopping import EarlyStopping

from .model.model import *
from .model.loss import *
from .model.parse import *
from .tools.analyze import *

def parse_cmd_arguments():
  parser = argparse.ArgumentParser(description="Evaluate a model or parse raw texts.")
  parser.add_argument('model_dir', type=str, help='Model directory that contains model.pt & config.json.')
  parser.add_argument('--data-file', type=str, help='File that contains VictorNLP format data. default: stdin(only raw texts)')
  parser.add_argument('-a', '--analyze', type=str, action='append', choices=[fn for fn in globals().keys() if fn.startswith('analyze')])
  parser.add_argument('--save-result', help='Print VictorNLP-format results to a file.')

  args = parser.parse_args()
  return args

def main():
  args = parse_cmd_arguments()

  # Load configuration file
  config = None
  config_path = args.model_dir + '/config.json'
  with open(config_path) as config_file:
    config = json.load(config_file)
  assert config
  
  train_config = config['train']
  language_config = config['language'][train_config['language']]
  embedding_config = config['embedding']
  parser_config = config['parser'][train_config['model']]
  
  # Set frequent constant variables
  language = train_config['language']
  parser_model = train_config['model']
  now = datetime.now().strftime(u'%Y%m%d %H:%M:%S')
  title = train_config['title'] if 'title' in train_config else now + ' ' + language + ' ' + parser_model
  train_config['title'] = title

  # Set logger
  file_formatter = logging.Formatter(fmt="%(message)s")
  stream_formatter = logging.Formatter(fmt="%(asctime)s %(message)s")

  fileHandler = logging.FileHandler(filename=args.model_dir + '/parse_{}.log'.format(now), encoding='utf-8')
  fileHandler.setFormatter(file_formatter)
  streamHandler = logging.StreamHandler()
  streamHandler.setFormatter(stream_formatter)
  logger = logging.getLogger()
  logger.addHandler(fileHandler)
  logger.addHandler(streamHandler)
  logger.setLevel(logging.INFO)

  logger.info(title)

  # Prepare data
  logger.info('Preparing data...')

  # Load label data
  labels_path = language_config['corpus']['labels']
  with open(labels_path) as type_label_file:
    type_label = json.load(type_label_file)['dp_labels']

  # Prepare evaluation data if file is given
  from_file = bool(args.data_file)
  pos_tagger = globals()['pos_tag_' + language]
  preprocessors = [preprocessor_WordCount, pos_tagger, preprocessor_DependencyParsing]

  if from_file:
    with open(args.data_file, 'r') as data_file:
      dataset = VictorNLPDataset(json.load(data_file), preprocessors)
      # Prepare DataLoader instances
      loader = DataLoader(dataset, train_config['batch_size'], shuffle=True, collate_fn=VictorNLPDataset.collate_fn)
      if args.analyze:
        # If evaluation mode, input must contain gold dependency tags.
        for i in range(len(dataset)):
          assert 'dependency' in dataset[i]
  else:
    logger.info('Receiving data from stdin...')
  
  # Create parser module
  logger.info('Preparing models...')
  device = torch.device(train_config['device'])
  embeddings = [globals()[embedding_type](embedding_config[embedding_type]).to(device) for embedding_type in language_config['embedding']]
  parser = globals()[parser_model](embeddings, type_label, parser_config)
  parser = parser.to(device)
  parser.load_state_dict(torch.load(args.model_dir + '/model.pt'))

  parse_fn = globals()[train_config['parse_fn']]

  # Evaluation
  with torch.no_grad():
    if from_file:
      # From file data

      # Parse and log time
      before = datetime.now()
      logger.info('Started at...' + before.strftime(u'%Y%m%d %H:%M:%S'))
      for batch in tqdm(loader):
        parse_fn(parser, batch, language_config['parse'])
      after = datetime.now()
      logger.info('Finished at...' + after.strftime(u'%Y%m%d %H:%M:%S'))
      seconds = (after - before).total_seconds()
      logger.info('Total time: %.2fs (%.2f sents/s)', seconds, len(dataset)/seconds)
      logger.info('')

      # Run analysis functions
      if not args.analyze:
        return
      analyzers = [globals()[analyzer] for analyzer in args.analyze]
      for analyzer in analyzers:
        result = analyzer(dataset)
        stream_logger.info('-'*40)
        logger.info(analyzer.__name__.replace('analyze_', ''))
        if isinstance(result, dict):
          # Dictionary results
          for key, value in result.items():
            logger.info('  {}: {}'.format(key, value))
        else:
          # Text results(TSV, pd.dataframe, ...)
          logger.info('\n' + str(result))
        logger.info('-'*40)
        logger.info('')

      # Save result if needed
      if args.save_result:
        with open(args.model_dir + 'parse_result_{}.log'.format(now), 'w') as out_file:
          json.dump(inputs, out_file, indent=4)

    else:
      # From stdin
      while True:
        sentence = input()
        # Conversion to VictorNLP format
        sentence = [{'text': sentence}]
        for preprocess in preprocessors:
          sentence = preprocess(sentence)

        # Parse and log time
        before = datetime.now()
        logger.info('Started at...' + before.strftime(u'%Y%m%d %H:%M:%S'))
        parse_fn(parser, sentence, language_config['parse'])
        after = datetime.now()
        logger.info('Finisheded at...' + after.strftime(u'%Y%m%d %H:%M:%S'))
        seconds = (after - before).total_seconds()
        logger.info('Total time: %.2fs (%.2f sents/s)', seconds, len(dataset)/seconds)
        logger.info('')
        
        # Format and print result
        sentence = sentence[0]
        logger.info('; ' + sentence['text'])
        logger.info('# ' + ' '.join(
          ['+'.join(
            [morph['text'] + '/' + morph['pos_tag']
            for morph in wordphr])
          for wordphr in sentence['pos']]
        ))
        wordphrs = ['[ROOT]'] + sentence['text'].split()
        for arc in sentence['dependency']:
          logger.info('{}\t{}\t{}\t{}\t-> {}'.format(arc['dep'], arc['head'], arc['label'], wordphrs[arc['dep']], wordphrs[arc['head']]))
        logger.info('')

if __name__ == '__main__':
  main()