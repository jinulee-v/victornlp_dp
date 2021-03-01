"""
@module loss
Various loss functions for dependency parsing.

Lee(2020)(available in repository) introduces each loss functions briefly. In the same paper, loss_LH was proved to be most effective for Korean data.

loss_*(parser, inputs)
  @param parser *Parser object. Refer to 'model.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  
  @return Loss value.
"""

import torch
import torch.nn as nn

def loss_NLL(parser, inputs):
  """
  Negative Log-Likelihood loss function.
  
  This function only backpropagates from p(head_golden), completely ignoring the possibilities assigned to errors.
  
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs)
  lengths = torch.zeros(batch_size, dtype=torch.long).detach().to(device)
  for i, input in enumerate(inputs):
    lengths[i] = input['word_count'] + 1
  max_length = torch.max(lengths)

  golden_heads = torch.zeros((batch_size, max_length, 1), dtype=torch.long).detach().to(device)
  golden_labels = torch.zeros((batch_size, 1, max_length, max_length), dtype=torch.long).detach().to(device)
  for i, input in enumerate(inputs):
    for j, arc in enumerate(input['dependency']):
      golden_heads[i, j+1, 0] = arc['head']
      golden_labels[i, 0, j+1, :] = parser.labels_stoi[arc['label']]
  
  loss_arc = arc_attention.squeeze(1).gather(2, golden_heads).squeeze(2)
  loss_type = type_attention.gather(1, golden_labels).squeeze(1).gather(2, golden_heads).squeeze(2)
  
  return -(torch.sum(loss_arc) + torch.sum(loss_type)) / (torch.sum(lengths) - batch_size)


def loss_XBCE(parser, inputs):
  """
  eXtended Binary Cross Entropy loss function.
  
  @param parser *Parser object. Refer to 'model.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs)
  
  lengths = torch.zeros(batch_size, dtype=torch.long).detach().to(device)
  for i, input in enumerate(inputs):
    lengths[i] = input['word_count'] + 1
  max_length = torch.max(lengths)

  golden_heads = torch.zeros((batch_size, max_length, 1), dtype=torch.long).detach().to(device)
  golden_labels = torch.zeros((batch_size, 1, max_length, max_length), dtype=torch.long).detach().to(device)
  golden_head_mask = torch.zeros_like(arc_attention, dtype=torch.float).detach().to(device)
  golden_label_mask = torch.zeros_like(type_attention, dtype=torch.float).detach().to(device)
  for i, input in enumerate(inputs):
    for j, arc in enumerate(input['dependency']):
      golden_heads[i, j+1, 0] = arc['head']
      golden_labels[i, 0, j+1, :] = parser.labels_stoi[arc['label']]
      golden_head_mask[i, 0, j+1, :lengths[i]] = 1
      golden_head_mask[i, 0, j+1, arc['head']] = 0
      golden_label_mask[i, :, j+1, arc['head']] = 1
      golden_label_mask[i, parser.labels_stoi[arc['label']], j+1, arc['head']] = 0
  
  loss_arc = torch.sum(arc_attention.squeeze(1).gather(2, golden_heads).squeeze(2))
  loss_arc += torch.sum(torch.log(1+(1e-6)-torch.exp(arc_attention)) * golden_head_mask)
  
  loss_type = torch.sum(type_attention.gather(1, golden_labels).squeeze(1).gather(2, golden_heads).squeeze(2))
  loss_type += torch.sum(torch.log(1+(1e-6)-torch.exp(type_attention)) * golden_label_mask)
  
  return -(loss_arc + loss_type) / (torch.sum(lengths) - batch_size)