"""
@module kmdp_utils
Various utilities for KMDP.
"""

import torch

from .lexicals import label2head
from ..victornlp_utils.corpora.dataset import register_preprocessors

def generate_kmdp_lengths_mask(inputs, device):
  """
  Generates lengths/mask for KMDP parsing. 

  @return lengths Modified input for Parser::run().
  @return mask Modified input for Parser::run().
  """
  batch_size = len(inputs)

  lengths = torch.zeros(batch_size, dtype=torch.long, device=device).detach()
  max_len = 0
  for i, input in enumerate(inputs):
    length = input['pos'][-1][-1]['id'] + 1
    lengths[i] = length
    if max_len < length:
      max_len = length
  
  mask = torch.zeros(batch_size, 1, max_len, max_len, device=device).detach()
  for i, input in enumerate(inputs):
    length = input['pos'][-1][-1]['id'] + 1
    ids = []
    for wp in input['pos']:
      for morph in wp:
        if morph['pos_tag'] in label2head['all_heads']:
          ids.append(morph['id'])
    mask[i, 0, ids, 0] = 1
    for id in ids:
      mask[i, 0, id, ids] = 1
      mask[i, 0, ids, id] = 1
  
  return lengths, mask

@register_preprocessors('kmdp')
def prerocessor_ReplaceKMDP(inputs):
  for input in inputs:
    assert input['text']
    assert input['pos']
    assert len(input['pos']) == input['word_count']

    input['dependency'] = input['kmdp']
    input.pop('kmdp')
  
  return inputs