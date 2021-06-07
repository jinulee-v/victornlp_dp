"""
@module parse
Various parsing functions based on attention scores.

parse_*(parser, inputs, config)
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
"""

import torch
import torch.nn as nn

from . import register_parse_fn

@register_parse_fn('greedy')
def parse_greedy(parser, inputs, config, **kwargs):
  """
  Simple argmax parsing. No topologocal restrictions are applied during parsing, thus may generate improper structures. Use for testing.
  
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
  @param **kwargs Passed to parser.run().
  
  @return 'inputs' dictionary with parse tree information added.
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs, **kwargs)
  
  for i, input in enumerate(inputs):
    result = []
    length = input['word_count'] + 1
    for dep in range(1, length):
      head = torch.argmax(arc_attention[i][0][dep][:length]).item()
      label = parser.labels[torch.argmax(type_attention[i, :, dep, head]).reshape(-1)]
      result.append({
        'dep': dep,
        'head': head,
        'label': label
      })
      
    if 'dependency' in input:
      key = 'dependency_predict'
    else:
      key = 'dependency'
    input[key] = result
  
  return inputs
