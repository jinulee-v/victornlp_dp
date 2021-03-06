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

def parse_greedy(parser, inputs, config):
  """
  Simple argmax parsing. No topologocal restrictions are applied during parsing, thus may generate improper structures. Use for testing.
  
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
  
  @return 'inputs' dictionary with parse tree information added.
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs)
  
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


def parse_beam(parser, inputs, config):
  """
  Beam search implemented in Stack-Pointer Network(Ma, 2018) and Left-To-Right parser(Fernandez-Gonzalez, 2019).
  
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
  
  @return 'inputs' dictionary with parse tree information added.
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs)

  head_initial = config['head_initial']
  head_final = config['head_final']
  allow_projective = config['allow_projective']
  beam_size = config['beam_size']

  for i, input in enumerate(inputs):

    # Arc decision
    beam = [[0]]
    beam_scores = torch.ones(beam_size).to(device) * float('-inf')
    beam_scores[0] = 0.

    arc_att = torch.squeeze(arc_attention[i].clone(), 0)

    length = input['word_count'] + 1
    for dep_id in range(1, length):
      new_scores = beam_scores.unsqueeze(0) + arc_att[dep_id].unsqueeze(1)
      new_scores, indices = torch.sort(new_scores.view(-1), descending=True)
      indices = list(indices.cpu())
      for j, index in enumerate(indices):
        index = index.item()
        indices[j] = (index//beam_size, index%beam_size)

      new_beam = []
      new_beam_scores = torch.ones(beam_size).to(device) * float('-inf')
      for score, (head_id, beam_id) in zip(new_scores, indices):
        if len(new_beam) == beam_size or beam_id >= len(beam):
          break
        if head_id >= length:
          continue

        # Check topological conditions
        if head_initial:
          if head_id >= dep_id:
            continue
        if head_final:
          if (dep_id == length - 1) and (head_id != 0):
            continue
          elif (dep_id < length - 1) and (head_id <= dep_id):
            continue
        if not allow_projective:
          projective = False
          for child, parent in enumerate(beam[beam_id]):
            if parent < dep_id < child or child < dep_id < parent:
              if (head_id < child and head_id < parent) or (head_id > child and head_id > parent):
                projective = True
                break
            if parent < head_id < child or child < head_id < parent:
              if (dep_id < child and dep_id < parent) or (dep_id > child and dep_id > parent):
                projective = True
                break
          if projective:
            continue

        # Add current parse to list
        new_beam_scores[len(new_beam)] = score
        new_beam.append(beam[beam_id]+[head_id])
      # Update beam
      beam = new_beam
      beam_scores = new_beam_scores

    heads = beam[0]

    # Type decision
    types = []
    type_att = type_attention[i]
    for dep_id, head_id in enumerate(heads):
      types.append(torch.argmax(type_att[:, dep_id, head_id]))

    # Update result
    result = []
    for dep_id, (head_id, type_id) in enumerate(zip(heads, types)): # FIXME
      if dep_id == 0:
        continue
      result.append({'dep': dep_id, 'head': head_id, 'label': parser.labels[type_id]})
    if 'dependency' in input:
      key = 'dependency_predict'
    else:
      key = 'dependency'
    input[key] = result

  return inputs
  
def parse_MST(parser, inputs, config):
  """
  Wrapper function for Projective / Nonprojective / Head-Initial/Final MST Algorithms.
  Projective : Eisner Algorithm
  Non-Projective : Chu-Liu-Edmonds Algorithm
  Non-Projective + Head-Initial-Final: DP O(n^3)
  Note that Projectiveness and Head-Initial/Finality do not co-occur in most languages.
  
    @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
  
  @return 'inputs' dictionary with parse tree information added.
  """
  head_initial = config['head_initial']
  head_final = config['head_final']
  allow_projective = config['allow_projective']
  beam_size = config['beam_size']
  
  if allow_projective:
    if head_initial or head_final:
      raise "MST for projective & head-initial/final trees are not supported."
    else:
      inputs = _parse_MST_Eisner(parser, inputs, config)
  else:
    if head_initial or head_final:
      inputs = _parse_MST_DP(parser, inputs, config)
    else:
      inputs = _parse_MST_ChuLiuEdmonds(parser, inputs, config)
  
  return inputs


def _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
  """
  Backtracking step in Eisner's algorithm.
  - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
  an end position, and a direction flag (0 means left, 1 means right). This array contains
  the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
  - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
  an end position, and a direction flag (0 means left, 1 means right). This array contains
  the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
  - s is the current start of the span
  - t is the current end of the span
  - direction is 0 (left attachment) or 1 (right attachment)
  - complete is 1 if the current span is complete, and 0 otherwise
  - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
  head of each word.
  """
  if s == t:
    return
  if complete:
    r = complete_backtrack[s][t][direction]
    if direction == 0:
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
      return
    else:
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
      return
  else:
    r = incomplete_backtrack[s][t][direction]
    if direction == 0:
      heads[s] = t
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
      return
    else:
      heads[t] = s
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
      return

def _parse_MST_Eisner(parser, inputs, config):
  """
  Implementation of Eisner Algorithm(Eisner, 1996) which is capable of generating projective MST for a sequence.
  
  Code citation: Modified from https://github.com/daandouwe/perceptron-dependency-parser
  - Transplant from NumPy to PyTorch
  - Modified input / output details to fit VictorNLP format
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs)
  
  for i, input in enumerate(inputs):
    num_words = input['word_count']
    scores = arc_attention[i, 0, :num_words+1, :num_words+1].transpose(0, 1)

    # Initialize CKY table.
    complete = torch.zeros([num_words+1, num_words+1, 2]).to(device)  # s, t, direction (right=1).
    incomplete = torch.zeros([num_words+1, num_words+1, 2]).to(device)  # s, t, direction (right=1).
    complete_backtrack = -torch.ones([num_words+1, num_words+1, 2], dtype=torch.long)  # s, t, direction (right=1).
    incomplete_backtrack = -torch.ones([num_words+1, num_words+1, 2], dtype=torch.long)  # s, t, direction (right=1).

    incomplete[0, :, 0] -= float("inf")

    # Loop from smaller items to larger items.
    for k in range(1, num_words+1):
      for s in range(num_words-k+1):
        t = s + k

        # First, create incomplete items.
        # left tree
        incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s]
        incomplete[s, t, 0] = torch.max(incomplete_vals0)
        incomplete_backtrack[s, t, 0] = (s + torch.argmax(incomplete_vals0))
        # right tree
        incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t]
        incomplete[s, t, 1] = torch.max(incomplete_vals1)
        incomplete_backtrack[s, t, 1] = (s + torch.argmax(incomplete_vals1)).cpu()

        # Second, create complete items.
        # left tree
        complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
        complete[s, t, 0] = torch.max(complete_vals0)
        complete_backtrack[s, t, 0] = (s + torch.argmax(complete_vals0)).cpu()
        # right tree
        complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
        complete[s, t, 1] = torch.max(complete_vals1)
        complete_backtrack[s, t, 1] = (s + 1 + torch.argmax(complete_vals1)).cpu()

    value = complete[0][num_words][1]
    heads = -torch.ones(num_words + 1, dtype=torch.long)
    _backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, num_words, 1, 1, heads)

    value_proj = 0.0 # Total log score
    for m in range(1, num_words+1):
      h = heads[m]
      value_proj += scores[h, m]

    # Type decision
    types = []
    type_att = type_attention[i]
    for dep_id, head_id in enumerate(heads):
      types.append(torch.argmax(type_att[:, dep_id, head_id]))

    # Update result
    result = []
    for dep_id, (head_id, type_id) in enumerate(zip(heads, types)): # FIXME
      if dep_id == 0:
        continue
      result.append({'dep': dep_id, 'head': head_id, 'label': parser.labels[type_id]})
    if 'dependency' in input:
      key = 'dependency_predict'
    else:
      key = 'dependency'
    input[key] = result
    
  return inputs

            