"""
@module analyze
Tools for analyzong the parser behavior such as accuracy().
"""

from . import register_analysis_fn

@register_analysis_fn('accuracy')
def analyze_accuracy(inputs):
  """
  Calculates accuracy.
  
  @param inputs List of dictionaries. Refer to 'dataset.py'' for more details.
  
  @return Dictionary with keys 'uas' and 'las' in percentage(rounded for 4 digits).
  """
  total = 0
  unlabel = 0
  label = 0
  for input in inputs:
    assert 'dependency' in input
    assert 'dependency_predict' in input

    for golden, predict in zip(input['dependency'], input['dependency_predict']):
      if predict['dep'] != golden['dep']:
        continue
      total += 1
      if predict['head'] == golden['head']:
        unlabel += 1
        if predict['label'] == golden['label']:
          label += 1
  return {'uas': round(unlabel/total*100, 2), 'las': round(label/total*100, 2)}


@register_analysis_fn('accuracy_per_label')
def analyze_accuracy_per_label(inputs):
  """
  Calculates accuracy.
  
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  
  @return Dictionary containing label and accuracy(UAS) pair.
  """
  total = {}
  unlabel = {}
  for input in inputs:
    assert 'dependency' in input
    assert 'dependency_predict' in input

    for golden, predict in zip(input['dependency'], input['dependency_predict']):
      if predict['dep'] != golden['dep']:
        continue
      
      if golden['label'] not in total:
        total[golden['label']] = 0
        unlabel[golden['label']] = 0
      total[golden['label']] += 1
      if predict['head'] == golden['head']:
        unlabel[golden['label']] += 1

  return {label:unlabel[label]/total[label] for label in total.keys()}


@register_analysis_fn('accuracy_per_distance')
def analyze_accuracy_per_distance(inputs):
  """
  Calculates accuracy.
  
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  
  @return Dictionary containing label and accuracy(UAS) pair.
  """
  total = {}
  unlabel = {}
  for input in inputs:
    assert 'dependency' in input
    assert 'dependency_predict' in input

    for golden, predict in zip(input['dependency'], input['dependency_predict']):
      if predict['dep'] != golden['dep']:
        continue
      
      key = 'ROOT' if golden['head'] == 0 else abs(golden['head'] - golden['dep'])

      if key not in total:
        total[key] = 0
        unlabel[key] = 0
      total[key] += 1
      if predict['head'] == golden['head']:
        unlabel[key] += 1

  return {key:unlabel[key]/total[key] for key in total.keys()}