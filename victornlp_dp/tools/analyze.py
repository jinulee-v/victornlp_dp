"""
@module analyze
Tools for analyzong the parser behavior such as accuracy().
"""

def analyze_accuracy(inputs):
  """
  Calculates accuracy.
  
  @param inputs List of dictionaries. Refer to 'corpus.py'' for more details.
  
  @return Dictionary with keys 'uas' and 'las' in percentage(rounded for 4 digits).
  """
  total = 0
  unlabel = 0
  label = 0
  for input in inputs:
    assert 'dependency' in input
    assert 'dependency_predict' in input

    for golden, predict in zip(input['dependency'], input['dependency_predict']):
      assert predict['dep'] == golden['dep']
      total += 1
      if predict['head'] == golden['head']:
        unlabel += 1
        if predict['label'] == golden['label']:
          label += 1
  return {'uas': round(unlabel/total*100, 2), 'las': round(label/total*100, 2)}