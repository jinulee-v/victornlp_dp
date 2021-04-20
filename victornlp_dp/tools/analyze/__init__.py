def register_analysis(fn):
  if 'victornlp_dp_analysis' not in globals():
    globals()['victornlp_dp_analysis'] = {}
  victornlp_dp_analysis[fn.__name__] = fn

from .accuracy import *