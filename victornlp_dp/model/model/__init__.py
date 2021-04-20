def register_model(cls):
  if 'victornlp_dp_model' not in globals():
    globals()['victornlp_dp_model'] = {}
  victornlp_dp_model[cls.__name__] = cls

from .LeftToRightParser import *
from .DeepBiaffineParser import *