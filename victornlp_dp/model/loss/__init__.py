"""
@module loss
Various loss functions for dependency parsing.

loss_*(parser, inputs)
  @param parser *Parser object. Refer to 'model.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  
  @return Loss value.
"""

def register_loss_fn(fn):
  if 'victornlp_dp_loss_fn' not in globals():
    globals()['victornlp_dp_loss_fn'] = {}
  victornlp_dp_loss_fn[fn.__name__] = fn

from .loss_local import loss_NLL, loss_XBCE, loss_LH