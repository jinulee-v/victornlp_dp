"""
@module parse
Various parsing functions based on attention scores.

parse_*(parser, inputs, config)
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
"""

def register_parse_fn(fn):
  if 'victornlp_dp_parse_fn' not in globals():
    globals()['victornlp_dp_parse_fn'] = {}
  victornlp_dp_parse_fn[fn.__name__] = fn

from .parse_greedy import parse_greedy
from .parse_beam import parse_beam
from .parse_MST import parse_MST