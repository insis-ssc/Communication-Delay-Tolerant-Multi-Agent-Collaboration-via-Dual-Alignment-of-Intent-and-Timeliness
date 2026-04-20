REGISTRY = {}

from .rnn_agent import RNNAgent
from .maic_agent import MAICAgent
from .Code_Agent import Code_Agent
from .Code_seq_Agent import Code_seq_Agent


REGISTRY["rnn"] = RNNAgent
REGISTRY['maic'] = MAICAgent
REGISTRY['Code'] = Code_Agent
REGISTRY['Code_seq'] = Code_seq_Agent

