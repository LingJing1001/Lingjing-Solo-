"""Neural perception + online reasoning for CEAX unknown capability.

Modules:
  conv     — NumpyCNN / optional torch encode
  mlp      — OnlineMLP + ActionEffectNet + FailureMemory
  reasoner — NeuralReasonLoop (observe→hypothesize→experiment→summarize)
"""
from .conv import NumpyCNN, try_torch_encode
from .mlp import ActionEffectNet, FailureMemory, OnlineMLP
from .reasoner import NeuralReasonLoop

__all__ = [
    "NumpyCNN",
    "try_torch_encode",
    "OnlineMLP",
    "ActionEffectNet",
    "FailureMemory",
    "NeuralReasonLoop",
]
