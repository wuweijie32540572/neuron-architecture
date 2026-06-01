"""
Neuron Architecture - PyTorch Version
=====================================

改进版本，解决原版的关键问题：
1. GPU支持和批处理
2. 真正的自由能计算
3. 完整的神经调质动态
4. 真正的正交模式（Gram-Schmidt）
5. 时序编码和STDP
"""

from .sch_torch import SCHNeuronTorch, SCHLayer
from .pc_torch import PredictiveCodingLayer, TwoStagePCTrainer
from .ng_torch import DynamicNeuromodulator
from .esb_torch import ESBNeuronTorch, gram_schmidt

__all__ = [
    'SCHNeuronTorch',
    'SCHLayer',
    'PredictiveCodingLayer',
    'TwoStagePCTrainer',
    'DynamicNeuromodulator',
    'ESBNeuronTorch',
    'gram_schmidt'
]

__version__ = '2.0.0'
