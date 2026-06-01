"""
Novel Neuron Architectures for Overcoming LLM Limitations
=========================================================

A neuroscience-inspired framework implementing five novel neuron architectures
to address core limitations of Large Language Models:
- Catastrophic forgetting
- Lack of local learning rules
- Missing event-driven computation
- Symbol grounding problem

Architectures:
- SCH-Neuron: Spiking-Continuous Hybrid with adaptive threshold
- PC-Neuron: Predictive Coding with two-stage training
- HM-Neuron: Hippocampus-Cortex Memory with stability measure
- NG-Neuron: Neuromodulator-Gated learning (DA/5HT/ACh/NE)
- ESB-Neuron: Embodied-Symbolic Bridging

Contact: aiwuweijie@foxmail.com
License: MIT
"""

from .sch import AdaptiveThresholdSCH
from .pc import ResidualPCLayer
from .hm import PatternSeparationHM
from .ng import NormalizedNG
from .esb import ESBNeuronLayer
from .integrated import IntegratedSystem

__version__ = "1.0.0"
__all__ = [
    "AdaptiveThresholdSCH",
    "ResidualPCLayer", 
    "PatternSeparationHM",
    "NormalizedNG",
    "ESBNeuronLayer",
    "IntegratedSystem",
]
