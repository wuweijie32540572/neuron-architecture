"""
Integrated System: All Architectures Combined
===========================================

Complete system integrating all five neuron architectures:
- SCH: Event-driven sparse computation
- PC: Local predictive learning
- HM: Continual learning with memory
- NG: Dynamic neuromodulatory control
- ESB: Symbol grounding

System Flow:
-----------
    Input → SCH (spike encoding)
          → PC (prediction)
          → HM (memory retrieval)
          → NG (gating)
          → Output

Monitoring:
----------
Key metrics tracked:
- Sparsity: 1 - spike_rate
- Gradient norm: ||∇W||
- Free energy: prediction error²
- Effective learning rate
- Neuromodulator levels
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from .sch import AdaptiveThresholdSCH
from .pc import ResidualPCLayer
from .hm import PatternSeparationHM
from .ng import NormalizedNG


@dataclass
class MonitorMetrics:
    """System monitoring metrics."""
    sparsity: float = 0.0
    grad_norm: float = 0.0
    spike_rate: float = 0.0
    free_energy: float = 0.0
    da: float = 0.5
    effective_lr: float = 0.05


class IntegratedSystem:
    """
    Integrated system combining all architectures.
    
    Parameters
    ----------
    n_sch : int, default=256
        SCH neuron count
    n_pc : int, default=256
        PC state dimension
    n_hm : int, default=128
        HM hidden dimension
    """
    
    def __init__(
        self,
        n_sch: int = 256,
        n_pc: int = 256,
        n_hm: int = 128
    ):
        self.n_sch = n_sch
        self.n_pc = n_pc
        self.n_hm = n_hm
        
        self.sch = AdaptiveThresholdSCH(
            n_sch, tau_m=5.0, v_th_base=0.08, 
            adapt_strength=1.0, target_spike_rate=0.1
        )
        self.pc = ResidualPCLayer(n_pc, n_pc, lr=0.002)
        self.hm = PatternSeparationHM(
            n_sch, n_hm, 
            hippocampus_lr=0.02, 
            cortex_lr=0.008, 
            top_k=64
        )
        self.ng = NormalizedNG(base_lr=0.01)
        
        self.W_encode = np.random.randn(n_sch, 1) * 0.2
        self.W_decode = np.random.randn(1, n_hm) * 0.2
        
        self.prev_z = np.zeros(n_sch)
        self.current_task = 0
        
        self.metrics_history = []
        
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        params = {
            'sch': self.sch.n * self.sch.n,
            'pc': self.pc.W.size,
            'hm_hippo': self.hm.W_hippo.size,
            'hm_cortex': self.hm.W_cortex.size,
            'encode': self.W_encode.size,
            'decode': self.W_decode.size
        }
        params['total'] = sum(params.values())
        return params
    
    def forward(self, x: float) -> Dict:
        """
        Forward pass through system.
        
        Parameters
        ----------
        x : float
            Scalar input
        
        Returns
        -------
        dict
            Intermediate and final outputs
        """
        input_vec = self.W_encode * x
        
        spikes, continuous = self.sch.step(input_vec.flatten())
        
        pc_pred = self.pc.predict(self.prev_z)
        
        hm_out = self.hm.forward(self.sch.z)
        
        output = (self.W_decode @ hm_out)[0]
        
        return {
            'output': output,
            'spikes': spikes,
            'continuous': continuous,
            'pc_pred': pc_pred,
            'hm_out': hm_out
        }
    
    def learn(self, x: float, target: float) -> MonitorMetrics:
        """
        Learning step.
        
        Parameters
        ----------
        x : float
            Input
        target : float
            Target output
        
        Returns
        -------
        MonitorMetrics
            Current monitoring metrics
        """
        result = self.forward(x)
        
        output_error = target - result['output']
        reward = 1.0 - abs(output_error)
        
        self.ng.update_from_reward(reward)
        
        effective_lr = self.ng.compute_effective_lr()
        
        hm_target = result['hm_out'] + 0.1 * np.sign(output_error)
        self.hm.learn(self.sch.z, hm_target, task_id=self.current_task)
        
        self.pc.local_learn(self.prev_z, target=result['continuous'][:self.n_pc])
        
        self.prev_z = self.sch.z.copy()
        
        metrics = MonitorMetrics(
            sparsity=self.sch.get_sparsity(),
            grad_norm=self.pc.get_grad_norm(),
            spike_rate=np.mean(result['spikes']),
            free_energy=float(output_error ** 2),
            da=self.ng.da,
            effective_lr=effective_lr
        )
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
        
        return metrics
    
    def consolidate(self, n_replay: int = 80) -> int:
        """Run HM consolidation."""
        return self.hm.consolidate(n_replay)
    
    def get_average_metrics(self, n_last: int = 100) -> Dict[str, float]:
        """Get average recent metrics."""
        if len(self.metrics_history) == 0:
            return {}
        
        recent = self.metrics_history[-n_last:]
        return {
            'sparsity': np.mean([m.sparsity for m in recent]),
            'grad_norm': np.mean([m.grad_norm for m in recent]),
            'spike_rate': np.mean([m.spike_rate for m in recent]),
            'free_energy': np.mean([m.free_energy for m in recent]),
            'da': np.mean([m.da for m in recent]),
            'effective_lr': np.mean([m.effective_lr for m in recent])
        }
    
    def reset(self) -> None:
        """Reset all states."""
        self.sch.reset()
        self.prev_z = np.zeros(self.n_sch)
        self.ng.reset()
        self.metrics_history = []
