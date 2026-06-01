"""
SCH-Neuron: Spiking-Continuous Hybrid Neuron
============================================

Physics Foundation:
------------------
Based on the leaky integrate-and-fire (LIF) neuron model from computational neuroscience.

Membrane potential dynamics (differential equation):
    τ_m · dV/dt = -V + R_m · I(t)

Where:
- τ_m: membrane time constant (biological: ~10-20ms)
- V: membrane potential
- R_m: membrane resistance
- I(t): input current

Spiking mechanism:
    s(t) = H(V(t) - V_th)

Where H is the Heaviside step function.

Adaptive Threshold (Key Innovation):
-----------------------------------
To maintain target spike rate r_target, we use feedback control:

    V_th(t) = V_th_base × (1 + α · (r_current - r_target))

This prevents:
1. Sparse coding collapse (rate → 0)
2. Rate explosion (rate → 1)
3. Maintains event-driven computation advantage

Biological Analogy:
------------------
Similar to homeostatic plasticity in biological neurons:
- Intrinsic plasticity adjusts firing thresholds
- Maintains stable activity levels
- Prevents both silence and saturation

Electronic Implementation:
------------------------
Can be implemented with:
- RC circuit for membrane integration
- Comparator for threshold detection
- Reset switch for spike generation
- Digital feedback for threshold adaptation
"""

import numpy as np
from typing import Tuple, Optional


class AdaptiveThresholdSCH:
    """
    Spiking-Continuous Hybrid Neuron with Adaptive Threshold.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons in the layer
    tau_m : float, default=5.0
        Membrane time constant (ms)
    v_th_base : float, default=0.1
        Base spike threshold
    adapt_strength : float, default=1.0
        Adaptation strength for threshold control
    target_spike_rate : float, default=0.1
        Target spike rate for homeostatic control
    
    Attributes
    ----------
    v : ndarray
        Membrane potentials
    z : ndarray
        Spike traces (exponential moving average)
    v_th : ndarray
        Adaptive thresholds
    spike_history : list
        Recent spike rates for monitoring
    
    Examples
    --------
    >>> sch = AdaptiveThresholdSCH(n_neurons=256)
    >>> spikes, continuous = sch.step(input_current)
    >>> print(f"Spike rate: {np.mean(spikes):.3f}")
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau_m: float = 5.0,
        v_th_base: float = 0.1,
        adapt_strength: float = 1.0,
        target_spike_rate: float = 0.1
    ):
        self.n = n_neurons
        self.tau_m = tau_m
        self.v_th_base = v_th_base
        self.adapt_strength = adapt_strength
        self.target_spike_rate = target_spike_rate
        
        self.v = np.zeros(n_neurons)
        self.z = np.zeros(n_neurons)
        self.v_th = np.ones(n_neurons) * v_th_base
        
        self.spike_history: list = []
        
    def step(
        self,
        input_current: np.ndarray,
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one simulation step.
        
        Parameters
        ----------
        input_current : ndarray
            Input current to neurons
        dt : float, default=1.0
            Time step size
        
        Returns
        -------
        spikes : ndarray
            Binary spike outputs
        continuous : ndarray
            Continuous representation (tanh of normalized potential)
        """
        current_rate = (
            np.mean(self.spike_history) 
            if len(self.spike_history) > 10 
            else self.target_spike_rate
        )
        rate_error = current_rate - self.target_spike_rate
        
        adapt_factor = 1.0 + self.adapt_strength * rate_error
        adapt_factor = np.clip(adapt_factor, 0.5, 2.0)
        
        self.v_th = np.ones(self.n) * self.v_th_base * adapt_factor
        self.v_th = np.clip(self.v_th, 0.05, 0.5)
        
        dv = (-self.v + input_current) / self.tau_m
        self.v = self.v + dv * dt
        
        spikes = (self.v >= self.v_th).astype(float)
        self.v = np.where(spikes > 0, 0.0, self.v)
        
        decay = np.exp(-dt / self.tau_m)
        self.z = decay * self.z + spikes
        
        continuous = np.tanh(self.v / (self.v_th + 1e-6))
        
        self.spike_history.append(np.mean(spikes))
        if len(self.spike_history) > 100:
            self.spike_history.pop(0)
        
        return spikes, continuous
    
    def get_sparsity(self) -> float:
        """Get current sparsity (1 - spike_rate)."""
        if len(self.spike_history) == 0:
            return 1.0
        return 1.0 - np.mean(self.spike_history)
    
    def reset(self) -> None:
        """Reset neuron states."""
        self.v = np.zeros(self.n)
        self.z = np.zeros(self.n)
        self.v_th = np.ones(self.n) * self.v_th_base
        self.spike_history = []
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveThresholdSCH(n={self.n}, tau_m={self.tau_m}, "
            f"v_th_base={self.v_th_base}, target_rate={self.target_spike_rate})"
        )
