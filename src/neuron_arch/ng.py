"""
NG-Neuron: Neuromodulator-Gated Learning
=======================================

Neuroscience Foundation:
----------------------
The brain uses neuromodulators to dynamically regulate learning and behavior:

1. Dopamine (DA):
   - Reward prediction error (Schultz et al., 1997)
   - Signals unexpected rewards
   - Controls learning rate and motivation
   - DA = R_actual - R_predicted

2. Serotonin (5HT):
   - Mood and patience
   - Controls explore/exploit tradeoff
   - High 5HT: patience, exploitation
   - Low 5HT: impulsivity, exploration

3. Acetylcholine (ACh):
   - Attention and learning
   - Controls synaptic plasticity
   - High ACh: learning mode
   - Low ACh: consolidation mode

4. Norepinephrine (NE):
   - Arousal and vigilance
   - Controls neural gain
   - High NE: heightened sensitivity
   - Low NE: relaxed state

Gating Mechanism (Key Innovation):
---------------------------------
We use weighted sum gating (not product, which is too conservative):

    gate = w_DA·DA + w_5HT·5HT + w_ACh·ACh + w_NE·NE

Where weights are:
- w_DA = 0.4 (dominant, reward-driven)
- w_5HT = 0.2
- w_ACh = 0.2
- w_NE = 0.2

Effective Learning Rate:
-----------------------
Adaptive learning rate with decay:

    η(t) = η_base × gate × 2.0 / (1 + t·(1-DA))

Properties:
- High DA: Slow decay, sustained exploration
- Low DA: Fast decay, quick stabilization
- gate: Overall modulation strength

Explore/Exploit Tradeoff:
------------------------
DA controls the balance:

    High DA (exploration):
    - Learning rate decays slowly
    - Maintains plasticity
    - Searches for better solutions
    
    Low DA (exploitation):
    - Learning rate decays quickly
    - Stabilizes current solution
    - Fine-tunes existing knowledge

Biological Analogy:
------------------
Phasic DA signals (Schultz):
- Unexpected reward: DA spike → Increase learning
- Expected reward: No DA change → Normal learning
- Worse than expected: DA dip → Decrease learning

This implements:
- Temporal difference learning
- Adaptive step size
- Reward-modulated plasticity
"""

import numpy as np


class NormalizedNG:
    """
    Neuromodulator-Gated Learning System.
    
    Parameters
    ----------
    base_lr : float, default=0.01
        Base learning rate
    
    Attributes
    ----------
    da : float
        Dopamine level [0, 1]
    serotonin : float
        Serotonin level [0, 1]
    ach : float
        Acetylcholine level [0, 1]
    ne : float
        Norepinephrine level [0, 1]
    step_count : int
        Number of learning steps
    reward_prediction : float
        Predicted reward (for RPE calculation)
    """
    
    def __init__(self, base_lr: float = 0.01):
        self.base_lr = base_lr
        
        self.da = 0.5
        self.serotonin = 0.5
        self.ach = 0.5
        self.ne = 0.5
        
        self.step_count = 0
        self.reward_prediction = 0.5
        
    def compute_gate(self) -> float:
        """
        Compute gating signal from neuromodulators.
        
        Returns
        -------
        float
            Gate value in [0.1, 0.9]
        """
        weights = np.array([0.4, 0.2, 0.2, 0.2])
        signals = np.array([self.da, self.serotonin, self.ach, self.ne])
        gate = np.dot(weights, signals)
        return np.clip(gate, 0.1, 0.9)
    
    def compute_effective_lr(self) -> float:
        """
        Compute effective learning rate.
        
        Returns
        -------
        float
            Effective learning rate in [0.002, 0.02]
        """
        decay = 1.0 + self.step_count * (1.0 - self.da) * 0.005
        gate = self.compute_gate()
        effective_lr = self.base_lr * gate * 2.0 / decay
        return np.clip(effective_lr, 0.002, 0.02)
    
    def update_from_reward(self, reward: float) -> float:
        """
        Update neuromodulators based on reward.
        
        Parameters
        ----------
        reward : float
            Current reward signal
        
        Returns
        -------
        float
            Reward prediction error
        """
        rpe = reward - self.reward_prediction
        
        self.reward_prediction = 0.9 * self.reward_prediction + 0.1 * reward
        
        self.da = np.clip(0.5 + 0.3 * np.tanh(rpe), 0.1, 0.9)
        
        self.ach = np.clip(0.5 + 0.2 * (1.0 - abs(rpe)), 0.1, 0.9)
        
        self.step_count += 1
        
        return rpe
    
    def set_state(
        self,
        da: float = None,
        serotonin: float = None,
        ach: float = None,
        ne: float = None
    ) -> None:
        """
        Manually set neuromodulator levels.
        
        Parameters
        ----------
        da, serotonin, ach, ne : float, optional
            Neuromodulator levels [0, 1]
        """
        if da is not None:
            self.da = np.clip(da, 0.0, 1.0)
        if serotonin is not None:
            self.serotonin = np.clip(serotonin, 0.0, 1.0)
        if ach is not None:
            self.ach = np.clip(ach, 0.0, 1.0)
        if ne is not None:
            self.ne = np.clip(ne, 0.0, 1.0)
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.da = 0.5
        self.serotonin = 0.5
        self.ach = 0.5
        self.ne = 0.5
        self.step_count = 0
        self.reward_prediction = 0.5
    
    def __repr__(self) -> str:
        return (
            f"NormalizedNG(base_lr={self.base_lr}, "
            f"DA={self.da:.2f}, 5HT={self.serotonin:.2f}, "
            f"ACh={self.ach:.2f}, NE={self.ne:.2f})"
        )
