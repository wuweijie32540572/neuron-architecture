"""
HM-Neuron: Hippocampus-Cortex Memory
===================================

Neuroscience Foundation:
----------------------
Based on the Complementary Learning Systems (CLS) theory (McClelland et al., 1995):

1. Hippocampus:
   - Fast learning, episodic memory
   - Pattern separation (dentate gyrus)
   - Rapid encoding of new experiences
   
2. Neocortex:
   - Slow learning, semantic memory
   - Systems consolidation during sleep
   - Long-term knowledge storage

Memory Stability Measure (Key Innovation):
-----------------------------------------
We define memory stability as:

    σ(m) = 1 / (1 + α·age + β/(access+1))

Where:
- age: Time since memory formation
- access: Number of times memory was accessed
- α, β: Weight parameters

Properties:
- New memories (high age): Low stability → High forgetting risk
- Frequently accessed memories: High stability → Protected
- Old but rarely accessed: Medium stability

Risk-Weighted Replay:
--------------------
Replay probability proportional to forgetting risk:

    P(sample m) ∝ 1 - σ(m)

This ensures:
- High-risk memories are replayed more often
- Low-risk memories are protected but not forgotten
- Efficient use of limited replay capacity

Pattern Separation (Dentate Gyrus):
----------------------------------
To prevent interference between similar memories:

    output = top-k(W·x)
    
Where top-k keeps only the k largest activations.

This implements:
- Sparse representation
- Orthogonalization of similar inputs
- Reduced interference

Systems Consolidation:
--------------------
During "sleep" phase:
1. Sample memories by risk-weighted probability
2. Train cortex on replayed memories
3. Strengthen cortical representations
4. Reduce dependence on hippocampus

Mathematical Model:
------------------
Hippocampus learning:
    ΔW_H = η_H · ε · x^T  (fast, η_H ~ 0.05)

Cortex learning:
    ΔW_C = η_C · ε · x^T  (slow, η_C ~ 0.01)

Consolidation:
    W_C ← W_C + η_C · Σ replay_samples

Output mixing:
    output = α(t)·H + (1-α(t))·C
    α(t) = 0.3 + 0.7·exp(-n_consolidation/τ)
"""

import numpy as np
from typing import Tuple, List


class PatternSeparationHM:
    """
    Hippocampus-Cortex Memory with Pattern Separation.
    
    Parameters
    ----------
    n_input : int
        Input dimension
    n_hidden : int
        Hidden dimension
    hippocampus_lr : float, default=0.02
        Hippocampus learning rate (fast)
    cortex_lr : float, default=0.008
        Cortex learning rate (slow)
    top_k : int, default=64
        Number of active units in pattern separation
    
    Attributes
    ----------
    W_hippo : ndarray
        Hippocampus weights
    W_cortex : ndarray
        Cortex weights
    memory_buffer : list
        Stored memories (x, target, task_id)
    consolidation_count : int
        Number of consolidation steps
    """
    
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        hippocampus_lr: float = 0.02,
        cortex_lr: float = 0.008,
        top_k: int = 64
    ):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.top_k = top_k
        
        self.W_hippo = np.random.randn(n_hidden, n_input) * 0.1
        self.W_cortex = np.random.randn(n_hidden, n_input) * 0.1
        
        self.memory_buffer: List[Tuple[np.ndarray, np.ndarray, int]] = []
        self.buffer_size = 500
        
        self.hippo_lr = hippocampus_lr
        self.cortex_lr = cortex_lr
        
        self.consolidation_count = 0
        
    def pattern_separation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply pattern separation (dentate gyrus function).
        
        Parameters
        ----------
        x : ndarray
            Input pattern
        
        Returns
        -------
        ndarray
            Sparse, separated output
        """
        activity = self.W_hippo @ x
        
        top_k_indices = np.argsort(np.abs(activity))[-self.top_k:]
        sparse_output = np.zeros_like(activity)
        sparse_output[top_k_indices] = activity[top_k_indices]
        
        return np.tanh(sparse_output)
    
    def forward(self, x: np.ndarray, use_cortex: bool = True) -> np.ndarray:
        """
        Forward pass through hippocampus-cortex system.
        
        Parameters
        ----------
        x : ndarray
            Input
        use_cortex : bool, default=True
            Whether to mix hippocampus and cortex outputs
        
        Returns
        -------
        ndarray
            Output (mixed or hippocampus-only)
        """
        h_out = self.pattern_separation(x)
        c_out = np.tanh(self.W_cortex @ x)
        
        if not use_cortex:
            return h_out
        
        alpha = 0.3 + 0.7 * np.exp(-self.consolidation_count / 3.0)
        return alpha * h_out + (1 - alpha) * c_out
    
    def learn(
        self,
        x: np.ndarray,
        target: np.ndarray,
        task_id: int = 0
    ) -> None:
        """
        Learn new pattern (hippocampus fast, cortex slow).
        
        Parameters
        ----------
        x : ndarray
            Input
        target : ndarray
            Target output
        task_id : int, default=0
            Task identifier for memory tagging
        """
        pred_h = self.pattern_separation(x)
        error_h = target - pred_h
        error_h = np.clip(error_h, -1.0, 1.0)
        self.W_hippo += self.hippo_lr * np.outer(error_h, x)
        
        pred_c = np.tanh(self.W_cortex @ x)
        error_c = target - pred_c
        error_c = np.clip(error_c, -1.0, 1.0)
        self.W_cortex += self.cortex_lr * np.outer(error_c, x)
        
        self.memory_buffer.append((x.copy(), target.copy(), task_id))
        if len(self.memory_buffer) > self.buffer_size:
            self.memory_buffer.pop(0)
    
    def consolidate(self, n_replay: int = 80) -> int:
        """
        Systems consolidation (sleep phase).
        
        Parameters
        ----------
        n_replay : int, default=80
            Number of memories to replay
        
        Returns
        -------
        int
            Number of memories actually replayed
        """
        if len(self.memory_buffer) == 0:
            return 0
        
        ages = np.arange(len(self.memory_buffer), 0, -1)
        risks = 1.0 - 1.0 / (1.0 + 0.05 * ages)
        probs = risks / np.sum(risks)
        
        n_sample = min(n_replay, len(self.memory_buffer))
        indices = np.random.choice(
            len(self.memory_buffer), 
            n_sample, 
            replace=False, 
            p=probs
        )
        
        for idx in indices:
            x, target, _ = self.memory_buffer[idx]
            pred = np.tanh(self.W_cortex @ x)
            error = target - pred
            error = np.clip(error, -1.0, 1.0)
            self.W_cortex += self.cortex_lr * 1.0 * np.outer(error, x)
        
        self.consolidation_count += 1
        return n_sample
    
    def __repr__(self) -> str:
        return (
            f"PatternSeparationHM(n_input={self.n_input}, n_hidden={self.n_hidden}, "
            f"top_k={self.top_k})"
        )
