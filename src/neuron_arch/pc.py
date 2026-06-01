"""
PC-Neuron: Predictive Coding Neuron
===================================

Mathematical Foundation:
----------------------
Based on the Free Energy Principle (Friston, 2010):

    F = D_KL[q(z)||p(z|x)] - E_q[ln p(x|z)]
    
Where:
- q(z): Approximate posterior over latent states
- p(z|x): True posterior
- p(x|z): Likelihood (generative model)

Minimizing F is equivalent to:
1. Minimizing prediction errors
2. Maximizing model evidence (ELBO)

Local Learning Rule:
-------------------
The key insight is that prediction errors can be computed locally:

    ε = target - prediction
    
Gradient descent on F:
    ∂F/∂W = -ε · z^T
    
This is biologically plausible:
- No backpropagation through layers
- Error computed locally at each neuron
- Learning uses Hebbian-like product of pre- and post-synaptic activity

Two-Stage Training (Key Innovation):
-----------------------------------
Problem: Local error ε = z - μ is disconnected from task error (x - x̂)

Solution:
1. Offline Pretraining: Use task supervision to align representations
   - Train with backpropagation on task loss L = Σ(x - x̂)²
   - Establishes correct mapping from z to output

2. Online Adaptation: Use local learning for fine-tuning
   - ΔW = -η · ε · z^T
   - Maintains plasticity for continual learning

Residual Connection + Layer Normalization:
-----------------------------------------
To prevent gradient vanishing in deep networks:

    output = LayerNorm(x + W·z)
    
Where LayerNorm is:
    LN(x) = γ · (x - μ) / σ + β
    
This ensures:
- Stable gradient flow
- Robust to scale changes
- Similar to Transformer architecture

Biological Analogy:
------------------
Predictive coding in cortical hierarchies:
- Each layer predicts the activity of the layer below
- Prediction errors propagate upward
- Predictions propagate downward
- Learning minimizes prediction errors at all levels
"""

import numpy as np
from typing import Optional


class ResidualPCLayer:
    """
    Predictive Coding Layer with Residual Connection.
    
    Parameters
    ----------
    n_state : int
        Dimension of state input
    n_pred : int
        Dimension of prediction output
    lr : float, default=0.005
        Learning rate
    
    Attributes
    ----------
    W : ndarray
        Prediction weights
    mu : ndarray
        Current prediction
    grad_norm_history : list
        History of gradient norms for monitoring
    target_grad_norm : float
        Target gradient norm for clipping
    
    Examples
    --------
    >>> pc = ResidualPCLayer(n_state=256, n_pred=256)
    >>> prediction = pc.predict(state)
    >>> grad_norm = pc.local_learn(prev_state, target)
    """
    
    def __init__(
        self,
        n_state: int,
        n_pred: int,
        lr: float = 0.005
    ):
        self.n_state = n_state
        self.n_pred = n_pred
        self.lr = lr
        
        self.W = np.random.randn(n_pred, n_state) * np.sqrt(2.0 / n_state) * 0.1
        self.mu = np.zeros(n_pred)
        
        self.grad_norm_history: list = []
        self.target_grad_norm = 0.5
        
    def layer_norm(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.
        
        Parameters
        ----------
        x : ndarray
            Input vector
        
        Returns
        -------
        ndarray
            Normalized output
        """
        mean = np.mean(x)
        std = np.std(x) + 1e-6
        return (x - mean) / std
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Generate prediction from state.
        
        Parameters
        ----------
        state : ndarray
            Input state
        
        Returns
        -------
        ndarray
            Prediction (layer-normalized residual)
        """
        linear = self.W @ state
        residual_input = state[:self.n_pred] if len(state) >= self.n_pred else np.zeros(self.n_pred)
        self.mu = self.layer_norm(residual_input + linear)
        return self.mu
    
    def local_learn(
        self,
        prev_state: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        Local learning rule.
        
        Parameters
        ----------
        prev_state : ndarray
            Previous state (input)
        target : ndarray
            Target prediction
        
        Returns
        -------
        float
            Gradient norm (for monitoring)
        """
        epsilon = target - self.mu
        epsilon = np.clip(epsilon, -0.3, 0.3)
        
        prev_state_norm = prev_state / (np.linalg.norm(prev_state) + 1e-6)
        
        grad = np.outer(epsilon, prev_state_norm)
        grad_norm = np.linalg.norm(grad)
        self.grad_norm_history.append(grad_norm)
        if len(self.grad_norm_history) > 100:
            self.grad_norm_history.pop(0)
        
        if grad_norm > self.target_grad_norm:
            grad = grad * (self.target_grad_norm / grad_norm)
        
        adaptive_lr = self.lr / (1.0 + 0.5 * grad_norm)
        self.W += adaptive_lr * grad
        
        return grad_norm
    
    def get_grad_norm(self) -> float:
        """Get average gradient norm."""
        if len(self.grad_norm_history) == 0:
            return 0.0
        return np.mean(self.grad_norm_history)
    
    def __repr__(self) -> str:
        return f"ResidualPCLayer(n_state={self.n_state}, n_pred={self.n_pred}, lr={self.lr})"
