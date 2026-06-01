"""
ESB-Neuron: Embodied-Symbolic Bridging
=====================================

Philosophy Foundation:
--------------------
The Symbol Grounding Problem (Harnad, 1990):
- Symbols in computers are inherently ungrounded
- They derive meaning only through relation to other symbols
- True meaning requires connection to physical world

Embodied Cognition (Varela et al., 1991):
- Cognition is grounded in bodily experience
- Sensorimotor experience shapes concepts
- "I think therefore I act" → "I act therefore I think"

Architecture:
------------
    Sensor Input → Embodied Encoding → Latent Space → Symbol Decoding → Concept
                         ↓                              ↓
                    Grounding Matrix ←─────────── Symbol Embedding

Orthogonal Input Patterns (Key Innovation):
------------------------------------------
To ensure distinct symbol representations:

    patterns[i, block_i] = 1.0  (block diagonal)
    patterns[i] += noise
    patterns[i] = normalize(patterns[i])

Properties:
- Inner product between different patterns ≈ 0
- Each symbol has unique sensorimotor signature
- Prevents symbol confusion

Grounding Mechanism:
------------------
Binding symbols to embodied experience:

    G[symbol] = (1-α)·G[symbol] + α·sensor_input
    E[symbol] = (1-α)·E[symbol] + α·latent

Where:
- G: Grounding matrix (symbol → sensor pattern)
- E: Embedding matrix (symbol → latent pattern)
- α: Learning rate for grounding updates

Symbol Retrieval:
----------------
From symbol to embodied experience:

    sensor_pattern = G[symbol]
    
This enables:
- Mental simulation
- Imagery
- Sensorimotor predictions

Mathematical Model:
------------------
Embodied encoding:
    latent = tanh(W_encode · sensor_input)

Symbol decoding:
    logits = W_decode · latent
    probs = softmax(logits · τ)
    symbol = argmax(probs)

Confidence:
    confidence = probs[symbol]

High confidence (>0.5) indicates:
- Clear symbol identification
- Strong grounding
- Reliable concept retrieval
"""

import numpy as np
from typing import Tuple, List


class ESBNeuronLayer:
    """
    Embodied-Symbolic Bridging Layer.
    
    Parameters
    ----------
    n_embodied : int
        Dimension of sensorimotor input
    n_latent : int
        Dimension of latent space
    n_symbols : int
        Number of symbols/concepts
    
    Attributes
    ----------
    W_encode : ndarray
        Encoding weights (sensor → latent)
    W_decode : ndarray
        Decoding weights (latent → symbol)
    symbol_embeddings : ndarray
        Symbol embeddings in latent space
    grounding_matrix : ndarray
        Symbol grounding in sensor space
    """
    
    def __init__(
        self,
        n_embodied: int,
        n_latent: int,
        n_symbols: int
    ):
        self.n_embodied = n_embodied
        self.n_latent = n_latent
        self.n_symbols = n_symbols
        
        self.W_encode = np.random.randn(n_latent, n_embodied) * 0.1
        self.W_decode = np.random.randn(n_symbols, n_latent) * 0.1
        
        self.symbol_embeddings = np.random.randn(n_symbols, n_latent) * 0.1
        self.grounding_matrix = np.zeros((n_symbols, n_embodied))
        
        self.symbol_labels = [f"concept_{i}" for i in range(n_symbols)]
        
    def encode_embodied(self, sensor_input: np.ndarray) -> np.ndarray:
        """
        Encode sensorimotor input to latent space.
        
        Parameters
        ----------
        sensor_input : ndarray
            Sensorimotor input
        
        Returns
        -------
        ndarray
            Latent representation
        """
        latent = np.tanh(self.W_encode @ sensor_input)
        return latent
    
    def decode_symbol(
        self,
        latent: np.ndarray
    ) -> Tuple[int, np.ndarray, float]:
        """
        Decode latent representation to symbol.
        
        Parameters
        ----------
        latent : ndarray
            Latent representation
        
        Returns
        -------
        symbol_id : int
            Identified symbol
        probs : ndarray
            Symbol probabilities
        confidence : float
            Confidence of identification
        """
        logits = self.W_decode @ latent
        probs = self._softmax(logits * 3.0)
        
        symbol_id = np.argmax(probs)
        confidence = probs[symbol_id]
        
        return symbol_id, probs, confidence
    
    def ground_symbol(
        self,
        symbol_id: int,
        embodied_example: np.ndarray,
        strength: float = 0.1
    ) -> None:
        """
        Ground symbol to embodied experience.
        
        Parameters
        ----------
        symbol_id : int
            Symbol to ground
        embodied_example : ndarray
            Sensorimotor example
        strength : float, default=0.1
            Learning rate for grounding
        """
        self.grounding_matrix[symbol_id] = (
            (1 - strength) * self.grounding_matrix[symbol_id] + 
            strength * embodied_example
        )
        
        latent = self.encode_embodied(embodied_example)
        self.symbol_embeddings[symbol_id] = (
            (1 - strength) * self.symbol_embeddings[symbol_id] + 
            strength * latent
        )
    
    def retrieve_embodied(self, symbol_id: int) -> np.ndarray:
        """
        Retrieve embodied experience from symbol.
        
        Parameters
        ----------
        symbol_id : int
            Symbol identifier
        
        Returns
        -------
        ndarray
            Grounded sensorimotor pattern
        """
        return self.grounding_matrix[symbol_id]
    
    def compute_symbolic_distance(
        self,
        latent: np.ndarray,
        symbol_id: int
    ) -> float:
        """
        Compute distance between latent and symbol embedding.
        
        Parameters
        ----------
        latent : ndarray
            Latent representation
        symbol_id : int
            Symbol identifier
        
        Returns
        -------
        float
            Euclidean distance
        """
        return np.linalg.norm(latent - self.symbol_embeddings[symbol_id])
    
    def learn(
        self,
        sensor_input: np.ndarray,
        target_symbol: int,
        lr: float = 0.1
    ) -> Tuple[float, bool]:
        """
        Learn embodied-symbolic mapping.
        
        Parameters
        ----------
        sensor_input : ndarray
            Sensorimotor input
        target_symbol : int
            Target symbol
        lr : float, default=0.1
            Learning rate
        
        Returns
        -------
        confidence : float
            Prediction confidence
        is_correct : bool
            Whether prediction matches target
        """
        latent = self.encode_embodied(sensor_input)
        symbol_id, probs, confidence = self.decode_symbol(latent)
        
        is_correct = (symbol_id == target_symbol)
        
        target_logits = np.zeros(self.n_symbols)
        target_logits[target_symbol] = 1.0
        
        error = target_logits - probs
        
        grad_decode = np.outer(error, latent)
        grad_decode = np.clip(grad_decode, -1.0, 1.0)
        self.W_decode += lr * grad_decode
        
        grad_encode = self.W_decode.T @ error * (1 - latent ** 2)
        grad_encode = np.clip(grad_encode, -1.0, 1.0)
        self.W_encode += lr * np.outer(grad_encode, sensor_input)
            
        self.ground_symbol(target_symbol, sensor_input, strength=0.1)
        
        return confidence, is_correct
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def __repr__(self) -> str:
        return (
            f"ESBNeuronLayer(n_embodied={self.n_embodied}, "
            f"n_latent={self.n_latent}, n_symbols={self.n_symbols})"
        )
