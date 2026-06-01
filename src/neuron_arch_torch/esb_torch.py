"""
PyTorch版本：ESB-Neuron with True Orthogonal Patterns
======================================================

改进：
1. 使用Gram-Schmidt正交化确保真正正交
2. GPU支持
3. 批处理

Mathematical Foundation:
----------------------
符号接地问题（Harnad, 1990）：
符号必须通过感知经验获得意义，而非仅通过符号间关系。

正交模式要求：
⟨p_i, p_j⟩ = δ_ij

使用Gram-Schmidt正交化：
u_1 = v_1
u_2 = v_2 - ⟨v_2, u_1⟩/⟨u_1, u_1⟩ · u_1
...
u_n = v_n - Σ⟨v_n, u_i⟩/⟨u_i, u_i⟩ · u_i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


def gram_schmidt(vectors: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Gram-Schmidt正交化。
    
    Parameters
    ----------
    vectors : Tensor
        输入向量，形状 (n_vectors, dim)
    normalize : bool
        是否归一化
    
    Returns
    -------
    orthogonal : Tensor
        正交化后的向量
    """
    n_vectors, dim = vectors.shape
    orthogonal = torch.zeros_like(vectors)
    
    for i in range(n_vectors):
        v = vectors[i].clone()
        
        for j in range(i):
            proj = torch.dot(v, orthogonal[j]) / (torch.dot(orthogonal[j], orthogonal[j]) + 1e-8)
            v = v - proj * orthogonal[j]
        
        if normalize:
            norm = torch.norm(v)
            if norm > 1e-8:
                v = v / norm
        
        orthogonal[i] = v
    
    return orthogonal


class ESBNeuronTorch(nn.Module):
    """
    具身-符号桥接层，使用真正正交的模式。
    
    Parameters
    ----------
    n_embodied : int
        具身输入维度
    n_latent : int
        潜在空间维度
    n_symbols : int
        符号数量
    device : str
        设备
    """
    
    def __init__(
        self,
        n_embodied: int,
        n_latent: int,
        n_symbols: int,
        device: str = 'auto'
    ):
        super().__init__()
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.n_embodied = n_embodied
        self.n_latent = n_latent
        self.n_symbols = n_symbols
        
        self.W_encode = nn.Parameter(
            torch.randn(n_latent, n_embodied, device=self.device) * np.sqrt(2.0 / n_embodied) * 0.1
        )
        self.W_decode = nn.Parameter(
            torch.randn(n_symbols, n_latent, device=self.device) * np.sqrt(2.0 / n_latent) * 0.1
        )
        
        self.register_buffer(
            'symbol_embeddings',
            torch.randn(n_symbols, n_latent, device=self.device) * 0.1
        )
        self.register_buffer(
            'grounding_matrix',
            torch.zeros(n_symbols, n_embodied, device=self.device)
        )
        
        self._init_orthogonal_patterns()
        
        self.symbol_labels = [f"concept_{i}" for i in range(n_symbols)]
        
    def _init_orthogonal_patterns(self) -> None:
        """初始化真正正交的模式"""
        if self.n_symbols > self.n_embodied:
            print(f"警告: 符号数({self.n_symbols}) > 维度({self.n_embodied})，无法完全正交")
        
        n = min(self.n_symbols, self.n_embodied)
        
        raw_patterns = torch.randn(self.n_symbols, self.n_embodied, device=self.device)
        
        raw_patterns[:n] = torch.eye(n, self.n_embodied, device=self.device)
        
        orthogonal_patterns = gram_schmidt(raw_patterns, normalize=True)
        
        self.register_buffer('orthogonal_patterns', orthogonal_patterns)
        
        with torch.no_grad():
            for i in range(self.n_symbols):
                self.grounding_matrix[i] = orthogonal_patterns[i]
                latent = torch.tanh(self.W_encode @ orthogonal_patterns[i])
                self.symbol_embeddings[i] = latent
    
    def verify_orthogonality(self) -> Tuple[torch.Tensor, float]:
        """
        验证模式的正交性。
        
        Returns
        -------
        inner_products : Tensor
            内积矩阵
        orthogonality_error : float
            正交性误差（非对角线元素的RMS）
        """
        patterns = self.orthogonal_patterns
        
        inner_products = patterns @ patterns.T
        
        n = patterns.shape[0]
        mask = ~torch.eye(n, dtype=bool, device=self.device)
        off_diagonal = inner_products[mask]
        
        orthogonality_error = torch.sqrt(torch.mean(off_diagonal ** 2)).item()
        
        return inner_products, orthogonality_error
    
    def encode_embodied(self, sensor_input: torch.Tensor) -> torch.Tensor:
        """
        编码具身输入到潜在空间。
        
        Parameters
        ----------
        sensor_input : Tensor
            形状 (batch, n_embodied) 或 (n_embodied,)
        
        Returns
        -------
        latent : Tensor
            潜在表示
        """
        if sensor_input.dim() == 1:
            sensor_input = sensor_input.unsqueeze(0)
        
        sensor_input = sensor_input.to(self.device)
        latent = torch.tanh(F.linear(sensor_input, self.W_encode))
        
        return latent
    
    def decode_symbol(
        self,
        latent: torch.Tensor,
        temperature: float = 3.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解码潜在表示到符号。
        
        Parameters
        ----------
        latent : Tensor
            潜在表示
        temperature : float
            Softmax温度
        
        Returns
        -------
        symbol_ids : Tensor
            符号ID
        probs : Tensor
            符号概率
        confidences : Tensor
            置信度
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        
        latent = latent.to(self.device)
        
        logits = F.linear(latent, self.W_decode)
        probs = F.softmax(logits * temperature, dim=-1)
        
        symbol_ids = torch.argmax(probs, dim=-1)
        confidences = probs[torch.arange(probs.size(0)), symbol_ids]
        
        return symbol_ids, probs, confidences
    
    def ground_symbol(
        self,
        symbol_id: int,
        embodied_example: torch.Tensor,
        strength: float = 0.1
    ) -> None:
        """
        将符号接地到具身经验。
        
        Parameters
        ----------
        symbol_id : int
            符号ID
        embodied_example : Tensor
            具身示例
        strength : float
            学习率
        """
        if embodied_example.dim() == 1:
            embodied_example = embodied_example.unsqueeze(0)
        
        embodied_example = embodied_example.to(self.device)
        
        with torch.no_grad():
            self.grounding_matrix[symbol_id] = (
                (1 - strength) * self.grounding_matrix[symbol_id] +
                strength * embodied_example.mean(dim=0)
            )
            
            latent = self.encode_embodied(embodied_example).mean(dim=0)
            self.symbol_embeddings[symbol_id] = (
                (1 - strength) * self.symbol_embeddings[symbol_id] +
                strength * latent
            )
    
    def retrieve_embodied(self, symbol_id: int) -> torch.Tensor:
        """
        从符号检索具身经验。
        
        Parameters
        ----------
        symbol_id : int
            符号ID
        
        Returns
        -------
        embodied : Tensor
            接地的具身模式
        """
        return self.grounding_matrix[symbol_id]
    
    def forward(
        self,
        sensor_input: torch.Tensor,
        target_symbols: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Parameters
        ----------
        sensor_input : Tensor
            具身输入
        target_symbols : Tensor, optional
            目标符号（用于训练）
        
        Returns
        -------
        symbol_ids : Tensor
        probs : Tensor
        confidences : Tensor
        """
        latent = self.encode_embodied(sensor_input)
        symbol_ids, probs, confidences = self.decode_symbol(latent)
        
        return symbol_ids, probs, confidences
    
    def learn(
        self,
        sensor_input: torch.Tensor,
        target_symbol: int,
        lr: float = 0.1
    ) -> Tuple[float, bool]:
        """
        学习具身-符号映射。
        
        Parameters
        ----------
        sensor_input : Tensor
            具身输入
        target_symbol : int
            目标符号
        lr : float
            学习率
        
        Returns
        -------
        confidence : float
        is_correct : bool
        """
        if sensor_input.dim() == 1:
            sensor_input = sensor_input.unsqueeze(0)
        
        sensor_input = sensor_input.to(self.device)
        
        latent = self.encode_embodied(sensor_input)
        symbol_ids, probs, confidences = self.decode_symbol(latent)
        
        symbol_id = symbol_ids[0].item()
        confidence = confidences[0].item()
        is_correct = (symbol_id == target_symbol)
        
        target_logits = torch.zeros(self.n_symbols, device=self.device)
        target_logits[target_symbol] = 1.0
        
        error = target_logits - probs[0]
        
        grad_decode = torch.outer(error, latent[0])
        grad_decode = torch.clamp(grad_decode, -1.0, 1.0)
        
        with torch.no_grad():
            self.W_decode += lr * grad_decode
        
        grad_encode = self.W_decode.T @ error * (1 - latent[0] ** 2)
        grad_encode = torch.clamp(grad_encode, -1.0, 1.0)
        
        with torch.no_grad():
            self.W_encode += lr * torch.outer(grad_encode, sensor_input[0])
        
        self.ground_symbol(target_symbol, sensor_input[0], strength=0.1)
        
        return confidence, is_correct
    
    def get_orthogonal_patterns(self) -> torch.Tensor:
        """获取正交模式"""
        return self.orthogonal_patterns
    
    def to(self, device):
        """移动到指定设备"""
        super().to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        return self
