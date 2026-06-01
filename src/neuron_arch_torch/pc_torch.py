"""
PyTorch版本：PC-Neuron with True Free Energy Computation
=========================================================

改进：
1. 真正的自由能计算和监控
2. GPU支持
3. 批处理
4. 自动微分

Mathematical Foundation:
----------------------
自由能原理（Friston, 2010）：

F = D_KL[q(z)||p(z|x)] - E_q[ln p(x|z)]

对于高斯近似后验 q(z) = N(μ, σ²)：
F = 0.5 * Σ[(μ - μ_prior)²/σ²_prior + σ²/σ²_prior - ln(σ²/σ²_prior) - 1]
    + 0.5 * Σ[(x - μ_pred)²/σ²_obs]

最小化F等价于：
1. 最小化预测误差
2. 保持后验与先验接近（正则化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class PredictiveCodingLayer(nn.Module):
    """
    预测编码层，实现真正的自由能最小化。
    
    Parameters
    ----------
    n_state : int
        状态维度
    n_pred : int
        预测维度
    lr : float
        学习率
    sigma_prior : float
        先验方差
    sigma_obs : float
        观测方差
    """
    
    def __init__(
        self,
        n_state: int,
        n_pred: int,
        lr: float = 0.01,
        sigma_prior: float = 1.0,
        sigma_obs: float = 0.1,
        device: str = 'auto'
    ):
        super().__init__()
        
        self.n_state = n_state
        self.n_pred = n_pred
        self.lr = lr
        self.sigma_prior = sigma_prior
        self.sigma_obs = sigma_obs
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.W = nn.Parameter(
            torch.randn(n_pred, n_state, device=self.device) * np.sqrt(2.0 / n_state) * 0.1
        )
        
        self.register_buffer('mu', torch.zeros(n_state, device=self.device))
        self.register_buffer('sigma', torch.ones(n_state, device=self.device))
        
        self.register_buffer('mu_prior', torch.zeros(n_state, device=self.device))
        
        self.free_energy_history: List[float] = []
        self.prediction_error_history: List[float] = []
        
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        生成预测。
        
        Parameters
        ----------
        state : Tensor
            状态，形状 (batch, n_state) 或 (n_state,)
        
        Returns
        -------
        prediction : Tensor
            预测
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        linear = F.linear(state, self.W)
        
        residual = state[:, :self.n_pred] if state.size(1) >= self.n_pred else torch.zeros(state.size(0), self.n_pred, device=self.device)
        
        prediction = self._layer_norm(residual + linear)
        
        return prediction
    
    def compute_free_energy(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算自由能及其分量。
        
        Parameters
        ----------
        mu : Tensor
            后验均值
        sigma : Tensor
            后验方差
        prediction : Tensor
            预测值
        target : Tensor
            目标值
        
        Returns
        -------
        free_energy : Tensor
            总自由能
        components : Dict
            自由能各分量
        """
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(0)
        
        kl_term = 0.5 * torch.sum(
            (mu - self.mu_prior.unsqueeze(0))**2 / self.sigma_prior**2 +
            sigma**2 / self.sigma_prior**2 -
            torch.log(sigma**2 / self.sigma_prior**2 + 1e-8) -
            1,
            dim=1
        )
        
        prediction_error = target - prediction
        pe_term = 0.5 * torch.sum(
            prediction_error**2 / self.sigma_obs**2,
            dim=1
        )
        
        free_energy = kl_term + pe_term
        
        components = {
            'kl_divergence': kl_term.mean(),
            'prediction_error_term': pe_term.mean(),
            'total': free_energy.mean()
        }
        
        return free_energy.mean(), components
    
    def local_learn(
        self,
        state: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[float, Dict[str, float]]:
        """
        局部学习，最小化自由能。
        
        Parameters
        ----------
        state : Tensor
            输入状态
        target : Tensor
            目标预测
        
        Returns
        -------
        grad_norm : float
            梯度范数
        metrics : Dict
            学习指标
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        
        state = state.to(self.device)
        target = target.to(self.device)
        
        prediction = self.predict(state)
        
        free_energy, fe_components = self.compute_free_energy(
            self.mu, self.sigma, prediction, target
        )
        
        self.free_energy_history.append(free_energy.item())
        
        epsilon = target - prediction
        epsilon = torch.clamp(epsilon, -0.5, 0.5)
        
        state_norm = state / (torch.norm(state, dim=1, keepdim=True) + 1e-6)
        
        grad = torch.bmm(epsilon.unsqueeze(2), state_norm.unsqueeze(1)).mean(dim=0)
        grad_norm = torch.norm(grad).item()
        
        if grad_norm > 0.5:
            grad = grad * (0.5 / grad_norm)
        
        with torch.no_grad():
            self.W += self.lr * grad
        
        pe_mse = F.mse_loss(prediction, target).item()
        self.prediction_error_history.append(pe_mse)
        
        metrics = {
            'free_energy': free_energy.item(),
            'kl_divergence': fe_components['kl_divergence'].item(),
            'prediction_error_term': fe_components['prediction_error_term'].item(),
            'grad_norm': grad_norm,
            'prediction_mse': pe_mse
        }
        
        return grad_norm, metrics
    
    def _layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        """层归一化"""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        return (x - mean) / std
    
    def get_free_energy_trend(self) -> Dict[str, float]:
        """获取自由能趋势"""
        if len(self.free_energy_history) < 10:
            return {'trend': 0.0, 'current': 0.0}
        
        recent = self.free_energy_history[-50:]
        trend = (recent[-1] - recent[0]) / len(recent)
        
        return {
            'trend': trend,
            'current': recent[-1],
            'mean': np.mean(recent)
        }
    
    def reset(self) -> None:
        """重置状态"""
        self.mu.zero_()
        self.sigma.fill_(1.0)
        self.free_energy_history = []
        self.prediction_error_history = []


class TwoStagePCTrainer:
    """
    两阶段训练器。
    
    Stage 1: 离线预训练，使用任务监督信号
    Stage 2: 在线适应，使用局部学习规则
    """
    
    def __init__(
        self,
        pc_layer: PredictiveCodingLayer,
        pretrain_lr: float = 0.01,
        online_lr: float = 0.005
    ):
        self.pc = pc_layer
        self.pretrain_lr = pretrain_lr
        self.online_lr = online_lr
        
        self.optimizer = torch.optim.Adam(
            [pc_layer.W],
            lr=pretrain_lr
        )
        
        self.stage = 'pretrain'
        
    def pretrain_step(
        self,
        state: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        预训练步骤（使用反向传播）。
        """
        self.stage = 'pretrain'
        self.pc.train()
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        
        state = state.to(self.pc.device)
        target = target.to(self.pc.device)
        
        self.optimizer.zero_grad()
        
        prediction = self.pc.predict(state)
        loss = F.mse_loss(prediction, target)
        
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'stage': 'pretrain'
        }
    
    def online_step(
        self,
        state: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        在线适应步骤（使用局部学习）。
        """
        self.stage = 'online'
        self.pc.eval()
        
        original_lr = self.pc.lr
        self.pc.lr = self.online_lr
        
        grad_norm, metrics = self.pc.local_learn(state, target)
        
        self.pc.lr = original_lr
        
        metrics['stage'] = 'online'
        return metrics
    
    def switch_to_online(self) -> None:
        """切换到在线学习模式"""
        self.stage = 'online'
        self.pc.eval()
