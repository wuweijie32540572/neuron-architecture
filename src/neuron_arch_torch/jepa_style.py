"""
JEPA-Style Predictive Coding Layer
==================================

基于LeCun的JEPA (Joint Embedding Predictive Architecture) 思想实现。

关键组件：
1. 编码器 + 目标编码器（EMA更新）
2. 预测器
3. 表征坍缩防护（SIGReg）

参考：
- LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence"
- Assran, M., et al. (2023). "I-JEPA" CVPR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class SIGReg(nn.Module):
    """
    Simplified InfoMax Regularization for preventing representation collapse.
    
    基于V-JEPA的实现，惩罚协方差矩阵的非对角元素。
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z1, z2 : Tensor
            表征，形状 (batch, dim)
        
        Returns
        -------
        loss : Tensor
            正则化损失
        """
        batch_size = z1.size(0)
        
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        
        cov = (z1.T @ z2) / (batch_size - 1 + self.eps)
        
        if cov.size(0) > 1:
            off_diag = cov.flatten()[:-1].view(cov.size(0) - 1, cov.size(0) + 1)[:, 1:]
            loss = off_diag.pow(2).sum()
        else:
            loss = torch.tensor(0.0, device=z1.device)
        
        return loss


class JEPAPredictiveCoder(nn.Module):
    """
    JEPA-Style Predictive Coding Layer.
    
    架构：
    - 编码器 E: 输入 -> 表征
    - 目标编码器 Ē: E的EMA
    - 预测器 P: 上下文表征 -> 目标表征
    
    训练：
    1. 编码上下文: z_ctx = E(x_ctx)
    2. 预测目标: z_pred = P(z_ctx)
    3. 目标表征: z_tgt = Ē(x_tgt) [不计算梯度]
    4. 损失: ||z_pred - z_tgt||² + λ * SIGReg
    
    参数：
    ----------
    input_dim : int
        输入维度
    hidden_dim : int
        隐藏层维度
    latent_dim : int
        潜在表征维度
    momentum : float
        EMA动量（目标编码器更新）
    sigreg_weight : float
        SIGReg权重
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        momentum: float = 0.996,
        sigreg_weight: float = 0.1,
        device: str = 'auto'
    ):
        super().__init__()
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.momentum = momentum
        self.sigreg_weight = sigreg_weight
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        ).to(self.device)
        
        self.target_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        ).to(self.device)
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        ).to(self.device)
        
        self.sigreg = SIGReg()
        
        self._init_target_encoder()
        
        self.step_count = 0
    
    def _init_target_encoder(self):
        """初始化目标编码器为编码器的副本"""
        for param, target_param in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_param.data.copy_(param.data)
    
    @torch.no_grad()
    def update_target_encoder(self):
        """EMA更新目标编码器"""
        for param, target_param in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_param.data = self.momentum * target_param.data + (1 - self.momentum) * param.data
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入"""
        x = x.to(self.device)
        return self.encoder(x)
    
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """目标编码（不计算梯度）"""
        x = x.to(self.device)
        with torch.no_grad():
            return self.target_encoder(x)
    
    def predict(self, z_context: torch.Tensor) -> torch.Tensor:
        """从上下文表征预测目标表征"""
        return self.predictor(z_context)
    
    def forward(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播。
        
        参数：
        ----------
        x_context : Tensor
            上下文输入
        x_target : Tensor
            目标输入
        
        返回：
        ----------
        loss : Tensor
            总损失
        metrics : dict
            各项指标
        """
        x_context = x_context.to(self.device)
        x_target = x_target.to(self.device)
        
        z_context = self.encode(x_context)
        
        z_pred = self.predict(z_context)
        
        z_target = self.encode_target(x_target)
        
        pred_loss = F.mse_loss(z_pred, z_target)
        
        sigreg_loss = self.sigreg(z_context, z_target)
        
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        
        metrics = {
            'prediction_loss': pred_loss.item(),
            'sigreg_loss': sigreg_loss.item(),
            'total_loss': loss.item(),
            'z_context_norm': z_context.norm().item(),
            'z_target_norm': z_target.norm().item()
        }
        
        return loss, metrics
    
    def train_step(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> dict:
        """
        训练步骤。
        
        参数：
        ----------
        x_context, x_target : Tensor
            输入
        optimizer : Optimizer
            优化器
        
        返回：
        ----------
        metrics : dict
            训练指标
        """
        self.train()
        optimizer.zero_grad()
        
        loss, metrics = self.forward(x_context, x_target)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        self.update_target_encoder()
        
        self.step_count += 1
        
        return metrics
    
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """获取输入的表征"""
        self.eval()
        with torch.no_grad():
            return self.encode(x)


class JEPATrainer:
    """
    JEPA训练器，支持自监督学习。
    """
    
    def __init__(
        self,
        model: JEPAPredictiveCoder,
        lr: float = 0.001,
        weight_decay: float = 0.05
    ):
        self.model = model
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=lr * 0.01
        )
        
        self.history = []
    
    def create_masks(
        self,
        batch_size: int,
        input_dim: int,
        context_ratio: float = 0.6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建上下文和目标的mask。
        
        对于向量输入，随机选择一部分作为上下文，其余作为目标。
        """
        n_context = int(input_dim * context_ratio)
        
        context_mask = torch.zeros(batch_size, input_dim, device=self.model.device)
        target_mask = torch.zeros(batch_size, input_dim, device=self.model.device)
        
        for i in range(batch_size):
            perm = torch.randperm(input_dim)
            context_idx = perm[:n_context]
            target_idx = perm[n_context:]
            
            context_mask[i, context_idx] = 1.0
            target_mask[i, target_idx] = 1.0
        
        return context_mask, target_mask
    
    def train_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        context_ratio: float = 0.6
    ) -> dict:
        """
        训练一个epoch。
        
        参数：
        ----------
        data_loader : DataLoader
            数据加载器
        context_ratio : float
            上下文比例
        
        返回：
        ----------
        epoch_metrics : dict
            epoch指标
        """
        self.model.train()
        
        all_metrics = []
        
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(self.model.device)
            
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            batch_size, input_dim = x.shape
            
            context_mask, target_mask = self.create_masks(
                batch_size, input_dim, context_ratio
            )
            
            x_context = x * context_mask
            x_target = x * target_mask
            
            metrics = self.model.train_step(x_context, x_target, self.optimizer)
            all_metrics.append(metrics)
        
        self.scheduler.step()
        
        epoch_metrics = {
            'prediction_loss': np.mean([m['prediction_loss'] for m in all_metrics]),
            'sigreg_loss': np.mean([m['sigreg_loss'] for m in all_metrics]),
            'total_loss': np.mean([m['total_loss'] for m in all_metrics]),
            'lr': self.scheduler.get_last_lr()[0]
        }
        
        self.history.append(epoch_metrics)
        
        return epoch_metrics


if __name__ == '__main__':
    print("JEPA-Style Predictive Coding Layer")
    print("="*50)
    
    model = JEPAPredictiveCoder(
        input_dim=64,
        hidden_dim=128,
        latent_dim=32
    )
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"设备: {model.device}")
    
    x_context = torch.randn(8, 64)
    x_target = torch.randn(8, 64)
    
    loss, metrics = model(x_context, x_target)
    
    print(f"\n前向传播测试:")
    print(f"  预测损失: {metrics['prediction_loss']:.4f}")
    print(f"  SIGReg损失: {metrics['sigreg_loss']:.4f}")
    print(f"  总损失: {metrics['total_loss']:.4f}")
    
    print("\n✓ JEPA-Style实现完成")
