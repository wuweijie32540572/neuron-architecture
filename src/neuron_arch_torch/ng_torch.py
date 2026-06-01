"""
PyTorch版本：NG-Neuron with Full Dynamic Neuromodulation
=========================================================

改进：
1. 所有神经调质都有动态更新
2. 基于神经科学的正确模型
3. GPU支持

Neuroscience Foundation:
----------------------
多巴胺 (DA):
    - 奖励预测误差: DA = R_actual - R_predicted
    - 控制学习率和探索/利用

血清素 (5HT):
    - 与等待时间、耐心相关
    - 高5HT: 愿意等待，利用
    - 低5HT: 冲动，探索
    - 更新: 5HT = f(reward_history, waiting_cost)

乙酰胆碱 (ACh):
    - 注意和学习控制
    - 高ACh: 学习模式，高可塑性
    - 低ACh: 巩固模式，低可塑性
    - 更新: ACh = f(novelty, uncertainty)

去甲肾上腺素 (NE):
    - 唤醒和增益控制
    - 高NE: 高增益，对输入敏感
    - 低NE: 低增益，忽略噪声
    - 更新: NE = f(surprise, arousal)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np


class DynamicNeuromodulator(nn.Module):
    """
    动态神经调质系统，所有调质都有基于神经科学的动态更新。
    
    Parameters
    ----------
    base_lr : float
        基础学习率
    device : str
        设备
    """
    
    def __init__(
        self,
        base_lr: float = 0.01,
        device: str = 'auto'
    ):
        super().__init__()
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.base_lr = base_lr
        
        self.register_buffer('da', torch.tensor(0.5, device=self.device))
        self.register_buffer('serotonin', torch.tensor(0.5, device=self.device))
        self.register_buffer('ach', torch.tensor(0.5, device=self.device))
        self.register_buffer('ne', torch.tensor(0.5, device=self.device))
        
        self.register_buffer('reward_prediction', torch.tensor(0.5, device=self.device))
        self.register_buffer('reward_history_mean', torch.tensor(0.5, device=self.device))
        self.register_buffer('novelty_estimate', torch.tensor(0.5, device=self.device))
        self.register_buffer('arousal_baseline', torch.tensor(0.5, device=self.device))
        
        self.step_count = 0
        self.reward_history: list = []
        self.surprise_history: list = []
        
        self.da_tau = 0.9
        self.serotonin_tau = 0.95
        self.ach_tau = 0.9
        self.ne_tau = 0.85
        
    def update_dopamine(self, reward: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新多巴胺水平（奖励预测误差）。
        
        基于Schultz et al. (1997)的发现：
        - 意外奖励: DA激增
        - 预期奖励: DA不变
        - 意外惩罚: DA下降
        
        Parameters
        ----------
        reward : Tensor
            当前奖励
        
        Returns
        -------
        rpe : Tensor
            奖励预测误差
        da_new : Tensor
            新的DA水平
        """
        reward = reward.to(self.device)
        
        rpe = reward - self.reward_prediction
        
        self.reward_prediction = self.da_tau * self.reward_prediction + (1 - self.da_tau) * reward
        
        da_target = 0.5 + 0.4 * torch.tanh(rpe * 2.0)
        self.da = self.da_tau * self.da + (1 - self.da_tau) * da_target
        self.da = torch.clamp(self.da, 0.1, 0.9)
        
        return rpe, self.da
    
    def update_serotonin(self, reward: torch.Tensor, waiting_cost: float = 0.0) -> torch.Tensor:
        """
        更新血清素水平（耐心和等待）。
        
        基于Daw et al.的研究：
        - 高平均奖励: 高5HT（愿意等待）
        - 高等待成本: 低5HT（冲动）
        
        Parameters
        ----------
        reward : Tensor
            当前奖励
        waiting_cost : float
            等待成本
        
        Returns
        -------
        serotonin_new : Tensor
            新的5HT水平
        """
        reward = reward.to(self.device)
        
        self.reward_history.append(reward.item())
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        
        self.reward_history_mean = torch.tensor(
            np.mean(self.reward_history),
            device=self.device
        )
        
        serotonin_target = 0.3 + 0.5 * torch.sigmoid(
            (self.reward_history_mean - 0.5) * 4.0 - waiting_cost
        )
        
        self.serotonin = self.serotonin_tau * self.serotonin + (1 - self.serotonin_tau) * serotonin_target
        self.serotonin = torch.clamp(self.serotonin, 0.1, 0.9)
        
        return self.serotonin
    
    def update_acetylcholine(
        self,
        prediction_error: torch.Tensor,
        novelty: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        更新乙酰胆碱水平（注意和学习）。
        
        基于Hasselmo的研究：
        - 高预测误差/新颖性: 高ACh（学习模式）
        - 低预测误差: 低ACh（巩固模式）
        
        Parameters
        ----------
        prediction_error : Tensor
            预测误差
        novelty : Tensor, optional
            新颖性信号
        
        Returns
        -------
        ach_new : Tensor
            新的ACh水平
        """
        prediction_error = prediction_error.to(self.device)
        
        pe_magnitude = torch.abs(prediction_error).mean()
        
        if novelty is not None:
            novelty = novelty.to(self.device)
            self.novelty_estimate = 0.9 * self.novelty_estimate + 0.1 * novelty
        
        ach_target = 0.2 + 0.6 * torch.sigmoid(
            pe_magnitude * 3.0 + self.novelty_estimate * 2.0 - 1.0
        )
        
        self.ach = self.ach_tau * self.ach + (1 - self.ach_tau) * ach_target
        self.ach = torch.clamp(self.ach, 0.1, 0.9)
        
        return self.ach
    
    def update_norepinephrine(
        self,
        surprise: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        更新去甲肾上腺素水平（唤醒和增益）。
        
        基于Aston-Jones et al.的研究：
        - 高惊讶/不确定性: 高NE（高唤醒）
        - 低惊讶: 低NE（放松）
        
        Parameters
        ----------
        surprise : Tensor
            惊讶信号
        uncertainty : Tensor, optional
            不确定性
        
        Returns
        -------
        ne_new : Tensor
            新的NE水平
        """
        surprise = surprise.to(self.device)
        
        self.surprise_history.append(surprise.item() if surprise.numel() == 1 else surprise.mean().item())
        if len(self.surprise_history) > 100:
            self.surprise_history.pop(0)
        
        surprise_deviation = abs(float(np.mean(self.surprise_history[-20:])) - float(self.arousal_baseline.item()))
        
        ne_target = 0.3 + 0.5 * torch.tanh(torch.tensor(surprise_deviation * 4.0, device=self.device))
        
        if uncertainty is not None:
            uncertainty = uncertainty.to(self.device)
            ne_target = ne_target + 0.2 * torch.tanh(uncertainty)
        
        self.ne = self.ne_tau * self.ne + (1 - self.ne_tau) * ne_target
        self.ne = torch.clamp(self.ne, 0.1, 0.9)
        
        return self.ne
    
    def update_all(
        self,
        reward: torch.Tensor,
        prediction_error: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
        waiting_cost: float = 0.0,
        novelty: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        更新所有神经调质。
        
        Parameters
        ----------
        reward : Tensor
            奖励
        prediction_error : Tensor
            预测误差
        surprise : Tensor, optional
            惊讶信号
        waiting_cost : float
            等待成本
        novelty : Tensor, optional
            新颖性
        uncertainty : Tensor, optional
            不确定性
        
        Returns
        -------
        updates : Dict
            所有更新后的值
        """
        rpe, da = self.update_dopamine(reward)
        serotonin = self.update_serotonin(reward, waiting_cost)
        ach = self.update_acetylcholine(prediction_error, novelty)
        
        if surprise is None:
            surprise = torch.abs(prediction_error)
        ne = self.update_norepinephrine(surprise, uncertainty)
        
        self.step_count += 1
        
        return {
            'da': da,
            'serotonin': serotonin,
            'ach': ach,
            'ne': ne,
            'rpe': rpe
        }
    
    def compute_gate(self) -> torch.Tensor:
        """
        计算门控信号。
        
        使用加权和（非乘积，避免过于保守）：
        gate = w_DA·DA + w_5HT·5HT + w_ACh·ACh + w_NE·NE
        
        Returns
        -------
        gate : Tensor
            门控值
        """
        w_da = 0.35
        w_serotonin = 0.2
        w_ach = 0.25
        w_ne = 0.2
        
        gate = w_da * self.da + w_serotonin * self.serotonin + w_ach * self.ach + w_ne * self.ne
        
        return torch.clamp(gate, 0.1, 0.9)
    
    def compute_effective_lr(self) -> torch.Tensor:
        """
        计算有效学习率。
        
        η(t) = η_base × gate × decay
        
        其中decay由DA控制：
        - 高DA: 慢衰减（探索）
        - 低DA: 快衰减（利用）
        
        Returns
        -------
        effective_lr : Tensor
        """
        decay = 1.0 + self.step_count * (1.0 - self.da.item()) * 0.005
        gate = self.compute_gate()
        
        effective_lr = self.base_lr * gate * 2.0 / decay
        
        return torch.clamp(effective_lr, torch.tensor(0.001, device=self.device), torch.tensor(0.05, device=self.device))
    
    def get_state(self) -> Dict[str, float]:
        """获取当前状态"""
        return {
            'da': self.da.item(),
            'serotonin': self.serotonin.item(),
            'ach': self.ach.item(),
            'ne': self.ne.item(),
            'gate': self.compute_gate().item(),
            'effective_lr': self.compute_effective_lr().item(),
            'reward_prediction': self.reward_prediction.item()
        }
    
    def reset(self) -> None:
        """重置状态"""
        self.da.fill_(0.5)
        self.serotonin.fill_(0.5)
        self.ach.fill_(0.5)
        self.ne.fill_(0.5)
        self.reward_prediction.fill_(0.5)
        self.reward_history_mean.fill_(0.5)
        self.novelty_estimate.fill_(0.5)
        self.arousal_baseline.fill_(0.5)
        self.step_count = 0
        self.reward_history = []
        self.surprise_history = []
