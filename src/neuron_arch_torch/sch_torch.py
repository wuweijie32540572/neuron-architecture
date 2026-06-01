"""
PyTorch版本：SCH-Neuron with GPU Support and Temporal Coding
=============================================================

改进：
1. GPU支持
2. 批处理
3. 时序编码（脉冲时间戳）
4. STDP可塑性
5. 自动微分

Physics Foundation:
------------------
膜电位动力学（LIF模型）：
    τ_m · dV/dt = -V + R_m · I(t)

STDP学习规则：
    Δw = A₊·exp(-Δt/τ₊)  if Δt > 0 (post after pre)
    Δw = -A₋·exp(Δt/τ₋)  if Δt < 0 (pre after post)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class SCHNeuronTorch(nn.Module):
    """
    PyTorch版本的SCH神经元，支持GPU、批处理、时序编码和STDP。
    
    Parameters
    ----------
    n_neurons : int
        神经元数量
    tau_m : float
        膜时间常数 (ms)
    v_th_base : float
        基础阈值
    target_spike_rate : float
        目标脉冲率
    enable_stdp : bool
        是否启用STDP
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau_m: float = 5.0,
        v_th_base: float = 0.1,
        target_spike_rate: float = 0.1,
        enable_stdp: bool = True,
        device: str = 'auto'
    ):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.tau_m = tau_m
        self.v_th_base = v_th_base
        self.target_spike_rate = target_spike_rate
        self.enable_stdp = enable_stdp
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.register_buffer('v', torch.zeros(n_neurons, device=self.device))
        self.register_buffer('v_th', torch.ones(n_neurons, device=self.device) * v_th_base)
        self.register_buffer('spike_trace', torch.zeros(n_neurons, device=self.device))
        
        self.spike_times: List[torch.Tensor] = []
        self.current_time = 0.0
        
        if enable_stdp:
            self.stdp_A_plus = 0.01
            self.stdp_A_minus = 0.012
            self.stdp_tau_plus = 20.0
            self.stdp_tau_minus = 20.0
            self.register_buffer('pre_spike_times', torch.zeros(n_neurons, device=self.device))
            self.register_buffer('post_spike_times', torch.zeros(n_neurons, device=self.device))
        
        self.spike_history: List[float] = []
        
    def forward(
        self,
        input_current: torch.Tensor,
        dt: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播，支持批处理。
        
        Parameters
        ----------
        input_current : Tensor
            输入电流，形状 (batch, n_neurons) 或 (n_neurons,)
        dt : float
            时间步长
        
        Returns
        -------
        spikes : Tensor
            脉冲输出 (batch, n_neurons) 或 (n_neurons,)
        continuous : Tensor
            连续表示
        spike_times : Tensor
            脉冲时间戳
        """
        if input_current.dim() == 1:
            input_current = input_current.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = input_current.size(0)
        input_current = input_current.to(self.device)
        
        current_rate = torch.tensor(
            np.mean(self.spike_history[-100:]) if len(self.spike_history) > 10 
            else self.target_spike_rate,
            device=self.device
        )
        rate_error = current_rate - self.target_spike_rate
        adapt_factor = torch.clamp(1.0 + rate_error, 0.5, 2.0)
        self.v_th = torch.ones_like(self.v_th) * self.v_th_base * adapt_factor
        self.v_th = torch.clamp(self.v_th, 0.05, 0.5)
        
        dv = (-self.v.unsqueeze(0).expand(batch_size, -1) + input_current) / self.tau_m
        self.v = self.v + dv.mean(dim=0) * dt
        
        spikes = (self.v.unsqueeze(0).expand(batch_size, -1) >= self.v_th.unsqueeze(0)).float()
        
        v_reset = torch.where(spikes > 0, torch.zeros_like(self.v), self.v.unsqueeze(0).expand(batch_size, -1))
        self.v = v_reset.mean(dim=0)
        
        decay = np.exp(-dt / self.tau_m)
        self.spike_trace = decay * self.spike_trace + spikes.mean(dim=0)
        
        continuous = torch.tanh(self.v / (self.v_th + 1e-6))
        
        spike_times_batch = torch.zeros_like(spikes)
        if self.enable_stdp:
            current_time_tensor = torch.tensor(self.current_time, device=self.device)
            for b in range(batch_size):
                fired = spikes[b] > 0
                if fired.any():
                    spike_times_batch[b, fired] = current_time_tensor
                    self.post_spike_times[fired] = current_time_tensor
        
        self.current_time += dt
        
        spike_rate = spikes.mean().item()
        self.spike_history.append(spike_rate)
        if len(self.spike_history) > 100:
            self.spike_history.pop(0)
        
        if squeeze_output:
            return spikes.squeeze(0), continuous, spike_times_batch.squeeze(0)
        
        return spikes, continuous, spike_times_batch
    
    def compute_stdp(
        self,
        pre_spike_time: torch.Tensor,
        post_spike_time: torch.Tensor
    ) -> torch.Tensor:
        """
        计算STDP权重更新。
        
        Parameters
        ----------
        pre_spike_time : Tensor
            突触前脉冲时间
        post_spike_time : Tensor
            突触后脉冲时间
        
        Returns
        -------
        delta_w : Tensor
            权重更新量
        """
        if not self.enable_stdp:
            return torch.zeros_like(pre_spike_time)
        
        delta_t = post_spike_time - pre_spike_time
        
        delta_w = torch.where(
            delta_t > 0,
            self.stdp_A_plus * torch.exp(-delta_t / self.stdp_tau_plus),
            -self.stdp_A_minus * torch.exp(delta_t / self.stdp_tau_minus)
        )
        
        return delta_w
    
    def get_sparsity(self) -> float:
        """获取稀疏度"""
        if len(self.spike_history) == 0:
            return 1.0
        return 1.0 - np.mean(self.spike_history[-50:])
    
    def reset(self) -> None:
        """重置状态"""
        self.v.zero_()
        self.v_th.fill_(self.v_th_base)
        self.spike_trace.zero_()
        self.spike_history = []
        self.spike_times = []
        self.current_time = 0.0
        if self.enable_stdp:
            self.pre_spike_times.zero_()
            self.post_spike_times.zero_()
    
    def to(self, device):
        """移动到指定设备"""
        super().to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        return self


class SCHLayer(nn.Module):
    """
    SCH神经元层，包含输入权重。
    
    Parameters
    ----------
    n_input : int
        输入维度
    n_output : int
        输出维度（神经元数量）
    """
    
    def __init__(
        self,
        n_input: int,
        n_output: int,
        tau_m: float = 5.0,
        v_th_base: float = 0.1,
        target_spike_rate: float = 0.1,
        enable_stdp: bool = True,
        device: str = 'auto'
    ):
        super().__init__()
        
        self.n_input = n_input
        self.n_output = n_output
        
        self.W = nn.Parameter(
            torch.randn(n_output, n_input) * np.sqrt(2.0 / n_input) * 0.1
        )
        
        self.neurons = SCHNeuronTorch(
            n_neurons=n_output,
            tau_m=tau_m,
            v_th_base=v_th_base,
            target_spike_rate=target_spike_rate,
            enable_stdp=enable_stdp,
            device=device
        )
        
    def forward(
        self,
        x: torch.Tensor,
        dt: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Parameters
        ----------
        x : Tensor
            输入，形状 (batch, n_input) 或 (n_input,)
        dt : float
            时间步长
        
        Returns
        -------
        spikes : Tensor
        continuous : Tensor
        spike_times : Tensor
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.to(self.W.device)
        current = F.linear(x, self.W)
        
        spikes, continuous, spike_times = self.neurons(current, dt)
        
        if self.neurons.enable_stdp and self.training:
            with torch.no_grad():
                pre_times = x.abs().argmax(dim=1).float()
                delta_w = self.neurons.compute_stdp(
                    pre_times.unsqueeze(1).expand(-1, self.n_output),
                    spike_times
                )
                self.W.data += delta_w.mean(dim=0).unsqueeze(1) * 0.01
        
        return spikes, continuous, spike_times
    
    def get_sparsity(self) -> float:
        return self.neurons.get_sparsity()
    
    def reset(self) -> None:
        self.neurons.reset()
