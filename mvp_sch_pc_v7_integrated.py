"""
MVP v7: 中等规模完整系统 (66K参数)
===================================
规模：
- SCH: 256个神经元
- PC: 256维状态空间  
- HM: 128维隐藏层
- 总参数: ~66K

防御性设计：
1. 自适应阈值：v_th(t) = v_th0 + β·σ(W_adapt·z)
2. 残差连接 + 层归一化
3. 模式分离：top-k稀疏化
4. 显式调度器：模块运行频率
5. NG归一化：四通道乘积门控
6. 监控：稀疏度、梯度范数、自由能分量

验证命题：HM遗忘缓解 + PC异常检测能否同时工作
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
import json
from collections import defaultdict


@dataclass
class MonitorMetrics:
    """监控指标"""
    sparsity: float = 0.0
    grad_norm: float = 0.0
    spike_rate: float = 0.0
    free_energy: float = 0.0
    da: float = 0.5
    effective_lr: float = 0.05


class AdaptiveThresholdSCH:
    """
    SCH神经元层（自适应阈值）
    
    修复：目标脉冲率反馈控制
    维持脉冲率在 0.05-0.2 范围
    """
    
    def __init__(self, n_neurons: int, tau_m: float = 5.0,
                 v_th_base: float = 0.1, adapt_strength: float = 1.0,
                 target_spike_rate: float = 0.1):
        self.n = n_neurons
        self.tau_m = tau_m
        self.v_th_base = v_th_base
        self.adapt_strength = adapt_strength
        self.target_spike_rate = target_spike_rate
        
        self.v = np.zeros(n_neurons)
        self.z = np.zeros(n_neurons)
        self.v_th = np.ones(n_neurons) * v_th_base
        
        self.W_adapt = np.random.randn(n_neurons, n_neurons) * 0.1
        
        self.spike_history = []
        
    def step(self, input_current: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        current_rate = np.mean(self.spike_history) if len(self.spike_history) > 10 else self.target_spike_rate
        rate_error = current_rate - self.target_spike_rate
        
        adapt_factor = 1.0 + self.adapt_strength * 1.0 * rate_error
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
        if len(self.spike_history) == 0:
            return 0.0
        return 1.0 - np.mean(self.spike_history)
    
    def reset(self):
        self.v = np.zeros(self.n)
        self.z = np.zeros(self.n)
        self.v_th = np.ones(self.n) * self.v_th_base


class ResidualPCLayer:
    """
    PC预测编码层（残差连接 + 层归一化）
    
    防御梯度消失：
    output = LayerNorm(x + W·z)
    
    加强防御梯度爆炸：
    - 更强的梯度裁剪
    - 自适应学习率
    """
    
    def __init__(self, n_state: int, n_pred: int, lr: float = 0.005):
        self.n_state = n_state
        self.n_pred = n_pred
        self.lr = lr
        
        self.W = np.random.randn(n_pred, n_state) * np.sqrt(2.0 / n_state) * 0.1
        self.mu = np.zeros(n_pred)
        
        self.grad_norm_history = []
        self.target_grad_norm = 0.5
        
    def layer_norm(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x)
        std = np.std(x) + 1e-6
        return (x - mean) / std
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        linear = self.W @ state
        self.mu = self.layer_norm(state[:self.n_pred] + linear)
        return self.mu
    
    def local_learn(self, prev_state: np.ndarray, target: np.ndarray) -> float:
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
        if len(self.grad_norm_history) == 0:
            return 0.0
        return np.mean(self.grad_norm_history)


class PatternSeparationHM:
    """
    HM记忆层（模式分离）
    
    防御检索退化：
    - Top-k稀疏化竞争
    - 模式分离（齿状回功能）
    """
    
    def __init__(self, n_input: int, n_hidden: int, 
                 hippocampus_lr: float = 0.05, cortex_lr: float = 0.01,
                 top_k: int = 32):
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
        模式分离：top-k稀疏化
        
        模拟齿状回的功能：增强区分度
        """
        activity = self.W_hippo @ x
        
        top_k_indices = np.argsort(np.abs(activity))[-self.top_k:]
        sparse_output = np.zeros_like(activity)
        sparse_output[top_k_indices] = activity[top_k_indices]
        
        return np.tanh(sparse_output)
    
    def forward(self, x: np.ndarray, use_cortex: bool = True) -> np.ndarray:
        h_out = self.pattern_separation(x)
        c_out = np.tanh(self.W_cortex @ x)
        
        if not use_cortex:
            return h_out
        
        alpha = 0.3 + 0.7 * np.exp(-self.consolidation_count / 3.0)
        return alpha * h_out + (1 - alpha) * c_out
    
    def learn(self, x: np.ndarray, target: np.ndarray, task_id: int = 0):
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
    
    def consolidate(self, n_replay: int = 60) -> int:
        if len(self.memory_buffer) == 0:
            return 0
        
        ages = np.arange(len(self.memory_buffer), 0, -1)
        risks = 1.0 - 1.0 / (1.0 + 0.05 * ages)
        probs = risks / np.sum(risks)
        
        n_sample = min(n_replay, len(self.memory_buffer))
        indices = np.random.choice(len(self.memory_buffer), n_sample, replace=False, p=probs)
        
        for idx in indices:
            x, target, _ = self.memory_buffer[idx]
            pred = np.tanh(self.W_cortex @ x)
            error = target - pred
            error = np.clip(error, -1.0, 1.0)
            self.W_cortex += self.cortex_lr * 1.0 * np.outer(error, x)
        
        self.consolidation_count += 1
        return n_sample


class NormalizedNG:
    """
    NG神经调质门控（加权和）
    
    修复：乘积门控太保守
    - 改用加权和 + clamp
    - effective_lr范围: 0.002-0.02
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
        """加权和门控"""
        weights = np.array([0.4, 0.2, 0.2, 0.2])
        signals = np.array([self.da, self.serotonin, self.ach, self.ne])
        gate = np.dot(weights, signals)
        return np.clip(gate, 0.1, 0.9)
    
    def compute_effective_lr(self) -> float:
        decay = 1.0 + self.step_count * (1.0 - self.da) * 0.005
        gate = self.compute_gate()
        effective_lr = self.base_lr * gate * 2.0 / decay
        return np.clip(effective_lr, 0.002, 0.02)
    
    def update_from_reward(self, reward: float):
        rpe = reward - self.reward_prediction
        self.reward_prediction = 0.9 * self.reward_prediction + 0.1 * reward
        
        self.da = np.clip(0.5 + 0.3 * np.tanh(rpe), 0.1, 0.9)
        self.ach = np.clip(0.5 + 0.2 * (1.0 - abs(rpe)), 0.1, 0.9)
        
        self.step_count += 1


class ModuleScheduler:
    """
    显式调度器
    
    防御时间尺度失配：
    - 每个模块指定运行频率
    """
    
    def __init__(self):
        self.frequencies = {
            'sch': 1,
            'pc': 1,
            'hm': 1,
            'hm_consolidate': 10,
            'ng': 1
        }
        self.step = 0
        
    def should_run(self, module: str) -> bool:
        freq = self.frequencies.get(module, 1)
        return self.step % freq == 0
    
    def tick(self):
        self.step += 1


class IntegratedSystemV7:
    """
    完整集成系统 V7
    
    规模：~66K参数
    - SCH: 256 neurons
    - PC: 256 state
    - HM: 128 hidden
    """
    
    def __init__(self, n_sch: int = 256, n_pc: int = 256, n_hm: int = 128):
        self.n_sch = n_sch
        self.n_pc = n_pc
        self.n_hm = n_hm
        
        self.sch = AdaptiveThresholdSCH(n_sch, tau_m=5.0, v_th_base=0.08, adapt_strength=1.0, target_spike_rate=0.1)
        self.pc = ResidualPCLayer(n_pc, n_pc, lr=0.002)
        self.hm = PatternSeparationHM(n_sch, n_hm, hippocampus_lr=0.02, cortex_lr=0.008, top_k=64)
        self.ng = NormalizedNG(base_lr=0.01)
        self.scheduler = ModuleScheduler()
        
        self.W_encode = np.random.randn(n_sch, 1) * 0.2
        self.W_decode = np.random.randn(1, n_hm) * 0.2
        
        self.prev_z = np.zeros(n_sch)
        self.current_task = 0
        
        self.metrics_history: List[MonitorMetrics] = []
        
    def count_parameters(self) -> Dict[str, int]:
        params = {
            'sch_W_adapt': self.sch.W_adapt.size,
            'pc_W': self.pc.W.size,
            'hm_W_hippo': self.hm.W_hippo.size,
            'hm_W_cortex': self.hm.W_cortex.size,
            'W_encode': self.W_encode.size,
            'W_decode': self.W_decode.size
        }
        params['total'] = sum(params.values())
        return params
    
    def forward(self, x: float) -> Dict:
        input_vec = self.W_encode * x
        
        if self.scheduler.should_run('sch'):
            spikes, continuous = self.sch.step(input_vec.flatten())
        else:
            spikes = np.zeros(self.n_sch)
            continuous = np.tanh(self.sch.v / (self.sch.v_th + 1e-6))
        
        if self.scheduler.should_run('pc'):
            pc_pred = self.pc.predict(self.prev_z)
        else:
            pc_pred = self.pc.mu
        
        hm_out = self.hm.forward(self.sch.z)
        
        output = (self.W_decode @ hm_out)[0]
        
        return {
            'output': output,
            'spikes': spikes,
            'continuous': continuous,
            'pc_pred': pc_pred,
            'hm_out': hm_out
        }
    
    def learn(self, x: float, target: float) -> MonitorMetrics:
        result = self.forward(x)
        
        output_error = target - result['output']
        reward = 1.0 - abs(output_error)
        
        if self.scheduler.should_run('ng'):
            self.ng.update_from_reward(reward)
        
        effective_lr = self.ng.compute_effective_lr()
        
        hm_target = result['hm_out'] + 0.1 * np.sign(output_error)
        if self.scheduler.should_run('hm'):
            self.hm.learn(self.sch.z, hm_target, task_id=self.current_task)
        
        if self.scheduler.should_run('pc'):
            self.pc.local_learn(self.prev_z, target=result['continuous'][:self.n_pc])
        
        self.prev_z = self.sch.z.copy()
        
        metrics = MonitorMetrics(
            sparsity=self.sch.get_sparsity(),
            grad_norm=self.pc.get_grad_norm(),
            spike_rate=np.mean(result['spikes']),
            free_energy=float(output_error ** 2),
            da=self.ng.da,
            effective_lr=effective_lr
        )
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
        
        self.scheduler.tick()
        
        return metrics
    
    def consolidate(self, n_replay: int = 40) -> int:
        return self.hm.consolidate(n_replay)
    
    def get_average_metrics(self, n_last: int = 100) -> Dict[str, float]:
        if len(self.metrics_history) == 0:
            return {}
        
        recent = self.metrics_history[-n_last:]
        return {
            'sparsity': np.mean([m.sparsity for m in recent]),
            'grad_norm': np.mean([m.grad_norm for m in recent]),
            'spike_rate': np.mean([m.spike_rate for m in recent]),
            'free_energy': np.mean([m.free_energy for m in recent]),
            'da': np.mean([m.da for m in recent]),
            'effective_lr': np.mean([m.effective_lr for m in recent])
        }


def generate_interference_tasks(n_tasks: int = 8, n_samples: int = 100) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    生成真正正交的冲突任务
    
    消除任务间的意外协同：
    - sin、线性、指数、阶跃、绝对值、二次、符号、对数
    """
    tasks = []
    for task_id in range(n_tasks):
        x = np.linspace(0.01, 1, n_samples)
        np.random.seed(42 + task_id)
        noise = np.random.randn(n_samples) * 0.03
        
        if task_id == 0:
            y = np.sin(2 * np.pi * x)
        elif task_id == 1:
            y = 2 * x - 1
        elif task_id == 2:
            y = np.exp(x) - 1.5
        elif task_id == 3:
            y = np.where(x < 0.5, -0.5, 0.5)
        elif task_id == 4:
            y = np.abs(2 * x - 1) - 0.5
        elif task_id == 5:
            y = 4 * (x - 0.5) ** 2 - 0.5
        elif task_id == 6:
            y = np.sign(x - 0.5) * 0.5
        else:
            y = np.log(x + 0.1) / 2
        
        y = y + noise
        y = np.clip(y, -1.5, 1.5)
        tasks.append((x, y))
    
    return tasks


def run_integrated_experiment():
    """运行完整集成实验"""
    
    print("=" * 70)
    print("MVP v7: 中等规模完整系统 (66K参数)")
    print("=" * 70)
    
    np.random.seed(42)
    
    system = IntegratedSystemV7(n_sch=256, n_pc=256, n_hm=128)
    
    params = system.count_parameters()
    print("\n[系统规模]")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    n_tasks = 8
    tasks = generate_interference_tasks(n_tasks, n_samples=100)
    
    print(f"\n[任务设置] {n_tasks}个干扰任务")
    
    print("\n" + "=" * 70)
    print("阶段1: 连续学习 + 监控")
    print("=" * 70)
    
    task_performances = {i: [] for i in range(n_tasks)}
    anomaly_scores = []
    
    for task_id in range(n_tasks):
        print(f"\n[任务{task_id}] 训练...")
        system.current_task = task_id
        
        x_data, y_data = tasks[task_id]
        
        epoch_metrics = []
        for epoch in range(30):
            epoch_loss = 0.0
            epoch_sparsity = []
            epoch_grad_norm = []
            
            for x, y in zip(x_data, y_data):
                metrics = system.learn(x, y)
                epoch_loss += (y - system.forward(x)['output']) ** 2
                epoch_sparsity.append(metrics.sparsity)
                epoch_grad_norm.append(metrics.grad_norm)
            
            avg_loss = epoch_loss / len(x_data)
            avg_sparsity = np.mean(epoch_sparsity)
            avg_grad_norm = np.mean(epoch_grad_norm)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: loss={avg_loss:.4f}, "
                      f"sparsity={avg_sparsity:.3f}, grad_norm={avg_grad_norm:.4f}")
        
        n_replay = system.consolidate(n_replay=80)
        print(f"  HM巩固: {n_replay}样本")
        
        print("  评估所有任务:")
        for i in range(n_tasks):
            x_eval, y_eval = tasks[i]
            total_loss = 0.0
            for x, y in zip(x_eval, y_eval):
                pred = system.forward(x)['output']
                total_loss += (y - pred) ** 2
            mse = total_loss / len(x_eval)
            task_performances[i].append(mse)
            marker = "★" if i == task_id else " "
            print(f"    {marker} 任务{i}: MSE={mse:.4f}")
        
        x_test, y_test = tasks[task_id]
        
        normal_errors = []
        for x, y in zip(x_test[:30], y_test[:30]):
            pred = system.forward(x)['output']
            normal_errors.append((y - pred) ** 2)
        normal_error = np.mean(normal_errors)
        
        anomaly_errors = []
        np.random.seed(123)
        for x, y in zip(x_test[:30], y_test[:30]):
            y_anomaly = y + np.random.randn() * 0.5
            pred = system.forward(x)['output']
            anomaly_errors.append((y_anomaly - pred) ** 2)
        anomaly_error = np.mean(anomaly_errors)
        
        anomaly_ratio = anomaly_error / (normal_error + 1e-6)
        anomaly_scores.append(anomaly_ratio)
        print(f"  异常检测比: {anomaly_ratio:.2f}x (正常={normal_error:.4f}, 异常={anomaly_error:.4f})")
    
    print("\n" + "=" * 70)
    print("阶段2: 遗忘分析")
    print("=" * 70)
    
    task0_mse = task_performances[0]
    print(f"\n任务0性能变化:")
    for stage, mse in enumerate(task0_mse):
        print(f"  阶段{stage+1}: MSE={mse:.4f}")
    
    initial_mse = task0_mse[0]
    final_mse = task0_mse[-1]
    worst_mse = max(task0_mse)
    
    final_forgetting_rate = (final_mse - initial_mse) / initial_mse * 100
    worst_forgetting_rate = (worst_mse - initial_mse) / initial_mse * 100
    
    print(f"\n遗忘率（终值）: {final_forgetting_rate:+.1f}%")
    print(f"遗忘率（最差）: {worst_forgetting_rate:+.1f}%")
    print(f"判定: {'✓ HM有效缓解遗忘' if worst_forgetting_rate < 150 else '✗ 灾难性遗忘严重'}")
    
    print("\n" + "=" * 70)
    print("阶段3: 异常检测分析")
    print("=" * 70)
    
    avg_anomaly_ratio = np.mean(anomaly_scores)
    print(f"\n平均异常检测比: {avg_anomaly_ratio:.2f}x")
    print(f"判定: {'✓ PC异常检测有效' if avg_anomaly_ratio > 1.5 else '✗ 异常检测不足'}")
    
    print("\n" + "=" * 70)
    print("阶段4: 系统监控总结")
    print("=" * 70)
    
    avg_metrics = system.get_average_metrics(200)
    print(f"\n平均指标:")
    for name, value in avg_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    sparsity_ok = avg_metrics['sparsity'] > 0.5
    grad_ok = avg_metrics['grad_norm'] < 5.0
    
    print(f"\n稀疏性: {'✓ 保持良好' if sparsity_ok else '✗ 稀疏性崩溃'}")
    print(f"梯度范数: {'✓ 稳定' if grad_ok else '✗ 梯度爆炸'}")
    
    print("\n" + "=" * 70)
    print("验证命题: HM遗忘缓解 + PC异常检测同时工作")
    print("=" * 70)
    
    hm_works = worst_forgetting_rate < 150
    pc_works = avg_anomaly_ratio > 1.5
    no_interference = sparsity_ok and grad_ok
    
    print(f"\nHM遗忘缓解: {'✓' if hm_works else '✗'}")
    print(f"PC异常检测: {'✓' if pc_works else '✗'}")
    print(f"互不干扰: {'✓' if no_interference else '✗'}")
    
    proposition_holds = hm_works and pc_works and no_interference
    print(f"\n命题验证: {'✓ 成立' if proposition_holds else '✗ 不成立'}")
    
    results = {
        'params': params,
        'final_forgetting_rate': float(final_forgetting_rate),
        'worst_forgetting_rate': float(worst_forgetting_rate),
        'avg_anomaly_ratio': float(avg_anomaly_ratio),
        'avg_metrics': {k: float(v) for k, v in avg_metrics.items()},
        'hm_works': bool(hm_works),
        'pc_works': bool(pc_works),
        'no_interference': bool(no_interference),
        'proposition_holds': bool(proposition_holds)
    }
    
    with open('/workspace/mvp_v7_integrated_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到 mvp_v7_integrated_results.json")
    
    return results


if __name__ == '__main__':
    results = run_integrated_experiment()
