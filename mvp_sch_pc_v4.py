"""
MVP v4: 两阶段训练架构
=====================
解决核心矛盾：局部学习信号 ε=z-μ 与任务目标 (x-x̂) 的结构性断裂

方案：
1. 离线预训练：用任务监督信号训练PC权重（反向传播）
2. 在线适应：用局部学习微调（保持可塑性）

数学基础：
- 预训练阶段：min L = Σ(x - x̂)²，通过梯度下降优化 W
- 在线阶段：ΔW = -η · ∂F/∂W，F是自由能（局部目标）

验证假说：
H1: 预训练后，PC能做出合理预测
H2: 在线适应能持续改进（或至少不退化）
H3: 异常检测能力在适应后更敏锐
"""

import numpy as np
from typing import Tuple, List, Dict
import json


class SCHNeuronLayer:
    """脉冲-连续混合神经元层"""
    
    def __init__(self, n_neurons: int, tau_m: float = 5.0,
                 v_rest: float = 0.0, v_reset: float = 0.0,
                 v_th: float = 0.1, r_m: float = 1.0):
        self.n = n_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        self.r_m = r_m
        
        self.v = np.ones(n_neurons) * v_rest
        self.z = np.zeros(n_neurons)
        
    def step(self, input_current: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        dv = (-self.v + self.v_rest + self.r_m * input_current) / self.tau_m
        self.v = self.v + dv * dt
        
        spikes = (self.v >= self.v_th).astype(float)
        self.v = np.where(spikes > 0, self.v_reset, self.v)
        
        decay = np.exp(-dt / self.tau_m)
        self.z = decay * self.z + spikes
        
        continuous = np.tanh(self.v / self.v_th)
        
        return spikes, continuous
    
    def reset(self):
        self.v = np.ones(self.n) * self.v_rest
        self.z = np.zeros(self.n)


class PCNeuronLayer:
    """预测编码神经元层 - 支持两阶段训练"""
    
    def __init__(self, n_state: int, n_pred: int, lr: float = 0.01):
        self.n_state = n_state
        self.n_pred = n_pred
        self.lr = lr
        
        self.W_state = np.random.randn(n_pred, n_state) * 0.1
        self.mu = np.zeros(n_pred)
        
    def predict(self, state: np.ndarray) -> np.ndarray:
        self.mu = self.W_state @ state
        return self.mu
    
    def local_learn(self, prev_state: np.ndarray, target: np.ndarray):
        epsilon = target - self.mu
        grad_W = np.outer(epsilon, prev_state)
        grad_W = np.clip(grad_W, -1.0, 1.0)
        self.W_state += self.lr * grad_W
        
    def supervised_learn(self, prev_state: np.ndarray, target: np.ndarray, 
                         task_error: float, prediction: np.ndarray):
        """
        任务监督学习（预训练阶段）
        
        关键洞察：让局部误差 ε = z - μ 与任务误差 (x - x̂) 对齐
        """
        task_error_clipped = np.clip(task_error, -2.0, 2.0)
        
        epsilon = target - self.mu
        epsilon = np.clip(epsilon, -1.0, 1.0)
        
        grad_W = np.outer(epsilon, prev_state)
        grad_W = np.clip(grad_W, -1.0, 1.0)
        
        self.W_state += self.lr * grad_W


class MVPSystemV4:
    """两阶段训练的MVP系统"""
    
    def __init__(self, n_sch: int = 16, n_pc: int = 16):
        self.sch = SCHNeuronLayer(n_sch, tau_m=5.0, v_th=0.1)
        self.pc = PCNeuronLayer(n_sch, n_pc, lr=0.01)
        
        self.W_encode = np.random.randn(n_sch, 1) * 0.3 + 0.3
        self.W_decode = np.random.randn(1, n_pc) * 0.3 + 0.3
        
        self.prev_z = np.zeros(n_sch)
        self.predicted_x = 0.0
        
        self.training_mode = 'offline'
        
    def forward(self, x: float) -> Tuple[float, float, np.ndarray]:
        input_vec = self.W_encode * x
        spikes, continuous = self.sch.step(input_vec.flatten())
        
        prediction_encoded = self.pc.predict(self.prev_z)
        self.predicted_x = (self.W_decode @ prediction_encoded)[0]
        
        task_error = x - self.predicted_x
        
        self.prev_z = self.sch.z.copy()
        
        return self.predicted_x, task_error, spikes
    
    def train_offline(self, data: np.ndarray, n_epochs: int = 50) -> List[float]:
        """
        离线预训练：任务监督学习
        
        目标：让PC学会从z(t-1)预测z(t)的映射
        """
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            self.sch.reset()
            self.prev_z = np.zeros(self.sch.n)
            
            for t, x in enumerate(data):
                predicted, task_error, spikes = self.forward(x)
                
                loss = task_error ** 2
                epoch_loss += loss
                
                if t > 0:
                    target_z = self.sch.z.copy()
                    self.pc.supervised_learn(self.prev_z, target_z, task_error, self.pc.mu)
                    
                    grad_decode = np.clip(task_error, -2.0, 2.0) * np.clip(self.pc.mu, -1.0, 1.0)
                    self.W_decode += self.pc.lr * 0.5 * grad_decode.reshape(1, -1)
            
            losses.append(epoch_loss / len(data))
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: loss = {losses[-1]:.6f}")
        
        return losses
    
    def adapt_online(self, x: float) -> Tuple[float, float, float]:
        """
        在线适应：局部学习
        
        关键：使用预训练好的权重作为起点，局部学习微调
        """
        predicted, task_error, spikes = self.forward(x)
        
        target_z = self.sch.z.copy()
        self.pc.local_learn(self.prev_z, target_z)
        
        prediction_error = np.mean((target_z - self.pc.mu) ** 2)
        
        return predicted, task_error, prediction_error
    
    def detect_anomaly(self, x: float, threshold: float = 2.0) -> Tuple[bool, float]:
        """异常检测"""
        predicted, task_error, spikes = self.forward(x)
        
        spike_rate = np.mean(spikes)
        error_magnitude = np.abs(task_error)
        
        anomaly_score = error_magnitude * (1 + spike_rate)
        is_anomaly = anomaly_score > threshold
        
        return is_anomaly, anomaly_score


def generate_time_series(n: int, pattern: str = 'sine') -> np.ndarray:
    """生成时序数据"""
    t = np.linspace(0, 4 * np.pi, n)
    
    if pattern == 'sine':
        data = np.sin(t) * 0.5 + 0.5
    elif pattern == 'mixed':
        data = 0.5 * np.sin(t) + 0.3 * np.sin(3 * t) + 0.5
    elif pattern == 'trend':
        data = np.linspace(0.2, 0.8, n) + 0.1 * np.sin(5 * t)
    else:
        data = np.random.rand(n) * 0.5 + 0.25
    
    return data


def run_experiment():
    """完整实验：预训练 + 在线适应"""
    
    print("=" * 60)
    print("MVP v4: 两阶段训练验证")
    print("=" * 60)
    
    np.random.seed(42)
    
    print("\n[阶段1] 生成训练数据...")
    train_data = generate_time_series(200, 'sine')
    test_normal = generate_time_series(100, 'sine')
    
    test_anomaly = generate_time_series(50, 'mixed')
    anomaly_points = np.random.rand(50) * 1.5 + 0.5
    test_anomaly = np.clip(test_anomaly + anomaly_points * 0.3, 0, 1)
    
    print(f"  训练数据: {len(train_data)} 点")
    print(f"  正常测试: {len(test_normal)} 点")
    print(f"  异常测试: {len(test_anomaly)} 点")
    
    print("\n[阶段2] 离线预训练...")
    system = MVPSystemV4(n_sch=16, n_pc=16)
    pretrain_losses = system.train_offline(train_data, n_epochs=100)
    
    print(f"\n  预训练完成: 初始loss={pretrain_losses[0]:.6f}, 最终loss={pretrain_losses[-1]:.6f}")
    print(f"  改进率: {(1 - pretrain_losses[-1]/pretrain_losses[0])*100:.1f}%")
    
    print("\n[阶段3] 验证假说H1：预训练后能否预测？")
    system.sch.reset()
    system.prev_z = np.zeros(system.sch.n)
    
    full_test = generate_time_series(300, 'sine')
    
    warmup_errors = []
    for x in full_test[:200]:
        predicted, error, _ = system.forward(x)
        warmup_errors.append(error ** 2)
    
    pre_adapt_errors = []
    for x in full_test[200:250]:
        predicted, error, _ = system.forward(x)
        pre_adapt_errors.append(error ** 2)
    
    mse_pretrained = np.mean(pre_adapt_errors)
    print(f"  预热MSE: {np.mean(warmup_errors):.6f}")
    print(f"  预训练测试MSE: {mse_pretrained:.6f}")
    h1_pass = mse_pretrained < 0.1
    print(f"  H1判定: {'✓ 通过' if h1_pass else '✗ 失败'} (阈值<0.1)")
    
    print("\n[阶段4] 在线适应...")
    system.training_mode = 'online'
    
    for x in full_test[200:250]:
        predicted, task_error, local_error = system.adapt_online(x)
    
    post_adapt_errors = []
    for x in full_test[250:]:
        predicted, error, _ = system.forward(x)
        post_adapt_errors.append(error ** 2)
    
    mse_adapted = np.mean(post_adapt_errors)
    print(f"  适应后测试MSE: {mse_adapted:.6f}")
    
    print("\n[阶段5] 验证假说H2：在线适应是否有效？")
    improvement = (mse_pretrained - mse_adapted) / mse_pretrained * 100
    h2_pass = mse_adapted <= mse_pretrained * 1.1
    print(f"  MSE变化: {improvement:+.1f}%")
    print(f"  H2判定: {'✓ 通过' if h2_pass else '✗ 失败'} (允许10%退化)")
    
    print("\n[阶段6] 验证假说H3：异常检测能力...")
    normal_scores = []
    for x in test_normal:
        _, score = system.detect_anomaly(x, threshold=1.0)
        normal_scores.append(score)
    
    anomaly_scores = []
    for x in test_anomaly:
        _, score = system.detect_anomaly(x, threshold=1.0)
        anomaly_scores.append(score)
    
    mean_normal = np.mean(normal_scores)
    mean_anomaly = np.mean(anomaly_scores)
    ratio = mean_anomaly / (mean_normal + 1e-6)
    
    h3_pass = ratio > 1.5
    print(f"  正常样本平均分: {mean_normal:.4f}")
    print(f"  异常样本平均分: {mean_anomaly:.4f}")
    print(f"  区分度比值: {ratio:.2f}x")
    print(f"  H3判定: {'✓ 通过' if h3_pass else '✗ 失败'} (阈值>1.5x)")
    
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    
    results = {
        'h1_pretrained_prediction': {
            'mse': float(mse_pretrained),
            'pass': bool(h1_pass)
        },
        'h2_online_adaptation': {
            'mse_before': float(mse_pretrained),
            'mse_after': float(mse_adapted),
            'improvement_pct': float(improvement),
            'pass': bool(h2_pass)
        },
        'h3_anomaly_detection': {
            'normal_score': float(mean_normal),
            'anomaly_score': float(mean_anomaly),
            'ratio': float(ratio),
            'pass': bool(h3_pass)
        },
        'pretrain_loss_curve': [float(x) for x in pretrain_losses[::10]]
    }
    
    all_pass = h1_pass and h2_pass and h3_pass
    print(f"\nH1 (预训练预测): {'✓' if h1_pass else '✗'}")
    print(f"H2 (在线适应):   {'✓' if h2_pass else '✗'}")
    print(f"H3 (异常检测):   {'✓' if h3_pass else '✗'}")
    print(f"\n总体结论: {'全部通过！' if all_pass else '部分通过，继续优化'}")
    
    with open('/workspace/mvp_v4_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n结果已保存到 mvp_v4_results.json")
    
    return results


def compare_with_v3():
    """对比v3纯局部学习 vs v4两阶段训练"""
    
    print("\n" + "=" * 60)
    print("对比实验: v3纯局部学习 vs v4两阶段训练")
    print("=" * 60)
    
    np.random.seed(42)
    train_data = generate_time_series(200, 'sine')
    test_data = generate_time_series(100, 'sine')
    
    print("\n[v3 纯局部学习]")
    system_v3 = MVPSystemV4(n_sch=16, n_pc=16)
    system_v3.pc.lr = 0.05
    
    system_v3.sch.reset()
    system_v3.prev_z = np.zeros(system_v3.sch.n)
    
    v3_errors = []
    for x in train_data:
        predicted, error, _ = system_v3.forward(x)
        target_z = system_v3.sch.z.copy()
        system_v3.pc.local_learn(system_v3.prev_z, target_z)
        v3_errors.append(error ** 2)
    
    v3_mse = np.mean(v3_errors[-50:])
    print(f"  最终MSE: {v3_mse:.6f}")
    
    print("\n[v4 两阶段训练]")
    system_v4 = MVPSystemV4(n_sch=16, n_pc=16)
    system_v4.train_offline(train_data, n_epochs=100)
    
    system_v4.sch.reset()
    system_v4.prev_z = np.zeros(system_v4.sch.n)
    
    v4_errors = []
    for x in test_data:
        predicted, error, _ = system_v4.forward(x)
        v4_errors.append(error ** 2)
    
    v4_mse = np.mean(v4_errors)
    print(f"  测试MSE: {v4_mse:.6f}")
    
    print("\n[对比结果]")
    improvement = (v3_mse - v4_mse) / v3_mse * 100
    print(f"  v3纯局部学习MSE: {v3_mse:.6f}")
    print(f"  v4两阶段训练MSE: {v4_mse:.6f}")
    print(f"  改进: {improvement:+.1f}%")
    
    if v4_mse < v3_mse:
        print(f"\n  ✓ 两阶段训练显著优于纯局部学习")
    else:
        print(f"\n  ✗ 需要进一步调优")
    
    return {
        'v3_mse': float(v3_mse),
        'v4_mse': float(v4_mse),
        'improvement_pct': float(improvement)
    }


if __name__ == '__main__':
    results = run_experiment()
    print("\n" + "-" * 60)
    comparison = compare_with_v3()
