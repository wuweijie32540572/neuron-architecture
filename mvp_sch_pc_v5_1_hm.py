"""
MVP v5.1: HM记忆层升级版
========================
修复问题：
1. 实现记忆稳定性度量 σ(m) = 1/(1 + α·age + β/(access+1))
2. 风险加权重放: risk(m) ∝ 1-σ(m)，优先巩固高风险记忆
3. 扩展到8个任务，验证带HM遗忘率<50%
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass, field
import json


@dataclass
class MemoryItem:
    """记忆项：带元数据"""
    x: np.ndarray
    target: np.ndarray
    age: int = 0
    access: int = 0
    task_id: int = 0
    importance: float = 1.0
    
    def stability(self, alpha: float = 0.1, beta: float = 1.0) -> float:
        """记忆稳定性度量
        
        σ(m) = 1 / (1 + α·age + β/(access+1))
        
        - age越大，稳定性越低（旧记忆容易被覆盖）
        - access越大，稳定性越高（频繁访问的记忆更稳固）
        """
        return 1.0 / (1.0 + alpha * self.age + beta / (self.access + 1))
    
    def forgetting_risk(self, alpha: float = 0.1, beta: float = 1.0) -> float:
        """遗忘风险 = 1 - 稳定性"""
        return 1.0 - self.stability(alpha, beta)


class HippocampusLayerV2:
    """海马体V2：带稳定性度量的记忆管理"""
    
    def __init__(self, n_input: int, n_output: int, lr: float = 0.1):
        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        
        self.W = np.random.randn(n_output, n_input) * 0.1
        self.memory_buffer: List[MemoryItem] = []
        self.buffer_size = 400
        
        self.alpha_stability = 0.05
        self.beta_stability = 2.0
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x
    
    def learn(self, x: np.ndarray, target: np.ndarray, task_id: int = 0):
        pred = self.forward(x)
        error = target - pred
        error = np.clip(error, -1.0, 1.0)
        
        grad = np.outer(error, x)
        grad = np.clip(grad, -1.0, 1.0)
        self.W += self.lr * grad
        
        for m in self.memory_buffer:
            m.age += 1
        
        new_memory = MemoryItem(
            x=x.copy(),
            target=target.copy(),
            age=0,
            access=0,
            task_id=task_id,
            importance=1.0
        )
        self.memory_buffer.append(new_memory)
        
        if len(self.memory_buffer) > self.buffer_size:
            self._prune_memories()
    
    def _prune_memories(self):
        """按稳定性修剪记忆"""
        if len(self.memory_buffer) <= self.buffer_size:
            return
        
        stabilities = [m.stability(self.alpha_stability, self.beta_stability) 
                       for m in self.memory_buffer]
        
        sorted_indices = np.argsort(stabilities)
        
        n_keep = self.buffer_size // 2
        keep_indices = set(sorted_indices[-n_keep:])
        
        high_risk_indices = sorted_indices[:n_keep]
        for idx in high_risk_indices:
            if np.random.rand() < 0.3:
                keep_indices.add(idx)
        
        self.memory_buffer = [self.memory_buffer[i] for i in sorted(keep_indices)]
    
    def get_risk_weighted_replay(self, n: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:
        """风险加权重放采样"""
        if len(self.memory_buffer) == 0:
            return [], {}
        
        risks = [m.forgetting_risk(self.alpha_stability, self.beta_stability) 
                 for m in self.memory_buffer]
        
        total_risk = sum(risks)
        if total_risk == 0:
            probs = [1.0 / len(risks)] * len(risks)
        else:
            probs = [r / total_risk for r in risks]
        
        n_sample = min(n, len(self.memory_buffer))
        indices = np.random.choice(len(self.memory_buffer), n_sample, replace=False, p=probs)
        
        samples = []
        for idx in indices:
            m = self.memory_buffer[idx]
            m.access += 1
            m.age = 0
            samples.append((m.x, m.target))
        
        stats = {
            'n_sampled': len(samples),
            'avg_risk': np.mean([risks[i] for i in indices]),
            'max_risk': max(risks),
            'min_risk': min(risks),
            'risk_distribution': {
                'high_risk_ratio': sum(1 for r in risks if r > 0.5) / len(risks),
                'low_risk_ratio': sum(1 for r in risks if r <= 0.5) / len(risks)
            }
        }
        
        return samples, stats


class CortexLayerV2:
    """皮层V2：慢速学习，长期记忆"""
    
    def __init__(self, n_input: int, n_output: int, lr: float = 0.01):
        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        
        self.W = np.random.randn(n_output, n_input) * 0.1
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x
    
    def learn(self, x: np.ndarray, target: np.ndarray):
        pred = self.forward(x)
        error = target - pred
        error = np.clip(error, -1.0, 1.0)
        
        grad = np.outer(error, x)
        grad = np.clip(grad, -1.0, 1.0)
        self.W += self.lr * grad
    
    def consolidate(self, samples: List[Tuple[np.ndarray, np.ndarray]], 
                    importance_weights: Optional[List[float]] = None):
        """系统巩固：用重放样本训练"""
        for i, (x, target) in enumerate(samples):
            weight = importance_weights[i] if importance_weights else 1.0
            pred = self.forward(x)
            error = target - pred
            error = np.clip(error, -1.0, 1.0)
            
            grad = np.outer(error, x)
            grad = np.clip(grad, -1.0, 1.0)
            self.W += self.lr * weight * grad


class HMNeuronLayerV2:
    """海马体-皮层混合记忆层V2"""
    
    def __init__(self, n_input: int, n_output: int, 
                 hippocampus_lr: float = 0.05,
                 cortex_lr: float = 0.02):
        self.hippocampus = HippocampusLayerV2(n_input, n_output, lr=hippocampus_lr)
        self.cortex = CortexLayerV2(n_input, n_output, lr=cortex_lr)
        
        self.consolidation_count = 0
        
    def forward(self, x: np.ndarray, use_both: bool = True) -> np.ndarray:
        h_out = self.hippocampus.forward(x)
        c_out = self.cortex.forward(x)
        
        if not use_both:
            return c_out
        
        base_alpha = 0.3
        decay_alpha = 0.7 * np.exp(-self.consolidation_count / 3.0)
        alpha = base_alpha + decay_alpha
        alpha = np.clip(alpha, 0.0, 1.0)
        
        return alpha * h_out + (1 - alpha) * c_out
    
    def learn(self, x: np.ndarray, target: np.ndarray, task_id: int = 0):
        self.hippocampus.learn(x, target, task_id)
        self.cortex.learn(x, target)
    
    def sleep(self, n_replay: int = 50) -> Tuple[int, Dict]:
        """睡眠阶段：风险加权的系统巩固"""
        samples, stats = self.hippocampus.get_risk_weighted_replay(n_replay)
        
        if samples:
            importance = [1.0 + stats['avg_risk']] * len(samples)
            self.cortex.consolidate(samples, importance)
        
        self.consolidation_count += 1
        
        return len(samples), stats


class SimpleNeuronLayer:
    """简单神经元层（无HM，作为对照）"""
    
    def __init__(self, n_input: int, n_output: int, lr: float = 0.05):
        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        
        self.W = np.random.randn(n_output, n_input) * 0.1
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x
    
    def learn(self, x: np.ndarray, target: np.ndarray):
        pred = self.forward(x)
        error = target - pred
        error = np.clip(error, -1.0, 1.0)
        
        grad = np.outer(error, x)
        grad = np.clip(grad, -1.0, 1.0)
        self.W += self.lr * grad


class SystemWithHMV2:
    """带HM记忆层V2的完整系统"""
    
    def __init__(self, n_features: int = 8, n_hidden: int = 4):
        self.hm = HMNeuronLayerV2(n_features, n_hidden, 
                                  hippocampus_lr=0.1, 
                                  cortex_lr=0.01)
        self.output = SimpleNeuronLayer(n_hidden, 1, lr=0.05)
        
        self.W_encode = np.random.randn(n_features, 1) * 0.3 + 0.3
        self.current_task = 0
        
    def forward(self, x: float) -> float:
        x_vec = np.array([[x]])
        encoded = self.W_encode @ x_vec
        hidden = self.hm.forward(encoded.flatten())
        output = self.output.forward(hidden)
        return output[0]
    
    def learn(self, x: float, target: float):
        x_vec = np.array([[x]])
        encoded = self.W_encode @ x_vec
        
        hidden = self.hm.forward(encoded.flatten())
        pred = self.output.forward(hidden)[0]
        
        error = target - pred
        error = np.clip(error, -2.0, 2.0)
        
        target_hidden = hidden + 0.1 * np.sign(error)
        self.output.learn(hidden, np.array([target]))
        
        self.hm.learn(encoded.flatten(), target_hidden, task_id=self.current_task)
    
    def sleep(self, n_replay: int = 60) -> Tuple[int, Dict]:
        return self.hm.sleep(n_replay)


class SystemWithoutHM:
    """无HM记忆层的对照系统"""
    
    def __init__(self, n_features: int = 8, n_hidden: int = 4):
        self.hidden = SimpleNeuronLayer(n_features, n_hidden, lr=0.05)
        self.output = SimpleNeuronLayer(n_hidden, 1, lr=0.05)
        
        self.W_encode = np.random.randn(n_features, 1) * 0.3 + 0.3
        
    def forward(self, x: float) -> float:
        x_vec = np.array([[x]])
        encoded = self.W_encode @ x_vec
        hidden = self.hidden.forward(encoded.flatten())
        output = self.output.forward(hidden)
        return output[0]
    
    def learn(self, x: float, target: float):
        x_vec = np.array([[x]])
        encoded = self.W_encode @ x_vec
        
        hidden = self.hidden.forward(encoded.flatten())
        pred = self.output.forward(hidden)[0]
        
        error = target - pred
        error = np.clip(error, -2.0, 2.0)
        
        target_hidden = hidden + 0.1 * np.sign(error)
        self.output.learn(hidden, np.array([target]))
        
        self.hidden.learn(encoded.flatten(), target_hidden)


def generate_task_data(task_id: int, n: int, noise_level: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
    """生成8个干扰任务"""
    x = np.linspace(0, 1, n)
    np.random.seed(42 + task_id)
    noise = np.random.randn(n) * noise_level
    
    if task_id == 0:
        y = np.sin(2 * np.pi * x)
    elif task_id == 1:
        y = -np.sin(2 * np.pi * x)
    elif task_id == 2:
        y = np.sin(4 * np.pi * x)
    elif task_id == 3:
        y = -np.sin(4 * np.pi * x)
    elif task_id == 4:
        y = 2 * x - 1
    elif task_id == 5:
        y = 1 - 2 * x
    elif task_id == 6:
        y = np.sin(2 * np.pi * x + np.pi / 2)
    elif task_id == 7:
        y = np.sin(2 * np.pi * x + 3 * np.pi / 2)
    else:
        y = np.sin(2 * np.pi * x * (task_id + 1))
    
    y = y + noise
    y = np.clip(y, -1.5, 1.5)
    
    return x, y


def train_on_task(system, x_data: np.ndarray, y_data: np.ndarray, n_epochs: int = 80) -> float:
    """在单个任务上训练"""
    total_loss = 0.0
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for x, y in zip(x_data, y_data):
            pred = system.forward(x)
            system.learn(x, y)
            epoch_loss += (y - pred) ** 2
        
        total_loss = epoch_loss / len(x_data)
    
    return total_loss


def evaluate_task(system, x_data: np.ndarray, y_data: np.ndarray) -> float:
    """评估在任务上的性能"""
    total_loss = 0.0
    
    for x, y in zip(x_data, y_data):
        pred = system.forward(x)
        total_loss += (y - pred) ** 2
    
    return total_loss / len(x_data)


def run_forgetting_experiment():
    """运行遗忘实验"""
    
    print("=" * 70)
    print("HM记忆层V2遗忘实验（8任务 + 风险加权重放）")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_tasks = 8
    task_names = [
        "sin(2πx)", "-sin(2πx)", 
        "sin(4πx)", "-sin(4πx)",
        "2x-1", "1-2x",
        "sin(2πx+π/2)", "sin(2πx+3π/2)"
    ]
    
    print("\n[准备] 生成8个干扰任务...")
    tasks = [generate_task_data(i, 100, noise_level=0.03) for i in range(n_tasks)]
    for i, (name, (x, y)) in enumerate(zip(task_names, tasks)):
        print(f"  任务{i}: {name}")
    
    print("\n" + "=" * 70)
    print("实验组: 带HM记忆层V2（风险加权重放）")
    print("=" * 70)
    
    np.random.seed(42)
    system_hm = SystemWithHMV2(n_features=8, n_hidden=4)
    
    hm_performance = {i: [] for i in range(n_tasks)}
    hm_replay_stats = []
    
    for task_id in range(n_tasks):
        print(f"\n[阶段{task_id+1}] 训练任务{task_id} ({task_names[task_id]})...")
        
        system_hm.current_task = task_id
        x_train, y_train = tasks[task_id]
        train_loss = train_on_task(system_hm, x_train, y_train, n_epochs=80)
        print(f"  训练损失: {train_loss:.6f}")
        
        n_replay, stats = system_hm.sleep(n_replay=60)
        hm_replay_stats.append(stats)
        print(f"  系统巩固: 重放 {n_replay} 个样本")
        print(f"    平均风险: {stats['avg_risk']:.3f}")
        print(f"    高风险比例: {stats['risk_distribution']['high_risk_ratio']:.1%}")
        
        print("  评估所有任务:")
        for i in range(n_tasks):
            x_eval, y_eval = tasks[i]
            perf = evaluate_task(system_hm, x_eval, y_eval)
            hm_performance[i].append(perf)
            marker = "★" if i == task_id else " "
            print(f"    {marker} 任务{i}: MSE = {perf:.6f}")
    
    print("\n" + "=" * 70)
    print("对照组: 无HM记忆层")
    print("=" * 70)
    
    np.random.seed(42)
    system_no_hm = SystemWithoutHM(n_features=8, n_hidden=4)
    
    no_hm_performance = {i: [] for i in range(n_tasks)}
    
    for task_id in range(n_tasks):
        print(f"\n[阶段{task_id+1}] 训练任务{task_id} ({task_names[task_id]})...")
        
        x_train, y_train = tasks[task_id]
        train_loss = train_on_task(system_no_hm, x_train, y_train, n_epochs=80)
        print(f"  训练损失: {train_loss:.6f}")
        
        print("  评估所有任务:")
        for i in range(n_tasks):
            x_eval, y_eval = tasks[i]
            perf = evaluate_task(system_no_hm, x_eval, y_eval)
            no_hm_performance[i].append(perf)
            marker = "★" if i == task_id else " "
            print(f"    {marker} 任务{i}: MSE = {perf:.6f}")
    
    print("\n" + "=" * 70)
    print("遗忘分析")
    print("=" * 70)
    
    print("\n任务0 (sin(2πx)) 的性能变化:")
    print("-" * 60)
    print(f"{'阶段':<8} {'带HM':<12} {'无HM':<12} {'HM优势':<12}")
    print("-" * 60)
    
    hm_task0 = hm_performance[0]
    no_hm_task0 = no_hm_performance[0]
    
    for stage, (hm_val, no_hm_val) in enumerate(zip(hm_task0, no_hm_task0)):
        if no_hm_val > 0:
            advantage = (no_hm_val - hm_val) / no_hm_val * 100
        else:
            advantage = 0
        print(f"阶段{stage+1:<5} {hm_val:<12.6f} {no_hm_val:<12.6f} {advantage:+.1f}%")
    
    final_hm = hm_task0[-1]
    final_no_hm = no_hm_task0[-1]
    initial_hm = hm_task0[0]
    initial_no_hm = no_hm_task0[0]
    
    if initial_hm > 0:
        hm_forgetting = (final_hm - initial_hm) / initial_hm * 100
    else:
        hm_forgetting = 0
    
    if initial_no_hm > 0:
        no_hm_forgetting = (final_no_hm - initial_no_hm) / initial_no_hm * 100
    else:
        no_hm_forgetting = 0
    
    print("\n遗忘率:")
    print(f"  带HM: {hm_forgetting:+.1f}%")
    print(f"  无HM: {no_hm_forgetting:+.1f}%")
    print(f"  HM缓解效果: {no_hm_forgetting - hm_forgetting:.1f}个百分点")
    
    hm_target_met = hm_forgetting < 50
    print(f"\n  目标: 带HM遗忘率<50%: {'✓ 达成' if hm_target_met else '✗ 未达成'}")
    
    print("\n所有任务最终性能对比:")
    print("-" * 60)
    print(f"{'任务':<15} {'带HM':<10} {'无HM':<10} {'HM优势':<10}")
    print("-" * 60)
    
    total_hm = 0
    total_no_hm = 0
    
    for i, name in enumerate(task_names):
        short_name = name[:10]
        hm_val = hm_performance[i][-1]
        no_hm_val = no_hm_performance[i][-1]
        if no_hm_val > 0:
            advantage = (no_hm_val - hm_val) / no_hm_val * 100
        else:
            advantage = 0
        print(f"{short_name:<13} {hm_val:<10.4f} {no_hm_val:<10.4f} {advantage:+.1f}%")
        total_hm += hm_val
        total_no_hm += no_hm_val
    
    print("-" * 60)
    if total_no_hm > 0:
        total_advantage = (total_no_hm - total_hm) / total_no_hm * 100
    else:
        total_advantage = 0
    print(f"{'平均':<13} {total_hm/n_tasks:<10.4f} {total_no_hm/n_tasks:<10.4f} {total_advantage:+.1f}%")
    
    results = {
        'n_tasks': n_tasks,
        'hm_performance': {k: v for k, v in hm_performance.items()},
        'no_hm_performance': {k: v for k, v in no_hm_performance.items()},
        'forgetting_rate': {
            'with_hm': float(hm_forgetting),
            'without_hm': float(no_hm_forgetting),
            'mitigation': float(no_hm_forgetting - hm_forgetting)
        },
        'target_met': bool(hm_target_met),
        'replay_stats': hm_replay_stats
    }
    
    with open('/workspace/mvp_v5_1_hm_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n结果已保存到 mvp_v5_1_hm_results.json")
    
    return results


if __name__ == '__main__':
    results = run_forgetting_experiment()
