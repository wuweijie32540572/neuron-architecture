"""
MVP v5: HM记忆层验证（修正版）
=============================
验证海马体-皮层架构是否缓解灾难性遗忘

关键修正：
- 使用真正干扰的任务（不同函数映射）
- 加入噪声防止过拟合
- 测量真正的遗忘而非正向迁移
"""

import numpy as np
from typing import Tuple, List, Dict
import json


class HippocampusLayer:
    """海马体：快速学习，短期记忆"""
    
    def __init__(self, n_input: int, n_output: int, lr: float = 0.1):
        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        
        self.W = np.random.randn(n_output, n_input) * 0.1
        self.memory_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self.buffer_size = 200
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x
    
    def learn(self, x: np.ndarray, target: np.ndarray):
        pred = self.forward(x)
        error = target - pred
        error = np.clip(error, -1.0, 1.0)
        
        grad = np.outer(error, x)
        grad = np.clip(grad, -1.0, 1.0)
        self.W += self.lr * grad
        
        self.memory_buffer.append((x.copy(), target.copy()))
        if len(self.memory_buffer) > self.buffer_size:
            self.memory_buffer.pop(0)
    
    def get_replay_samples(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        if len(self.memory_buffer) == 0:
            return []
        indices = np.random.choice(len(self.memory_buffer), min(n, len(self.memory_buffer)), replace=False)
        return [self.memory_buffer[i] for i in indices]


class CortexLayer:
    """皮层：慢速学习，长期记忆"""
    
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
    
    def consolidate(self, samples: List[Tuple[np.ndarray, np.ndarray]]):
        for x, target in samples:
            self.learn(x, target)


class HMNeuronLayer:
    """海马体-皮层混合记忆层"""
    
    def __init__(self, n_input: int, n_output: int, 
                 hippocampus_lr: float = 0.05,
                 cortex_lr: float = 0.02,
                 tau_consolidation: float = 10.0):
        self.hippocampus = HippocampusLayer(n_input, n_output, lr=hippocampus_lr)
        self.cortex = CortexLayer(n_input, n_output, lr=cortex_lr)
        
        self.tau = tau_consolidation
        self.time_since_consolidation = 0.0
        self.consolidation_interval = 20
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
    
    def learn(self, x: np.ndarray, target: np.ndarray):
        self.hippocampus.learn(x, target)
        self.cortex.learn(x, target)
        
        self.time_since_consolidation += 1
    
    def sleep(self, n_replay: int = 50):
        samples = self.hippocampus.get_replay_samples(n_replay)
        self.cortex.consolidate(samples)
        
        self.time_since_consolidation = 0.0
        self.consolidation_count += 1
        
        return len(samples)


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


class SystemWithHM:
    """带HM记忆层的完整系统"""
    
    def __init__(self, n_features: int = 16, n_hidden: int = 16):
        self.hm = HMNeuronLayer(n_features, n_hidden, 
                                hippocampus_lr=0.1, 
                                cortex_lr=0.01,
                                tau_consolidation=20.0)
        self.output = SimpleNeuronLayer(n_hidden, 1, lr=0.05)
        
        self.W_encode = np.random.randn(n_features, 1) * 0.3 + 0.3
        
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
        
        self.hm.learn(encoded.flatten(), target_hidden)
    
    def sleep(self, n_replay: int = 50):
        return self.hm.sleep(n_replay)


class SystemWithoutHM:
    """无HM记忆层的对照系统"""
    
    def __init__(self, n_features: int = 16, n_hidden: int = 16):
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


def generate_task_data(task_id: int, n: int, noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成真正干扰的任务数据 - 极端冲突版本
    
    任务设计：
    - 任务0: y = sin(2πx) 周期映射
    - 任务1: y = -sin(2πx) 反向周期（与任务0完全冲突）
    - 任务2: y = 2x - 1 线性（与周期冲突）
    - 任务3: y = sin(2πx + π) = -sin(2πx) 相位偏移（再次冲突）
    
    这些任务在权重空间上强烈冲突，学习新任务必然破坏旧任务
    """
    x = np.linspace(0, 1, n)
    np.random.seed(42 + task_id)
    noise = np.random.randn(n) * noise_level
    
    if task_id == 0:
        y = np.sin(2 * np.pi * x)
    elif task_id == 1:
        y = -np.sin(2 * np.pi * x)
    elif task_id == 2:
        y = 2 * x - 1
    elif task_id == 3:
        y = np.sin(2 * np.pi * x + np.pi)
    else:
        y = np.sin(2 * np.pi * x * (task_id + 1))
    
    y = y + noise
    y = np.clip(y, -1.5, 1.5)
    
    return x, y


def train_on_task(system, x_data: np.ndarray, y_data: np.ndarray, n_epochs: int = 50) -> float:
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
    print("HM记忆层遗忘实验（真正干扰任务）")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_tasks = 4
    task_names = ["周期sin(2πx)", "反向-sin(2πx)", "线性2x-1", "相位sin(2πx+π)"]
    
    print("\n[准备] 生成干扰任务数据...")
    tasks = [generate_task_data(i, 100, noise_level=0.05) for i in range(n_tasks)]
    for i, (name, (x, y)) in enumerate(zip(task_names, tasks)):
        print(f"  任务{i} ({name}):")
        print(f"    输入: [{x.min():.3f}, {x.max():.3f}], 目标: [{y.min():.3f}, {y.max():.3f}]")
    
    print("\n" + "=" * 70)
    print("实验组: 带HM记忆层")
    print("=" * 70)
    
    np.random.seed(42)
    system_hm = SystemWithHM(n_features=8, n_hidden=4)
    
    hm_performance = {i: [] for i in range(n_tasks)}
    
    for task_id in range(n_tasks):
        print(f"\n[阶段{task_id+1}] 训练任务{task_id} ({task_names[task_id]})...")
        
        x_train, y_train = tasks[task_id]
        train_loss = train_on_task(system_hm, x_train, y_train, n_epochs=100)
        print(f"  训练损失: {train_loss:.6f}")
        
        n_replay = system_hm.sleep(n_replay=40)
        print(f"  系统巩固: 重放 {n_replay} 个样本")
        
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
        train_loss = train_on_task(system_no_hm, x_train, y_train, n_epochs=100)
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
    
    print("\n任务0 (周期映射) 的性能变化:")
    print("-" * 60)
    print(f"{'阶段':<10} {'带HM':<15} {'无HM':<15} {'HM优势':<15}")
    print("-" * 60)
    
    hm_task0 = hm_performance[0]
    no_hm_task0 = no_hm_performance[0]
    
    for stage, (hm_val, no_hm_val) in enumerate(zip(hm_task0, no_hm_task0)):
        if no_hm_val > 0:
            advantage = (no_hm_val - hm_val) / no_hm_val * 100
        else:
            advantage = 0
        print(f"阶段{stage+1:<6} {hm_val:<15.6f} {no_hm_val:<15.6f} {advantage:+.1f}%")
    
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
    
    hm_effective = hm_forgetting < no_hm_forgetting and (no_hm_forgetting - hm_forgetting) > 10
    print(f"\n  判定: {'✓ HM有效缓解遗忘（差距>10%）' if hm_effective else '✗ HM效果不显著'}")
    
    print("\n所有任务最终性能对比:")
    print("-" * 60)
    print(f"{'任务':<18} {'带HM':<12} {'无HM':<12} {'HM优势':<12}")
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
        print(f"{short_name:<16} {hm_val:<12.6f} {no_hm_val:<12.6f} {advantage:+.1f}%")
        total_hm += hm_val
        total_no_hm += no_hm_val
    
    print("-" * 60)
    if total_no_hm > 0:
        total_advantage = (total_no_hm - total_hm) / total_no_hm * 100
    else:
        total_advantage = 0
    print(f"{'平均':<16} {total_hm/4:<12.6f} {total_no_hm/4:<12.6f} {total_advantage:+.1f}%")
    
    results = {
        'hm_performance': {k: v for k, v in hm_performance.items()},
        'no_hm_performance': {k: v for k, v in no_hm_performance.items()},
        'forgetting_rate': {
            'with_hm': float(hm_forgetting),
            'without_hm': float(no_hm_forgetting),
            'mitigation': float(no_hm_forgetting - hm_forgetting)
        },
        'final_avg_mse': {
            'with_hm': float(total_hm / 4),
            'without_hm': float(total_no_hm / 4)
        },
        'hm_effective': bool(hm_effective)
    }
    
    with open('/workspace/mvp_v5_hm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到 mvp_v5_hm_results.json")
    
    return results


if __name__ == '__main__':
    results = run_forgetting_experiment()
