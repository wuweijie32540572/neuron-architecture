"""
MVP v6.2: DA机制重新设计
========================
核心修正：
1. DA控制探索/利用权衡，而非简单学习率缩放
2. 自适应学习率：η(t) = η_base / (1 + t·(1-DA))
3. 奖励预测误差驱动：DA = sigmoid(R_actual - R_predicted)
4. 验证：高DA初期快（探索），低DA后期稳（利用）
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class NeuromodulatorState:
    """神经调质状态"""
    da: float = 0.5
    serotonin: float = 0.5
    ach: float = 0.5
    ne: float = 0.5
    
    def to_array(self) -> np.ndarray:
        return np.array([self.da, self.serotonin, self.ach, self.ne])


class NGNeuronLayerV3:
    """
    神经调质门控神经元层V3
    
    DA机制重新设计：
    - 探索阶段（高DA）：高学习率，快速探索
    - 利用阶段（低DA）：低学习率，精细调整
    - 自适应衰减：η(t) = η_base / (1 + t·(1-DA))
    """
    
    def __init__(self, n_input: int, n_output: int, 
                 base_lr: float = 0.05,
                 alpha_da: float = 1.0,
                 beta_5ht: float = 2.0,
                 gamma_ach: float = 0.5,
                 delta_ne: float = 0.5):
        self.n_input = n_input
        self.n_output = n_output
        self.base_lr = base_lr
        
        self.alpha_da = alpha_da
        self.beta_5ht = beta_5ht
        self.gamma_ach = gamma_ach
        self.delta_ne = delta_ne
        
        self.W = np.random.randn(n_output, n_input) * 0.1
        self.nm_state = NeuromodulatorState()
        
        self.step_count = 0
        self.reward_prediction = 0.5
        
    def forward(self, x: np.ndarray, nm: Optional[NeuromodulatorState] = None) -> np.ndarray:
        if nm is not None:
            self.nm_state = nm
        
        gain = 1.0 + self.delta_ne * self.nm_state.ne
        output = gain * (self.W @ x)
        
        return output
    
    def compute_effective_lr(self) -> float:
        """
        自适应学习率
        
        η(t) = η_base / (1 + t·(1-DA))
        
        - 高DA (接近1)：衰减慢，保持探索能力
        - 低DA (接近0)：衰减快，快速稳定
        """
        decay_factor = 1.0 + self.step_count * (1.0 - self.nm_state.da) * 0.01
        effective_lr = self.base_lr / decay_factor
        
        explore_boost = 1.0 + self.alpha_da * self.nm_state.da * 0.5
        effective_lr *= explore_boost
        
        return np.clip(effective_lr, 0.001, 0.3)
    
    def update_da_from_rpe(self, reward: float):
        """
        从奖励预测误差更新DA
        
        DA = sigmoid(R_actual - R_predicted)
        
        - 正RPE（意外奖励）：DA↑ → 继续探索
        - 负RPE（预期未实现）：DA↓ → 开始利用
        """
        rpe = reward - self.reward_prediction
        
        self.reward_prediction = 0.9 * self.reward_prediction + 0.1 * reward
        
        da_update = 1.0 / (1.0 + np.exp(-5.0 * rpe))
        self.nm_state.da = 0.7 * self.nm_state.da + 0.3 * da_update
        self.nm_state.da = np.clip(self.nm_state.da, 0.0, 1.0)
        
        return rpe
    
    def learn(self, x: np.ndarray, target: np.ndarray, 
              reward: float = 0.5, nm: Optional[NeuromodulatorState] = None) -> Dict:
        
        if nm is not None:
            self.nm_state = nm
        
        rpe = self.update_da_from_rpe(reward)
        
        effective_lr = self.compute_effective_lr()
        
        pred = self.forward(x)
        error = target - pred
        
        if self.nm_state.ach > 0.5:
            sparsity_threshold = self.gamma_ach * self.nm_state.ach
            mask = np.abs(error) > sparsity_threshold
            error = error * mask
        
        explore_prob = 1.0 / (1.0 + np.exp(-self.beta_5ht * (self.nm_state.serotonin - 0.5)))
        if np.random.rand() < explore_prob * 0.1 * self.nm_state.da:
            noise = np.random.randn(*error.shape) * 0.2
            error = error + noise
        
        grad = np.outer(error, x)
        grad = np.clip(grad, -1.0, 1.0)
        
        self.W += effective_lr * grad
        
        self.step_count += 1
        
        return {
            'effective_lr': effective_lr,
            'rpe': rpe,
            'da': self.nm_state.da,
            'step': self.step_count
        }
    
    def set_neuromodulators(self, da: float = None, serotonin: float = None, 
                            ach: float = None, ne: float = None):
        if da is not None:
            self.nm_state.da = np.clip(da, 0.0, 1.0)
        if serotonin is not None:
            self.nm_state.serotonin = np.clip(serotonin, 0.0, 1.0)
        if ach is not None:
            self.nm_state.ach = np.clip(ach, 0.0, 1.0)
        if ne is not None:
            self.nm_state.ne = np.clip(ne, 0.0, 1.0)
    
    def reset_counters(self):
        self.step_count = 0
        self.reward_prediction = 0.5


def generate_orthogonal_patterns(n_patterns: int, dim: int) -> np.ndarray:
    """生成正交输入模式"""
    patterns = np.zeros((n_patterns, dim))
    block_size = dim // n_patterns
    
    for i in range(n_patterns):
        start = i * block_size
        end = start + block_size if i < n_patterns - 1 else dim
        patterns[i, start:end] = 1.0
        patterns[i] += np.random.randn(dim) * 0.05
    
    patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)
    return patterns


class SimpleSystemV3:
    """简单系统用于DA实验"""
    
    def __init__(self, n_input: int = 16, n_hidden: int = 8):
        self.hidden = NGNeuronLayerV3(n_input, n_hidden, base_lr=0.05)
        self.output = NGNeuronLayerV3(n_hidden, 1, base_lr=0.05)
        
        self.W_encode = np.random.randn(n_input, 1) * 0.3 + 0.3
        
    def forward(self, x: np.ndarray) -> float:
        encoded = x
        hidden = self.hidden.forward(encoded)
        output = self.output.forward(hidden)
        return output[0]
    
    def learn(self, x: np.ndarray, target: float) -> Dict:
        encoded = x
        
        hidden = self.hidden.forward(encoded)
        pred = self.output.forward(hidden)[0]
        
        error = target - pred
        reward = 1.0 - abs(error)
        
        hidden_target = hidden + 0.1 * np.sign(error)
        
        hidden_result = self.hidden.learn(encoded.flatten(), hidden_target, reward)
        output_result = self.output.learn(hidden, np.array([target]), reward)
        
        return {
            'prediction': pred,
            'error': float(error),
            'reward': float(reward),
            'effective_lr': output_result['effective_lr'],
            'da': output_result['da'],
            'rpe': output_result['rpe']
        }
    
    def set_da(self, da: float):
        self.hidden.set_neuromodulators(da=da)
        self.output.set_neuromodulators(da=da)
    
    def reset(self):
        self.hidden.reset_counters()
        self.output.reset_counters()


def run_da_experiment():
    """运行DA机制实验"""
    
    print("=" * 70)
    print("DA机制重新设计实验")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_input = 16
    test_pattern = generate_orthogonal_patterns(1, n_input)[0]
    target_output = 0.8
    
    print("\n[实验1] DA对学习率衰减的影响")
    print("-" * 60)
    
    print("\n理论预测：")
    print("  η(t) = η_base / (1 + t·(1-DA))")
    print("  - 高DA: 衰减慢，保持探索能力")
    print("  - 低DA: 衰减快，快速稳定")
    
    print("\n学习率随时间变化:")
    print(f"{'步数':<8} {'高DA(0.9)':<12} {'中DA(0.5)':<12} {'低DA(0.1)':<12}")
    print("-" * 50)
    
    base_lr = 0.05
    for t in [0, 10, 50, 100, 200, 500]:
        lr_high = base_lr / (1 + t * (1 - 0.9) * 0.01) * (1 + 1.0 * 0.9 * 0.5)
        lr_mid = base_lr / (1 + t * (1 - 0.5) * 0.01) * (1 + 1.0 * 0.5 * 0.5)
        lr_low = base_lr / (1 + t * (1 - 0.1) * 0.01) * (1 + 1.0 * 0.1 * 0.5)
        print(f"{t:<8} {lr_high:<12.4f} {lr_mid:<12.4f} {lr_low:<12.4f}")
    
    print("\n[实验2] DA对收敛的影响（固定DA）")
    print("-" * 60)
    
    results_by_da = {}
    
    for da_level in ['high', 'mid', 'low']:
        np.random.seed(42)
        system = SimpleSystemV3(n_input=16, n_hidden=8)
        
        if da_level == 'high':
            system.set_da(0.9)
        elif da_level == 'mid':
            system.set_da(0.5)
        else:
            system.set_da(0.1)
        
        errors = []
        effective_lrs = []
        rewards = []
        
        for _ in range(200):
            result = system.learn(test_pattern, target_output)
            errors.append(abs(result['error']))
            effective_lrs.append(result['effective_lr'])
            rewards.append(result['reward'])
        
        results_by_da[da_level] = {
            'errors': errors,
            'effective_lrs': effective_lrs,
            'rewards': rewards,
            'final_error': errors[-1],
            'convergence_speed': np.mean(errors[:20]) / (np.mean(errors[-20:]) + 1e-6)
        }
    
    print("\n学习曲线:")
    print(f"{'步数':<8} {'高DA误差':<12} {'中DA误差':<12} {'低DA误差':<12}")
    print("-" * 50)
    for i in [0, 19, 49, 99, 199]:
        high_err = results_by_da['high']['errors'][i]
        mid_err = results_by_da['mid']['errors'][i]
        low_err = results_by_da['low']['errors'][i]
        print(f"{i+1:<8} {high_err:<12.4f} {mid_err:<12.4f} {low_err:<12.4f}")
    
    print("\n收敛分析:")
    for da_level in ['high', 'mid', 'low']:
        final_err = results_by_da[da_level]['final_error']
        conv_speed = results_by_da[da_level]['convergence_speed']
        print(f"  {da_level}DA: 最终误差={final_err:.4f}, 收敛速度={conv_speed:.2f}x")
    
    print("\n[实验3] 动态DA（奖励预测误差驱动）")
    print("-" * 60)
    
    np.random.seed(42)
    system_dynamic = SimpleSystemV3(n_input=16, n_hidden=8)
    system_dynamic.set_da(0.5)
    
    errors_dynamic = []
    da_values = []
    effective_lrs_dynamic = []
    rpes = []
    
    for i in range(200):
        result = system_dynamic.learn(test_pattern, target_output)
        errors_dynamic.append(abs(result['error']))
        da_values.append(result['da'])
        effective_lrs_dynamic.append(result['effective_lr'])
        rpes.append(result['rpe'])
    
    print("\n动态DA变化:")
    print(f"{'步数':<8} {'误差':<10} {'DA':<10} {'有效学习率':<12} {'RPE':<10}")
    print("-" * 55)
    for i in [0, 19, 49, 99, 199]:
        print(f"{i+1:<8} {errors_dynamic[i]:<10.4f} {da_values[i]:<10.4f} "
              f"{effective_lrs_dynamic[i]:<12.4f} {rpes[i]:<10.4f}")
    
    print("\n[实验4] 验证：高DA初期快，低DA后期稳")
    print("-" * 60)
    
    high_early = np.mean(results_by_da['high']['errors'][:30])
    high_late = np.mean(results_by_da['high']['errors'][-30:])
    low_early = np.mean(results_by_da['low']['errors'][:30])
    low_late = np.mean(results_by_da['low']['errors'][-30:])
    
    print(f"\n初期误差（前30步）:")
    print(f"  高DA: {high_early:.4f}")
    print(f"  低DA: {low_early:.4f}")
    
    print(f"\n后期误差（后30步）:")
    print(f"  高DA: {high_late:.4f}")
    print(f"  低DA: {low_late:.4f}")
    
    early_benefit = (low_early - high_early) / low_early * 100
    late_benefit = (high_late - low_late) / high_late * 100
    
    print(f"\n相对优势:")
    print(f"  初期高DA优势: {early_benefit:+.1f}%")
    print(f"  后期低DA优势: {late_benefit:+.1f}%")
    
    da_correct = early_benefit > 0 and late_benefit > 0
    
    print("\n" + "=" * 70)
    print("实验总结")
    print("=" * 70)
    
    print(f"\n验证结果:")
    print(f"  初期高DA更快: {'✓' if early_benefit > 0 else '✗'} ({early_benefit:+.1f}%)")
    print(f"  后期低DA更稳: {'✓' if late_benefit > 0 else '✗'} ({late_benefit:+.1f}%)")
    print(f"\n总体判定: {'✓ DA机制正确' if da_correct else '✗ DA机制需进一步优化'}")
    
    results = {
        'fixed_da_results': {
            'high': {
                'final_error': float(results_by_da['high']['final_error']),
                'convergence_speed': float(results_by_da['high']['convergence_speed'])
            },
            'mid': {
                'final_error': float(results_by_da['mid']['final_error']),
                'convergence_speed': float(results_by_da['mid']['convergence_speed'])
            },
            'low': {
                'final_error': float(results_by_da['low']['final_error']),
                'convergence_speed': float(results_by_da['low']['convergence_speed'])
            }
        },
        'dynamic_da': {
            'final_error': float(errors_dynamic[-1]),
            'final_da': float(da_values[-1])
        },
        'validation': {
            'early_high_da_benefit_pct': float(early_benefit),
            'late_low_da_benefit_pct': float(late_benefit),
            'da_mechanism_correct': bool(da_correct)
        }
    }
    
    with open('/workspace/mvp_v6_2_da_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到 mvp_v6_2_da_results.json")
    
    return results


if __name__ == '__main__':
    results = run_da_experiment()
