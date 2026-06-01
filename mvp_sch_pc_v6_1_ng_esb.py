"""
MVP v6.1: NG+ESB 修复版
======================
修复问题：
1. 正交输入模式：确保不同符号的输入在空间上正交
2. 奖励信号与误差挂钩：reward = -|预测误差|
3. 验证符号识别的真实准确率（置信度应>>0.25）
4. 验证DA学习曲线（高DA应更快收敛）
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


class NGNeuronLayerV2:
    """
    神经调质门控神经元层V2
    
    修复：DA与奖励预测误差挂钩
    """
    
    def __init__(self, n_input: int, n_output: int, 
                 base_lr: float = 0.05,
                 alpha_da: float = 0.5,
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
        
    def forward(self, x: np.ndarray, nm: Optional[NeuromodulatorState] = None) -> np.ndarray:
        if nm is not None:
            self.nm_state = nm
        
        gain = 1.0 + self.delta_ne * self.nm_state.ne
        output = gain * (self.W @ x)
        
        return output
    
    def learn(self, x: np.ndarray, target: np.ndarray, 
              prediction_error: float = 0.0, nm: Optional[NeuromodulatorState] = None,
              update_da: bool = True) -> float:
        """
        学习
        
        修复：DA由预测误差驱动
        - 正DA：正奖励预测误差（比预期好）→ 加速学习
        - 负DA：负奖励预测误差（比预期差）→ 减速学习
        """
        if nm is not None:
            self.nm_state = nm
        
        if update_da:
            reward_prediction_error = 0.5 - prediction_error
            self.nm_state.da = np.clip(0.5 + 0.3 * reward_prediction_error, 0.0, 1.0)
        
        effective_lr = self.base_lr * (1.0 + self.alpha_da * (self.nm_state.da - 0.5))
        effective_lr = np.clip(effective_lr, 0.001, 0.2)
        
        pred = self.forward(x)
        error = target - pred
        
        if self.nm_state.ach > 0.5:
            sparsity_threshold = self.gamma_ach * self.nm_state.ach
            mask = np.abs(error) > sparsity_threshold
            error = error * mask
        
        explore_prob = 1.0 / (1.0 + np.exp(-self.beta_5ht * (self.nm_state.serotonin - 0.5)))
        if np.random.rand() < explore_prob * 0.1:
            noise = np.random.randn(*error.shape) * 0.1
            error = error + noise
        
        grad = np.outer(error, x)
        grad = np.clip(grad, -1.0, 1.0)
        
        self.W += effective_lr * grad
        
        return effective_lr
    
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


class ESBNeuronLayerV2:
    """
    具身-符号桥接神经元层V2
    
    修复：使用正交输入模式
    """
    
    def __init__(self, n_embodied: int, n_latent: int, n_symbols: int):
        self.n_embodied = n_embodied
        self.n_latent = n_latent
        self.n_symbols = n_symbols
        
        self.W_encode = np.random.randn(n_latent, n_embodied) * 0.1
        self.W_decode = np.random.randn(n_symbols, n_latent) * 0.1
        
        self.symbol_embeddings = np.random.randn(n_symbols, n_latent) * 0.1
        self.grounding_matrix = np.zeros((n_symbols, n_embodied))
        
        self.symbol_labels = [f"concept_{i}" for i in range(n_symbols)]
        
    def encode_embodied(self, sensor_input: np.ndarray) -> np.ndarray:
        latent = np.tanh(self.W_encode @ sensor_input)
        return latent
    
    def decode_symbol(self, latent: np.ndarray) -> Tuple[int, np.ndarray, float]:
        logits = self.W_decode @ latent
        probs = self._softmax(logits * 3.0)
        
        symbol_id = np.argmax(probs)
        confidence = probs[symbol_id]
        
        return symbol_id, probs, confidence
    
    def ground_symbol(self, symbol_id: int, embodied_example: np.ndarray, 
                      strength: float = 0.1):
        self.grounding_matrix[symbol_id] = (
            (1 - strength) * self.grounding_matrix[symbol_id] + 
            strength * embodied_example
        )
        
        latent = self.encode_embodied(embodied_example)
        self.symbol_embeddings[symbol_id] = (
            (1 - strength) * self.symbol_embeddings[symbol_id] + 
            strength * latent
        )
    
    def retrieve_embodied(self, symbol_id: int) -> np.ndarray:
        return self.grounding_matrix[symbol_id]
    
    def compute_symbolic_distance(self, latent: np.ndarray, symbol_id: int) -> float:
        return np.linalg.norm(latent - self.symbol_embeddings[symbol_id])
    
    def learn(self, sensor_input: np.ndarray, target_symbol: int, 
              lr: float = 0.05) -> Tuple[float, bool]:
        """学习具身-符号映射，返回(置信度, 是否正确)"""
        latent = self.encode_embodied(sensor_input)
        symbol_id, probs, confidence = self.decode_symbol(latent)
        
        is_correct = (symbol_id == target_symbol)
        
        target_logits = np.zeros(self.n_symbols)
        target_logits[target_symbol] = 1.0
        
        error = target_logits - probs
        
        grad_decode = np.outer(error, latent)
        grad_decode = np.clip(grad_decode, -1.0, 1.0)
        self.W_decode += lr * grad_decode
        
        grad_encode = self.W_decode.T @ error * (1 - latent ** 2)
        grad_encode = np.clip(grad_encode, -1.0, 1.0)
        self.W_encode += lr * np.outer(grad_encode, sensor_input)
            
        self.ground_symbol(target_symbol, sensor_input, strength=0.1)
        
        return confidence, is_correct
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)


def generate_orthogonal_patterns(n_patterns: int, dim: int) -> np.ndarray:
    """
    生成正交输入模式
    
    确保不同符号的输入在空间上正交，模型必须学会区分
    """
    patterns = np.zeros((n_patterns, dim))
    
    block_size = dim // n_patterns
    
    for i in range(n_patterns):
        start = i * block_size
        end = start + block_size if i < n_patterns - 1 else dim
        patterns[i, start:end] = 1.0
        
        patterns[i] += np.random.randn(dim) * 0.05
    
    patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)
    
    return patterns


class IntegratedNGESBSystemV2:
    """集成NG和ESB的完整系统V2"""
    
    def __init__(self, n_embodied: int = 16, n_latent: int = 8, 
                 n_symbols: int = 4, n_hidden: int = 8):
        self.ng = NGNeuronLayerV2(n_embodied, n_hidden, base_lr=0.02)
        self.esb = ESBNeuronLayerV2(n_embodied, n_latent, n_symbols)
        
        self.output_layer = NGNeuronLayerV2(n_hidden, 1, base_lr=0.02)
        
    def forward(self, sensor_input: np.ndarray, 
                nm: Optional[NeuromodulatorState] = None) -> Dict:
        ng_hidden = self.ng.forward(sensor_input, nm)
        output = self.output_layer.forward(ng_hidden, nm)
        
        latent = self.esb.encode_embodied(sensor_input)
        symbol_id, symbol_probs, confidence = self.esb.decode_symbol(latent)
        
        return {
            'output': output[0],
            'symbol_id': symbol_id,
            'symbol_probs': symbol_probs,
            'confidence': confidence,
            'latent': latent,
            'hidden': ng_hidden
        }
    
    def learn(self, sensor_input: np.ndarray, target_output: float,
              target_symbol: int, nm: Optional[NeuromodulatorState] = None) -> Dict:
        result = self.forward(sensor_input, nm)
        
        output_error = target_output - result['output']
        prediction_error = abs(output_error)
        
        reward = -prediction_error
        
        hidden = result['hidden']
        target_hidden = hidden + 0.1 * np.sign(output_error)
        
        lr1 = self.ng.learn(sensor_input, target_hidden, prediction_error, nm, update_da=False)
        lr2 = self.output_layer.learn(hidden, np.array([target_output]), prediction_error, nm, update_da=False)
        
        symbol_conf, symbol_correct = self.esb.learn(sensor_input, target_symbol, lr=0.1)
        
        return {
            'output_error': float(output_error),
            'prediction_error': float(prediction_error),
            'effective_lr': float(lr1),
            'symbol_confidence': float(symbol_conf),
            'symbol_correct': bool(symbol_correct),
            'reward': float(reward)
        }
    
    def set_state(self, da: float = None, serotonin: float = None,
                  ach: float = None, ne: float = None):
        self.ng.set_neuromodulators(da, serotonin, ach, ne)
        self.output_layer.set_neuromodulators(da, serotonin, ach, ne)


def run_ng_esb_experiment():
    """运行NG+ESB实验"""
    
    print("=" * 70)
    print("NG神经调质门控V2 + ESB具身-符号桥接V2 实验")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_embodied = 16
    n_latent = 8
    n_symbols = 4
    symbol_names = ["安全", "警告", "危险", "正常"]
    
    system = IntegratedNGESBSystemV2(n_embodied, n_latent, n_symbols, n_hidden=8)
    
    print("\n[实验1] 正交输入模式验证")
    print("-" * 60)
    
    patterns = generate_orthogonal_patterns(n_symbols, n_embodied)
    
    print("\n正交模式内积矩阵:")
    inner_products = patterns @ patterns.T
    for i in range(n_symbols):
        row = "  ".join([f"{inner_products[i,j]:.3f}" for j in range(n_symbols)])
        print(f"  [{row}]")
    
    print("\n模式正交性验证:")
    off_diag_sum = 0.0
    for i in range(n_symbols):
        for j in range(n_symbols):
            if i != j:
                off_diag_sum += abs(inner_products[i, j])
    print(f"  非对角线元素和: {off_diag_sum:.4f} (理想=0)")
    
    print("\n[实验2] 符号识别训练（正交模式）")
    print("-" * 60)
    
    target_symbols = list(range(n_symbols))
    
    print("\n训练中...")
    for epoch in range(100):
        epoch_correct = 0
        for pattern_idx, target_sym in enumerate(target_symbols):
            result = system.learn(patterns[pattern_idx], 0.5, target_sym)
            if result['symbol_correct']:
                epoch_correct += 1
        
        if (epoch + 1) % 20 == 0:
            accuracy = epoch_correct / n_symbols * 100
            print(f"  Epoch {epoch+1}: 准确率={accuracy:.0f}%")
    
    print("\n测试符号识别:")
    correct = 0
    total_confidence = 0.0
    for i, (pattern, target_sym) in enumerate(zip(patterns, target_symbols)):
        result = system.forward(pattern)
        predicted_sym = result['symbol_id']
        confidence = result['confidence']
        is_correct = predicted_sym == target_sym
        correct += int(is_correct)
        total_confidence += confidence
        print(f"  模式{i} ({symbol_names[target_sym]}): "
              f"预测={symbol_names[predicted_sym]}, "
              f"置信度={confidence:.3f}, "
              f"{'✓' if is_correct else '✗'}")
    
    accuracy = correct / n_symbols * 100
    avg_confidence = total_confidence / n_symbols
    print(f"\n符号识别准确率: {accuracy:.0f}%")
    print(f"平均置信度: {avg_confidence:.3f} (随机=0.25)")
    
    confidence_valid = avg_confidence > 0.4
    print(f"判定: {'✓ 置信度有效（>0.4）' if confidence_valid else '✗ 置信度不足'}")
    
    print("\n[实验3] Grounding验证")
    print("-" * 60)
    
    grounding_norms = []
    for sym_id in range(n_symbols):
        grounded = system.esb.retrieve_embodied(sym_id)
        norm = np.linalg.norm(grounded)
        grounding_norms.append(norm)
        print(f"  符号'{symbol_names[sym_id]}' 的具身grounding: 范数={norm:.4f}")
    
    grounding_variance = np.var(grounding_norms)
    print(f"\nGrounding范数方差: {grounding_variance:.6f}")
    print(f"判定: {'✓ Grounding有区分' if grounding_variance > 0.01 else '⚠ Grounding趋同'}")
    
    print("\n[实验4] DA对学习的影响（奖励=-|误差|）")
    print("-" * 60)
    
    results_by_da = {}
    
    for da_level in ['high', 'low']:
        system2 = IntegratedNGESBSystemV2(n_embodied, n_latent, n_symbols, n_hidden=8)
        
        if da_level == 'high':
            system2.set_state(da=0.9)
        else:
            system2.set_state(da=0.1)
        
        target_output = 0.8
        test_pattern = patterns[0]
        
        errors = []
        rewards = []
        
        for _ in range(50):
            result = system2.forward(test_pattern)
            error = abs(target_output - result['output'])
            errors.append(error)
            
            learn_result = system2.learn(test_pattern, target_output, 0)
            rewards.append(learn_result['reward'])
        
        results_by_da[da_level] = {
            'errors': errors,
            'rewards': rewards,
            'final_error': errors[-1],
            'convergence_speed': sum(errors[:10]) / sum(errors[-10:]) if sum(errors[-10:]) > 0 else 1.0
        }
    
    print("\n学习曲线 (目标输出=0.8):")
    print(f"{'轮次':<8} {'高DA误差':<12} {'低DA误差':<12} {'差异':<12}")
    print("-" * 50)
    for i in [0, 9, 24, 49]:
        high_err = results_by_da['high']['errors'][i]
        low_err = results_by_da['low']['errors'][i]
        diff = low_err - high_err
        print(f"{i+1:<8} {high_err:<12.4f} {low_err:<12.4f} {diff:+.4f}")
    
    high_final = results_by_da['high']['final_error']
    low_final = results_by_da['low']['final_error']
    
    print(f"\n最终误差:")
    print(f"  高DA: {high_final:.4f}")
    print(f"  低DA: {low_final:.4f}")
    
    da_correct = high_final < low_final
    print(f"判定: {'✓ 高DA收敛更快（符合预期）' if da_correct else '✗ DA效果异常'}")
    
    print("\n" + "=" * 70)
    print("实验总结")
    print("=" * 70)
    
    all_pass = confidence_valid and da_correct
    
    print(f"\n符号识别准确率: {accuracy:.0f}%")
    print(f"平均置信度: {avg_confidence:.3f} {'✓' if confidence_valid else '✗'}")
    print(f"DA学习效果: {'✓ 正确' if da_correct else '✗ 异常'}")
    print(f"\n总体判定: {'✓ 全部通过' if all_pass else '⚠ 部分问题'}")
    
    results = {
        'symbol_accuracy': float(accuracy),
        'avg_confidence': float(avg_confidence),
        'confidence_valid': bool(confidence_valid),
        'grounding_variance': float(grounding_variance),
        'da_learning_correct': bool(da_correct),
        'final_errors': {
            'high_DA': float(high_final),
            'low_DA': float(low_final)
        },
        'all_pass': bool(all_pass)
    }
    
    with open('/workspace/mvp_v6_1_ng_esb_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到 mvp_v6_1_ng_esb_results.json")
    
    return results


if __name__ == '__main__':
    results = run_ng_esb_experiment()
