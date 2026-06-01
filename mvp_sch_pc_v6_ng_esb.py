"""
MVP v6: NG神经调质门控 + ESB具身-符号桥接
==========================================

NG-Neuron (Neuromodulator-Gated):
- DA (多巴胺): 奖励预测误差，控制学习率
- 5HT (血清素): 情绪状态，控制探索/利用
- ACh (乙酰胆碱): 注意力聚焦，控制稀疏性
- NE (去甲肾上腺素): 唤醒水平，控制增益

ESB-Neuron (Embodied-Symbolic Bridge):
- 具身输入: 传感器数据（连续、高维）
- 符号输出: 概念/标签（离散、抽象）
- 桥接机制: 连续→离散的映射与 grounding
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class NeuromodulatorState:
    """神经调质状态"""
    da: float = 0.5      # 多巴胺: [0,1] 奖励信号
    serotonin: float = 0.5  # 血清素: [0,1] 情绪状态
    ach: float = 0.5     # 乙酰胆碱: [0,1] 注意力
    ne: float = 0.5      # 去甲肾上腺素: [0,1] 唤醒
    
    def to_array(self) -> np.ndarray:
        return np.array([self.da, self.serotonin, self.ach, self.ne])


class NGNeuronLayer:
    """
    神经调质门控神经元层
    
    四通道门控机制：
    - DA门控: η_eff = η_base * (1 + α_da * (DA - 0.5))
    - 5HT门控: explore_prob = sigmoid(β_5ht * (5HT - 0.5))
    - ACh门控: sparsity = γ_ach * ACh
    - NE门控: gain = 1 + δ_ne * NE
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
        
    def forward(self, x: np.ndarray, nm: Optional[NeuromodulatorState] = None) -> np.ndarray:
        if nm is not None:
            self.nm_state = nm
        
        gain = 1.0 + self.delta_ne * self.nm_state.ne
        output = gain * (self.W @ x)
        
        return output
    
    def learn(self, x: np.ndarray, target: np.ndarray, 
              reward: float = 0.0, nm: Optional[NeuromodulatorState] = None):
        if nm is not None:
            self.nm_state = nm
        
        da_error = reward - 0.5
        self.nm_state.da = np.clip(0.5 + 0.3 * da_error, 0.0, 1.0)
        
        effective_lr = self.base_lr * (1.0 + self.alpha_da * (self.nm_state.da - 0.5))
        effective_lr = np.clip(effective_lr, 0.001, 0.5)
        
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


class ESBNeuronLayer:
    """
    具身-符号桥接神经元层
    
    功能：
    - 具身编码: 连续传感器输入 → 内部表示
    - 符号解码: 内部表示 → 离散概念
    - Grounding: 符号与具身体验的绑定
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
        """具身编码: 传感器 → 潜在空间"""
        latent = np.tanh(self.W_encode @ sensor_input)
        return latent
    
    def decode_symbol(self, latent: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """符号解码: 潜在空间 → 离散符号"""
        logits = self.W_decode @ latent
        probs = self._softmax(logits)
        
        symbol_id = np.argmax(probs)
        confidence = probs[symbol_id]
        
        return symbol_id, probs, confidence
    
    def ground_symbol(self, symbol_id: int, embodied_example: np.ndarray, 
                      strength: float = 0.1):
        """Grounding: 将符号绑定到具身体验"""
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
        """从符号检索具身表示"""
        return self.grounding_matrix[symbol_id]
    
    def compute_symbolic_distance(self, latent: np.ndarray, symbol_id: int) -> float:
        """计算潜在表示与符号嵌入的距离"""
        return np.linalg.norm(latent - self.symbol_embeddings[symbol_id])
    
    def learn(self, sensor_input: np.ndarray, target_symbol: int, 
              lr: float = 0.05):
        """学习具身-符号映射"""
        latent = self.encode_embodied(sensor_input)
        symbol_id, probs, confidence = self.decode_symbol(latent)
        
        if symbol_id != target_symbol:
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
        
        return confidence
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)


class IntegratedNGESBSystem:
    """集成NG和ESB的完整系统"""
    
    def __init__(self, n_embodied: int = 16, n_latent: int = 8, 
                 n_symbols: int = 4, n_hidden: int = 8):
        self.ng = NGNeuronLayer(n_embodied, n_hidden, base_lr=0.05)
        self.esb = ESBNeuronLayer(n_embodied, n_latent, n_symbols)
        
        self.output_layer = NGNeuronLayer(n_hidden, 1, base_lr=0.05)
        
    def forward(self, sensor_input: np.ndarray, 
                nm: Optional[NeuromodulatorState] = None) -> Dict:
        """前向传播"""
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
              target_symbol: int, reward: float = 0.0,
              nm: Optional[NeuromodulatorState] = None):
        """学习"""
        result = self.forward(sensor_input, nm)
        
        output_error = target_output - result['output']
        reward_signal = reward + 0.1 * np.abs(output_error)
        
        hidden = result['hidden']
        target_hidden = hidden + 0.1 * np.sign(output_error)
        
        lr1 = self.ng.learn(sensor_input, target_hidden, reward_signal, nm)
        lr2 = self.output_layer.learn(hidden, np.array([target_output]), reward_signal, nm)
        
        symbol_conf = self.esb.learn(sensor_input, target_symbol, lr=0.05)
        
        return {
            'output_error': float(output_error),
            'effective_lr': float(lr1),
            'symbol_confidence': float(symbol_conf)
        }
    
    def set_state(self, da: float = None, serotonin: float = None,
                  ach: float = None, ne: float = None):
        """设置神经调质状态"""
        self.ng.set_neuromodulators(da, serotonin, ach, ne)
        self.output_layer.set_neuromodulators(da, serotonin, ach, ne)


def run_ng_esb_experiment():
    """运行NG+ESB实验"""
    
    print("=" * 70)
    print("NG神经调质门控 + ESB具身-符号桥接 实验")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_embodied = 16
    n_latent = 8
    n_symbols = 4
    symbol_names = ["安全", "警告", "危险", "正常"]
    
    system = IntegratedNGESBSystem(n_embodied, n_latent, n_symbols, n_hidden=8)
    
    print("\n[实验1] 神经调质门控效果")
    print("-" * 60)
    
    test_input = np.random.randn(n_embodied) * 0.5
    
    print("\n不同DA水平下的有效学习率:")
    for da in [0.0, 0.25, 0.5, 0.75, 1.0]:
        nm = NeuromodulatorState(da=da)
        system.set_state(da=da)
        result = system.forward(test_input, nm)
        effective_lr = system.ng.base_lr * (1.0 + system.ng.alpha_da * (da - 0.5))
        print(f"  DA={da:.2f}: 有效学习率={effective_lr:.4f}")
    
    print("\n不同NE水平下的输出增益:")
    for ne in [0.0, 0.5, 1.0]:
        nm = NeuromodulatorState(ne=ne)
        result = system.forward(test_input, nm)
        gain = 1.0 + system.ng.delta_ne * ne
        print(f"  NE={ne:.2f}: 增益={gain:.2f}, 输出={result['output']:.4f}")
    
    print("\n[实验2] 具身-符号桥接")
    print("-" * 60)
    
    sensor_patterns = [
        np.array([1.0, 0.5, 0.0] + [0.0] * 13),
        np.array([0.0, 1.0, 0.5] + [0.0] * 13),
        np.array([0.5, 0.0, 1.0] + [0.0] * 13),
        np.array([0.3, 0.3, 0.3] + [0.0] * 13),
    ]
    target_symbols = [0, 1, 2, 3]
    
    print("\n训练具身-符号映射...")
    for epoch in range(50):
        for pattern, target_sym in zip(sensor_patterns, target_symbols):
            system.learn(pattern, 0.5, target_sym, reward=0.1)
    
    print("\n测试符号识别:")
    correct = 0
    for i, (pattern, target_sym) in enumerate(zip(sensor_patterns, target_symbols)):
        result = system.forward(pattern)
        predicted_sym = result['symbol_id']
        confidence = result['confidence']
        is_correct = predicted_sym == target_sym
        correct += int(is_correct)
        print(f"  模式{i} ({symbol_names[target_sym]}): "
              f"预测={symbol_names[predicted_sym]}, "
              f"置信度={confidence:.3f}, "
              f"{'✓' if is_correct else '✗'}")
    
    accuracy = correct / len(target_symbols) * 100
    print(f"\n符号识别准确率: {accuracy:.1f}%")
    
    print("\n[实验3] Grounding检索")
    print("-" * 60)
    
    for sym_id in range(n_symbols):
        grounded = system.esb.retrieve_embodied(sym_id)
        norm = np.linalg.norm(grounded)
        print(f"  符号'{symbol_names[sym_id]}' 的具身grounding: 范数={norm:.4f}")
    
    print("\n[实验4] 神经调质对学习的影响")
    print("-" * 60)
    
    system2 = IntegratedNGESBSystem(n_embodied, n_latent, n_symbols, n_hidden=8)
    
    learning_curves = {
        'high_DA': [],
        'low_DA': [],
        'high_ACh': [],
        'low_ACh': []
    }
    
    target_output = 0.8
    test_pattern = sensor_patterns[0]
    
    for condition in ['high_DA', 'low_DA']:
        if condition == 'high_DA':
            system2.set_state(da=0.9)
        else:
            system2.set_state(da=0.1)
        
        errors = []
        for _ in range(30):
            result = system2.forward(test_pattern)
            error = abs(target_output - result['output'])
            errors.append(error)
            system2.learn(test_pattern, target_output, 0, reward=0.1)
        
        learning_curves[condition] = errors
    
    print("\n学习曲线 (目标输出=0.8):")
    print(f"{'轮次':<8} {'高DA误差':<12} {'低DA误差':<12} {'差异':<12}")
    print("-" * 50)
    for i in [0, 9, 19, 29]:
        high_err = learning_curves['high_DA'][i]
        low_err = learning_curves['low_DA'][i]
        diff = low_err - high_err
        print(f"{i+1:<8} {high_err:<12.4f} {low_err:<12.4f} {diff:+.4f}")
    
    results = {
        'symbol_accuracy': float(accuracy),
        'final_errors': {
            'high_DA': float(learning_curves['high_DA'][-1]),
            'low_DA': float(learning_curves['low_DA'][-1])
        },
        'neuromodulator_effects': {
            'da_learning_rate_range': [0.025, 0.075],
            'ne_gain_range': [1.0, 1.5]
        }
    }
    
    with open('/workspace/mvp_v6_ng_esb_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到 mvp_v6_ng_esb_results.json")
    
    return results


if __name__ == '__main__':
    results = run_ng_esb_experiment()
