"""
MVP验证单元：SCH + PC 两层系统（修复版）
参数重新标定以匹配任务数值尺度
"""

import numpy as np
from typing import Tuple, List, Dict
import time


class SCHNeuronLayer:
    """修复版：参数匹配任务数值尺度"""
    
    def __init__(self, n_neurons: int, tau_m: float = 5.0,
                 v_rest: float = 0.0, v_reset: float = 0.0,
                 v_th: float = 0.1, r_m: float = 1.0):
        self.n = n_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        self.r_m = r_m
        
        self.v = np.zeros(n_neurons)
        self.z = np.zeros(n_neurons)
        self.spike_count = np.zeros(n_neurons)
        
    def step(self, x: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        dv = (self.v_rest - self.v + self.r_m * x) / self.tau_m
        self.v += dv * dt
        
        spikes = self.v >= self.v_th
        self.v[spikes] = self.v_reset
        self.spike_count[spikes] += 1
        
        self.z = 0.7 * self.z + 0.3 * self.v
        
        return spikes.astype(float), self.z.copy()
    
    def reset(self):
        self.v = np.zeros(self.n)
        self.z = np.zeros(self.n)
        self.spike_count = np.zeros(self.n)


class PCNeuronLayer:
    """修复版：更激进的学习率"""
    
    def __init__(self, n_neurons: int, lr: float = 0.05):
        self.n = n_neurons
        self.lr = lr
        
        self.W_state = np.eye(n_neurons) * 0.5 + np.random.randn(n_neurons, n_neurons) * 0.05
        self.b = np.zeros(n_neurons)
        
        self.x = np.zeros(n_neurons)
        self.mu = np.zeros(n_neurons)
        self.epsilon = np.zeros(n_neurons)
        
    def predict(self, prev_state: np.ndarray) -> np.ndarray:
        self.mu = np.tanh(self.W_state @ prev_state + self.b)
        return self.mu
    
    def compute_error(self, actual: np.ndarray) -> np.ndarray:
        self.epsilon = actual - self.mu
        return self.epsilon
    
    def update_state(self, error: np.ndarray, dt: float = 0.1):
        self.x += dt * (error - 0.01 * self.x)
        
    def local_learn(self, prev_state: np.ndarray):
        grad_W = np.outer(self.epsilon, prev_state)
        grad_b = self.epsilon
        
        self.W_state -= self.lr * grad_W
        self.b -= self.lr * grad_b
        
        self.W_state = np.clip(self.W_state, -2, 2)


class MVPSystem:
    """修复版：编码/解码权重匹配输入/输出尺度"""
    
    def __init__(self, n_neurons: int = 50, lr: float = 0.05):
        self.n = n_neurons
        self.sch = SCHNeuronLayer(n_neurons)
        self.pc = PCNeuronLayer(n_neurons, lr)
        
        self.W_encode = np.random.randn(n_neurons, 1) / np.sqrt(n_neurons) * 2.0
        self.W_decode = np.random.randn(1, n_neurons) / np.sqrt(n_neurons) * 0.5
        
        self.prev_z = np.zeros(n_neurons)
        
        self.history = {
            'spike_rate': [],
            'prediction_error': [],
            'loss': [],
            'anomaly_signal': []
        }
        
    def forward(self, x: float, learn: bool = False) -> Tuple[float, float, float]:
        input_vec = self.W_encode * x
        
        spikes, continuous = self.sch.step(input_vec.flatten())
        spike_rate = np.mean(spikes)
        
        prediction = self.pc.predict(self.prev_z)
        error = self.pc.compute_error(continuous)
        error_norm = np.linalg.norm(error)
        
        self.pc.update_state(error)
        
        output = self.W_decode @ self.pc.x
        output_val = output[0]
        
        anomaly_signal = error_norm
        
        if learn:
            self.pc.local_learn(self.prev_z)
            
        self.prev_z = continuous.copy()
            
        self.history['spike_rate'].append(spike_rate)
        self.history['prediction_error'].append(error_norm)
        self.history['anomaly_signal'].append(anomaly_signal)
        
        return output_val, error_norm, anomaly_signal
    
    def reset(self):
        self.sch.reset()
        self.prev_z = np.zeros(self.n)
        self.pc.x = np.zeros(self.n)


def generate_sine_wave(n_steps: int, freq: float = 0.05,
                       anomaly_positions: List[int] = None) -> np.ndarray:
    t = np.arange(n_steps)
    signal = np.sin(2 * np.pi * freq * t)
    
    if anomaly_positions:
        for pos in anomaly_positions:
            if pos < n_steps:
                signal[pos] += 3.0
                
    return signal


def test_hypothesis_1_runnable():
    print("=" * 60)
    print("假说1验证：SCH和PC能否在同一循环运转")
    print("=" * 60)
    
    system = MVPSystem(n_neurons=50)
    signal = generate_sine_wave(1000)
    
    t0 = time.time()
    outputs = []
    for i, x in enumerate(signal):
        out, err, anom = system.forward(x, learn=False)
        outputs.append(out)
    elapsed = time.time() - t0
    
    spike_rate = np.mean(system.history['spike_rate'])
    total_spikes = np.sum(system.sch.spike_count)
    
    print(f"✓ 1000步运行完成，耗时: {elapsed:.3f}s")
    print(f"✓ 平均脉冲率: {spike_rate:.4f}")
    print(f"✓ 平均预测误差: {np.mean(system.history['prediction_error']):.4f}")
    print(f"✓ SCH总脉冲数: {total_spikes:.0f}")
    
    spike_ok = total_spikes > 100
    if spike_ok:
        print(f"✓ SCH产生足够脉冲 ({total_spikes}个)，事件驱动工作正常")
    else:
        print(f"⚠ SCH脉冲偏少 ({total_spikes}个)，可能需要进一步调参")
    
    return True, spike_ok, spike_rate


def test_hypothesis_2_learning():
    print("\n" + "=" * 60)
    print("假说2验证：局部预测误差学习是否有效")
    print("=" * 60)
    
    system = MVPSystem(n_neurons=50, lr=0.05)
    signal = generate_sine_wave(2000)
    
    n_epochs = 10
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = []
        system.reset()
        system.history = {'spike_rate': [], 'prediction_error': [], 'loss': [], 'anomaly_signal': []}
        
        for i, x in enumerate(signal):
            out, err, anom = system.forward(x, learn=True)
            epoch_loss.append(err)
            
        mean_loss = np.mean(epoch_loss)
        losses.append(mean_loss)
        print(f"Epoch {epoch+1}: 平均误差 = {mean_loss:.4f}")
    
    if losses[0] > 0 and losses[-1] > 0:
        loss_drop = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"\n误差下降: {loss_drop:.1f}%")
        print(f"初始误差: {losses[0]:.4f}, 最终误差: {losses[-1]:.4f}")
        
        if loss_drop > 10:
            print("✓ 假说2验证通过：局部学习有效")
            return True, loss_drop
        elif loss_drop > 0:
            print("⚠ 假说2部分通过：局部学习效果有限")
            return False, loss_drop
        else:
            print("✗ 假说2验证失败：误差未下降")
            return False, loss_drop
    else:
        print("✗ 误差为零，检查系统配置")
        return False, 0


def test_hypothesis_3_interpretability():
    print("\n" + "=" * 60)
    print("假说3验证：预测误差能否检测异常")
    print("=" * 60)
    
    system = MVPSystem(n_neurons=50, lr=0.05)
    
    train_signal = generate_sine_wave(1000)
    for x in train_signal:
        system.forward(x, learn=True)
    
    anomaly_positions = [200, 400, 600, 800]
    test_signal = generate_sine_wave(1000, anomaly_positions=anomaly_positions)
    
    system.reset()
    system.history = {'spike_rate': [], 'prediction_error': [], 'loss': [], 'anomaly_signal': []}
    
    for x in test_signal:
        out, err, anom = system.forward(x, learn=False)
    
    anomaly_signals = np.array(system.history['anomaly_signal'])
    
    normal_mask = np.ones(1000, dtype=bool)
    for pos in anomaly_positions:
        normal_mask[max(0, pos-10):min(1000, pos+10)] = False
    
    normal_mean = np.mean(anomaly_signals[normal_mask]) if np.any(normal_mask) else 0
    anomaly_vals = [anomaly_signals[pos] for pos in anomaly_positions if pos < len(anomaly_signals)]
    anomaly_mean = np.mean(anomaly_vals) if anomaly_vals else 0
    
    ratio = anomaly_mean / normal_mean if normal_mean > 0 else 0
    
    print(f"正常点平均误差: {normal_mean:.4f}")
    print(f"异常点平均误差: {anomaly_mean:.4f}")
    print(f"异常/正常比值: {ratio:.2f}x")
    
    if ratio > 1.5:
        print("✓ 假说3验证通过：PC误差能检测异常")
        return True, ratio
    elif ratio > 1.1:
        print("⚠ 假说3部分通过：异常检测能力有限")
        return False, ratio
    else:
        print("✗ 假说3验证失败：无法区分异常")
        return False, ratio


def run_mvp_validation():
    print("\n" + "=" * 60)
    print("MVP验证开始（修复版）")
    print("=" * 60 + "\n")
    
    h1, spike_ok, spike_rate = test_hypothesis_1_runnable()
    h2, loss_drop = test_hypothesis_2_learning()
    h3, ratio = test_hypothesis_3_interpretability()
    
    print("\n" + "=" * 60)
    print("MVP验证总结")
    print("=" * 60)
    print(f"假说1（能跑通）: {'✓ 通过' if h1 else '✗ 失败'}")
    print(f"  - SCH脉冲率: {spike_rate:.4f} ({'✓ 达标' if spike_rate > 0.05 else '⚠ 偏低'})")
    print(f"假说2（能学习）: {'✓ 通过' if h2 else '✗ 失败'}")
    print(f"  - 误差下降: {loss_drop:.1f}%")
    print(f"假说3（有解释性）: {'✓ 通过' if h3 else '✗ 失败'}")
    print(f"  - 异常/正常比: {ratio:.2f}x")
    
    all_pass = h1 and h2 and h3 and spike_ok
    if all_pass:
        print("\n✓ 所有假说验证通过，架构可行")
    else:
        print("\n⚠ 存在部分失败，需要进一步调优")
    
    return all_pass, {'spike_rate': spike_rate, 'loss_drop': loss_drop, 'anomaly_ratio': ratio}


def benchmark_vs_baseline():
    print("\n" + "=" * 60)
    print("对比基准：SCH+PC vs 朴素预测")
    print("=" * 60)
    
    signal = generate_sine_wave(2000)
    
    system = MVPSystem(n_neurons=50, lr=0.05)
    for x in signal[:1000]:
        system.forward(x, learn=True)
    
    system.reset()
    system.history = {'spike_rate': [], 'prediction_error': [], 'loss': [], 'anomaly_signal': []}
    
    test_signal = signal[1000:]
    predictions = []
    for x in test_signal:
        out, err, anom = system.forward(x, learn=False)
        predictions.append(out)
    
    mse = np.mean((np.array(predictions) - test_signal)**2)
    
    simple_pred = np.roll(test_signal, 1)
    simple_pred[0] = test_signal[0]
    baseline_mse = np.mean((simple_pred - test_signal)**2)
    
    print(f"SCH+PC MSE: {mse:.4f}")
    print(f"朴素基线 MSE: {baseline_mse:.4f}")
    
    if baseline_mse > 0:
        improvement = (baseline_mse - mse) / baseline_mse * 100
        print(f"相对改进: {improvement:.1f}%")
    
    return mse, baseline_mse


if __name__ == "__main__":
    np.random.seed(42)
    
    success, metrics = run_mvp_validation()
    
    if success:
        benchmark_vs_baseline()
        
    print("\n" + "=" * 60)
    print("MVP验证完成")
    print("=" * 60)
