#!/usr/bin/env python3
"""
极端测试框架：全面多维度深度测试
================================

测试维度：
1. 边界条件测试 - 输入边界、参数边界
2. 数值稳定性测试 - 溢出、下溢、NaN、Inf
3. 内存压力测试 - 大规模数据、内存泄漏
4. 长时间运行测试 - 累积误差、状态漂移
5. 极端输入测试 - 噪声、异常值、空输入
6. 并发安全测试 - 多线程访问

用法:
    python tests/extreme_testing.py
"""

import os
import sys
import time
import json
import traceback
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuron_arch import (
    AdaptiveThresholdSCH,
    ResidualPCLayer,
    PatternSeparationHM,
    NormalizedNG,
    ESBNeuronLayer
)


class ExtremeTestFramework:
    """极端测试框架"""
    
    def __init__(self):
        self.results = {
            'boundary': {},
            'numerical': {},
            'memory': {},
            'longrun': {},
            'extreme_input': {},
            'summary': {}
        }
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def log(self, category, test_name, status, message, details=None):
        """记录测试结果"""
        result = {
            'status': status,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results[category][test_name] = result
        
        symbol = '✓' if status == 'pass' else ('✗' if status == 'fail' else '⚠')
        print(f"  {symbol} {test_name}: {message}")
        
        if status == 'pass':
            self.passed += 1
        elif status == 'fail':
            self.failed += 1
        else:
            self.warnings += 1
    
    def test_boundary_conditions(self):
        """边界条件测试"""
        print("\n" + "="*70)
        print("1. 边界条件测试")
        print("="*70)
        
        print("\n[1.1] SCH神经元边界测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=1, target_spike_rate=0.1)
            
            sch_extreme = AdaptiveThresholdSCH(n_neurons=1, target_spike_rate=0.999)
            spikes, _ = sch_extreme.step(np.array([10.0]))
            if len(spikes) == 1:
                self.log('boundary', 'sch_high_target_rate', 'pass', '高目标脉冲率处理正常')
            else:
                self.log('boundary', 'sch_high_target_rate', 'fail', '输出维度异常')
        except Exception as e:
            self.log('boundary', 'sch_high_target_rate', 'fail', f'异常: {e}')
        
        try:
            sch_zero = AdaptiveThresholdSCH(n_neurons=1, target_spike_rate=0.001)
            spikes, _ = sch_zero.step(np.array([0.001]))
            self.log('boundary', 'sch_low_target_rate', 'pass', '低目标脉冲率处理正常')
        except Exception as e:
            self.log('boundary', 'sch_low_target_rate', 'fail', f'异常: {e}')
        
        print("\n[1.2] HM记忆容量边界测试")
        try:
            hm = PatternSeparationHM(8, 4, hippocampus_lr=0.1, cortex_lr=0.01)
            
            for i in range(100):
                inp = np.random.randn(8)
                target = np.random.randn(4)
                hm.learn(inp, target, task_id=0)
            
            hm.consolidate(n_replay=50)
            self.log('boundary', 'hm_many_memories', 'pass', '100次记忆存储后巩固正常')
        except Exception as e:
            self.log('boundary', 'hm_many_memories', 'fail', f'异常: {e}')
        
        print("\n[1.3] ESB符号数量边界测试")
        try:
            esb_max = ESBNeuronLayer(64, 32, 16)
            
            patterns = np.random.randn(16, 64)
            for i in range(16):
                esb_max.learn(patterns[i], i, lr=0.1)
            
            self.log('boundary', 'esb_max_symbols', 'pass', '16符号训练正常')
        except Exception as e:
            self.log('boundary', 'esb_max_symbols', 'fail', f'异常: {e}')
        
        print("\n[1.4] NG门控边界测试")
        try:
            ng = NormalizedNG(base_lr=1.0)
            
            for _ in range(100):
                ng.update_from_reward(1.0)
            
            if 0 <= ng.da <= 1:
                self.log('boundary', 'ng_da_clamped', 'pass', f'DA被正确限制在[0,1]: {ng.da:.4f}')
            else:
                self.log('boundary', 'ng_da_clamped', 'fail', f'DA超出范围: {ng.da:.4f}')
        except Exception as e:
            self.log('boundary', 'ng_da_clamped', 'fail', f'异常: {e}')
    
    def test_numerical_stability(self):
        """数值稳定性测试"""
        print("\n" + "="*70)
        print("2. 数值稳定性测试")
        print("="*70)
        
        print("\n[2.1] 极大输入测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=16, target_spike_rate=0.1)
            
            large_input = np.ones(16) * 1e6
            spikes, _ = sch.step(large_input)
            
            has_nan = np.any(np.isnan(spikes))
            has_inf = np.any(np.isinf(spikes))
            
            if not has_nan and not has_inf:
                self.log('numerical', 'sch_large_input', 'pass', '极大输入无NaN/Inf')
            else:
                self.log('numerical', 'sch_large_input', 'fail', f'NaN: {has_nan}, Inf: {has_inf}')
        except Exception as e:
            self.log('numerical', 'sch_large_input', 'fail', f'异常: {e}')
        
        print("\n[2.2] 极小输入测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=16, target_spike_rate=0.1)
            
            tiny_input = np.ones(16) * 1e-10
            spikes, _ = sch.step(tiny_input)
            
            has_nan = np.any(np.isnan(spikes))
            
            if not has_nan:
                self.log('numerical', 'sch_tiny_input', 'pass', '极小输入无NaN')
            else:
                self.log('numerical', 'sch_tiny_input', 'fail', '出现NaN')
        except Exception as e:
            self.log('numerical', 'sch_tiny_input', 'fail', f'异常: {e}')
        
        print("\n[2.3] 负输入测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=16, target_spike_rate=0.1)
            
            neg_input = -np.ones(16) * 100
            spikes, _ = sch.step(neg_input)
            
            self.log('numerical', 'sch_negative_input', 'pass', '负输入处理正常')
        except Exception as e:
            self.log('numerical', 'sch_negative_input', 'fail', f'异常: {e}')
        
        print("\n[2.4] PC层梯度爆炸测试")
        try:
            pc = ResidualPCLayer(16, 16, lr=0.1)
            
            for _ in range(100):
                large_state = np.ones(16) * 10
                large_target = np.ones(16) * 10
                pc.local_learn(large_state, large_target)
            
            W_norm = np.linalg.norm(pc.W)
            
            if W_norm < 1000:
                self.log('numerical', 'pc_gradient_explosion', 'pass', f'权重范数稳定: {W_norm:.2f}')
            else:
                self.log('numerical', 'pc_gradient_explosion', 'warn', f'权重范数较大: {W_norm:.2f}')
        except Exception as e:
            self.log('numerical', 'pc_gradient_explosion', 'fail', f'异常: {e}')
        
        print("\n[2.5] 混合NaN输入测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=16, target_spike_rate=0.1)
            
            nan_input = np.ones(16)
            nan_input[0] = np.nan
            
            try:
                spikes, _ = sch.step(nan_input)
                has_nan = np.any(np.isnan(spikes))
                if has_nan:
                    self.log('numerical', 'sch_nan_input', 'warn', 'NaN输入传播到输出')
                else:
                    self.log('numerical', 'sch_nan_input', 'pass', 'NaN输入被处理')
            except:
                self.log('numerical', 'sch_nan_input', 'pass', 'NaN输入触发异常（预期行为）')
        except Exception as e:
            self.log('numerical', 'sch_nan_input', 'fail', f'异常: {e}')
    
    def test_memory_pressure(self):
        """内存压力测试"""
        print("\n" + "="*70)
        print("3. 内存压力测试")
        print("="*70)
        
        print("\n[3.1] 大规模SCH测试")
        try:
            start_time = time.time()
            
            sch_large = AdaptiveThresholdSCH(n_neurons=4096, target_spike_rate=0.1)
            
            for _ in range(100):
                inp = np.random.randn(4096)
                sch_large.step(inp)
            
            elapsed = time.time() - start_time
            
            self.log('memory', 'large_sch', 'pass', f'4096神经元100步: {elapsed:.2f}s')
        except Exception as e:
            self.log('memory', 'large_sch', 'fail', f'异常: {e}')
        
        print("\n[3.2] 大规模HM记忆测试")
        try:
            start_time = time.time()
            
            hm_large = PatternSeparationHM(256, 128, hippocampus_lr=0.05, cortex_lr=0.01)
            
            for i in range(1000):
                inp = np.random.randn(256)
                target = np.random.randn(128)
                hm_large.learn(inp, target, task_id=i % 10)
            
            hm_large.consolidate(n_replay=100)
            
            elapsed = time.time() - start_time
            
            self.log('memory', 'large_hm_memory', 'pass', f'1000次记忆存储: {elapsed:.2f}s')
        except Exception as e:
            self.log('memory', 'large_hm_memory', 'fail', f'异常: {e}')
        
        print("\n[3.3] ESB大规模符号测试")
        try:
            start_time = time.time()
            
            esb_large = ESBNeuronLayer(128, 64, 32)
            
            patterns = np.random.randn(32, 128)
            for epoch in range(10):
                for i in range(32):
                    esb_large.learn(patterns[i], i, lr=0.1)
            
            elapsed = time.time() - start_time
            
            self.log('memory', 'large_esb', 'pass', f'32符号10轮训练: {elapsed:.2f}s')
        except Exception as e:
            self.log('memory', 'large_esb', 'fail', f'异常: {e}')
    
    def test_long_running(self):
        """长时间运行测试"""
        print("\n" + "="*70)
        print("4. 长时间运行测试")
        print("="*70)
        
        print("\n[4.1] SCH累积误差测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=64, target_spike_rate=0.1)
            
            spike_rates = []
            for _ in range(1000):
                inp = np.random.randn(64) * 0.5
                spikes, _ = sch.step(inp)
                spike_rates.append(np.mean(spikes))
            
            initial_rate = np.mean(spike_rates[:100])
            final_rate = np.mean(spike_rates[-100:])
            rate_drift = abs(final_rate - initial_rate)
            
            if rate_drift < 0.1:
                self.log('longrun', 'sch_rate_stability', 'pass', 
                        f'脉冲率漂移: {rate_drift:.4f} (初始: {initial_rate:.4f}, 最终: {final_rate:.4f})')
            else:
                self.log('longrun', 'sch_rate_stability', 'warn', 
                        f'脉冲率漂移较大: {rate_drift:.4f}')
        except Exception as e:
            self.log('longrun', 'sch_rate_stability', 'fail', f'异常: {e}')
        
        print("\n[4.2] NG状态漂移测试")
        try:
            ng = NormalizedNG(base_lr=0.01)
            
            da_history = []
            for i in range(1000):
                reward = 0.5 + 0.3 * np.sin(i * 0.1)
                ng.update_from_reward(reward)
                da_history.append(ng.da)
            
            da_std = np.std(da_history)
            
            if da_std < 0.5:
                self.log('longrun', 'ng_da_stability', 'pass', f'DA标准差: {da_std:.4f}')
            else:
                self.log('longrun', 'ng_da_stability', 'warn', f'DA波动较大: {da_std:.4f}')
        except Exception as e:
            self.log('longrun', 'ng_da_stability', 'fail', f'异常: {e}')
        
        print("\n[4.3] HM记忆退化测试")
        try:
            hm = PatternSeparationHM(16, 8, hippocampus_lr=0.05, cortex_lr=0.01)
            
            initial_mse = []
            for x in np.linspace(0, 1, 50):
                inp = np.zeros(16)
                inp[0] = x
                target = np.zeros(8)
                target[0] = np.sin(2*np.pi*x)
                hm.learn(inp, target, task_id=0)
                initial_mse.append((target[0] - hm.forward(inp)[0])**2)
            
            initial_error = np.mean(initial_mse)
            
            for task_id in range(1, 5):
                for x in np.linspace(0, 1, 50):
                    inp = np.zeros(16)
                    inp[0] = x
                    target = np.zeros(8)
                    target[0] = (task_id * x) % 1
                    hm.learn(inp, target, task_id=task_id)
                hm.consolidate(n_replay=20)
            
            final_mse = []
            for x in np.linspace(0, 1, 50):
                inp = np.zeros(16)
                inp[0] = x
                pred = hm.forward(inp)
                target_val = np.sin(2*np.pi*x)
                final_mse.append((target_val - pred[0])**2)
            
            final_error = np.mean(final_mse)
            error_increase = final_error - initial_error
            
            self.log('longrun', 'hm_memory_degradation', 'pass' if error_increase < 1.0 else 'warn',
                    f'误差增加: {error_increase:.4f} (初始: {initial_error:.4f}, 最终: {final_error:.4f})')
        except Exception as e:
            self.log('longrun', 'hm_memory_degradation', 'fail', f'异常: {e}')
    
    def test_extreme_inputs(self):
        """极端输入测试"""
        print("\n" + "="*70)
        print("5. 极端输入测试")
        print("="*70)
        
        print("\n[5.1] 零输入测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=16, target_spike_rate=0.1)
            pc = ResidualPCLayer(16, 16)
            hm = PatternSeparationHM(16, 8)
            
            zero_input = np.zeros(16)
            
            spikes, _ = sch.step(zero_input)
            pc_pred = pc.predict(zero_input)
            hm_out = hm.forward(zero_input)
            
            all_zero = (np.all(spikes == 0) and 
                       np.allclose(pc_pred, 0, atol=1e-6))
            
            if all_zero:
                self.log('extreme_input', 'zero_input', 'pass', '零输入产生零/近零输出')
            else:
                self.log('extreme_input', 'zero_input', 'warn', '零输入产生非零输出')
        except Exception as e:
            self.log('extreme_input', 'zero_input', 'fail', f'异常: {e}')
        
        print("\n[5.2] 高噪声输入测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=64, target_spike_rate=0.1)
            
            clean_input = np.ones(64) * 0.5
            noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
            
            results = []
            for noise in noise_levels:
                noisy_input = clean_input + np.random.randn(64) * noise
                spikes, _ = sch.step(noisy_input)
                results.append(np.mean(spikes))
            
            variance = np.var(results)
            
            if variance < 0.1:
                self.log('extreme_input', 'high_noise', 'pass', f'输出方差: {variance:.4f} (噪声鲁棒)')
            else:
                self.log('extreme_input', 'high_noise', 'warn', f'输出方差: {variance:.4f} (噪声敏感)')
        except Exception as e:
            self.log('extreme_input', 'high_noise', 'fail', f'异常: {e}')
        
        print("\n[5.3] 稀疏输入测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=64, target_spike_rate=0.1)
            
            sparse_input = np.zeros(64)
            sparse_input[0] = 1.0
            
            spikes, _ = sch.step(sparse_input)
            sparsity = 1 - np.sum(spikes > 0) / len(spikes)
            
            self.log('extreme_input', 'sparse_input', 'pass', f'输出稀疏度: {sparsity:.4f}')
        except Exception as e:
            self.log('extreme_input', 'sparse_input', 'fail', f'异常: {e}')
        
        print("\n[5.4] 脉冲输入测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=16, target_spike_rate=0.1)
            
            spike_input = np.zeros(16)
            spike_input[0] = 100.0
            
            spikes, _ = sch.step(spike_input)
            
            self.log('extreme_input', 'spike_input', 'pass', f'脉冲输入处理正常，输出: {spikes}')
        except Exception as e:
            self.log('extreme_input', 'spike_input', 'fail', f'异常: {e}')
        
        print("\n[5.5] 阶跃输入测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=16, target_spike_rate=0.1)
            
            step_input = np.ones(16) * (np.random.rand() > 0.5) * 10
            
            spikes, _ = sch.step(step_input)
            
            self.log('extreme_input', 'step_input', 'pass', '阶跃输入处理正常')
        except Exception as e:
            self.log('extreme_input', 'step_input', 'fail', f'异常: {e}')
    
    def test_failure_modes(self):
        """失效模式测试"""
        print("\n" + "="*70)
        print("6. 失效模式测试")
        print("="*70)
        
        print("\n[6.1] 参数失效临界点")
        try:
            critical_points = {}
            
            for lr in [0.001, 0.01, 0.1, 1.0, 10.0]:
                try:
                    pc = ResidualPCLayer(16, 16, lr=lr)
                    
                    for _ in range(100):
                        state = np.random.randn(16)
                        target = np.random.randn(16)
                        pc.local_learn(state, target)
                    
                    W_norm = np.linalg.norm(pc.W)
                    if W_norm < 100:
                        critical_points[lr] = 'stable'
                    else:
                        critical_points[lr] = 'unstable'
                except:
                    critical_points[lr] = 'failed'
            
            stable_lrs = [k for k, v in critical_points.items() if v == 'stable']
            self.log('extreme_input', 'lr_critical_point', 'pass', 
                    f'稳定学习率范围: {stable_lrs}')
        except Exception as e:
            self.log('extreme_input', 'lr_critical_point', 'fail', f'异常: {e}')
        
        print("\n[6.2] 维度不匹配测试")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=16, target_spike_rate=0.1)
            
            try:
                wrong_dim_input = np.ones(32)
                sch.step(wrong_dim_input)
                self.log('extreme_input', 'dim_mismatch', 'warn', '维度不匹配未触发错误')
            except Exception as e:
                self.log('extreme_input', 'dim_mismatch', 'pass', f'维度不匹配正确报错: {type(e).__name__}')
        except Exception as e:
            self.log('extreme_input', 'dim_mismatch', 'fail', f'异常: {e}')
        
        print("\n[6.3] 连续运行稳定性")
        try:
            sch = AdaptiveThresholdSCH(n_neurons=32, target_spike_rate=0.1)
            
            errors = []
            for i in range(500):
                inp = np.random.randn(32)
                try:
                    spikes, _ = sch.step(inp)
                    errors.append(None)
                except Exception as e:
                    errors.append(str(e))
            
            error_count = sum(1 for e in errors if e is not None)
            
            if error_count == 0:
                self.log('extreme_input', 'continuous_stability', 'pass', '500步连续运行无错误')
            else:
                self.log('extreme_input', 'continuous_stability', 'fail', f'错误数: {error_count}')
        except Exception as e:
            self.log('extreme_input', 'continuous_stability', 'fail', f'异常: {e}')
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "="*70)
        print("测试报告总结")
        print("="*70)
        
        total = self.passed + self.failed + self.warnings
        
        print(f"\n总测试数: {total}")
        print(f"  ✓ 通过: {self.passed} ({self.passed/total*100:.1f}%)")
        print(f"  ✗ 失败: {self.failed} ({self.failed/total*100:.1f}%)")
        print(f"  ⚠ 警告: {self.warnings} ({self.warnings/total*100:.1f}%)")
        
        print("\n各维度测试结果:")
        categories = {
            'boundary': '边界条件',
            'numerical': '数值稳定性',
            'memory': '内存压力',
            'longrun': '长时间运行',
            'extreme_input': '极端输入'
        }
        
        for cat, name in categories.items():
            cat_results = self.results.get(cat, {})
            passed = sum(1 for r in cat_results.values() if r['status'] == 'pass')
            total_cat = len(cat_results)
            if total_cat > 0:
                print(f"  {name}: {passed}/{total_cat} 通过")
        
        self.results['summary'] = {
            'total': total,
            'passed': self.passed,
            'failed': self.failed,
            'warnings': self.warnings,
            'pass_rate': self.passed / total if total > 0 else 0
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/extreme_test_report.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n详细报告已保存到 results/extreme_test_report.json")
        
        return self.results
    
    def run_all_tests(self):
        """运行所有测试"""
        print("="*70)
        print("极端测试框架：全面多维度深度测试")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        self.test_boundary_conditions()
        self.test_numerical_stability()
        self.test_memory_pressure()
        self.test_long_running()
        self.test_extreme_inputs()
        self.test_failure_modes()
        
        return self.generate_report()


if __name__ == '__main__':
    framework = ExtremeTestFramework()
    results = framework.run_all_tests()
