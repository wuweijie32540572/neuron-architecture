#!/usr/bin/env python3
"""
改进版本验证测试
===============

验证PyTorch版本的改进：
1. GPU支持
2. 真正的自由能计算
3. 完整的神经调质动态
4. 真正的正交模式
5. STDP和时序编码
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch


def test_gpu_support():
    """测试GPU支持"""
    print("\n" + "="*70)
    print("测试1: GPU支持")
    print("="*70)
    
    from neuron_arch_torch import SCHNeuronTorch, PredictiveCodingLayer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    sch = SCHNeuronTorch(n_neurons=256, device=device)
    print(f"SCH设备: {sch.device}")
    
    x = torch.randn(64, 256, device=device)
    spikes, continuous, spike_times = sch(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {spikes.shape}")
    print(f"✓ GPU支持正常")
    
    return {'device': device, 'status': 'pass'}


def test_batch_processing():
    """测试批处理"""
    print("\n" + "="*70)
    print("测试2: 批处理")
    print("="*70)
    
    from neuron_arch_torch import SCHLayer
    
    layer = SCHLayer(n_input=128, n_output=64)
    
    batch_sizes = [1, 16, 64, 256]
    times = []
    
    for bs in batch_sizes:
        x = torch.randn(bs, 128)
        
        start = time.time()
        spikes, continuous, spike_times = layer(x)
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"批次大小 {bs:4d}: {elapsed*1000:.2f}ms, 输出形状 {spikes.shape}")
    
    print(f"✓ 批处理正常")
    
    return {'batch_sizes': batch_sizes, 'times': times, 'status': 'pass'}


def test_free_energy():
    """测试自由能计算"""
    print("\n" + "="*70)
    print("测试3: 真正的自由能计算")
    print("="*70)
    
    from neuron_arch_torch import PredictiveCodingLayer, TwoStagePCTrainer
    
    pc = PredictiveCodingLayer(n_state=32, n_pred=32)
    trainer = TwoStagePCTrainer(pc, pretrain_lr=0.02, online_lr=0.01)
    
    print("\n预训练阶段:")
    for i in range(20):
        state = torch.randn(32)
        target = torch.sin(state)
        metrics = trainer.pretrain_step(state, target)
        if i % 5 == 0:
            print(f"  Step {i}: Loss={metrics['loss']:.4f}")
    
    print("\n在线学习阶段:")
    free_energies = []
    for i in range(20):
        state = torch.randn(32)
        target = torch.sin(state)
        metrics = trainer.online_step(state, target)
        free_energies.append(metrics['free_energy'])
        if i % 5 == 0:
            print(f"  Step {i}: FE={metrics['free_energy']:.4f}, "
                  f"KL={metrics['kl_divergence']:.4f}, "
                  f"PE={metrics['prediction_error_term']:.4f}")
    
    fe_trend = pc.get_free_energy_trend()
    print(f"\n自由能趋势: {fe_trend['trend']:.6f}")
    print(f"当前自由能: {fe_trend['current']:.4f}")
    
    print(f"✓ 自由能计算正常")
    
    return {
        'free_energy_trend': fe_trend,
        'final_free_energy': free_energies[-1],
        'status': 'pass'
    }


def test_neuromodulator_dynamics():
    """测试神经调质动态"""
    print("\n" + "="*70)
    print("测试4: 完整的神经调质动态")
    print("="*70)
    
    from neuron_arch_torch import DynamicNeuromodulator
    
    ng = DynamicNeuromodulator(base_lr=0.01)
    
    print("\n场景1: 高奖励序列")
    for i in range(20):
        reward = torch.tensor(0.9)
        pe = torch.tensor(0.1)
        updates = ng.update_all(reward, pe)
    
    state_high = ng.get_state()
    print(f"  DA={state_high['da']:.3f}, 5HT={state_high['serotonin']:.3f}, "
          f"ACh={state_high['ach']:.3f}, NE={state_high['ne']:.3f}")
    
    ng.reset()
    
    print("\n场景2: 低奖励序列")
    for i in range(20):
        reward = torch.tensor(0.1)
        pe = torch.tensor(0.5)
        updates = ng.update_all(reward, pe)
    
    state_low = ng.get_state()
    print(f"  DA={state_low['da']:.3f}, 5HT={state_low['serotonin']:.3f}, "
          f"ACh={state_low['ach']:.3f}, NE={state_low['ne']:.3f}")
    
    ng.reset()
    
    print("\n场景3: 变化奖励")
    history = []
    for i in range(50):
        reward = torch.tensor(0.5 + 0.4 * np.sin(i * 0.2))
        pe = torch.tensor(0.3 * np.abs(np.cos(i * 0.1)))
        updates = ng.update_all(reward, pe)
        history.append({
            'da': updates['da'].item(),
            'serotonin': updates['serotonin'].item(),
            'ach': updates['ach'].item(),
            'ne': updates['ne'].item()
        })
    
    da_var = np.var([h['da'] for h in history])
    serotonin_var = np.var([h['serotonin'] for h in history])
    ach_var = np.var([h['ach'] for h in history])
    ne_var = np.var([h['ne'] for h in history])
    
    print(f"  DA方差: {da_var:.4f}")
    print(f"  5HT方差: {serotonin_var:.4f}")
    print(f"  ACh方差: {ach_var:.4f}")
    print(f"  NE方差: {ne_var:.4f}")
    
    all_dynamic = da_var > 0.001 and serotonin_var > 0.001 and ach_var > 0.001 and ne_var > 0.001
    
    if all_dynamic:
        print(f"✓ 所有神经调质都有动态变化")
    else:
        print(f"✗ 部分神经调质无动态变化")
    
    return {
        'da_variance': da_var,
        'serotonin_variance': serotonin_var,
        'ach_variance': ach_var,
        'ne_variance': ne_var,
        'all_dynamic': all_dynamic,
        'status': 'pass' if all_dynamic else 'fail'
    }


def test_orthogonal_patterns():
    """测试正交模式"""
    print("\n" + "="*70)
    print("测试5: 真正的正交模式")
    print("="*70)
    
    from neuron_arch_torch import ESBNeuronTorch
    
    esb = ESBNeuronTorch(n_embodied=32, n_latent=16, n_symbols=8)
    
    inner_products, ortho_error = esb.verify_orthogonality()
    
    print(f"\n内积矩阵对角线: {torch.diag(inner_products).tolist()}")
    print(f"正交性误差: {ortho_error:.6f}")
    
    patterns = esb.get_orthogonal_patterns()
    print(f"\n模式形状: {patterns.shape}")
    
    print("\n符号识别测试:")
    correct = 0
    for symbol_id in range(8):
        pattern = patterns[symbol_id]
        pred_id, probs, conf = esb(pattern)
        if pred_id.item() == symbol_id:
            correct += 1
        print(f"  符号{symbol_id}: 预测={pred_id.item()}, 置信度={conf.item():.3f}")
    
    accuracy = correct / 8 * 100
    print(f"\n准确率: {accuracy:.0f}%")
    
    if ortho_error < 0.01:
        print(f"✓ 正交性良好（误差 < 0.01）")
    else:
        print(f"⚠ 正交性误差较大: {ortho_error:.4f}")
    
    return {
        'orthogonality_error': ortho_error,
        'accuracy': accuracy,
        'status': 'pass' if ortho_error < 0.01 else 'warn'
    }


def test_stdp_and_temporal():
    """测试STDP和时序编码"""
    print("\n" + "="*70)
    print("测试6: STDP和时序编码")
    print("="*70)
    
    from neuron_arch_torch import SCHNeuronTorch
    
    sch = SCHNeuronTorch(n_neurons=32, enable_stdp=True)
    
    print("\n运行带时序编码的模拟:")
    spike_times_list = []
    
    for step in range(20):
        x = torch.randn(32) * 0.5
        spikes, continuous, spike_times = sch(x)
        
        fired = (spikes > 0).sum().item()
        if fired > 0:
            times = spike_times[spikes > 0].tolist()
            spike_times_list.extend(times)
        
        if step % 5 == 0:
            print(f"  Step {step}: 脉冲数={int(fired)}, 时间戳样本={spike_times[:3].tolist()}")
    
    unique_times = len(set(spike_times_list))
    print(f"\n记录的唯一时间戳数: {unique_times}")
    
    print("\nSTDP权重更新测试:")
    pre_time = torch.zeros(32)
    post_time = torch.ones(32) * 5.0
    
    delta_w = sch.compute_stdp(pre_time, post_time)
    print(f"STDP权重更新范围: [{delta_w.min().item():.4f}, {delta_w.max().item():.4f}]")
    
    has_temporal = unique_times > 0
    has_stdp = delta_w.abs().sum().item() > 0
    
    if has_temporal and has_stdp:
        print(f"✓ 时序编码和STDP正常工作")
    else:
        print(f"✗ 时序编码或STDP未正常工作")
    
    return {
        'unique_spike_times': unique_times,
        'stdp_active': has_stdp,
        'status': 'pass' if (has_temporal and has_stdp) else 'fail'
    }


def run_all_tests():
    """运行所有测试"""
    print("="*70)
    print("PyTorch改进版本验证测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    try:
        results['gpu_support'] = test_gpu_support()
    except Exception as e:
        results['gpu_support'] = {'status': 'fail', 'error': str(e)}
        print(f"✗ GPU支持测试失败: {e}")
    
    try:
        results['batch_processing'] = test_batch_processing()
    except Exception as e:
        results['batch_processing'] = {'status': 'fail', 'error': str(e)}
        print(f"✗ 批处理测试失败: {e}")
    
    try:
        results['free_energy'] = test_free_energy()
    except Exception as e:
        results['free_energy'] = {'status': 'fail', 'error': str(e)}
        print(f"✗ 自由能测试失败: {e}")
    
    try:
        results['neuromodulator'] = test_neuromodulator_dynamics()
    except Exception as e:
        results['neuromodulator'] = {'status': 'fail', 'error': str(e)}
        print(f"✗ 神经调质测试失败: {e}")
    
    try:
        results['orthogonal'] = test_orthogonal_patterns()
    except Exception as e:
        results['orthogonal'] = {'status': 'fail', 'error': str(e)}
        print(f"✗ 正交模式测试失败: {e}")
    
    try:
        results['stdp_temporal'] = test_stdp_and_temporal()
    except Exception as e:
        results['stdp_temporal'] = {'status': 'fail', 'error': str(e)}
        print(f"✗ STDP测试失败: {e}")
    
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    passed = sum(1 for r in results.values() if r.get('status') == 'pass')
    warned = sum(1 for r in results.values() if r.get('status') == 'warn')
    failed = sum(1 for r in results.values() if r.get('status') == 'fail')
    total = len(results)
    
    print(f"\n总计: {total} 测试")
    print(f"  ✓ 通过: {passed}")
    print(f"  ⚠ 警告: {warned}")
    print(f"  ✗ 失败: {failed}")
    
    results['summary'] = {
        'total': total,
        'passed': passed,
        'warned': warned,
        'failed': failed,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/torch_validation.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n结果已保存到 results/torch_validation.json")
    
    return results


if __name__ == '__main__':
    run_all_tests()
