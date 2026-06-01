#!/usr/bin/env python3
"""
一键复现所有实验结果
==================

运行所有关键实验并生成报告。

用法:
    python run_all_experiments.py

输出:
    - 控制台日志
    - results/experiment_report.json
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np


def run_experiment(name, func):
    """运行单个实验并记录结果"""
    print(f"\n{'='*70}")
    print(f"实验: {name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    try:
        result = func()
        elapsed = time.time() - start_time
        result['status'] = 'success'
        result['elapsed_seconds'] = elapsed
        print(f"\n✓ 完成 ({elapsed:.1f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ 失败: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'elapsed_seconds': elapsed
        }


def experiment_two_stage_training():
    """实验1: 两阶段训练 vs 纯局部学习"""
    from neuron_arch import ResidualPCLayer
    
    np.random.seed(42)
    
    # 生成数据
    data = np.sin(np.linspace(0, 4*np.pi, 200)) * 0.5 + 0.5
    
    # 纯局部学习
    pc_local = ResidualPCLayer(16, 16, lr=0.05)
    errors_local = []
    
    prev_z = np.zeros(16)
    for x in data:
        state = prev_z
        target = np.zeros(16)
        target[:8] = x
        pred = pc_local.predict(state)
        pc_local.local_learn(state, target)
        prev_z = target
        errors_local.append(np.mean((target - pred)**2))
    
    mse_local = np.mean(errors_local[-50:])
    
    # 两阶段训练
    pc_two_stage = ResidualPCLayer(16, 16, lr=0.02)
    
    # 预训练
    for epoch in range(50):
        for x in data:
            state = np.zeros(16)
            target = np.zeros(16)
            target[:8] = x
            pred = pc_two_stage.predict(state)
            pc_two_stage.local_learn(state, target)
    
    # 在线适应
    errors_two_stage = []
    prev_z = np.zeros(16)
    for x in data:
        state = prev_z
        target = np.zeros(16)
        target[:8] = x
        pred = pc_two_stage.predict(state)
        pc_two_stage.local_learn(state, target)
        prev_z = target
        errors_two_stage.append(np.mean((target - pred)**2))
    
    mse_two_stage = np.mean(errors_two_stage[-50:])
    
    improvement = (mse_local - mse_two_stage) / mse_local * 100
    
    print(f"纯局部学习 MSE: {mse_local:.4f}")
    print(f"两阶段训练 MSE: {mse_two_stage:.4f}")
    print(f"改进: {improvement:.1f}%")
    
    return {
        'mse_local': float(mse_local),
        'mse_two_stage': float(mse_two_stage),
        'improvement_pct': float(improvement)
    }


def experiment_hm_forgetting():
    """实验2: HM遗忘缓解"""
    from neuron_arch import PatternSeparationHM
    
    np.random.seed(42)
    
    # 生成正交任务
    def generate_task(task_id, n=100):
        x = np.linspace(0, 1, n)
        np.random.seed(42 + task_id)
        noise = np.random.randn(n) * 0.03
        
        if task_id == 0:
            y = np.sin(2*np.pi*x)
        elif task_id == 1:
            y = 2*x - 1
        elif task_id == 2:
            y = np.exp(x) - 1.5
        else:
            y = np.abs(2*x - 1) - 0.5
        
        return x, y + noise
    
    tasks = [generate_task(i) for i in range(4)]
    
    # 无HM
    from neuron_arch.integrated import IntegratedSystem
    
    # 简化测试：只测量遗忘
    hm = PatternSeparationHM(16, 8, hippocampus_lr=0.05, cortex_lr=0.01)
    
    task0_performances = []
    
    for task_id in range(4):
        x_data, y_data = tasks[task_id]
        
        # 训练当前任务
        for x, y in zip(x_data, y_data):
            inp = np.zeros(16)
            inp[0] = x
            target = np.zeros(8)
            target[0] = y
            hm.learn(inp, target, task_id=task_id)
        
        hm.consolidate(n_replay=20)
        
        # 评估任务0
        x0, y0 = tasks[0]
        total_error = 0
        for x, y in zip(x0[:20], y0[:20]):
            inp = np.zeros(16)
            inp[0] = x
            pred = hm.forward(inp)
            total_error += (y - pred[0])**2
        task0_performances.append(total_error / 20)
    
    initial_mse = task0_performances[0]
    worst_mse = max(task0_performances)
    forgetting_rate = (worst_mse - initial_mse) / initial_mse * 100
    
    print(f"任务0初始MSE: {initial_mse:.4f}")
    print(f"任务0最差MSE: {worst_mse:.4f}")
    print(f"遗忘率: {forgetting_rate:+.1f}%")
    
    return {
        'initial_mse': float(initial_mse),
        'worst_mse': float(worst_mse),
        'forgetting_rate': float(forgetting_rate),
        'task0_trajectory': [float(x) for x in task0_performances]
    }


def experiment_spike_rate():
    """实验3: SCH脉冲率控制"""
    from neuron_arch import AdaptiveThresholdSCH
    
    np.random.seed(42)
    
    sch = AdaptiveThresholdSCH(64, target_spike_rate=0.1)
    
    for _ in range(100):
        inp = np.random.randn(64) * 0.5
        sch.step(inp)
    
    final_rate = np.mean(sch.spike_history)
    sparsity = sch.get_sparsity()
    
    print(f"目标脉冲率: 0.10")
    print(f"实际脉冲率: {final_rate:.4f}")
    print(f"稀疏性: {sparsity:.4f}")
    
    rate_ok = 0.05 < final_rate < 0.2
    
    return {
        'target_rate': 0.1,
        'actual_rate': float(final_rate),
        'sparsity': float(sparsity),
        'in_target_range': bool(rate_ok)
    }


def experiment_ng_gating():
    """实验4: NG门控机制"""
    from neuron_arch import NormalizedNG
    
    ng = NormalizedNG(base_lr=0.01)
    
    results = []
    
    # 高奖励
    for _ in range(50):
        ng.update_from_reward(1.0)
        results.append({
            'reward': 1.0,
            'da': ng.da,
            'effective_lr': ng.compute_effective_lr()
        })
    
    high_da = ng.da
    high_lr = ng.compute_effective_lr()
    
    ng.reset()
    
    # 低奖励
    for _ in range(50):
        ng.update_from_reward(0.0)
        results.append({
            'reward': 0.0,
            'da': ng.da,
            'effective_lr': ng.compute_effective_lr()
        })
    
    low_da = ng.da
    low_lr = ng.compute_effective_lr()
    
    print(f"高奖励: DA={high_da:.3f}, lr={high_lr:.4f}")
    print(f"低奖励: DA={low_da:.3f}, lr={low_lr:.4f}")
    print(f"DA差异: {high_da - low_da:.3f}")
    
    return {
        'high_reward': {'da': float(high_da), 'lr': float(high_lr)},
        'low_reward': {'da': float(low_da), 'lr': float(low_lr)},
        'da_difference': float(high_da - low_da)
    }


def experiment_symbol_recognition():
    """实验5: ESB符号识别"""
    from neuron_arch import ESBNeuronLayer
    
    np.random.seed(42)
    
    esb = ESBNeuronLayer(16, 8, 4)
    
    # 生成正交模式
    patterns = np.zeros((4, 16))
    for i in range(4):
        patterns[i, i*4:(i+1)*4] = 1.0
        patterns[i] += np.random.randn(16) * 0.05
        patterns[i] /= np.linalg.norm(patterns[i])
    
    # 训练
    for epoch in range(100):
        for symbol_id in range(4):
            esb.learn(patterns[symbol_id], symbol_id, lr=0.1)
    
    # 测试
    correct = 0
    confidences = []
    for symbol_id in range(4):
        latent = esb.encode_embodied(patterns[symbol_id])
        pred_id, probs, conf = esb.decode_symbol(latent)
        if pred_id == symbol_id:
            correct += 1
        confidences.append(conf)
    
    accuracy = correct / 4 * 100
    avg_confidence = np.mean(confidences)
    
    print(f"准确率: {accuracy:.0f}%")
    print(f"平均置信度: {avg_confidence:.3f} (随机=0.25)")
    
    return {
        'accuracy': float(accuracy),
        'avg_confidence': float(avg_confidence),
        'random_baseline': 0.25
    }


def main():
    """运行所有实验"""
    print("="*70)
    print("一键复现所有实验")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    results = {}
    
    results['two_stage_training'] = run_experiment(
        "两阶段训练 vs 纯局部学习",
        experiment_two_stage_training
    )
    
    results['hm_forgetting'] = run_experiment(
        "HM遗忘缓解",
        experiment_hm_forgetting
    )
    
    results['spike_rate'] = run_experiment(
        "SCH脉冲率控制",
        experiment_spike_rate
    )
    
    results['ng_gating'] = run_experiment(
        "NG门控机制",
        experiment_ng_gating
    )
    
    results['symbol_recognition'] = run_experiment(
        "ESB符号识别",
        experiment_symbol_recognition
    )
    
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)
    
    summary = {
        'two_stage_improvement': results['two_stage_training'].get('improvement_pct', 0),
        'hm_forgetting_rate': results['hm_forgetting'].get('forgetting_rate', 0),
        'spike_rate_ok': results['spike_rate'].get('in_target_range', False),
        'symbol_accuracy': results['symbol_recognition'].get('accuracy', 0)
    }
    
    print(f"\n两阶段训练改进: {summary['two_stage_improvement']:.1f}%")
    print(f"HM遗忘率: {summary['hm_forgetting_rate']:+.1f}%")
    print(f"脉冲率达标: {'✓' if summary['spike_rate_ok'] else '✗'}")
    print(f"符号识别准确率: {summary['symbol_accuracy']:.0f}%")
    
    results['summary'] = summary
    results['timestamp'] = datetime.now().isoformat()
    
    with open('results/experiment_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到 results/experiment_report.json")
    
    return results


if __name__ == '__main__':
    main()
