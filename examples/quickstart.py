"""
Example: Quick Start with Neuron Architectures
=============================================
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from neuron_arch import (
    AdaptiveThresholdSCH,
    ResidualPCLayer,
    PatternSeparationHM,
    NormalizedNG,
    IntegratedSystem
)


def example_sch():
    """Example: SCH-Neuron for sparse encoding."""
    print("\n" + "="*60)
    print("SCH-Neuron Example")
    print("="*60)
    
    sch = AdaptiveThresholdSCH(n_neurons=64, target_spike_rate=0.1)
    
    for step in range(50):
        input_current = np.random.randn(64) * 0.5
        spikes, continuous = sch.step(input_current)
        
        if step % 10 == 0:
            print(f"Step {step}: spike_rate={np.mean(spikes):.3f}, "
                  f"sparsity={sch.get_sparsity():.3f}")
    
    print(f"\nFinal spike rate: {np.mean(sch.spike_history):.3f}")


def example_pc():
    """Example: PC-Neuron for predictive learning."""
    print("\n" + "="*60)
    print("PC-Neuron Example")
    print("="*60)
    
    pc = ResidualPCLayer(n_state=32, n_pred=32, lr=0.01)
    
    target = np.random.randn(32)
    
    for epoch in range(20):
        state = np.random.randn(32)
        prediction = pc.predict(state)
        error = np.linalg.norm(target - prediction)
        pc.local_learn(state, target)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: prediction_error={error:.4f}")


def example_hm():
    """Example: HM-Neuron for memory."""
    print("\n" + "="*60)
    print("HM-Neuron Example")
    print("="*60)
    
    hm = PatternSeparationHM(n_input=32, n_hidden=16, top_k=8)
    
    for task_id in range(3):
        print(f"\nTask {task_id}:")
        
        for _ in range(10):
            x = np.random.randn(32)
            target = np.random.randn(16)
            hm.learn(x, target, task_id=task_id)
        
        n_replay = hm.consolidate(n_replay=5)
        print(f"  Consolidated {n_replay} memories")
        print(f"  Buffer size: {len(hm.memory_buffer)}")


def example_ng():
    """Example: NG-Neuron for adaptive learning."""
    print("\n" + "="*60)
    print("NG-Neuron Example")
    print("="*60)
    
    ng = NormalizedNG(base_lr=0.01)
    
    rewards = [1.0, 0.5, 0.0, 0.8, 0.3]
    
    for i, reward in enumerate(rewards):
        ng.update_from_reward(reward)
        effective_lr = ng.compute_effective_lr()
        
        print(f"Step {i}: reward={reward:.1f}, DA={ng.da:.3f}, "
              f"effective_lr={effective_lr:.4f}")


def example_integrated():
    """Example: Full integrated system."""
    print("\n" + "="*60)
    print("Integrated System Example")
    print("="*60)
    
    system = IntegratedSystem(n_sch=64, n_pc=64, n_hm=32)
    
    params = system.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    
    print("\nTraining:")
    for epoch in range(10):
        x = np.random.rand()
        target = np.random.rand()
        
        metrics = system.learn(x, target)
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch}: sparsity={metrics.sparsity:.3f}, "
                  f"grad_norm={metrics.grad_norm:.3f}")
    
    system.consolidate(n_replay=20)
    
    avg_metrics = system.get_average_metrics()
    print(f"\nAverage metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    example_sch()
    example_pc()
    example_hm()
    example_ng()
    example_integrated()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
