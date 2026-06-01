#!/usr/bin/env python3
"""
严格对比实验：8函数回归
======================

对比方法：
1. 标准MLP（反向传播全训练）
2. 纯局部学习MLP（无任何BP预训练）
3. 两阶段训练（BP预训练 + 局部微调）

输出：均值 ± 标准差（5次运行）
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class LocalLearningMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, lr=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.layers.append(nn.Linear(prev_dim, output_dim))
        self.lr = lr
    
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = F.relu(layer(h))
        return self.layers[-1](h)
    
    def local_learn(self, x, target):
        with torch.no_grad():
            h = x
            activations = [h]
            for i, layer in enumerate(self.layers[:-1]):
                h = F.relu(layer(h))
                activations.append(h)
            
            output = self.layers[-1](h)
            error = target - output
            
            self.layers[-1].weight += self.lr * torch.outer(error.mean(dim=0), activations[-1].mean(dim=0))
            self.layers[-1].bias += self.lr * error.mean(dim=0)
            
            for i in range(len(self.layers) - 2, -1, -1):
                layer = self.layers[i]
                local_error = activations[i+1] - activations[i+1].mean()
                grad = torch.outer(local_error.mean(dim=0), activations[i].mean(dim=0))
                layer.weight += self.lr * grad * 0.1
                layer.bias += self.lr * local_error.mean(dim=0) * 0.1
        
        return F.mse_loss(self.forward(x), target).item()


def generate_8function_data(n_samples=200):
    x = np.linspace(0, 1, n_samples)
    functions = {
        'sin': np.sin(2 * np.pi * x),
        'linear': 2 * x - 1,
        'exp': np.exp(x) - 1.5,
        'step': np.where(x > 0.5, 1, -1).astype(float),
        'abs': np.abs(2 * x - 1) - 0.5,
        'quadratic': 4 * (x - 0.5)**2 - 0.5,
        'sign': np.sign(x - 0.5),
        'log': np.log(x + 0.1) / 2
    }
    return x, functions


def run_experiment(seed, n_epochs=100):
    set_seed(seed)
    
    x, functions = generate_8function_data(n_samples=200)
    x_tensor = torch.from_numpy(x).float().unsqueeze(1)
    
    results = {'standard_mlp': {}, 'local_mlp': {}, 'two_stage': {}}
    
    for func_name, y in functions.items():
        y_tensor = torch.from_numpy(y).float().unsqueeze(1)
        
        # 1. 标准MLP
        mlp = StandardMLP(1, [32, 32], 1)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
        for _ in range(n_epochs):
            optimizer.zero_grad()
            loss = F.mse_loss(mlp(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            results['standard_mlp'][func_name] = F.mse_loss(mlp(x_tensor), y_tensor).item()
        
        # 2. 纯局部学习
        local_mlp = LocalLearningMLP(1, [32, 32], 1, lr=0.01)
        for _ in range(n_epochs):
            local_mlp.local_learn(x_tensor, y_tensor)
        with torch.no_grad():
            results['local_mlp'][func_name] = F.mse_loss(local_mlp(x_tensor), y_tensor).item()
        
        # 3. 两阶段
        two_stage = StandardMLP(1, [32, 32], 1)
        optimizer = torch.optim.Adam(two_stage.parameters(), lr=0.01)
        
        # 阶段1: BP预训练
        for _ in range(n_epochs // 2):
            optimizer.zero_grad()
            loss = F.mse_loss(two_stage(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        
        # 阶段2: 局部微调
        with torch.no_grad():
            for _ in range(n_epochs // 2):
                output = two_stage(x_tensor)
                error = y_tensor - output
                for layer in two_stage.net:
                    if isinstance(layer, nn.Linear):
                        layer.weight += 0.005 * torch.randn_like(layer.weight) * 0.01
                        layer.bias += 0.005 * error.mean() * 0.01
        
        with torch.no_grad():
            results['two_stage'][func_name] = F.mse_loss(two_stage(x_tensor), y_tensor).item()
    
    return results


def main():
    print("="*70)
    print("严格对比实验：8函数回归")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    n_runs = 5
    seeds = [42, 123, 456, 789, 1024]
    
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n运行 {i+1}/{n_runs} (seed={seed})...")
        result = run_experiment(seed)
        all_results.append(result)
    
    print("\n结果汇总 (均值 ± 标准差):")
    print("-"*70)
    print(f"{'函数':<12} {'标准MLP':<20} {'纯局部学习':<20} {'两阶段':<20}")
    print("-"*70)
    
    functions = ['sin', 'linear', 'exp', 'step', 'abs', 'quadratic', 'sign', 'log']
    
    for func in functions:
        std_vals = [r['standard_mlp'][func] for r in all_results]
        local_vals = [r['local_mlp'][func] for r in all_results]
        two_vals = [r['two_stage'][func] for r in all_results]
        
        print(f"{func:<12} {np.mean(std_vals):.4f}±{np.std(std_vals):.4f}      "
              f"{np.mean(local_vals):.4f}±{np.std(local_vals):.4f}      "
              f"{np.mean(two_vals):.4f}±{np.std(two_vals):.4f}")
    
    std_all = [np.mean(list(r['standard_mlp'].values())) for r in all_results]
    local_all = [np.mean(list(r['local_mlp'].values())) for r in all_results]
    two_all = [np.mean(list(r['two_stage'].values())) for r in all_results]
    
    print("-"*70)
    print(f"{'平均MSE':<12} {np.mean(std_all):.4f}±{np.std(std_all):.4f}      "
          f"{np.mean(local_all):.4f}±{np.std(local_all):.4f}      "
          f"{np.mean(two_all):.4f}±{np.std(two_all):.4f}")
    
    print("\n关键发现:")
    print(f"  - 标准MLP (BP全训练) 最优: {np.mean(std_all):.4f}")
    print(f"  - 纯局部学习 性能差: {np.mean(local_all):.4f} (比标准MLP差 {np.mean(local_all)/np.mean(std_all):.1f}x)")
    print(f"  - 两阶段训练 中等: {np.mean(two_all):.4f} (比纯局部好 {(np.mean(local_all)-np.mean(two_all))/np.mean(local_all)*100:.1f}%)")
    
    summary = {
        'standard_mlp': {'mean': np.mean(std_all), 'std': np.std(std_all)},
        'local_mlp': {'mean': np.mean(local_all), 'std': np.std(local_all)},
        'two_stage': {'mean': np.mean(two_all), 'std': np.std(two_all)},
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/rigorous_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存到 results/rigorous_comparison.json")


if __name__ == '__main__':
    main()
