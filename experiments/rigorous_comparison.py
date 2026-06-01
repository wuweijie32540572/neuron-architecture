#!/usr/bin/env python3
"""
严格对比实验：合理基线 + 多次运行统计
=====================================

对比方法：
1. 标准MLP（反向传播全训练）
2. 纯局部学习MLP（无任何BP预训练）
3. 两阶段训练（BP预训练 + 局部微调）

任务：8函数回归 + Split MNIST持续学习

输出：均值 ± 标准差（5次运行）
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class StandardMLP(nn.Module):
    """标准MLP，使用反向传播全训练"""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class LocalLearningMLP(nn.Module):
    """纯局部学习MLP，无BP预训练"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, lr=0.01):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.layers.append(nn.Linear(prev_dim, output_dim))
        
        self.lr = lr
        self.n_layers = len(self.layers)
    
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = F.relu(layer(h))
        return self.layers[-1](h)
    
    def local_learn(self, x, target):
        """局部学习规则（无反向传播）"""
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
        
        return F.mse_loss(output, target).item()


class TwoStageMLP(nn.Module):
    """两阶段训练：BP预训练 + 局部微调"""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.mlp = StandardMLP(input_dim, hidden_dims, output_dim)
        self.pretrained = False
    
    def forward(self, x):
        return self.mlp(x)
    
    def pretrain(self, x, target, lr=0.01):
        """第一阶段：BP预训练"""
        self.mlp.train()
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        optimizer.zero_grad()
        output = self.mlp(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def local_finetune(self, x, target, lr=0.005):
        """第二阶段：局部微调"""
        with torch.no_grad():
            output = self.mlp(x)
            error = target - output
            
            for layer in self.mlp.net:
                if isinstance(layer, nn.Linear):
                    grad = torch.outer(error.mean(dim=0), torch.ones(layer.weight.shape[1]) * 0.1)
                    layer.weight += lr * grad * 0.1
                    layer.bias += lr * error.mean(dim=0) * 0.1
        
        return F.mse_loss(output, target).item()


def generate_8function_data(n_samples=200):
    """生成8函数回归数据"""
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


def run_8function_experiment(seed, n_epochs=100):
    """运行8函数回归实验"""
    set_seed(seed)
    
    x, functions = generate_8function_data(n_samples=200)
    x_tensor = torch.from_numpy(x).float().unsqueeze(1)
    
    results = {
        'standard_mlp': {},
        'local_mlp': {},
        'two_stage': {}
    }
    
    for func_name, y in functions.items():
        y_tensor = torch.from_numpy(y).float().unsqueeze(1)
        
        # 1. 标准MLP (BP全训练)
        mlp = StandardMLP(1, [32, 32], 1)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
        
        for _ in range(n_epochs):
            optimizer.zero_grad()
            output = mlp(x_tensor)
            loss = F.mse_loss(output, y_tensor)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            final_loss = F.mse_loss(mlp(x_tensor), y_tensor).item()
        results['standard_mlp'][func_name] = final_loss
        
        # 2. 纯局部学习MLP
        local_mlp = LocalLearningMLP(1, [32, 32], 1, lr=0.01)
        
        for _ in range(n_epochs):
            local_mlp.local_learn(x_tensor, y_tensor)
        
        with torch.no_grad():
            final_loss = F.mse_loss(local_mlp(x_tensor), y_tensor).item()
        results['local_mlp'][func_name] = final_loss
        
        # 3. 两阶段训练
        two_stage = TwoStageMLP(1, [32, 32], 1)
        
        # 阶段1: BP预训练
        for _ in range(n_epochs // 2):
            two_stage.pretrain(x_tensor, y_tensor, lr=0.01)
        
        # 阶段2: 局部微调
        for _ in range(n_epochs // 2):
            two_stage.local_finetune(x_tensor, y_tensor, lr=0.005)
        
        with torch.no_grad():
            final_loss = F.mse_loss(two_stage(x_tensor), y_tensor).item()
        results['two_stage'][func_name] = final_loss
    
    return results


def run_split_mnist(seed, n_epochs_per_task=5):
    """运行Split MNIST持续学习实验"""
    set_seed(seed)
    
    if not HAS_TORCHVISION:
        print("torchvision未安装，跳过Split MNIST")
        return None
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    def get_task_data(digits):
        train_idx = torch.isin(train_data.targets, torch.tensor(digits))
        test_idx = torch.isin(test_data.targets, torch.tensor(digits))
        
        train_x = train_data.data[train_idx].float().unsqueeze(1) / 255.0
        train_y = train_data.targets[train_idx]
        test_x = test_data.data[test_idx].float().unsqueeze(1) / 255.0
        test_y = test_data.targets[test_idx]
        
        return (train_x, train_y), (test_x, test_y)
    
    task1_train, task1_test = get_task_data([0, 1, 2, 3, 4])
    task2_train, task2_test = get_task_data([5, 6, 7, 8, 9])
    
    results = {
        'standard_mlp': {},
        'local_mlp': {},
        'two_stage': {}
    }
    
    input_dim = 28 * 28
    
    # 1. 标准MLP
    mlp = StandardMLP(input_dim, [256, 256], 10)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    
    # 训练任务1
    train_x1 = task1_train[0].view(-1, input_dim)
    train_y1 = task1_train[1]
    dataset1 = TensorDataset(train_x1, train_y1)
    loader1 = DataLoader(dataset1, batch_size=64, shuffle=True)
    
    for _ in range(n_epochs_per_task):
        for x, y in loader1:
            optimizer.zero_grad()
            output = mlp(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
    
    # 测试任务1
    test_x1 = task1_test[0].view(-1, input_dim)
    test_y1 = task1_test[1]
    with torch.no_grad():
        pred = mlp(test_x1).argmax(dim=1)
        acc1_after_task1 = (pred == test_y1).float().mean().item()
    
    # 训练任务2
    train_x2 = task2_train[0].view(-1, input_dim)
    train_y2 = task2_train[1]
    dataset2 = TensorDataset(train_x2, train_y2)
    loader2 = DataLoader(dataset2, batch_size=64, shuffle=True)
    
    for _ in range(n_epochs_per_task):
        for x, y in loader2:
            optimizer.zero_grad()
            output = mlp(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
    
    # 测试任务1（遗忘）
    with torch.no_grad():
        pred = mlp(test_x1).argmax(dim=1)
        acc1_after_task2 = (pred == test_y1).float().mean().item()
    
    results['standard_mlp'] = {
        'task1_after_task1': acc1_after_task1,
        'task1_after_task2': acc1_after_task2,
        'forgetting': acc1_after_task1 - acc1_after_task2
    }
    
    # 2. 纯局部学习MLP
    local_mlp = LocalLearningMLP(input_dim, [256, 256], 10, lr=0.001)
    
    # 训练任务1
    for _ in range(n_epochs_per_task):
        for i in range(0, len(train_x1), 64):
            x = train_x1[i:i+64]
            y = train_y1[i:i+64]
            target = F.one_hot(y, 10).float()
            local_mlp.local_learn(x, target)
    
    # 测试任务1
    with torch.no_grad():
        pred = local_mlp(test_x1).argmax(dim=1)
        acc1_after_task1_local = (pred == test_y1).float().mean().item()
    
    # 训练任务2
    for _ in range(n_epochs_per_task):
        for i in range(0, len(train_x2), 64):
            x = train_x2[i:i+64]
            y = train_y2[i:i+64]
            target = F.one_hot(y, 10).float()
            local_mlp.local_learn(x, target)
    
    # 测试任务1（遗忘）
    with torch.no_grad():
        pred = local_mlp(test_x1).argmax(dim=1)
        acc1_after_task2_local = (pred == test_y1).float().mean().item()
    
    results['local_mlp'] = {
        'task1_after_task1': acc1_after_task1_local,
        'task1_after_task2': acc1_after_task2_local,
        'forgetting': acc1_after_task1_local - acc1_after_task2_local
    }
    
    # 3. 两阶段训练
    two_stage = TwoStageMLP(input_dim, [256, 256], 10)
    
    # 阶段1: BP预训练任务1
    for _ in range(n_epochs_per_task // 2):
        for x, y in loader1:
            target = F.one_hot(y, 10).float()
            two_stage.pretrain(x, target, lr=0.001)
    
    # 阶段2: 局部微调任务1
    for _ in range(n_epochs_per_task // 2):
        for i in range(0, len(train_x1), 64):
            x = train_x1[i:i+64]
            y = train_y1[i:i+64]
            target = F.one_hot(y, 10).float()
            two_stage.local_finetune(x, target, lr=0.0005)
    
    # 测试任务1
    with torch.no_grad():
        pred = two_stage(test_x1).argmax(dim=1)
        acc1_after_task1_two = (pred == test_y1).float().mean().item()
    
    # 训练任务2
    for _ in range(n_epochs_per_task // 2):
        for x, y in loader2:
            target = F.one_hot(y, 10).float()
            two_stage.pretrain(x, target, lr=0.001)
    
    for _ in range(n_epochs_per_task // 2):
        for i in range(0, len(train_x2), 64):
            x = train_x2[i:i+64]
            y = train_y2[i:i+64]
            target = F.one_hot(y, 10).float()
            two_stage.local_finetune(x, target, lr=0.0005)
    
    # 测试任务1（遗忘）
    with torch.no_grad():
        pred = two_stage(test_x1).argmax(dim=1)
        acc1_after_task2_two = (pred == test_y1).float().mean().item()
    
    results['two_stage'] = {
        'task1_after_task1': acc1_after_task1_two,
        'task1_after_task2': acc1_after_task2_two,
        'forgetting': acc1_after_task1_two - acc1_after_task2_two
    }
    
    return results


def compute_statistics(all_results, method, key=None):
    """计算均值和标准差"""
    if key:
        values = [r[method][key] for r in all_results if r and method in r and key in r[method]]
    else:
        values = [r[method] for r in all_results if r and method in r]
    
    if len(values) == 0:
        return None
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'n': len(values)
    }


def main():
    """运行所有实验"""
    print("="*70)
    print("严格对比实验：合理基线 + 多次运行统计")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    n_runs = 5
    seeds = [42, 123, 456, 789, 1024]
    
    # 8函数回归实验
    print("\n" + "="*70)
    print("实验1: 8函数回归")
    print("="*70)
    
    all_func_results = []
    for i, seed in enumerate(seeds):
        print(f"\n运行 {i+1}/{n_runs} (seed={seed})...")
        result = run_8function_experiment(seed)
        all_func_results.append(result)
    
    print("\n结果汇总 (均值 ± 标准差):")
    print("-"*70)
    print(f"{'函数':<12} {'标准MLP':<20} {'纯局部学习':<20} {'两阶段':<20}")
    print("-"*70)
    
    functions = ['sin', 'linear', 'exp', 'step', 'abs', 'quadratic', 'sign', 'log']
    
    for func in functions:
        std_stats = compute_statistics(all_func_results, 'standard_mlp', func)
        local_stats = compute_statistics(all_func_results, 'local_mlp', func)
        two_stats = compute_statistics(all_func_results, 'two_stage', func)
        
        std_str = f"{std_stats['mean']:.4f}±{std_stats['std']:.4f}" if std_stats else "N/A"
        local_str = f"{local_stats['mean']:.4f}±{local_stats['std']:.4f}" if local_stats else "N/A"
        two_str = f"{two_stats['mean']:.4f}±{two_stats['std']:.4f}" if two_stats else "N/A"
        
        print(f"{func:<12} {std_str:<20} {local_str:<20} {two_str:<20}")
    
    # 计算总体MSE
    std_all = [np.mean(list(r['standard_mlp'].values())) for r in all_func_results]
    local_all = [np.mean(list(r['local_mlp'].values())) for r in all_func_results]
    two_all = [np.mean(list(r['two_stage'].values())) for r in all_func_results]
    
    print("-"*70)
    print(f"{'平均MSE':<12} {np.mean(std_all):.4f}±{np.std(std_all):.4f}      "
          f"{np.mean(local_all):.4f}±{np.std(local_all):.4f}      "
          f"{np.mean(two_all):.4f}±{np.std(two_all):.4f}")
    
    # Split MNIST实验
    print("\n" + "="*70)
    print("实验2: Split MNIST持续学习")
    print("="*70)
    
    all_mnist_results = []
    for i, seed in enumerate(seeds):
        print(f"\n运行 {i+1}/{n_runs} (seed={seed})...")
        result = run_split_mnist(seed)
        if result:
            all_mnist_results.append(result)
    
    if all_mnist_results:
        print("\n结果汇总 (均值 ± 标准差):")
        print("-"*70)
        print(f"{'方法':<20} {'任务1初始准确率':<20} {'任务1最终准确率':<20} {'遗忘率':<20}")
        print("-"*70)
        
        for method, name in [('standard_mlp', '标准MLP'), ('local_mlp', '纯局部学习'), ('two_stage', '两阶段')]:
            init_stats = compute_statistics(all_mnist_results, method, 'task1_after_task1')
            final_stats = compute_statistics(all_mnist_results, method, 'task1_after_task2')
            forget_stats = compute_statistics(all_mnist_results, method, 'forgetting')
            
            init_str = f"{init_stats['mean']:.4f}±{init_stats['std']:.4f}" if init_stats else "N/A"
            final_str = f"{final_stats['mean']:.4f}±{final_stats['std']:.4f}" if final_stats else "N/A"
            forget_str = f"{forget_stats['mean']:.4f}±{forget_stats['std']:.4f}" if forget_stats else "N/A"
            
            print(f"{name:<20} {init_str:<20} {final_str:<20} {forget_str:<20}")
    
    # 保存结果
    summary = {
        '8function': {
            'standard_mlp': {'mean': np.mean(std_all), 'std': np.std(std_all)},
            'local_mlp': {'mean': np.mean(local_all), 'std': np.std(local_all)},
            'two_stage': {'mean': np.mean(two_all), 'std': np.std(two_all)},
            'raw_results': all_func_results
        },
        'split_mnist': {
            'raw_results': all_mnist_results
        },
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/rigorous_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n结果已保存到 results/rigorous_comparison.json")
    
    return summary


if __name__ == '__main__':
    main()
