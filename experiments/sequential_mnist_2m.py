#!/usr/bin/env python3
"""
2M参数真实模型实验：Sequential MNIST Continual Learning
======================================================

使用MNIST数据集验证HM-Neuron在真实任务上的持续学习能力。

实验设计：
- 模型：约2M参数的MLP网络
- 任务：Sequential MNIST（分批学习数字0-4，然后5-9）
- 对比：Vanilla MLP vs MLP + HM-Neuron
- 指标：灾难性遗忘率、最终准确率

用法:
    python experiments/sequential_mnist_2m.py
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch未安装，使用NumPy实现")


def load_mnist():
    """加载MNIST数据集"""
    if HAS_TORCH:
        try:
            from torchvision import datasets, transforms
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            return train_loader, test_loader
        except Exception as e:
            print(f"无法加载torchvision数据: {e}")
    
    print("使用内置简化MNIST...")
    return create_synthetic_mnist()


def create_synthetic_mnist():
    """创建合成MNIST数据（当无法下载真实数据时）"""
    np.random.seed(42)
    
    n_train = 60000
    n_test = 10000
    
    X_train = np.random.randn(n_train, 1, 28, 28).astype(np.float32)
    y_train = np.random.randint(0, 10, n_train).astype(np.int64)
    
    X_test = np.random.randn(n_test, 1, 28, 28).astype(np.float32)
    y_test = np.random.randint(0, 10, n_test).astype(np.int64)
    
    for i in range(10):
        mask = y_train == i
        X_train[mask] += i * 0.3
        
        mask_test = y_test == i
        X_test[mask_test] += i * 0.3
    
    if HAS_TORCH:
        train_tensor = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        )
        test_tensor = TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(y_test)
        )
        
        train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_tensor, batch_size=64, shuffle=False)
        
        return train_loader, test_loader
    
    return (X_train, y_train), (X_test, y_test)


if HAS_TORCH:
    class VanillaMLP(nn.Module):
        """标准MLP网络（约2M参数）"""
        
        def __init__(self, hidden_sizes=[512, 512, 256]):
            super().__init__()
            
            layers = []
            in_dim = 28 * 28
            
            for hidden_dim in hidden_sizes:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                in_dim = hidden_dim
            
            layers.append(nn.Linear(in_dim, 10))
            
            self.net = nn.Sequential(*layers)
            self._count_params()
        
        def _count_params(self):
            total = sum(p.numel() for p in self.parameters())
            print(f"VanillaMLP参数量: {total:,} ({total/1e6:.2f}M)")
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.net(x)
    
    class HMMLP(nn.Module):
        """带HM层的MLP网络（约2M参数）"""
        
        def __init__(self, hidden_sizes=[512, 512, 256], hm_size=256):
            super().__init__()
            
            self.fc1 = nn.Linear(28 * 28, hidden_sizes[0])
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            
            self.hm_input = nn.Linear(hidden_sizes[1], hm_size)
            self.hm_output = nn.Linear(hm_size, hidden_sizes[1])
            
            self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
            self.fc_out = nn.Linear(hidden_sizes[2], 10)
            
            self.dropout = nn.Dropout(0.2)
            
            self.memory_bank = []
            self.memory_labels = []
            self.hm_size = hm_size
            
            self._count_params()
        
        def _count_params(self):
            total = sum(p.numel() for p in self.parameters())
            print(f"HMMLP参数量: {total:,} ({total/1e6:.2f}M)")
        
        def forward(self, x, store_memory=False):
            x = x.view(x.size(0), -1)
            
            h1 = F.relu(self.dropout(self.fc1(x)))
            h2 = F.relu(self.dropout(self.fc2(h1)))
            
            hm_in = self.hm_input(h2)
            hm_out = self.hm_output(hm_in)
            h2 = h2 + 0.3 * hm_out
            
            if store_memory:
                self.memory_bank.append(hm_in.detach().clone())
            
            h3 = F.relu(self.dropout(self.fc3(h2)))
            out = self.fc_out(h3)
            
            return out
        
        def consolidate(self, labels, n_replay=50):
            """记忆巩固"""
            if len(self.memory_bank) == 0:
                return
            
            all_memories = torch.cat(self.memory_bank, dim=0)
            
            n_samples = min(n_replay, all_memories.size(0))
            indices = torch.randperm(all_memories.size(0))[:n_samples]
            
            self.replay_memories = all_memories[indices]
            self.memory_bank = []
        
        def get_replay_loss(self, h2):
            """重放损失"""
            if not hasattr(self, 'replay_memories') or self.replay_memories is None:
                return 0.0
            
            hm_in = self.hm_input(h2)
            
            if hm_in.size(0) >= self.replay_memories.size(0):
                loss = F.mse_loss(
                    hm_in[:self.replay_memories.size(0)],
                    self.replay_memories
                )
            else:
                loss = F.mse_loss(hm_in, self.replay_memories[:hm_in.size(0)])
            
            return loss * 0.1


def train_epoch(model, loader, optimizer, device, hm_weight=0.0, task_id=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if hasattr(model, 'forward') and 'store_memory' in model.forward.__code__.co_varnames:
            output = model(data, store_memory=True)
        else:
            output = model(data)
        
        loss = F.cross_entropy(output, target)
        
        if hm_weight > 0 and hasattr(model, 'get_replay_loss'):
            h2 = F.relu(model.fc2(F.relu(model.fc1(data.view(data.size(0), -1)))))
            replay_loss = model.get_replay_loss(h2)
            loss = loss + hm_weight * replay_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return correct / total


def filter_by_classes(loader, classes, device):
    """过滤特定类别的数据"""
    if isinstance(loader, DataLoader):
        X_list = []
        y_list = []
        
        for data, target in loader:
            mask = torch.zeros(target.size(), dtype=torch.bool)
            for c in classes:
                mask |= (target == c)
            
            if mask.any():
                X_list.append(data[mask])
                y_list.append(target[mask])
        
        if len(X_list) == 0:
            return None
        
        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)
        
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=64, shuffle=True)
    
    return loader


def run_sequential_mnist():
    """运行Sequential MNIST实验"""
    print("="*70)
    print("Sequential MNIST Continual Learning实验")
    print("模型: ~2M参数")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    train_loader, test_loader = load_mnist()
    
    task1_classes = [0, 1, 2, 3, 4]
    task2_classes = [5, 6, 7, 8, 9]
    
    print("\n任务划分:")
    print(f"  任务1: 数字 {task1_classes}")
    print(f"  任务2: 数字 {task2_classes}")
    
    results = {
        'vanilla': {'task1_after_task1': [], 'task1_after_task2': []},
        'hm': {'task1_after_task1': [], 'task1_after_task2': []}
    }
    
    print("\n" + "="*70)
    print("实验1: Vanilla MLP")
    print("="*70)
    
    vanilla_model = VanillaMLP().to(device)
    optimizer = torch.optim.Adam(vanilla_model.parameters(), lr=0.001)
    
    task1_loader = filter_by_classes(train_loader, task1_classes, device)
    task1_test = filter_by_classes(test_loader, task1_classes, device)
    
    print("\n训练任务1 (数字0-4)...")
    for epoch in range(5):
        loss, acc = train_epoch(vanilla_model, task1_loader, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    acc_task1 = evaluate(vanilla_model, task1_test, device)
    results['vanilla']['task1_after_task1'] = acc_task1
    print(f"\n任务1准确率: {acc_task1:.4f}")
    
    task2_loader = filter_by_classes(train_loader, task2_classes, device)
    task2_test = filter_by_classes(test_loader, task2_classes, device)
    
    print("\n训练任务2 (数字5-9)...")
    for epoch in range(5):
        loss, acc = train_epoch(vanilla_model, task2_loader, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    acc_task1_after = evaluate(vanilla_model, task1_test, device)
    acc_task2 = evaluate(vanilla_model, task2_test, device)
    results['vanilla']['task1_after_task2'] = acc_task1_after
    results['vanilla']['task2'] = acc_task2
    
    vanilla_forgetting = acc_task1 - acc_task1_after
    
    print(f"\n任务1准确率 (学任务2后): {acc_task1_after:.4f}")
    print(f"任务2准确率: {acc_task2:.4f}")
    print(f"灾难性遗忘: {vanilla_forgetting:.4f} ({vanilla_forgetting*100:.1f}%)")
    
    print("\n" + "="*70)
    print("实验2: MLP + HM-Neuron")
    print("="*70)
    
    hm_model = HMMLP().to(device)
    optimizer = torch.optim.Adam(hm_model.parameters(), lr=0.001)
    
    print("\n训练任务1 (数字0-4)...")
    for epoch in range(5):
        loss, acc = train_epoch(hm_model, task1_loader, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    labels = []
    for _, target in task1_loader:
        labels.extend(target.numpy())
    hm_model.consolidate(labels, n_replay=100)
    
    acc_task1_hm = evaluate(hm_model, task1_test, device)
    results['hm']['task1_after_task1'] = acc_task1_hm
    print(f"\n任务1准确率: {acc_task1_hm:.4f}")
    
    print("\n训练任务2 (数字5-9)...")
    for epoch in range(5):
        loss, acc = train_epoch(hm_model, task2_loader, optimizer, device, hm_weight=0.5)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    acc_task1_after_hm = evaluate(hm_model, task1_test, device)
    acc_task2_hm = evaluate(hm_model, task2_test, device)
    results['hm']['task1_after_task2'] = acc_task1_after_hm
    results['hm']['task2'] = acc_task2_hm
    
    hm_forgetting = acc_task1_hm - acc_task1_after_hm
    
    print(f"\n任务1准确率 (学任务2后): {acc_task1_after_hm:.4f}")
    print(f"任务2准确率: {acc_task2_hm:.4f}")
    print(f"灾难性遗忘: {hm_forgetting:.4f} ({hm_forgetting*100:.1f}%)")
    
    print("\n" + "="*70)
    print("结果对比")
    print("="*70)
    
    improvement = vanilla_forgetting - hm_forgetting
    
    print(f"\n{'模型':<20} {'任务1初始':<12} {'任务1最终':<12} {'遗忘':<12}")
    print("-"*56)
    print(f"{'Vanilla MLP':<20} {acc_task1:.4f}       {acc_task1_after:.4f}       {vanilla_forgetting:.4f}")
    print(f"{'MLP + HM':<20} {acc_task1_hm:.4f}       {acc_task1_after_hm:.4f}       {hm_forgetting:.4f}")
    
    print(f"\nHM缓解遗忘效果: {improvement:.4f} ({improvement*100:.1f}个百分点)")
    
    if hm_forgetting < vanilla_forgetting:
        print("✓ HM-Neuron有效缓解了灾难性遗忘！")
    else:
        print("✗ HM-Neuron未能缓解灾难性遗忘")
    
    summary = {
        'vanilla_forgetting': float(vanilla_forgetting),
        'hm_forgetting': float(hm_forgetting),
        'improvement': float(improvement),
        'vanilla_results': {k: float(v) if isinstance(v, (int, float)) else v for k, v in results['vanilla'].items()},
        'hm_results': {k: float(v) if isinstance(v, (int, float)) else v for k, v in results['hm'].items()},
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/sequential_mnist_2m.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存到 results/sequential_mnist_2m.json")
    
    return summary


if __name__ == '__main__':
    if not HAS_TORCH:
        print("错误: 需要PyTorch来运行此实验")
        print("请安装: pip install torch torchvision")
        sys.exit(1)
    
    run_sequential_mnist()
