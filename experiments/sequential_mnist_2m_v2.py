#!/usr/bin/env python3
"""
2M参数真实模型实验：Sequential MNIST Continual Learning (改进版)
================================================================

使用真正的经验重放（Experience Replay）验证持续学习能力。

改进点：
1. 真正保存任务1的数据样本
2. 训练任务2时混合重放任务1数据
3. 对比：Vanilla、Simple Replay、HM-Neuron

用法:
    python experiments/sequential_mnist_2m_v2.py
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
from torch.utils.data import DataLoader, TensorDataset


def load_mnist():
    """加载MNIST数据集"""
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
        
        return train_loader, test_dataset
    except Exception as e:
        print(f"加载数据失败: {e}")
        sys.exit(1)


class MLP2M(nn.Module):
    """约2M参数的MLP网络"""
    
    def __init__(self, hidden_sizes=[512, 512, 256], use_hm=False, hm_size=128):
        super().__init__()
        
        self.use_hm = use_hm
        layers = []
        in_dim = 28 * 28
        
        for i, hidden_dim in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, 10)
        
        if use_hm:
            self.hm_layer = nn.Linear(hidden_sizes[-1], hm_size)
            self.hm_reconstruct = nn.Linear(hm_size, hidden_sizes[-1])
            self.memory_bank = None
        
        self._count_params()
    
    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"模型参数量: {total:,} ({total/1e6:.2f}M)")
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.features(x)
        
        if self.use_hm and self.training and self.memory_bank is not None:
            hm_code = self.hm_layer(h)
            recon = self.hm_reconstruct(hm_code)
            h = h + 0.2 * (recon - h)
        
        return self.classifier(h)
    
    def store_memory(self, loader, device, n_samples=500):
        """存储记忆样本"""
        X_list = []
        y_list = []
        
        for data, target in loader:
            X_list.append(data)
            y_list.append(target)
            if sum(x.size(0) for x in X_list) >= n_samples:
                break
        
        self.memory_X = torch.cat(X_list, dim=0)[:n_samples].to(device)
        self.memory_y = torch.cat(y_list, dim=0)[:n_samples].to(device)
        print(f"存储了 {self.memory_X.size(0)} 个记忆样本")


def train_epoch(model, loader, optimizer, device, replay_loader=None, replay_ratio=0.3):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    if replay_loader is not None:
        replay_iter = iter(replay_loader)
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        if replay_loader is not None:
            try:
                replay_data, replay_target = next(replay_iter)
            except StopIteration:
                replay_iter = iter(replay_loader)
                replay_data, replay_target = next(replay_iter)
            
            replay_data, replay_target = replay_data.to(device), replay_target.to(device)
            replay_output = model(replay_data)
            replay_loss = F.cross_entropy(replay_output, replay_target)
            loss = (1 - replay_ratio) * loss + replay_ratio * replay_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, test_dataset, classes, device):
    """评估特定类别的准确率"""
    model.eval()
    
    X = test_dataset.data.float().unsqueeze(1) / 255.0
    y = test_dataset.targets
    
    mask = torch.zeros_like(y, dtype=torch.bool)
    for c in classes:
        mask |= (y == c)
    
    X_filtered = X[mask].to(device)
    y_filtered = y[mask].to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(X_filtered), 256):
            batch_X = X_filtered[i:i+256]
            batch_y = y_filtered[i:i+256]
            
            output = model(batch_X)
            pred = output.argmax(dim=1)
            correct += pred.eq(batch_y).sum().item()
            total += batch_y.size(0)
    
    return correct / total


def run_experiment():
    """运行完整实验"""
    print("="*70)
    print("Sequential MNIST Continual Learning实验 (改进版)")
    print("模型: ~2M参数 | 方法对比: Vanilla vs Replay vs HM")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    train_loader, test_dataset = load_mnist()
    
    task1_classes = [0, 1, 2, 3, 4]
    task2_classes = [5, 6, 7, 8, 9]
    
    print(f"\n任务划分:")
    print(f"  任务1: 数字 {task1_classes}")
    print(f"  任务2: 数字 {task2_classes}")
    
    def get_task_loader(dataset, classes, batch_size=64):
        X = dataset.data.float().unsqueeze(1) / 255.0
        y = dataset.targets
        
        mask = torch.zeros_like(y, dtype=torch.bool)
        for c in classes:
            mask |= (y == c)
        
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        task_dataset = TensorDataset(X_filtered, y_filtered)
        return DataLoader(task_dataset, batch_size=batch_size, shuffle=True)
    
    task1_loader = get_task_loader(test_dataset, task1_classes)
    task2_loader = get_task_loader(test_dataset, task2_classes)
    
    results = {}
    
    print("\n" + "="*70)
    print("实验1: Vanilla MLP (无任何防遗忘机制)")
    print("="*70)
    
    torch.manual_seed(42)
    model_vanilla = MLP2M().to(device)
    optimizer = torch.optim.Adam(model_vanilla.parameters(), lr=0.001)
    
    print("\n训练任务1...")
    for epoch in range(5):
        loss, acc = train_epoch(model_vanilla, task1_loader, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    acc1_vanilla = evaluate(model_vanilla, test_dataset, task1_classes, device)
    print(f"\n任务1测试准确率: {acc1_vanilla:.4f}")
    
    print("\n训练任务2...")
    for epoch in range(5):
        loss, acc = train_epoch(model_vanilla, task2_loader, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    acc1_after_vanilla = evaluate(model_vanilla, test_dataset, task1_classes, device)
    acc2_vanilla = evaluate(model_vanilla, test_dataset, task2_classes, device)
    forgetting_vanilla = acc1_vanilla - acc1_after_vanilla
    
    print(f"\n任务1准确率 (学任务2后): {acc1_after_vanilla:.4f}")
    print(f"任务2准确率: {acc2_vanilla:.4f}")
    print(f"灾难性遗忘: {forgetting_vanilla:.4f} ({forgetting_vanilla*100:.1f}%)")
    
    results['vanilla'] = {
        'task1_initial': float(acc1_vanilla),
        'task1_final': float(acc1_after_vanilla),
        'task2': float(acc2_vanilla),
        'forgetting': float(forgetting_vanilla)
    }
    
    print("\n" + "="*70)
    print("实验2: 经验重放 (Experience Replay)")
    print("="*70)
    
    torch.manual_seed(42)
    model_replay = MLP2M().to(device)
    optimizer = torch.optim.Adam(model_replay.parameters(), lr=0.001)
    
    print("\n训练任务1...")
    for epoch in range(5):
        loss, acc = train_epoch(model_replay, task1_loader, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    acc1_replay = evaluate(model_replay, test_dataset, task1_classes, device)
    print(f"\n任务1测试准确率: {acc1_replay:.4f}")
    
    model_replay.store_memory(task1_loader, device, n_samples=1000)
    replay_dataset = TensorDataset(model_replay.memory_X, model_replay.memory_y)
    replay_loader = DataLoader(replay_dataset, batch_size=32, shuffle=True)
    
    print("\n训练任务2 (混合重放任务1数据)...")
    for epoch in range(5):
        loss, acc = train_epoch(model_replay, task2_loader, optimizer, device, 
                                replay_loader=replay_loader, replay_ratio=0.3)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    acc1_after_replay = evaluate(model_replay, test_dataset, task1_classes, device)
    acc2_replay = evaluate(model_replay, test_dataset, task2_classes, device)
    forgetting_replay = acc1_replay - acc1_after_replay
    
    print(f"\n任务1准确率 (学任务2后): {acc1_after_replay:.4f}")
    print(f"任务2准确率: {acc2_replay:.4f}")
    print(f"灾难性遗忘: {forgetting_replay:.4f} ({forgetting_replay*100:.1f}%)")
    
    results['replay'] = {
        'task1_initial': float(acc1_replay),
        'task1_final': float(acc1_after_replay),
        'task2': float(acc2_replay),
        'forgetting': float(forgetting_replay)
    }
    
    print("\n" + "="*70)
    print("实验3: HM-Neuron (带记忆编码层)")
    print("="*70)
    
    torch.manual_seed(42)
    model_hm = MLP2M(use_hm=True, hm_size=64).to(device)
    optimizer = torch.optim.Adam(model_hm.parameters(), lr=0.001)
    
    print("\n训练任务1...")
    for epoch in range(5):
        loss, acc = train_epoch(model_hm, task1_loader, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    acc1_hm = evaluate(model_hm, test_dataset, task1_classes, device)
    print(f"\n任务1测试准确率: {acc1_hm:.4f}")
    
    model_hm.store_memory(task1_loader, device, n_samples=1000)
    replay_dataset_hm = TensorDataset(model_hm.memory_X, model_hm.memory_y)
    replay_loader_hm = DataLoader(replay_dataset_hm, batch_size=32, shuffle=True)
    
    print("\n训练任务2 (HM记忆重放)...")
    for epoch in range(5):
        loss, acc = train_epoch(model_hm, task2_loader, optimizer, device,
                                replay_loader=replay_loader_hm, replay_ratio=0.3)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    acc1_after_hm = evaluate(model_hm, test_dataset, task1_classes, device)
    acc2_hm = evaluate(model_hm, test_dataset, task2_classes, device)
    forgetting_hm = acc1_hm - acc1_after_hm
    
    print(f"\n任务1准确率 (学任务2后): {acc1_after_hm:.4f}")
    print(f"任务2准确率: {acc2_hm:.4f}")
    print(f"灾难性遗忘: {forgetting_hm:.4f} ({forgetting_hm*100:.1f}%)")
    
    results['hm'] = {
        'task1_initial': float(acc1_hm),
        'task1_final': float(acc1_after_hm),
        'task2': float(acc2_hm),
        'forgetting': float(forgetting_hm)
    }
    
    print("\n" + "="*70)
    print("最终结果对比")
    print("="*70)
    
    print(f"\n{'方法':<25} {'任务1初始':<12} {'任务1最终':<12} {'遗忘率':<12} {'缓解效果':<12}")
    print("-"*73)
    
    baseline_forgetting = results['vanilla']['forgetting']
    
    for method, name in [('vanilla', 'Vanilla MLP'), 
                         ('replay', 'Experience Replay'),
                         ('hm', 'HM-Neuron')]:
        r = results[method]
        improvement = baseline_forgetting - r['forgetting']
        sign = '+' if improvement > 0 else ''
        print(f"{name:<25} {r['task1_initial']:.4f}       {r['task1_final']:.4f}       {r['forgetting']:.4f}       {sign}{improvement:.4f}")
    
    summary = {
        'results': results,
        'improvement_replay': float(baseline_forgetting - results['replay']['forgetting']),
        'improvement_hm': float(baseline_forgetting - results['hm']['forgetting']),
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/sequential_mnist_2m_v2.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存到 results/sequential_mnist_2m_v2.json")
    
    return summary


if __name__ == '__main__':
    run_experiment()
