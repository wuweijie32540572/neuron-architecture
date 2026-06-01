#!/usr/bin/env python3
"""
JEPA-Style完整验证测试
=====================

验证：
1. SIGReg计算是否正确
2. 训练是否收敛
3. 表征是否坍缩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuron_arch_torch.jepa_style import JEPAPredictiveCoder, JEPATrainer, SIGReg


def test_sigreg():
    """测试SIGReg计算"""
    print("\n" + "="*60)
    print("测试1: SIGReg正则化")
    print("="*60)
    
    sigreg = SIGReg()
    
    # 测试1: 相同表征（应该接近0）
    z1 = torch.randn(32, 16)
    z2 = z1.clone()
    loss_same = sigreg(z1, z2)
    print(f"\n相同表征的SIGReg: {loss_same.item():.4f} (应该接近0)")
    
    # 测试2: 正交表征（应该较高）
    z1 = torch.randn(32, 16)
    z1 = F.normalize(z1, dim=1)
    z2 = torch.randn(32, 16)
    z2 = F.normalize(z2, dim=1)
    loss_ortho = sigreg(z1, z2)
    print(f"正交表征的SIGReg: {loss_ortho.item():.4f}")
    
    # 测试3: 随机表征
    z1 = torch.randn(32, 16)
    z2 = torch.randn(32, 16)
    loss_random = sigreg(z1, z2)
    print(f"随机表征的SIGReg: {loss_random.item():.4f}")
    
    # 测试4: 坍缩表征（常数）
    z1 = torch.ones(32, 16)
    z2 = torch.ones(32, 16)
    loss_collapse = sigreg(z1, z2)
    print(f"坍缩表征的SIGReg: {loss_collapse.item():.4f}")
    
    print("\n✓ SIGReg测试完成")


def test_jepa_training():
    """测试JEPA训练"""
    print("\n" + "="*60)
    print("测试2: JEPA训练收敛")
    print("="*60)
    
    # 创建模型
    model = JEPAPredictiveCoder(
        input_dim=32,
        hidden_dim=64,
        latent_dim=16,
        momentum=0.996,
        sigreg_weight=0.01  # 降低SIGReg权重
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 生成训练数据（有结构的）
    n_samples = 200
    X = torch.randn(n_samples, 32)
    # 添加一些结构
    X[:, :8] = torch.sin(torch.linspace(0, 4*np.pi, n_samples)).unsqueeze(1) * 0.5
    X[:, 8:16] = torch.cos(torch.linspace(0, 4*np.pi, n_samples)).unsqueeze(1) * 0.5
    
    print(f"\n训练数据形状: {X.shape}")
    print(f"训练参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    losses = []
    sigreg_losses = []
    
    print("\n训练进度:")
    for epoch in range(50):
        # 随机mask
        context_mask = (torch.rand(n_samples, 32) > 0.4).float()
        target_mask = 1 - context_mask
        
        x_context = X * context_mask
        x_target = X * target_mask
        
        model.train()
        optimizer.zero_grad()
        
        loss, metrics = model(x_context, x_target)
        
        loss.backward()
        optimizer.step()
        
        model.update_target_encoder()
        
        losses.append(metrics['prediction_loss'])
        sigreg_losses.append(metrics['sigreg_loss'])
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:2d}: pred_loss={metrics['prediction_loss']:.4f}, "
                  f"sigreg={metrics['sigreg_loss']:.4f}, total={metrics['total_loss']:.4f}")
    
    # 检查收敛
    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\n收敛分析:")
    print(f"  初始损失: {initial_loss:.4f}")
    print(f"  最终损失: {final_loss:.4f}")
    print(f"  改进: {improvement:.1f}%")
    
    if improvement > 10:
        print("  ✓ 损失下降，训练有效")
    else:
        print("  ⚠ 损失下降不明显")
    
    return model


def test_representation_quality(model):
    """测试表征质量"""
    print("\n" + "="*60)
    print("测试3: 表征质量")
    print("="*60)
    
    # 生成测试数据
    X_test = torch.randn(100, 32)
    
    model.eval()
    with torch.no_grad():
        z = model.encode(X_test)
        z_target = model.encode_target(X_test)
    
    # 检查表征坍缩
    z_std = z.std(dim=0).mean().item()
    z_target_std = z_target.std(dim=0).mean().item()
    
    print(f"\n表征标准差:")
    print(f"  编码器: {z_std:.4f}")
    print(f"  目标编码器: {z_target_std:.4f}")
    
    if z_std > 0.1:
        print("  ✓ 表征未坍缩")
    else:
        print("  ⚠ 表征可能坍缩")
    
    # 检查编码器和目标编码器的差异
    diff = (z - z_target).norm() / z.norm()
    print(f"\n编码器与目标编码器差异: {diff.item():.4f}")
    
    if diff < 0.1:
        print("  ✓ 目标编码器正确跟踪编码器")
    else:
        print("  ⚠ 目标编码器与编码器差异较大")


def test_comparison_with_original():
    """与原始PC-Neuron对比"""
    print("\n" + "="*60)
    print("测试4: 与原始PC-Neuron对比")
    print("="*60)
    
    # 简单任务：学习y = sin(x)
    n = 100
    x = torch.linspace(0, 2*np.pi, n).unsqueeze(1)
    y = torch.sin(x)
    
    # 原始PC方法
    print("\n原始PC-Neuron:")
    W = torch.randn(1, 1, requires_grad=True)
    lr = 0.1
    
    losses_original = []
    for _ in range(100):
        pred = x @ W
        loss = F.mse_loss(pred, y)
        
        grad = 2 * (pred - y).T @ x / n
        W = W - lr * grad
        
        losses_original.append(loss.item())
    
    print(f"  最终MSE: {losses_original[-1]:.4f}")
    
    # JEPA-style方法
    print("\nJEPA-Style:")
    model = JEPAPredictiveCoder(
        input_dim=1,
        hidden_dim=16,
        latent_dim=8,
        sigreg_weight=0.001
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    losses_jepa = []
    for _ in range(100):
        # 自监督：预测下一时刻
        x_context = x[:-1]
        x_target = x[1:]
        
        model.train()
        optimizer.zero_grad()
        loss, _ = model(x_context, x_target)
        loss.backward()
        optimizer.step()
        model.update_target_encoder()
        
        losses_jepa.append(loss.item())
    
    print(f"  最终损失: {losses_jepa[-1]:.4f}")
    
    print("\n对比完成")


if __name__ == '__main__':
    print("="*60)
    print("JEPA-Style完整验证测试")
    print("="*60)
    
    test_sigreg()
    model = test_jepa_training()
    test_representation_quality(model)
    test_comparison_with_original()
    
    print("\n" + "="*60)
    print("所有测试完成")
    print("="*60)
