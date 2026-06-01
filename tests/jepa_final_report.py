#!/usr/bin/env python3
"""
JEPA-Style 最终验证报告
======================

展示完整的JEPA实现效果
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuron_arch_torch.jepa_style import JEPAPredictiveCoder


def final_validation():
    print("="*70)
    print("JEPA-Style 最终验证报告")
    print("="*70)
    
    # 创建模型
    model = JEPAPredictiveCoder(
        input_dim=64,
        hidden_dim=128,
        latent_dim=32,
        momentum=0.99,
        sigreg_weight=0.01
    )
    
    print(f"\n模型配置:")
    print(f"  输入维度: 64")
    print(f"  隐藏维度: 128")
    print(f"  潜在维度: 32")
    print(f"  EMA动量: 0.99 Updat99")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  设备: {model.device}")
    
    # 生成有结构的数据
    print("\n" + "-"*70)
    print("数据生成")
    print("-"*70)
    
    n_samples = 500
    t = torch.linspace(0, 4*np.pi, n_samples)
    
    X = torch.zeros(n_samples, 64)
    X[:, 0] = torch.sin(t)
    X[:, 1] = torch.cos(t)
    X[:, 2] = torch.sin(2*t)
    X[:, 3] = torch.cos(2*t)
    X[:, 4:8] = t.unsqueeze(1).expand(-1, 4) / (4*np.pi)
    X[:, 8:16] = torch.randn(n_samples, 8) * 0.1
    X[:, 16:] = torch.randn(n_samples, 48) * 0.01
    
    print(f"  样本数: {n_samples}")
    print(f"  结构: sin/cos信号 + 线性趋势 + 噪声")
    
    # 训练
    print("\n" + "-"*70)
    print("训练进度")
    print("-"*70)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    losses = []
    sigreg_losses = []
    encoder_diffs = []
    
    for epoch in range(100):
        # 自监督mask
        context_mask = (torch.rand(n_samples, 64) > 0.4).float()
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
        
        # 测量编码器差异
        with torch.no_grad():
            z_enc = model.encode(X[:10])
            z_tgt = model.encode_target(X[:10])
            diff = (z_enc - z_tgt).norm() / (z_enc.norm() + 1e-6)
            encoder_diffs.append(diff.item())
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: loss={metrics['prediction_loss']:.4f}, "
                  f"sigreg={metrics['sigreg_loss']:.4f}, "
                  f"encoder_diff={diff.item():.4f}")
    
    # 结果分析
    print("\n" + "-"*70)
    print("结果分析")
    print("-"*70)
    
    print(f"\n损失变化:")
    print(f"  初始: {np.mean(losses[:5]):.4f}")
    print(f"  最终: {np.mean(losses[-5:]):.4f}")
    print(f"  改进: {(1 - np.mean(losses[-5:])/np.mean(losses[:5]))*100:.1f}%")
    
    print(f"\n编码器-目标编码器差异:")
    print(f"  初始: {encoder_diffs[0]:.4f}")
    print(f"  最终: {encoder_diffs[-1]:.4f}")
    
    print(f"\nSIGReg:")
    print(f"  初始: {sigreg_losses[0]:.4f}")
    print(f"  最终: {sigreg_losses[-1]:.4f}")
    
    # 表征质量
    model.eval()
    with torch.no_grad():
        z = model.encode(X)
    
    print(f"\n表征统计:")
    print(f"  均值: {z.mean().item():.4f}")
    print(f"  标准差: {z.std().item():.4f}")
    print(f"  最小值: {z.min().item():.4f}")
    print(f"  最大值: {z.max().item():.4f}")
    
    # 判断
    print("\n" + "-"*70)
    print("验证结果")
    print("-"*70)
    
    checks = []
    
    if np.mean(losses[-5:]) < np.mean(losses[:5]) * 0.5:
        checks.append(("损失显著下降 (>50%)", True))
    else:
        checks.append(("损失显著下降 (>50%)", False))
    
    if z.std().item() > 0.1:
        checks.append(("表征未坍缩 (std>0.1)", True))
    else:
        checks.append(("表征未坍缩 (std>0.1)", False))
    
    if encoder_diffs[-1] < 1.0:
        checks.append(("目标编码器合理跟踪 (diff<1.0)", True))
    else:
        checks.append(("目标编码器合理跟踪 (diff<1.0)", False))
    
    if sigreg_losses[-1] < 1.0:
        checks.append(("SIGReg稳定 (<1.0)", True))
    else:
        checks.append(("SIGReg稳定 (<1.0)", False))
    
    for name, passed in checks:
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}")
    
    all_passed = all(c[1] for c in checks)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ 所有验证通过！JEPA-Style实现有效")
        print("\n关键指标:")
        print(f"  - 损失改进: {(1 - np.mean(losses[-5:])/np.mean(losses[:5]))*100:.1f}%")
        print(f"  - 表征标准差: {z.std().item():.2f} (健康)")
        print(f"  - 目标编码器差异: {encoder_diffs[-1]:.2f} (正常滞后)")
    else:
        print("⚠ 部分验证未通过，需要进一步优化")
    print("="*70)
    
    return all_passed


if __name__ == '__main__':
    final_validation()
