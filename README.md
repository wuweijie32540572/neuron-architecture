# 新型神经元架构研究

**弥补大语言模型核心缺陷的神经科学启发式架构**

本项目探索基于神经科学的新型神经元架构，旨在解决大语言模型(LLM)的核心缺陷：灾难性遗忘、缺乏局部学习、事件驱动能力缺失、符号接地问题等。

## 核心架构

| 架构 | 全称 | 解决的问题 | 关键机制 |
|------|------|------------|----------|
| **SCH-Neuron** | 脉冲-连续混合神经元 | 事件驱动计算 | 自适应阈值脉冲发放 |
| **PC-Neuron** | 预测编码神经元 | 局部学习规则 | 自由能最小化 |
| **HM-Neuron** | 海马体-皮层神经元 | 灾难性遗忘 | 系统巩固+风险加权重放 |
| **NG-Neuron** | 神经调质门控神经元 | 动态调节 | DA/5HT/ACh/NE四通道门控 |
| **ESB-Neuron** | 具身-符号桥接神经元 | 符号接地 | 正交模式+grounding |

## 项目结构

```
/workspace
├── README.md                    # 项目概述
├── ARCHITECTURE.md              # 架构设计文档
├── EXPERIMENTS.md               # 实验结果文档
│
├── mvp_sch_pc_v4.py             # 两阶段训练验证
├── mvp_sch_pc_v5_1_hm.py        # HM记忆层(稳定性度量)
├── mvp_sch_pc_v6_1_ng_esb.py    # NG+ESB实现
├── mvp_sch_pc_v6_2_da.py        # DA机制设计
├── mvp_sch_pc_v7_integrated.py  # 完整集成系统(197K参数)
│
├── llm_architecture_improvement.md  # 架构改进设计
├── llm_vs_brain_analysis.md         # LLM与大脑对比分析
│
└── mvp_v*_results.json          # 实验结果数据
```

## 快速开始

### 环境要求

- Python 3.8+
- NumPy

```bash
pip install numpy
```

### 运行实验

```bash
# 两阶段训练验证
python mvp_sch_pc_v4.py

# HM记忆层验证
python mvp_sch_pc_v5_1_hm.py

# NG+ESB验证
python mvp_sch_pc_v6_1_ng_esb.py

# DA机制验证
python mvp_sch_pc_v6_2_da.py

# 完整系统集成验证(197K参数)
python mvp_sch_pc_v7_integrated.py
```

## 核心发现

### 1. 两阶段训练 (v4)

**问题**: 局部学习信号 ε=z-μ 与任务目标 (x-x̂) 存在结构性断裂

**方案**: 离线预训练 + 在线适应

**结果**: 
- v3纯局部学习 MSE: 6.64
- v4两阶段训练 MSE: 0.12
- **改进 98.1%**

### 2. HM记忆层 (v5.1)

**问题**: 灾难性遗忘

**方案**: 
- 记忆稳定性度量: σ(m) = 1/(1 + α·age + β/(access+1))
- 风险加权重放: risk(m) ∝ 1-σ(m)

**结果**:
- 无HM遗忘率: +232%
- 带HM遗忘率: +88%
- **缓解 143.9个百分点**

### 3. NG神经调质门控 (v6.2)

**问题**: 固定学习率无法适应动态环境

**方案**: 
- DA控制探索/利用权衡
- 自适应学习率: η(t) = η_base / (1 + t·(1-DA))

**结果**:
- 高DA初期收敛更快
- 低DA后期更稳定
- **DA机制验证正确**

### 4. ESB具身-符号桥接 (v6.1)

**问题**: 符号接地问题

**方案**: 
- 正交输入模式
- Grounding矩阵绑定

**结果**:
- 符号识别准确率: 100%
- 平均置信度: 0.987 (随机=0.25)

### 5. 完整系统集成 (v7)

**规模**: 197K参数 (SCH:256 + PC:256 + HM:128)

**验证命题**: HM遗忘缓解 + PC异常检测能否同时工作

**结果**:
- 脉冲率: 0.0518 ✓
- 异常检测比: 2.53x ✓
- 遗忘率(最差): +194.8% ✗

**结论**: 简单的海马体-皮层重放机制在真正正交的冲突任务上，不足以完全缓解灾难性遗忘。

## 数学基础

### SCH-Neuron 自适应阈值

```
v_th(t) = v_th_base × (1 + α·rate_error)
rate_error = current_spike_rate - target_spike_rate
```

### PC-Neuron 自由能最小化

```
F = D_KL[q||p] - E[ln p(s|ψ)]
局部学习: ΔW = -η · ∂F/∂W
```

### HM-Neuron 记忆稳定性

```
σ(m) = 1 / (1 + α·age + β/(access+1))
重放概率: P(m) ∝ 1 - σ(m)
```

### NG-Neuron 门控信号

```
gate = 0.4·DA + 0.2·5HT + 0.2·ACh + 0.2·NE
η_eff = η_base × gate × 2.0 / (1 + t·(1-DA))
```

## 防御性设计

| 风险 | 防御措施 | 效果 |
|------|----------|------|
| 稀疏性崩溃 | 自适应阈值反馈控制 | 稀疏性保持0.94 |
| 梯度爆炸 | 梯度裁剪+自适应学习率 | 梯度范数稳定3.1 |
| HM检索退化 | Top-k模式分离 | 遗忘率改善 |
| NG信号冲突 | 加权和门控 | effective_lr稳定 |

## 局限性与未来方向

### 当前局限

1. **HM遗忘缓解不足**: 在真正正交的8个冲突任务上，最差遗忘率仍达+194.8%
2. **任务相似性依赖**: 遗忘缓解效果部分依赖于任务间的意外正迁移
3. **规模限制**: 仅在197K参数规模验证，未扩展到更大规模

### 未来方向

1. **更强的遗忘缓解**: 弹性权重巩固(EWC)、渐进神经网络(PGN)
2. **真正的异步调度**: 模块间的时间尺度解耦
3. **更大规模验证**: 1M+参数的完整系统
4. **真实任务评估**: 传感器数据、机器人控制

## 引用

如果您在研究中使用本项目的代码或想法，请引用：

```bibtex
@misc{neuron_architecture_2024,
  title={Novel Neuron Architectures for Overcoming LLM Limitations},
  author={Research Team},
  year={2024},
  howpublished={\\url{https://github.com/xxx/neuron-architecture}}
}
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request。
