# 架构设计文档

本文档详细描述五种新型神经元架构的设计原理、数学基础和实现细节。

## 1. SCH-Neuron: 脉冲-连续混合神经元

### 设计动机

传统神经网络使用连续激活函数，无法实现事件驱动的稀疏计算。生物神经元通过脉冲发放进行信息传递，具有天然的稀疏性和事件驱动特性。

### 架构设计

```
输入电流 → 膜电位积分 → 阈值比较 → 脉冲发放
                ↓
           连续表示输出
```

### 数学模型

**膜电位动态**:
```
τ_m · dv/dt = -v + R_m · I(t)
```

**脉冲发放**:
```
s(t) = H(v(t) - v_th)  # H是Heaviside阶跃函数
```

**自适应阈值** (关键创新):
```
rate_error = current_spike_rate - target_spike_rate
adapt_factor = 1 + α · rate_error
v_th(t) = v_th_base · adapt_factor
v_th(t) = clip(v_th(t), 0.05, 0.5)
```

**连续表示**:
```
z(t) = decay · z(t-1) + s(t)  # 脉冲轨迹
continuous = tanh(v / v_th)   # 连续输出
```

### 防御性设计

**问题**: 维度增大时脉冲率不可控上升

**解决**: 目标脉冲率反馈控制，维持脉冲率在0.05-0.2范围

### 代码实现

```python
class AdaptiveThresholdSCH:
    def step(self, input_current):
        # 自适应阈值调整
        current_rate = np.mean(self.spike_history)
        rate_error = current_rate - self.target_spike_rate
        adapt_factor = 1.0 + self.adapt_strength * rate_error
        self.v_th = self.v_th_base * adapt_factor
        
        # 膜电位积分
        dv = (-self.v + input_current) / self.tau_m
        self.v = self.v + dv
        
        # 脉冲发放
        spikes = (self.v >= self.v_th).astype(float)
        self.v = np.where(spikes > 0, 0.0, self.v)
        
        return spikes, continuous
```

---

## 2. PC-Neuron: 预测编码神经元

### 设计动机

反向传播需要全局梯度传播，生物神经网络使用局部学习规则。预测编码理论提供了一种基于局部误差的学习框架。

### 架构设计

```
状态输入 → 预测生成 → 误差计算 → 局部学习
              ↓
         预测输出
```

### 数学模型

**预测生成**:
```
μ = W · z  # 预测值
```

**自由能**:
```
F = D_KL[q||p] - E[ln p(s|ψ)]
  ≈ Σ (target - μ)² / 2
```

**局部学习规则**:
```
ε = target - μ           # 预测误差
ΔW = -η · ε · z^T        # 局部梯度
```

**残差连接 + 层归一化** (防御梯度消失):
```
output = LayerNorm(x + W·z)
```

### 两阶段训练 (关键创新)

**问题**: 局部学习信号 ε=z-μ 与任务目标 (x-x̂) 存在结构性断裂

**解决**:
1. **离线预训练**: 用任务监督信号训练权重
2. **在线适应**: 用局部学习微调

```
预训练: min L = Σ(x - x̂)²  # 任务监督
在线:   ΔW = -η · ∂F/∂W    # 局部学习
```

### 代码实现

```python
class ResidualPCLayer:
    def predict(self, state):
        linear = self.W @ state
        self.mu = self.layer_norm(state + linear)
        return self.mu
    
    def local_learn(self, prev_state, target):
        epsilon = target - self.mu
        grad = np.outer(epsilon, prev_state)
        
        # 梯度裁剪
        if np.linalg.norm(grad) > self.target_grad_norm:
            grad = grad * (self.target_grad_norm / np.linalg.norm(grad))
        
        self.W += self.lr * grad
```

---

## 3. HM-Neuron: 海马体-皮层神经元

### 设计动机

灾难性遗忘是连续学习的核心挑战。海马体-皮层架构通过系统巩固缓解遗忘。

### 架构设计

```
            ┌─────────────┐
            │   海马体     │ 快速学习，短期记忆
            │ (Hippocampus)│ 模式分离
            └──────┬──────┘
                   │ 重放
                   ↓
            ┌─────────────┐
            │    皮层      │ 慢速学习，长期记忆
            │  (Cortex)   │ 系统巩固
            └─────────────┘
```

### 数学模型

**海马体学习**:
```
ΔW_H = η_H · ε · x^T  # 快速学习，η_H=0.05
```

**皮层学习**:
```
ΔW_C = η_C · ε · x^T  # 慢速学习，η_C=0.01
```

**记忆稳定性度量** (关键创新):
```
σ(m) = 1 / (1 + α·age + β/(access+1))

- age: 记忆年龄（步数）
- access: 访问次数
- α, β: 权重参数
```

**风险加权重放**:
```
risk(m) = 1 - σ(m)
P(采样m) ∝ risk(m)
```

**模式分离** (Top-k稀疏化):
```
activity = W_H · x
top_k_indices = argsort(|activity|)[-k:]
output[top_k_indices] = activity[top_k_indices]
```

### 代码实现

```python
class PatternSeparationHM:
    def pattern_separation(self, x):
        activity = self.W_hippo @ x
        top_k_indices = np.argsort(np.abs(activity))[-self.top_k:]
        sparse_output = np.zeros_like(activity)
        sparse_output[top_k_indices] = activity[top_k_indices]
        return np.tanh(sparse_output)
    
    def consolidate(self, n_replay):
        # 风险加权采样
        ages = np.arange(len(self.memory_buffer), 0, -1)
        risks = 1.0 - 1.0 / (1.0 + 0.05 * ages)
        probs = risks / np.sum(risks)
        
        indices = np.random.choice(len(self.memory_buffer), n_replay, p=probs)
        for idx in indices:
            x, target = self.memory_buffer[idx]
            # 皮层巩固
            self.cortex_learn(x, target)
```

---

## 4. NG-Neuron: 神经调质门控神经元

### 设计动机

生物大脑通过神经调质（多巴胺、血清素等）动态调节学习。传统神经网络使用固定学习率，无法适应动态环境。

### 架构设计

```
        ┌─────┐
        │ DA  │ 奖励预测误差 → 学习率门控
        ├─────┤
        │ 5HT │ 情绪状态 → 探索/利用权衡
        ├─────┤
        │ ACh │ 注意力 → 稀疏性门控
        ├─────┤
        │ NE  │ 唤醒水平 → 增益门控
        └─────┘
           ↓
      加权和门控
           ↓
      有效学习率
```

### 数学模型

**DA更新** (奖励预测误差驱动):
```
RPE = reward - reward_prediction
DA = sigmoid(α · RPE)
```

**门控信号** (加权和):
```
gate = 0.4·DA + 0.2·5HT + 0.2·ACh + 0.2·NE
gate = clip(gate, 0.1, 0.9)
```

**自适应学习率**:
```
η(t) = η_base · gate · 2.0 / (1 + t·(1-DA))
η(t) = clip(η(t), 0.002, 0.02)
```

**探索/利用权衡**:
```
- 高DA: 探索阶段，学习率衰减慢
- 低DA: 利用阶段，学习率衰减快
```

### 代码实现

```python
class NormalizedNG:
    def compute_gate(self):
        weights = np.array([0.4, 0.2, 0.2, 0.2])
        signals = np.array([self.da, self.serotonin, self.ach, self.ne])
        gate = np.dot(weights, signals)
        return np.clip(gate, 0.1, 0.9)
    
    def compute_effective_lr(self):
        decay = 1.0 + self.step_count * (1.0 - self.da) * 0.005
        gate = self.compute_gate()
        effective_lr = self.base_lr * gate * 2.0 / decay
        return np.clip(effective_lr, 0.002, 0.02)
    
    def update_from_reward(self, reward):
        rpe = reward - self.reward_prediction
        self.da = np.clip(0.5 + 0.3 * np.tanh(rpe), 0.1, 0.9)
```

---

## 5. ESB-Neuron: 具身-符号桥接神经元

### 设计动机

符号接地问题：神经网络的表示与物理世界没有直接联系。需要一种机制将传感器输入（具身）与概念（符号）绑定。

### 架构设计

```
传感器输入 → 具身编码 → 潜在空间 → 符号解码 → 离散概念
                ↓                      ↓
           Grounding矩阵 ←────────── 符号嵌入
```

### 数学模型

**正交输入模式** (确保区分度):
```
patterns[i, start:end] = 1.0  # 块对角
patterns[i] += noise · N(0,1)  # 小噪声
patterns[i] = patterns[i] / ||patterns[i]||  # 归一化
```

**具身编码**:
```
latent = tanh(W_encode · sensor_input)
```

**符号解码**:
```
logits = W_decode · latent
probs = softmax(logits · τ)  # τ是温度参数
symbol = argmax(probs)
```

**Grounding**:
```
grounding_matrix[symbol] = (1-α)·grounding_matrix[symbol] + α·sensor_input
symbol_embeddings[symbol] = (1-α)·symbol_embeddings[symbol] + α·latent
```

### 代码实现

```python
class ESBNeuronLayer:
    def encode_embodied(self, sensor_input):
        latent = np.tanh(self.W_encode @ sensor_input)
        return latent
    
    def decode_symbol(self, latent):
        logits = self.W_decode @ latent
        probs = self._softmax(logits * 3.0)  # 温度=3
        symbol_id = np.argmax(probs)
        confidence = probs[symbol_id]
        return symbol_id, probs, confidence
    
    def ground_symbol(self, symbol_id, embodied_example):
        self.grounding_matrix[symbol_id] = (
            0.9 * self.grounding_matrix[symbol_id] + 
            0.1 * embodied_example
        )
```

---

## 系统集成

### 完整架构

```
┌─────────────────────────────────────────────────────────┐
│                    输入层                                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  SCH-Neuron (自适应阈值脉冲发放)                         │
│  输出: spikes, continuous                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  PC-Neuron (预测编码 + 残差连接)                         │
│  输出: prediction                                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  HM-Neuron (海马体-皮层 + 模式分离)                      │
│  输出: memory_output                                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  NG-Neuron (神经调质门控)                                │
│  控制: effective_lr, gate                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    输出层                                │
└─────────────────────────────────────────────────────────┘
```

### 参数规模

```
SCH: 256 neurons  → W_adapt: 256×256 = 65,536
PC:  256 state    → W: 256×256 = 65,536
HM:  128 hidden   → W_hippo + W_cortex: 256×128×2 = 65,536
其他: W_encode + W_decode = 384

总计: ~197K 参数
```

### 调度器

```python
class ModuleScheduler:
    frequencies = {
        'sch': 1,              # 每步运行
        'pc': 1,               # 每步运行
        'hm': 1,               # 每步运行
        'hm_consolidate': 10,  # 每10步巩固
        'ng': 1                # 每步运行
    }
```

---

## 监控指标

| 指标 | 计算方式 | 目标范围 |
|------|----------|----------|
| 稀疏性 | 1 - spike_rate | >0.5 |
| 梯度范数 | \|\|grad\|\| | <5.0 |
| 脉冲率 | mean(spikes) | 0.05-0.2 |
| 自由能 | mean(error²) | - |
| 有效学习率 | η_eff | 0.002-0.02 |
| DA | 神经调质值 | 0.1-0.9 |
