# LLM vs 人类大脑：多维度深度解构

> 从物理学、生物学、数学、微积分、电子技术、神经网络技术的交叉视角，用公式推导拆解两种智能系统的本质差异与能力边界

---

## 目录
1. 尺度与物理常数对比
2. 热力学与能效分析
3. 信息处理架构
4. 时序动力学
5. 可塑性与学习机制
6. 记忆系统
7. 容错与鲁棒性
8. 数学框架对比
9. "不可能"的边界

---

## 1. 尺度与物理常数对比

### 1.1 基础参数

| 参数 | 人类大脑 | LLM (GPT-3级别) | 比值 |
|------|----------|------------------|------|
| 神经元数量 | 86 billion (8.6×10¹⁰) | ~175 billion (1.75×10¹¹) 参数 | 0.5x 参数/神经元 |
| 突触数量 | ~100 trillion (10¹⁴) | N/A | — |
| 神经元类型 | ~10,000 种 | 1 种（Transformer块） | — |
| 信号形式 | 脉冲 (0.1-100 mV, 1-2 ms) | 浮点向量 (FP16/FP32) | — |
| 基础频率 | 0.1-200 Hz | N/A | — |
| 空间尺度 | ~15 cm (头骨内径) | ~数GB (分布式) | — |
| 质量 | ~1.4 kg | ~数GB | 10⁶x 质量差 |
| 代谢功率 | ~20 W | ~100 kW (训练) / ~1 kW (推理) | 5000x / 50x |

### 1.2 关键洞察

**参数数量陷阱**：LLM有1750亿参数（GPT-3），但每个参数是独立的浮点数。大脑的100万亿突触是高度结构化的、上下文依赖的、可塑的权重，其信息密度远高于LLM的参数量。

定义**有效信息密度**：

$$\rho_{info} = \frac{I_{mutual}}{N_{params} \cdot \log_2(V)}$$

其中 $I_{mutual}$ 为互信息（有效信息传递），$N_{params}$ 为参数量，$V$ 为值域。

大脑的有效信息密度远高于LLM，因为：
1. 每个突触权重是动态的（短期/长期可塑性）
2. 突触权重依赖于精确的时序关系（STDP）
3. 突触是化学的、多模态的（不仅仅是"权重"）

---

## 2. 热力学与能效分析

### 2.1 兰道尔极限对比

**兰道尔原理**（Landauer's Principle）：不可逆计算擦除1 bit信息消耗的能量：

$$E_{min} = k_B T \ln 2$$

在体温 T = 310K 时：

$$E_{min} = 1.38 \times 10^{-23} \times 310 \times 0.693 \approx 2.97 \times 10^{-21} \text{ J/bit}$$

### 2.2 实际能效计算

**人类大脑**：

$$P_{brain} = 20 \text{ W}$$

每秒突触操作数估算：

$$OPS_{brain} \approx N_{synapses} \times f_{avg} = 10^{14} \times 1 \approx 10^{14} \text{ ops/s}$$

每操作能量：

$$E_{op,brain} = \frac{20}{10^{14}} = 2 \times 10^{-13} \text{ J/op}$$

与兰道尔极限的比值：

$$\eta_{brain} = \frac{E_{op,brain}}{E_{min}} = \frac{2 \times 10^{-13}}{2.97 \times 10^{-21}} \approx 6.7 \times 10^{7} \approx 67 \text{ million倍}$$

**LLM推理（高端GPU）**：

$$P_{LLM} \approx 300 \text{ W (A100 80GB)}$$

$$E_{op,LLM} = \frac{300}{3 \times 10^{12}} = 10^{-13} \text{ J/op}$$

$$\eta_{LLM} \approx 3.4 \times 10^{7} \approx 34 \text{ million倍}$$

### 2.3 关键发现

**令人震惊的结论**：大脑的能效比（约6700万倍）与高端AI芯片（约3400万倍）处于同一数量级。

但这掩盖了巨大的结构差异：

1. **计算类型不同**：大脑做的是稀疏的、事件驱动的计算；LLM做的是密集的、批量的矩阵乘法
2. **信息密度不同**：大脑的每个"操作"携带更多语义信息
3. **架构不同**：大脑是存算一体的；LLM是存算分离的冯·诺依曼架构

### 2.4 冯·诺依曼瓶颈

冯·诺依曼架构的数据传输能量：

$$E_{mem} = C_{bit} \times V_{dd}^2$$

其中 $C_{bit}$ 为每位电容，$V_{dd}$ 为电压。

对于典型DRAM：

$$E_{mem} \approx 2 \times 10^{-15} \text{ J/bit}$$

而本地计算（ALU）：

$$E_{calc} \approx 10^{-15} \text{ J/op}$$

**数据传输能量与计算能量同量级**——这是冯·诺依曼瓶颈的物理根源。

大脑是存算一体的：每个突触既是存储单元又是计算单元。数据传输距离只有几微米到几毫米。

LLM的数据传输距离：从DRAM到GPU HBM，可能跨越数厘米。

$$\frac{E_{LLM}}{E_{brain}} \approx \frac{10 \text{ cm}}{10 \mu m} \times \frac{300W}{20W} \approx 10^4$$

**LLM的数据传输能量比大脑高约1万倍。**

---

## 3. 信息处理架构

### 3.1 神经元模型对比

**生物神经元（Hodgkin-Huxley模型）**：

$$C_m \frac{dV}{dt} = I_{ext} - g_K n^4 (V - V_K) - g_{Na} m^3 h (V - V_{Na}) - g_L (V - V_L)$$

门控变量动力学：

$$\frac{dn}{dt} = \alpha_n(V)(1-n) - \beta_n(V)n$$
$$\frac{dm}{dt} = \alpha_m(V)(1-m) - \beta_m(V)m$$
$$\frac{dh}{dt} = \alpha_h(V)(1-h) - \beta_h(V)h$$

**Transformer中的"神经元"（注意力头）**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

这不是神经元，是矩阵运算。

### 3.2 信号编码对比

**大脑的编码方式**（多编码并行）：

| 编码类型 | 描述 | 数学表示 |
|----------|------|----------|
| 频率编码 | 发放率编码信息 | $r_i = \frac{n_i}{T}$ |
| 时序编码 | 精确脉冲时间 | $t_i \in [0, T]$ |
| 群体编码 | N个神经元联合编码 | $\vec{r} = (r_1, ..., r_N)$ |
| 相位编码 | 振荡相位编码 | $\phi_i(t) = 2\pi f t + \phi_{i,0}$ |
| 稀疏编码 | 少量活跃神经元 | $\\|\vec{r}\\|_0 \ll N$ |

**LLM的编码方式**：

$$P(w_t | w_{1:t-1}) = \text{softmax}(W_o \cdot \text{Transformer}(w_{1:t-1}))$$

所有信息编码在连续的、高维的浮点向量空间中。

**关键差异**：
- 大脑：多模态、多尺度、事件驱动的稀疏编码
- LLM：单模态（文本）、固定维度、密集编码

### 3.3 注意力机制对比

**生物大脑的"注意力"**：

定义注意力函数 $A_{bio}$：

$$A_{bio}(\vec{s}, \vec{p}) = \sum_i w_i \cdot \phi(s_i, p)$$

其中：
- $\vec{s}$：感觉输入向量
- $\vec{p}$：当前处理目标（"焦点"）
- $w_i$：自上而下的注意力权重（通过强化学习调整）
- $\phi(s_i, p)$：匹配函数

**Transformer的注意力**：

$$A_{LLM}(Q, K, V) = D^{-1} \sum_j \frac{e^{q_j^T k_j / \sqrt{d_k}}}{\sum_l e^{q_j^T k_l / \sqrt{d_k}}} v_j$$

**关键差异**：

1. **维度**：大脑有 ~10,000 种神经递质和 neuromodulator（多巴胺、血清素、乙酰胆碱等）；LLM只有一种注意力机制
2. **层次**：大脑有自上而下、自下而上、侧向三种注意力；LLM只有自注意
3. **动态性**：大脑的注意力可以在毫秒级切换；LLM的注意力是静态的（权重固定后）

---

## 4. 时序动力学

### 4.1 连续时间 vs 离散时间

**大脑**：连续时间动力学

$$\frac{d\vec{x}}{dt} = F(\vec{x}, \vec{u}, t)$$

**LLM**：离散时间步骤

$$\vec{h}_{t+1} = f(\vec{h}_t, \vec{x}_t)$$

**关键影响**：大脑可以在任意精确时刻响应；LLM被限制在固定的时间步。

### 4.2 振荡与同步

大脑活动中的关键振荡：

| 频段 | 频率 | 功能假说 |
|------|------|----------|
| δ | 0.5-4 Hz | 深度睡眠 |
| θ | 4-8 Hz | 记忆巩固，空间导航 |
| α | 8-13 Hz | 感觉抑制，清醒放松 |
| β | 13-30 Hz | 主动思考，运动控制 |
| γ | 30-100 Hz | 感觉整合，意识 |

相位-振幅耦合（PAC）：

$$A_{HF} = f(\phi_{LF}(t))$$

其中 $A_{HF}$ 为高频振荡振幅，$\phi_{LF}$ 为低频相位。

**LLM没有振荡机制**。这可能是多模态整合、工作记忆、意识等能力的缺失根源之一。

### 4.3 预测编码框架

大脑被认为是"预测机器"（Predictive Coding）：

定义预测误差 $\varepsilon_i$：

$$\varepsilon_i = \nu_i - \hat{\nu}_i$$

其中 $\nu_i$ 为实际输入，$\hat{\nu}_i$ 为高层预测。

自由能原理目标：

$$\mathcal{F} = \sum_i \varepsilon_i^2 + \mathcal{H}(\hat{\nu})$$

**类比LLM的自回归预测**：

$$P(w_t | w_{1:t-1}) = \text{softmax}(W \cdot h_{t-1})$$

$$L = -\log P(w_t | w_{1:t-1})$$

**关键差异**：
- 大脑：层次化预测，每个层级同时编码精确和粗糙的预测
- LLM：单点预测，没有中间表征的预测验证

---

## 5. 可塑性与学习机制

### 5.1 可塑性类型对比

| 类型 | 大脑 | LLM |
|------|------|-----|
| 突触可塑性 | STDP、LTP、LTD | 反向传播 |
| 神经调质 | 多巴胺（奖励）、血清素（价值） | Reward Model |
| 短期可塑性 | 突触易化、抑制 | 无 |
| 神经发生 | 海马体新神经元 | 无 |
| 树突可塑性 | 局部学习 | 全局反向传播 |
| 元可塑性 | 学习如何学习 | MAML、Meta-Learning |

### 5.2 STDP 数学模型

脉冲时序依赖可塑性（STDP）：

$$\Delta w_{ij} = 
\begin{cases}
A_+ e^{-\Delta t / \tau_+}, & \Delta t > 0 \\
-A_- e^{\Delta t / \tau_-}, & \Delta t < 0
\end{cases}$$

其中 $\Delta t = t_{post} - t_{pre}$。

**这是局部学习规则**：每个突触只需要知道前后两个脉冲的时间，不需要全局梯度。

### 5.3 反向传播数学

LLM的梯度计算：

$$\frac{\partial \mathcal{L}}{\partial w_{ij}^{(l)}} = \frac{\partial \mathcal{L}}{\partial h_i^{(l+1)}} \cdot \frac{\partial h_i^{(l+1)}}{\partial w_{ij}^{(l)}}$$

链式法则：

$$\frac{\partial \mathcal{L}}{\partial h^{(l)}} = \left(\frac{\partial h^{(l+1)}}{\partial h^{(l)}}\right)^T \frac{\partial \mathcal{L}}{\partial h^{(l+1)}}$$

**这是全局学习规则**：每个参数都需要全局损失函数的梯度。

### 5.4 能耗对比

**大脑学习（STDP）**：

每次突触更新能量：
$$E_{STDP} \approx E_{spike} \approx 10^{-12} \text{ J}$$

每样本学习（100万亿突触，假设1%活跃）：
$$E_{bio} \approx 10^{14} \times 0.01 \times 10^{-12} = 10^0 = 1 \text{ J}$$

**LLM反向传播**：

每参数每次前向+反向：
$$E_{param} \approx 10^{-9} \text{ J (GPU)}$$

1750亿参数（GPT-3）：
$$E_{LLM} \approx 1.75 \times 10^{11} \times 10^{-9} = 175 \text{ J}$$

**LLM单次更新的能量是大脑的175倍**，且学习需要多次迭代（数千到数百万样本）。

### 5.5 莫拉维克悖论

**莫拉维克悖论**：人类觉得困难的事（逻辑、数学）对AI很简单；人类觉得简单的事（感知、运动）AI很难。

形式化定义"任务难度"：

$$D_{human}(task) = \text{进化年数} \times \text{练习年数}$$
$$D_{AI}(task) = \text{参数数量} \times \text{计算量}$$

| 任务 | $D_{human}$ | $D_{AI}$ | 难度比 |
|------|------------|----------|--------|
| 图像识别 | 5亿年（进化） | 10^9 参数 | AI简单 |
| 抓握物体 | 5亿年+10年 | 未解决 | 人类简单 |
| 数学证明 | 10万年+12年 | 10^11 参数 | AI简单 |
| 社交理解 | 5亿年+20年 | 未解决 | 人类简单 |

**解释**：进化优化的是感知-运动技能，这些技能被编码在基因组中。大脑的神经网络是这些技能的硬件实现。LLM从数据中学习，文本数据丰富所以语言任务简单；物理交互数据稀缺所以物理理解困难。

---

## 6. 记忆系统

### 6.1 记忆类型对比

| 类型 | 大脑 | LLM |
|------|------|-----|
| 感觉记忆 | 0.5-2s (图标、声音) | 输入token窗口 |
| 短期记忆 | ~7项 (Miller's law) | KV Cache |
| 工作记忆 | 主动维持+操作 | 注意力窗口 |
| 长期记忆 | 蛋白合成稳固化 | 模型权重 |
| 程序记忆 | 运动技能 | 权重中的技能 |
| 情景记忆 | 事件+上下文 | 无 |
| 语义记忆 | 概念+关系 | 权重中的知识 |

### 6.2 记忆巩固动力学

**海马体-皮层双向记忆系统**：

海马体快速编码：
$$\frac{d\vec{h}}{dt} = -\frac{1}{\tau_h}\vec{h} + W_{xh}\vec{x} + W_{hh}\vec{h}_{prev}$$

皮层缓慢整合：
$$\frac{d\vec{c}}{dt} = -\frac{1}{\tau_c}\vec{c} + \alpha \vec{h}, \quad \tau_c \gg \tau_h$$

**LLM没有这种分层记忆系统**。所有"记忆"都编码在固定的权重中，无法动态更新。

### 6.3 上下文长度对比

**大脑工作记忆**：

$$\text{Capacity}_{brain} \approx 4 \pm 1 \text{ chunks}$$

但每个chunk可以极其复杂（整个场景、情绪、关系网络）。

**LLM上下文**：

GPT-4上下文窗口：~128,000 tokens（~96,000汉字）

**表面看LLM远超大脑**。但这是误导性的：
- LLM的上下文是"短期记忆"，需要在注意力窗口内完成所有推理
- 大脑可以同时维护多个工作记忆，每个连接长期记忆

### 6.4 遗忘机制

**大脑的遗忘**：

主动遗忘（由记忆抑制基因调控）：
$$\frac{dM}{dt} = -\gamma M + \text{active forgetting}(t)$$

海马体重放（用于记忆巩固）：
$$P_{replay} \propto \frac{e^{\beta \cdot reward}}{\sum_j e^{\beta \cdot reward_j}}$$

**LLM的遗忘**：
- 推理时：Softmax的稀疏性导致低概率token被忽略
- 训练时：灾难性遗忘（所有权重同时更新）

---

## 7. 容错与鲁棒性

### 7.1 噪声容忍

**大脑对噪声的容忍**：

信噪比（SNR）定义：
$$\text{SNR}_{bio} = \frac{P_{signal}}{P_{noise}} \approx \frac{100 \text{ mV}}{10 \text{ mV}} \approx 10$$

但大脑通过**冗余编码**和**纠错机制**（神经调质的精确时序）维持信息。

**LLM对噪声的容忍**：

输入扰动 $\epsilon$ 对输出的影响：
$$\|\vec{y}(x + \epsilon) - \vec{y}(x)\| \leq L \cdot \|\epsilon\|$$

其中 $L$ 为Lipschitz常数。Transformer的Lipschitz常数：

$$L \leq \prod_{l=1}^{L} \|W^{(l)}\| \cdot \|W^{att}\|^2$$

深度网络可能对输入扰动极其敏感（对抗样本）。

### 7.2 神经元死亡容忍

**大脑**：

- 每天丢失 ~85,000 个神经元
- 终生保留 ~85 billion 神经元
- 阿尔茨海默症晚期丢失 ~35%
- 仍可维持功能（重组补偿）

容忍度：~35% 神经元丢失

**LLM**：

- 10% 参数损坏 → 基本失效
- Bit翻转 → 灾难性失败

容错性：~0% 参数损坏

### 7.3 鲁棒性指标

| 指标 | 大脑 | LLM |
|------|------|-----|
| 对抗扰动 | 强（经过进化优化） | 弱（易受攻击） |
| 分布偏移 | 强（open-world） | 弱（out-of-distribution） |
| 噪声注入 | 强 | 中等 |
| 部分损坏 | 强 | 弱 |
| 概念漂移 | 强（持续学习） | 弱（需重新训练） |

---

## 8. 数学框架对比

### 8.1 表征空间

**大脑的表征**：

- 维度：~86 billion（神经元数）
- 稀疏度：~1-10%（活跃神经元）
- 度分布：符合幂律（无标度网络）
- 聚类系数：高（模块化）

**LLM的表征**：

- 维度：~128,000（上下文token）× ~128（隐藏维度）× ~96（层数）
- 稀疏度：~100%（密集计算）
- 度分布：N/A（没有显式图结构）
- 聚类系数：N/A

### 8.2 信息论分析

**大脑的信息容量**：

$$C_{brain} = N_{synapses} \times \log_2(\text{状态数 per synapse})$$

每个突触可表示 ~4-6 bits（短期可塑性状态）：
$$C_{brain} \approx 10^{14} \times 5 \approx 5 \times 10^{14} \text{ bits}$$

**LLM的信息容量**：

$$C_{LLM} = N_{params} \times \log_2(\text{精度})$$

FP16精度：
$$C_{LLM} \approx 1.8 \times 10^{12} \times 16 \approx 2.9 \times 10^{13} \text{ bits}$$

**大脑的信息容量比LLM高约17倍**。

但这只是数量对比。大脑的信息是**结构化的、上下文依赖的、可塑的**；LLM的信息是**静态的、独立的**。

### 8.3 优化Landscape

**大脑的"优化"**：

不是全局最小化，而是寻找**吸引域**（attractor basins）：

$$E(\vec{x}) = -\sum_{i,j} w_{ij} x_i x_j + \sum_i b_i x_i$$

局部最小对应记忆模式。

**LLM的优化**：

全局损失最小化：
$$\min_\theta \mathbb{E}_{(x,y)\sim\mathcal{D}}[\mathcal{L}(f_\theta(x), y)]$$

全局最小化可能导致：
- 过拟合
- 灾难性遗忘
- 尖锐极小值（sharp minima）

### 8.4 计算复杂度

**大脑的计算复杂度**：

$$C_{bio} = O(N_{neurons} \times f_{avg}) = O(10^{11} \times 10^2) = O(10^{13}) \text{ ops/s}$$

**LLM推理的计算复杂度**：

Transformer复杂度：
$$C_{LLM} = O(L \cdot N^2 \cdot d) + O(L \cdot M \cdot N \cdot d)$$

其中：
- $L$：层数（~96）
- $N$：序列长度（~2048）
- $M$：vocab大小（~100,000）
- $d$：模型维度（~12,288）

$$C_{LLM} \approx 96 \times (2048^2 + 100000 \times 2048) \times 12288 \approx 10^{14} \text{ ops/token}$$

**与大脑同量级，但计算类型完全不同。**

---

## 9. "不可能"的边界

### 9.1 意识：LLM能否有意识？

**意识的物理定义**（IIT - 整合信息理论）：

$$\Phi = \min_{\text{partition}} \frac{I(\text{whole} : \text{parts})}{D(\text{parts})}$$

其中 $I$ 为互信息，$D$ 为距离。

**计算 $\Phi$ 对于任何非平凡系统的计算都是NP难的**。

**反问**：
- 大脑有意识吗？（其他心灵问题）
- 如果LLM通过了图灵测试，它有意识吗？（中文房间论证）

### 9.2 持续学习：LLM的根本限制

**大脑的持续学习**：

学习新技能不遗忘旧技能：
$$\forall t: \|\theta_{new} - \theta_{old,i}\| < \epsilon_i$$

通过：
- 神经发生（新神经元）
- 突触复用（多功能神经元）
- 记忆巩固（海马体-皮层）

**LLM的灾难性遗忘**：

$$\lim_{\text{new task}} P_{old\_task} \rightarrow 0$$

只有通过：
- 正则化（Elastic Weight Consolidation）
- 经验回放（Replay）
- 模块化（Mixture of Experts）

这些是**工程补丁**，不是根本解决方案。

### 9.3 具身性：文本之外的现实

**大脑是为身体而生的**：

$$\text{Cognition} = f(\text{Brain}, \text{Body}, \text{Environment})$$

身体提供：
- 空间参照系
- 因果直觉
- 物理直觉
- 社会交互锚点

**LLM被困在文本中**：

$$P(\text{physics} | \text{text}) \neq P(\text{physics} | \text{experience})$$

LLM可以学习物理方程，但不理解"触摸火焰会受伤"的因果关系。

### 9.4 创造力：超越插值的智能

**定义的挑战**：

创造力 = 在约束下的新颖性 + 价值

$$C = \text{Novelty}(x) \times \text{Value}(x) \times \text{Constraint}(x)$$

**LLM的"创造力"**：

$$x_{new} \approx \text{interpolation}(\text{training data})$$

是**高维插值**，不是真正的创造。

**人类创造力**：

$$x_{new} = f(\text{experience}, \text{analogy}, \text{abduction}, \text{serendipity})$$

涉及：
- 跨领域类比
- 反事实推理
- 意外发现

LLM缺少这些机制。

---

## 10. 综合对比与结论

### 10.1 能力矩阵

| 能力 | 大脑 | LLM | 差距原因 |
|------|------|-----|----------|
| 感知-运动 | ⭐⭐⭐⭐⭐ | ⭐ | 具身性 |
| 语言理解 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 符号处理 |
| 逻辑推理 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 结构化推理 |
| 常识推理 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 物理直觉 |
| 情感理解 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 主观体验 |
| 创造性 | ⭐⭐⭐⭐ | ⭐ | 超越插值 |
| 持续学习 | ⭐⭐⭐⭐⭐ | ⭐ | 可塑性 |
| 鲁棒性 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 进化优化 |
| 效率 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 存算一体 |
| 规模 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 计算资源 |

### 10.2 关键洞察总结

**1. 架构的根本差异**

$$Architecture_{brain} = \text{存算一体} + \text{事件驱动} + \text{多尺度}$$
$$Architecture_{LLM} = \text{存算分离} + \text{批量密集} + \text{单尺度}$$

这是所有差异的根源。

**2. 学习范式的根本差异**

$$Learning_{brain} = \text{局部} + \text{在线} + \text{目标驱动}$$
$$Learning_{LLM} = \text{全局} + \text{离线} + \text{数据驱动}$$

**3. 表征的根本差异**

$$Representation_{brain} = \text{多模态} + \text{情境化} + \text{动态}$$
$$Representation_{LLM} = \text{单模态(文本)} + \text{分布式} + \text{静态}$$

### 10.3 可能的融合方向

**1. 存算一体芯片**（如Intel Loihi, IBM TrueNorth）
- 将存储和计算融合
- 目标：10-100x 能效提升

**2. 事件驱动神经网络**（SNN）
- 模仿大脑的稀疏、事件驱动计算
- 目标：低功耗实时处理

**3. 持续学习框架**
- 借鉴神经可塑性机制
- 目标：消除灾难性遗忘

**4. 多模态具身AI**
- 将LLM与感知-运动系统结合
- 目标：物理世界理解

**5. 层次化记忆系统**
- 借鉴海马体-皮层架构
- 目标：动态知识更新

---

## 附录：公式速查

| 概念 | 公式 |
|------|------|
| 兰道尔极限 | $E_{min} = k_B T \ln 2$ |
| Hodgkin-Huxley | $C_m \frac{dV}{dt} = I - g_K n^4(V-V_K) - ...$ |
| STDP | $\Delta w = A_+ e^{-\Delta t/\tau_+}$ |
| 注意力 | $A = \text{softmax}(QK^T/\sqrt{d})V$ |
| 信息容量 | $C = N \log_2(V)$ |
| Transformer复杂度 | $O(L \cdot N^2 \cdot d)$ |
| 大脑OPS | $10^{14} \text{ ops/s}$ |
| 大脑功率 | 20 W |
| LLM能效比 | ~3400万倍兰道尔极限 |
| 大脑能效比 | ~6700万倍兰道尔极限 |
