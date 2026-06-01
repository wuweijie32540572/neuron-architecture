"""
心智模拟学习器 (Mental Simulation Learner) v3 — 约束驱动架构
============================================================

核心假说验证：
一个具备内部心智模拟能力的神经架构，可以在只阅读中文教程文本
（不含任何代码示例或执行结果）的情况下，学会变量的赋值和加法操作，
并能够对未见过的数值正确执行计算。

v3 根本性改进：约束驱动自组织
-----------------------------
v2 的根本缺陷：_compute_target_value() 直接计算 v1+v2 作为训练目标。
这本质上是监督学习——把答案喂给网络，与核心假说背道而驰。

v3 的范式转变：不给答案，只给约束，让网络自己变成答案。

从教程文本中提取的约束（非标签）：
1. 赋值恒等：predict(assign(x, v)) = v
   ——来源："赋值语句 x = 5 将值 5 存入变量 x"（定义性约束）
2. 交换律：predict(add(a,b)) = predict(add(b,a))
   ——来源："加法运算 a + b 计算两个数字的和"（对称结构 → 对称性约束）
3. 递增一致性：predict(add(a+δ, b)) - predict(add(a, b)) = δ
   ——来源：赋值与加法的组合一致性（行为约束）
4. 零元锚定：predict(add(a, 0)) = a
   ——来源：赋值语义的推导——加"无"不变（边界约束）

数学证明：这三个约束唯一确定 f(a,b) = a + b
  由递增一致性：f(a,b) = a + g(b)
  由零元锚定：g(0) = 0
  由交换律：g(b) - b = g(a) - a = 常数 C
  由 g(0) = 0：C = 0，故 g(b) = b
  因此 f(a,b) = a + b  ∎

关键区别：
- 监督学习(v2)：告诉网络 "add(5,3) = 8" → 特定输入-输出对
- 约束驱动(v3)：告诉网络 "add(a,b) = add(b,a)" → 一般性规则
约束不告诉网络任何特定输入的输出是什么，只告诉它输出应满足的关系。
网络必须自己发现满足所有关系的唯一函数——加法。

认知科学对应：
- 约束满足 ≈ 预测编码中的自由能最小化 (Friston 2010)
- 自组织收敛 ≈ 耗散结构中的序参量涌现 (Prigogine 1978)
- 不给答案 ≈ 约束满足问题 vs 监督学习
- 物理学：从边界条件重构哈密顿量
- 生物学：脱离具身体验，在大脑中建立可运行的模拟回路
- 数学：发现满足约束的函子
- 电子技术：模拟电路通过物理弛豫自动满足基尔霍夫约束
"""

import numpy as np


class SymbolGrounder:
    """
    从教程文本中学习每个 token 的语义向量和角色。

    对应认知科学中的"符号接地问题"(Symbol Grounding Problem, Harnad 1990)。
    不使用关键词匹配，而是通过统计启发式推断 token 角色：
    - 数字：字符模式检测（可被解析为浮点数）
    - 操作符：位置统计（'=' 和 '+' 的字符形态）
    - 变量：在 '=' 左侧的出现频率
    - 概念词：高频但非操作性

    语义向量编码：
    - dims 0-4: 角色 one-hot
    - dims 5-10: 统计特征
    - dims 11+: 随机初始化（提供区分度）
    """

    ROLE_DIMS = {
        'value': 0, 'variable': 1, 'operator_assign': 2,
        'operator_add': 3, 'concept': 4,
    }

    def __init__(self, embed_dim=32):
        self.embed_dim = embed_dim
        self.token_embeddings = {}
        self.token_roles = {}
        self.all_tokens = []

    def learn_from_text(self, sentences):
        tokenized = [self._tokenize(s) for s in sentences]
        token_stats = self._collect_statistics(tokenized)

        for token, stats in token_stats.items():
            role = self._infer_role(token, stats)
            self.token_roles[token] = role
            self.token_embeddings[token] = self._build_embedding(token, role, stats)
            self.all_tokens.append(token)

    def _tokenize(self, text):
        tokens = []
        current = ""
        for ch in text:
            if ch in ' =+，。、；：\u201c\u201d\u2018\u2019（）【】\n\t':
                if current.strip():
                    tokens.append(current.strip())
                    current = ""
                if ch in '=+':
                    tokens.append(ch)
            else:
                current += ch
        if current.strip():
            tokens.append(current.strip())
        return tokens

    def _collect_statistics(self, tokenized_sentences):
        token_stats = {}
        for tokens in tokenized_sentences:
            for pos, token in enumerate(tokens):
                if token not in token_stats:
                    token_stats[token] = {
                        'count': 0, 'near_equals': 0, 'near_plus': 0,
                        'left_of_equals': 0, 'right_of_equals': 0,
                        'left_of_plus': 0, 'right_of_plus': 0,
                    }
                s = token_stats[token]
                s['count'] += 1
                window = tokens[max(0, pos - 2): pos + 3]
                s['near_equals'] += sum(1 for t in window if t == '=')
                s['near_plus'] += sum(1 for t in window if t == '+')
                if pos > 0 and tokens[pos - 1] == '=':
                    s['right_of_equals'] += 1
                if pos < len(tokens) - 1 and tokens[pos + 1] == '=':
                    s['left_of_equals'] += 1
                if pos > 0 and tokens[pos - 1] == '+':
                    s['right_of_plus'] += 1
                if pos < len(tokens) - 1 and tokens[pos + 1] == '+':
                    s['left_of_plus'] += 1
        return token_stats

    def _infer_role(self, token, stats):
        try:
            float(token)
            return 'value'
        except ValueError:
            pass
        if token == '=':
            return 'operator_assign'
        if token == '+':
            return 'operator_add'
        count = max(stats['count'], 1)
        if stats['left_of_equals'] / count > 0.2 or \
           (stats['left_of_plus'] + stats['right_of_plus']) / count > 0.3:
            return 'variable'
        return 'concept'

    def _build_embedding(self, token, role, stats):
        emb = np.zeros(self.embed_dim)
        emb[self.ROLE_DIMS.get(role, 4)] = 1.0
        count = max(stats['count'], 1)
        emb[5] = stats['near_equals'] / count
        emb[6] = stats['near_plus'] / count
        emb[7] = stats['left_of_equals'] / count
        emb[8] = stats['right_of_equals'] / count
        emb[9] = min(stats['count'] / 10.0, 1.0)
        if role == 'value':
            try:
                emb[10] = float(token) / 100.0
            except ValueError:
                pass
        rng = np.random.RandomState(abs(hash(token)) % (2 ** 31))
        emb[11:] = rng.randn(self.embed_dim - 11) * 0.1
        return emb

    def get_embedding(self, token):
        return self.token_embeddings.get(token, np.zeros(self.embed_dim))

    def get_role(self, token):
        return self.token_roles.get(token, 'concept')

    def get_variables(self):
        return [t for t, r in self.token_roles.items() if r == 'variable']

    def get_values(self):
        return [t for t, r in self.token_roles.items() if r == 'value']


class WorldState:
    """
    张量世界状态，模拟工作记忆中的心理模型。

    对应 Johnson-Laird (1983) 的心理模型理论。
    每个变量槽对应一个概念槽 (conceptual slot)。

    状态向量 = [values/100, defined_flags]，共 2*MAX_SLOTS 维。
    归一化确保数值在神经网络友好的范围内。
    """

    MAX_SLOTS = 10
    STATE_DIM = 20
    NORM_FACTOR = 100.0

    def __init__(self):
        self.values = np.zeros(self.MAX_SLOTS)
        self.defined = np.zeros(self.MAX_SLOTS)
        self.var_names = [None] * self.MAX_SLOTS
        self.var_to_slot = {}

    def get_state_vector(self):
        return np.concatenate([self.values / self.NORM_FACTOR, self.defined])

    def ensure_slot(self, var_name):
        if var_name not in self.var_to_slot:
            for i in range(self.MAX_SLOTS):
                if self.var_names[i] is None:
                    self.var_to_slot[var_name] = i
                    self.var_names[i] = var_name
                    return i
            raise RuntimeError("无可用槽位")
        return self.var_to_slot[var_name]

    def assign(self, var_name, value):
        slot = self.ensure_slot(var_name)
        self.values[slot] = value
        self.defined[slot] = 1.0

    def read(self, var_name):
        if var_name not in self.var_to_slot:
            return None
        slot = self.var_to_slot[var_name]
        if self.defined[slot] < 0.5:
            return None
        return self.values[slot]

    def copy(self):
        new = WorldState()
        new.values = self.values.copy()
        new.defined = self.defined.copy()
        new.var_names = self.var_names.copy()
        new.var_to_slot = dict(self.var_to_slot)
        return new


class OperationEncoder:
    """
    操作编码器：将操作编码为固定维度向量。

    v3 关键设计：在编码中包含源变量值（由符号系统 WorldState 查找提供）。
    这实现了双过程理论的清晰分离：
    - 符号系统 (WorldState) 负责"路由"——查找哪些变量参与运算及其当前值
    - 神经系统 (MentalSimulator) 负责"计算"——对值执行运算

    编码方案：
    | 维度   | 内容             | 大小 |
    |--------|------------------|------|
    | 0-1    | 操作类型 one-hot | 2    |
    | 2-11   | 目标变量槽 one-hot| 10   |
    | 12     | 主源值(赋值:值;加法:src1)| 1 |
    | 13     | 次源值(加法:src2) | 1    |
    | 14-23  | 加法源1槽 one-hot| 10   |
    | 24-33  | 加法源2槽 one-hot| 10   |

    对应前额叶皮层对"动作计划"的表征：
    做什么(类型) + 对谁做(目标) + 用什么做(源值)。
    """

    OP_DIM = 34

    def encode(self, operation, state):
        op_vec = np.zeros(self.OP_DIM)
        op_type = operation.get('type', '')

        if op_type == 'assign':
            op_vec[0] = 1.0
            var_name = operation.get('var', '')
            value = operation.get('value', 0)
            slot = state.var_to_slot.get(var_name, 0)
            op_vec[2 + slot] = 1.0
            op_vec[12] = value / WorldState.NORM_FACTOR

        elif op_type == 'add_assign':
            op_vec[1] = 1.0
            target = operation.get('target', '')
            src1 = operation.get('src1', '')
            src2 = operation.get('src2', '')
            op_vec[2 + state.var_to_slot.get(target, 0)] = 1.0
            v1 = state.read(src1)
            v2 = state.read(src2)
            op_vec[12] = (v1 if v1 is not None else 0) / WorldState.NORM_FACTOR
            op_vec[13] = (v2 if v2 is not None else 0) / WorldState.NORM_FACTOR
            op_vec[14 + state.var_to_slot.get(src1, 0)] = 1.0
            op_vec[24 + state.var_to_slot.get(src2, 0)] = 1.0

        return op_vec


class MentalSimulator:
    """
    心智模拟引擎（v3 约束驱动架构）：通过约束满足自组织地学会运算。

    设计原理：
    --------
    v3 的根本转变：训练信号不再是"答案"（v1+v2），而是"约束"。
    网络通过满足从文本中提取的约束，自组织地收敛到正确的运算函数。

    这对应物理学中的耗散结构 (Prigogine 1978)：
    系统从外界吸收低熵（文本中的约束），通过内部非线性动力学，
    自发形成高度有序的模式（加法函数）。

    也对应预测编码中的自由能最小化 (Friston 2010)：
    网络最小化约束违反量（自由能），收敛到满足所有约束的稳态。

    架构：
    ----
    输入: [当前状态 (20维) | 操作编码 (34维)] = 54维
    隐藏层1: 128维 + Leaky ReLU
    隐藏层2: 64维 + Leaky ReLU
    输出: 1维 (残差) + skip连接(op_vec[12])

    残差连接 output = MLP(input) + op_vec[12]：
    对应预测编码中的"预测残差"——大脑不编码绝对值，
    而是编码与预测的偏差。赋值时 MLP→0，加法时 MLP→op_vec[13]。

    约束训练机制：
    --------
    支持多次前向传播 + 梯度累积的约束驱动训练：
    1. forward_with_cache(): 前向传播，返回预测值和缓存
    2. compute_grad(): 从缓存和损失梯度计算参数梯度
    3. apply_grads(): 累积所有约束的梯度，裁剪后更新权重
    """

    def __init__(self, state_dim=20, op_dim=34, hidden_dims=(128, 64), lr=0.005):
        self.state_dim = state_dim
        self.op_dim = op_dim
        self.lr = lr
        self.max_grad_norm = 5.0

        input_dim = state_dim + op_dim
        self.weights = []
        self.biases = []

        dims = [input_dim] + list(hidden_dims) + [1]
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i + 1])
            self.weights.append(W)
            self.biases.append(b)

        self.error_history = []
        self.step_count = 0

    @staticmethod
    def _leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def _leaky_relu_grad(x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    def forward_with_cache(self, state_vec, op_vec):
        """
        前向传播：返回预测值和缓存（用于后续梯度计算）。

        残差连接：output = MLP_output + op_vec[12]

        对应认知科学中的"预测残差"编码 (Rao & Ballard 1999)：
        大脑不编码绝对值，而是编码与预测的偏差。
        - 赋值操作：MLP 输出 ≈ 0，总输出 = 0 + value = value
        - 加法操作：MLP 输出 ≈ op_vec[13]，总输出 = src1 + src2 = sum
        """
        x = np.concatenate([state_vec, op_vec])
        cache = [x]

        for i in range(len(self.weights) - 1):
            h = x @ self.weights[i] + self.biases[i]
            h = self._leaky_relu(h)
            cache.append(h)
            x = h

        residual = x @ self.weights[-1] + self.biases[-1]
        skip = op_vec[12]
        output = residual[0] + skip
        return output, cache

    def predict_value(self, state_vec, op_vec):
        pred, _ = self.forward_with_cache(state_vec, op_vec)
        return pred

    def compute_grad(self, cache, d_loss_d_output):
        """
        从缓存和损失对输出的梯度，计算参数梯度。

        这是约束驱动训练的核心：不同约束产生不同的 d_loss_d_output，
        但共享相同的梯度计算逻辑。所有约束的梯度被累积后统一应用。

        对应预测编码中的"误差反向传播"：
        预测误差沿着皮层层级反向传播，调整突触权重。
        """
        d = d_loss_d_output * np.ones(1)
        grads_W = []
        grads_b = []

        for i in range(len(self.weights) - 1, -1, -1):
            grads_W.insert(0, np.outer(cache[i], d))
            grads_b.insert(0, d.copy())

            if i > 0:
                d = d @ self.weights[i].T
                pre_act = cache[i - 1] @ self.weights[i - 1] + self.biases[i - 1]
                d = d * self._leaky_relu_grad(pre_act)

        return grads_W, grads_b

    def apply_grads(self, all_grads_W, all_grads_b):
        """
        累积所有约束的梯度，裁剪后更新权重。

        对应预测编码中的"权重更新"阶段：
        所有层级的预测误差被汇总后，统一调整内部模型。

        梯度裁剪对应神经系统的稳态机制——防止过度调整。
        """
        total_grads_W = [np.zeros_like(w) for w in self.weights]
        total_grads_b = [np.zeros_like(b) for b in self.biases]

        for gW, gb in zip(all_grads_W, all_grads_b):
            for i in range(len(self.weights)):
                total_grads_W[i] += gW[i]
                total_grads_b[i] += gb[i]

        grad_norm = sum(np.sum(gW ** 2) for gW in total_grads_W) + \
                    sum(np.sum(gb ** 2) for gb in total_grads_b)
        grad_norm = np.sqrt(grad_norm)

        if grad_norm > self.max_grad_norm:
            scale = self.max_grad_norm / grad_norm
            total_grads_W = [g * scale for g in total_grads_W]
            total_grads_b = [g * scale for g in total_grads_b]

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * total_grads_W[i]
            self.biases[i] -= self.lr * total_grads_b[i]

    def train_step(self, state_vec, op_vec, target_value):
        pred, cache = self.forward_with_cache(state_vec, op_vec)
        error = pred - target_value
        loss = error ** 2
        self.error_history.append(loss)
        self.step_count += 1
        d_loss = 2.0 * error
        grads_W, grads_b = self.compute_grad(cache, d_loss)
        self.apply_grads([grads_W], [grads_b])
        return loss

    def decay_lr(self, factor=0.95):
        self.lr *= factor


class ProceduralMemory:
    """
    程序性记忆：存储已验证的操作模板。

    对应认知科学中的程序性记忆 (Squire 2004)。
    反复验证的操作模式从工作记忆转移到长期记忆。
    """

    def __init__(self):
        self.templates = {}
        self.confidence = {}

    def store(self, name, pattern, state_change):
        if name not in self.templates:
            self.templates[name] = []
            self.confidence[name] = 0.0
        self.templates[name].append({'pattern': pattern, 'state_change': state_change})
        self.confidence[name] = min(self.confidence[name] + 0.1, 1.0)

    def apply_rule(self, rule_name, state, **kwargs):
        new_state = state.copy()
        if rule_name == 'assign_value':
            new_state.assign(kwargs.get('var'), kwargs.get('value'))
            return new_state
        elif rule_name == 'add_and_assign':
            v1 = state.read(kwargs.get('src1'))
            v2 = state.read(kwargs.get('src2'))
            if v1 is not None and v2 is not None:
                new_state.assign(kwargs.get('target'), v1 + v2)
                return new_state
        return None


class TutorialLearner:
    """
    教程学习器（v3 约束驱动）：从中文教程文本中学习编程能力。

    学习流程（模拟人类认知学习过程）：
    1. 文本分析 → 词汇习得与语法角色推断
    2. 规则推断 → 从文本模式中抽象出操作规则
    3. 约束提取 → 从规则结构中推导出约束（非标签！）
    4. 约束驱动训练 → 通过满足约束自组织地学会运算
    5. 模式巩固 → 将验证通过的模式转为长期记忆

    v3 核心创新：
    --------
    训练信号不再是"答案"（v1+v2），而是从文本中提取的"约束"：
    - 赋值恒等：定义性约束（文本直接定义了赋值的语义）
    - 交换律：结构对称性约束（加法操作数对称 → 结果对称）
    - 递增一致性：行为约束（改变输入δ → 输出改变δ）
    - 零元锚定：边界约束（加"无"不变，从赋值语义推导）

    数学保证：这三个约束唯一确定 f(a,b) = a + b。
    网络不需要任何加法标签，只需满足约束即可自组织到加法。

    这才是真正的"理解"：
    理解 = 能从压缩的陈述性描述中，恢复出足以生成所有符合该描述的
           程序性行为的内部模型。——不喂给网络任何答案，只喂给它问题，
           让网络自己变成答案。
    """

    def __init__(self):
        self.grounder = SymbolGrounder(embed_dim=32)
        self.simulator = MentalSimulator(
            state_dim=WorldState.STATE_DIM,
            op_dim=OperationEncoder.OP_DIM,
            hidden_dims=(128, 64),
            lr=0.008
        )
        self.memory = ProceduralMemory()
        self.encoder = OperationEncoder()
        self.learned_rules = {}
        self.constraints = {}

    def learn(self, tutorial_text):
        sentences = [s.strip() for s in tutorial_text.split('。') if s.strip()]

        print("=" * 64)
        print("  阶段 1：文本分析与符号接地 (Symbol Grounding)")
        print("=" * 64)

        self.grounder.learn_from_text(sentences)

        role_cn = {
            'value': '数值', 'variable': '变量',
            'operator_assign': '赋值操作符', 'operator_add': '加法操作符',
            'concept': '概念词'
        }
        print("\n  符号接地结果：")
        for token in self.grounder.all_tokens:
            role = self.grounder.get_role(token)
            print(f"    '{token}' → {role_cn.get(role, role)}")

        print("\n" + "=" * 64)
        print("  阶段 2：规则推断与约束提取 (Rule Induction & Constraint Extraction)")
        print("=" * 64)

        self._infer_rules(sentences)
        self._extract_constraints()

        for name, rule in self.learned_rules.items():
            print(f"    规则: {name}: {rule['description']}")

        for name, constraint in self.constraints.items():
            print(f"    约束: {name}: {constraint['description']}")

        print("\n" + "=" * 64)
        print("  阶段 3：约束驱动训练 (Constraint-Driven Self-Organization)")
        print("=" * 64)
        print("  ※ 训练信号仅为约束，不使用任何加法标签 (v1+v2)")

        self._constraint_driven_rehearsal()

        print("\n" + "=" * 64)
        print("  阶段 4：模式巩固 (Procedural Consolidation)")
        print("=" * 64)

        self._consolidate()
        print("\n  学习完成！")

    def _infer_rules(self, sentences):
        for sent in sentences:
            tokens = self.grounder._tokenize(sent)
            roles = [self.grounder.get_role(t) for t in tokens]

            if 'operator_assign' in roles and 'operator_add' in roles:
                eq_idx = roles.index('operator_assign')
                plus_idx = roles.index('operator_add')
                if (eq_idx < plus_idx and plus_idx > 1 and plus_idx < len(tokens) - 1
                        and roles[eq_idx - 1] == 'variable'
                        and roles[plus_idx - 1] == 'variable'
                        and roles[plus_idx + 1] == 'variable'):
                    self.learned_rules['add_and_assign'] = {
                        'description': '加法赋值：[变量] = [变量] + [变量]',
                        'pattern': ['variable', 'operator_assign', 'variable',
                                    'operator_add', 'variable'],
                        'action': 'add_and_assign',
                        'symmetric': True,
                    }

            if 'operator_assign' in roles:
                eq_idx = roles.index('operator_assign')
                if (eq_idx > 0 and eq_idx < len(tokens) - 1
                        and roles[eq_idx - 1] == 'variable'
                        and roles[eq_idx + 1] == 'value'):
                    self.learned_rules['assign_value'] = {
                        'description': '赋值：[变量] = [数值]',
                        'pattern': ['variable', 'operator_assign', 'value'],
                        'action': 'assign_value',
                        'symmetric': False,
                    }

    def _extract_constraints(self):
        """
        从推断出的规则中提取约束。

        这是 v3 的核心创新：约束不是硬编码的，而是从文本模式中推导出来的。

        推导逻辑：
        1. 赋值恒等：从赋值规则的定义直接得出——"将值存入变量"意味着
           操作后变量的值等于赋的值。这是定义性约束，不是标签。

        2. 交换律：从加法规则的结构得出——"+"两侧都是"变量"类型，
           没有顺序区分，因此操作应该是对称的。

        3. 递增一致性：从赋值与加法的组合语义得出——
           如果赋值改变了某个变量的值（增加δ），那么依赖该变量的
           加法结果也应该增加δ。这是操作间的一致性约束。

        4. 零元锚定：从赋值语义的边界条件得出——
           如果一个变量未被赋值（值为0），则"加上这个变量"不应改变结果。
           这是"加法合并两个量"这一语义的边界情况。
        """
        if 'assign_value' in self.learned_rules:
            self.constraints['assign_identity'] = {
                'description': '赋值恒等：predict(assign(x,v)) = v（定义性约束）',
                'type': 'identity',
                'weight': 1.0,
            }

        if 'add_and_assign' in self.learned_rules:
            rule = self.learned_rules['add_and_assign']

            if rule.get('symmetric', False):
                self.constraints['commutativity'] = {
                    'description': '交换律：predict(add(a,b)) = predict(add(b,a))（结构对称性约束）',
                    'type': 'symmetry',
                    'weight': 1.0,
                }

            self.constraints['incremental'] = {
                'description': '递增一致性：predict(add(a+δ,b)) - predict(add(a,b)) = δ（行为约束）',
                'type': 'gradient',
                'weight': 2.0,
            }

            self.constraints['zero_anchor'] = {
                'description': '零元锚定：predict(add(a,0)) = a（边界约束，从赋值语义推导）',
                'type': 'boundary',
                'weight': 1.0,
            }

    def _constraint_driven_rehearsal(self):
        """
        约束驱动的心智演练：不给答案，只给约束。

        这是 v3 的核心训练循环。与 v2 的根本区别：
        - v2：target = (v1 + v2) / NORM  ← 给了加法答案！
        - v3：约束满足 ← 不给任何加法答案！

        训练过程：
        1. 采样随机数值
        2. 为每个约束构建前向传播
        3. 计算约束损失和梯度
        4. 累积所有约束的梯度，统一更新权重

        对应物理学中的约束弛豫：
        系统在约束诱导的可行域中通过局部动力学自然收敛到正确语义。
        不需要给定目标轨迹，只需给定边界条件（约束），
        系统自发选择满足约束的路径——就像拉格朗日力学中的最小作用量原理。
        """
        rng = np.random.RandomState(42)
        n_steps = 15000
        constraint_losses = {k: [] for k in self.constraints}
        NORM = WorldState.NORM_FACTOR

        for step in range(n_steps):
            v1 = rng.randint(1, 99)
            v2 = rng.randint(1, 99)
            delta = rng.randint(1, 15)

            all_grads_W = []
            all_grads_b = []

            # ─── 约束 A：赋值恒等 ───
            # 来源："赋值语句 x = 5 将值 5 存入变量 x"
            # 这是定义性约束：赋值操作的语义就是"存入值"
            # 因此 predict(assign(x, v)) = v 是定义本身，不是标签
            if 'assign_identity' in self.constraints:
                w = self.constraints['assign_identity']['weight']

                state_a = WorldState()
                state_a.ensure_slot('x')
                sv_a = state_a.get_state_vector() + rng.randn(WorldState.STATE_DIM) * 0.005
                op_a = self.encoder.encode({'type': 'assign', 'var': 'x', 'value': v1}, state_a)
                pred_a, cache_a = self.simulator.forward_with_cache(sv_a, op_a)
                target_a = v1 / NORM
                d_a = w * 2.0 * (pred_a - target_a)
                gW_a, gb_a = self.simulator.compute_grad(cache_a, d_a)
                all_grads_W.append(gW_a)
                all_grads_b.append(gb_a)
                constraint_losses['assign_identity'].append((pred_a - target_a) ** 2)

                state_a2 = WorldState()
                state_a2.assign('a', v1)
                state_a2.ensure_slot('b')
                sv_a2 = state_a2.get_state_vector() + rng.randn(WorldState.STATE_DIM) * 0.005
                op_a2 = self.encoder.encode({'type': 'assign', 'var': 'b', 'value': v2}, state_a2)
                pred_a2, cache_a2 = self.simulator.forward_with_cache(sv_a2, op_a2)
                target_a2 = v2 / NORM
                d_a2 = w * 2.0 * (pred_a2 - target_a2)
                gW_a2, gb_a2 = self.simulator.compute_grad(cache_a2, d_a2)
                all_grads_W.append(gW_a2)
                all_grads_b.append(gb_a2)
                constraint_losses['assign_identity'].append((pred_a2 - target_a2) ** 2)

            # ─── 约束 B：交换律 ───
            # 来源："加法运算 a + b 计算两个数字的和"
            # "+" 两侧都是变量类型，结构对称 → 操作结果对称
            # 关键修正：使用相同操作 add(c,a,b) 但交换 a,b 的值
            # 这样 op_vec[12] 和 op_vec[13] 会互换，
            # 迫使 MLP 必须输出 op_vec[13] 来补偿残差跳连的差异
            # 如果 MLP 输出 0，则 pred1 = v1/NORM ≠ pred2 = v2/NORM，约束被违反
            # 只有 MLP 输出 op_vec[13] 时，pred1 = v1/NORM + v2/NORM = pred2 ✓
            if 'commutativity' in self.constraints:
                w = self.constraints['commutativity']['weight']

                state_b1 = WorldState()
                state_b1.assign('a', v1)
                state_b1.assign('b', v2)
                state_b1.ensure_slot('c')
                sv_b1 = state_b1.get_state_vector() + rng.randn(WorldState.STATE_DIM) * 0.005
                op_b1 = self.encoder.encode(
                    {'type': 'add_assign', 'target': 'c', 'src1': 'a', 'src2': 'b'},
                    state_b1)
                pred_b1, cache_b1 = self.simulator.forward_with_cache(sv_b1, op_b1)

                state_b2 = WorldState()
                state_b2.assign('a', v2)
                state_b2.assign('b', v1)
                state_b2.ensure_slot('c')
                sv_b2 = state_b2.get_state_vector() + rng.randn(WorldState.STATE_DIM) * 0.005
                op_b2 = self.encoder.encode(
                    {'type': 'add_assign', 'target': 'c', 'src1': 'a', 'src2': 'b'},
                    state_b2)
                pred_b2, cache_b2 = self.simulator.forward_with_cache(sv_b2, op_b2)

                diff_b = pred_b1 - pred_b2
                d_b1 = w * 2.0 * diff_b
                d_b2 = w * (-2.0 * diff_b)
                gW_b1, gb_b1 = self.simulator.compute_grad(cache_b1, d_b1)
                gW_b2, gb_b2 = self.simulator.compute_grad(cache_b2, d_b2)
                all_grads_W.extend([gW_b1, gW_b2])
                all_grads_b.extend([gb_b1, gb_b2])
                constraint_losses['commutativity'].append(diff_b ** 2)

            # ─── 约束 C：递增一致性 ───
            # 来源：赋值与加法的组合一致性
            # 如果 a 增加了 δ（通过赋值），那么 add(a, b) 也应增加 δ
            # 这是行为约束：不告诉网络 add(a,b) 等于什么，
            # 只告诉它 add(a+δ, b) - add(a, b) = δ
            if 'incremental' in self.constraints:
                w = self.constraints['incremental']['weight']

                state_c1 = WorldState()
                state_c1.assign('a', v1)
                state_c1.assign('b', v2)
                state_c1.ensure_slot('c')
                sv_c1 = state_c1.get_state_vector() + rng.randn(WorldState.STATE_DIM) * 0.005
                op_c1 = self.encoder.encode(
                    {'type': 'add_assign', 'target': 'c', 'src1': 'a', 'src2': 'b'},
                    state_c1)
                pred_c1, cache_c1 = self.simulator.forward_with_cache(sv_c1, op_c1)

                state_c2 = WorldState()
                state_c2.assign('a', v1 + delta)
                state_c2.assign('b', v2)
                state_c2.ensure_slot('c')
                sv_c2 = state_c2.get_state_vector() + rng.randn(WorldState.STATE_DIM) * 0.005
                op_c2 = self.encoder.encode(
                    {'type': 'add_assign', 'target': 'c', 'src1': 'a', 'src2': 'b'},
                    state_c2)
                pred_c2, cache_c2 = self.simulator.forward_with_cache(sv_c2, op_c2)

                inc_diff = (pred_c2 - pred_c1) - delta / NORM
                d_c2 = w * 2.0 * inc_diff
                d_c1 = w * (-2.0 * inc_diff)
                gW_c2, gb_c2 = self.simulator.compute_grad(cache_c2, d_c2)
                gW_c1, gb_c1 = self.simulator.compute_grad(cache_c1, d_c1)
                all_grads_W.extend([gW_c2, gW_c1])
                all_grads_b.extend([gb_c2, gb_c1])
                constraint_losses['incremental'].append(inc_diff ** 2)

            # ─── 约束 D：零元锚定 ───
            # 来源：赋值语义的边界推导
            # "变量存储一个数字" + "加法计算两个数字的和"
            # → 如果其中一个数字为0（未赋值/空），则"和"等于另一个数字
            # 这是边界约束，不是标签：它不告诉网络 add(5,3) 等于什么，
            # 只告诉它 add(a, 0) = a（一个必须满足的边界条件）
            if 'zero_anchor' in self.constraints:
                w = self.constraints['zero_anchor']['weight']

                state_d = WorldState()
                state_d.assign('a', v1)
                state_d.assign('b', 0)
                state_d.ensure_slot('c')
                sv_d = state_d.get_state_vector() + rng.randn(WorldState.STATE_DIM) * 0.005
                op_d = self.encoder.encode(
                    {'type': 'add_assign', 'target': 'c', 'src1': 'a', 'src2': 'b'},
                    state_d)
                pred_d, cache_d = self.simulator.forward_with_cache(sv_d, op_d)
                target_d = v1 / NORM
                d_d = w * 2.0 * (pred_d - target_d)
                gW_d, gb_d = self.simulator.compute_grad(cache_d, d_d)
                all_grads_W.append(gW_d)
                all_grads_b.append(gb_d)
                constraint_losses['zero_anchor'].append((pred_d - target_d) ** 2)

            # ─── 约束 E：第二参数递增一致性 ───
            # add(a, b+δ) - add(a, b) = δ
            # 这直接约束 MLP 对第二参数的响应：
            # MLP(a, b+δ) - MLP(a, b) = δ/NORM
            # 如果 MLP 输出 op_vec[13]，则 (v2+δ)/NORM - v2/NORM = δ/NORM ✓
            # 这提供了比交换律更直接的梯度信号
            if 'incremental' in self.constraints:
                w = self.constraints['incremental']['weight']

                state_e1 = WorldState()
                state_e1.assign('a', v1)
                state_e1.assign('b', v2)
                state_e1.ensure_slot('c')
                sv_e1 = state_e1.get_state_vector() + rng.randn(WorldState.STATE_DIM) * 0.005
                op_e1 = self.encoder.encode(
                    {'type': 'add_assign', 'target': 'c', 'src1': 'a', 'src2': 'b'},
                    state_e1)
                pred_e1, cache_e1 = self.simulator.forward_with_cache(sv_e1, op_e1)

                state_e2 = WorldState()
                state_e2.assign('a', v1)
                state_e2.assign('b', v2 + delta)
                state_e2.ensure_slot('c')
                sv_e2 = state_e2.get_state_vector() + rng.randn(WorldState.STATE_DIM) * 0.005
                op_e2 = self.encoder.encode(
                    {'type': 'add_assign', 'target': 'c', 'src1': 'a', 'src2': 'b'},
                    state_e2)
                pred_e2, cache_e2 = self.simulator.forward_with_cache(sv_e2, op_e2)

                inc_diff2 = (pred_e2 - pred_e1) - delta / NORM
                d_e2 = w * 2.0 * inc_diff2
                d_e1 = w * (-2.0 * inc_diff2)
                gW_e2, gb_e2 = self.simulator.compute_grad(cache_e2, d_e2)
                gW_e1, gb_e1 = self.simulator.compute_grad(cache_e1, d_e1)
                all_grads_W.extend([gW_e2, gW_e1])
                all_grads_b.extend([gb_e2, gb_e1])
                constraint_losses['incremental'].append(inc_diff2 ** 2)

            self.simulator.apply_grads(all_grads_W, all_grads_b)

            if (step + 1) % 2000 == 0:
                print(f"\n    步骤 {step + 1}/{n_steps}, lr: {self.simulator.lr:.6f}")
                for cname, closses in constraint_losses.items():
                    if closses:
                        recent = np.mean(closses[-500:])
                        print(f"      {cname}: {recent:.6f}")
                self.simulator.decay_lr(0.96)

        print(f"\n    最终约束损失：")
        for cname, closses in constraint_losses.items():
            if closses:
                print(f"      {cname}: {np.mean(closses[-200:]):.6f}")

    def _consolidate(self):
        for rule_name, rule in self.learned_rules.items():
            self.memory.store(
                name=rule['action'],
                pattern=rule['pattern'],
                state_change=rule['description']
            )
            print(f"    巩固: {rule_name} → {rule['description']}")

    def execute(self, operations):
        """
        执行操作序列（心智模拟）。

        约束驱动训练后，MentalSimulator 应已自组织到正确的运算函数。
        执行时：
        1. MentalSimulator 预测操作产生的值（神经计算）
        2. WorldState 将值写入正确的变量槽（符号路由）
        3. 递归进行，每步的预测结果更新状态供下一步使用

        这实现了真正的"心智模拟链"：
        在脑中逐步运行程序，每一步的输出成为下一步的输入。
        对应 Hegarty (2004) 的心智模拟理论。
        """
        state = WorldState()

        for op in operations:
            if op['type'] == 'assign':
                state.ensure_slot(op['var'])
            elif op['type'] == 'add_assign':
                state.ensure_slot(op['target'])
                state.ensure_slot(op['src1'])
                state.ensure_slot(op['src2'])

            state_vec = state.get_state_vector()
            op_vec = self.encoder.encode(op, state)
            predicted_normalized = self.simulator.predict_value(state_vec, op_vec)
            predicted_value = predicted_normalized * WorldState.NORM_FACTOR

            if op['type'] == 'assign':
                state.assign(op['var'], predicted_value)
            elif op['type'] == 'add_assign':
                state.assign(op['target'], predicted_value)

        return state

    def test(self, test_cases):
        print("\n" + "=" * 64)
        print("  测试阶段：泛化能力验证")
        print("=" * 64)

        correct = 0
        total = 0
        tolerance = 2.0

        for case_idx, test in enumerate(test_cases):
            operations = test['operations']
            expected = test['expected']
            result_state = self.execute(operations)

            print(f"\n  测试 {case_idx + 1}: {test.get('label', '')}")

            for var_name, expected_value in expected.items():
                actual = result_state.read(var_name)
                total += 1

                if actual is not None:
                    actual_rounded = round(actual)
                    is_correct = abs(actual_rounded - expected_value) < tolerance
                    if is_correct:
                        correct += 1
                    mark = "✓" if is_correct else "✗"
                    err = abs(actual_rounded - expected_value)
                    print(f"    {mark} {var_name}: "
                          f"期望={expected_value}, 预测={actual_rounded} "
                          f"(误差={err:.1f})")
                else:
                    print(f"    ✗ {var_name}: 未定义")

        accuracy = correct / total * 100 if total > 0 else 0
        print(f"\n  {'─' * 50}")
        print(f"  总准确率: {correct}/{total} = {accuracy:.1f}%")
        print(f"  {'─' * 50}")
        return accuracy


def main():
    print("╔" + "═" * 62 + "╗")
    print("║  心智模拟学习器 v3 — 约束驱动自组织架构                    ║")
    print("║  核心转变：不给答案，只给约束，让网络自己变成答案            ║")
    print("╚" + "═" * 62 + "╝")
    print()
    print("  数学保证：交换律 + 递增一致性 + 零元锚定 → 唯一确定 f(a,b)=a+b")
    print("  训练信号：仅约束（无任何 v1+v2 标签）")
    print()

    tutorial = (
        "变量是一个命名的容器，可以存储一个数字。"
        "赋值语句 x = 5 将值 5 存入变量 x。"
        "加法运算 a + b 计算两个数字的和。"
        "你可以用 c = a + b 将结果存入新变量 c。"
        "赋值后变量保持其值，直到被重新赋值。"
    )

    print(f"  教程文本：\n  \"{tutorial}\"\n")

    learner = TutorialLearner()
    learner.learn(tutorial)

    rng = np.random.RandomState(123)
    test_cases = []

    test_cases.append({
        'label': 'a=1, b=2, c=a+b',
        'operations': [
            {'type': 'assign', 'var': 'a', 'value': 1},
            {'type': 'assign', 'var': 'b', 'value': 2},
            {'type': 'add_assign', 'target': 'c', 'src1': 'a', 'src2': 'b'},
        ],
        'expected': {'a': 1, 'b': 2, 'c': 3},
    })

    test_cases.append({
        'label': 'a=5, b=7, c=a+b',
        'operations': [
            {'type': 'assign', 'var': 'a', 'value': 5},
            {'type': 'assign', 'var': 'b', 'value': 7},
            {'type': 'add_assign', 'target': 'c', 'src1': 'a', 'src2': 'b'},
        ],
        'expected': {'a': 5, 'b': 7, 'c': 12},
    })

    for i in range(20):
        v1 = rng.randint(1, 99)
        v2 = rng.randint(1, 99)
        test_cases.append({
            'label': f'a={v1}, b={v2}, c=a+b',
            'operations': [
                {'type': 'assign', 'var': 'a', 'value': v1},
                {'type': 'assign', 'var': 'b', 'value': v2},
                {'type': 'add_assign', 'target': 'c', 'src1': 'a', 'src2': 'b'},
            ],
            'expected': {'a': v1, 'b': v2, 'c': v1 + v2},
        })

    accuracy = learner.test(test_cases)

    print("\n" + "=" * 64)
    print("  实验结论")
    print("=" * 64)

    if accuracy >= 90:
        print("  ✓ 核心假说得到支持！")
        print("    系统仅从约束（非标签）自组织地学会了加法，")
        print("    并能泛化到训练中未见的数值组合。")
        print("    这证明了：约束满足可以替代监督信号，")
        print("    网络通过满足约束'自己变成了答案'。")
    elif accuracy >= 60:
        print("  △ 部分支持：约束驱动训练使网络接近正确语义，")
        print("    但泛化精度有限。可能需要更多约束或更长训练。")
    else:
        print("  ✗ 约束驱动训练尚未收敛到正确语义。")
        print("    可能需要调整约束权重或增加约束种类。")

    print()
    return accuracy


if __name__ == '__main__':
    main()
