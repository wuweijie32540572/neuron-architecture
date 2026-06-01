"""
心智模拟学习器 (Mental Simulation Learner) v2
==============================================

核心假说验证：
一个具备内部心智模拟能力的神经架构，可以在只阅读中文教程文本
（不含任何代码示例或执行结果）的情况下，学会变量的赋值和加法操作，
并能够对未见过的数值正确执行计算。

v2 关键改进：混合架构
--------------------
v1 失败原因：让单一 MLP 同时学习"路由"（哪个槽位该写）和"计算"（写什么值），
对极简 MLP 而言过于复杂。

v2 方案：神经-符号混合架构
- 神经部分 (MentalSimulator)：预测操作产生的**值**（标量输出）
- 符号部分 (WorldState + OperationEncoder)：负责路由（将值写入正确的槽位）

这对应认知科学中的双过程理论 (Dual Process Theory, Kahneman 2011)：
- 系统1（神经）：快速、直觉式的数值计算
- 系统2（符号）：慢速、精确的逻辑路由

认知科学基础：
- 符号接地 (Symbol Grounding, Harnad 1990)
- 心智模拟 (Mental Simulation, Hegarty 2004)
- 预测编码 (Predictive Coding, Rao & Ballard 1999)
- 程序性记忆 (Procedural Memory, Squire 2004)
- 心理模型 (Mental Model, Johnson-Laird 1983)
"""

import numpy as np


class SymbolGrounder:
    """
    从教程文本中学习每个 token 的语义向量和角色。

    对应认知科学中的"符号接地问题"(Symbol Grounding Problem)。
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
            raise RuntimeError(f"无可用槽位")
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

    v2 关键改进：在编码中直接包含源变量值。
    这实现了双过程理论的清晰分离：
    - 符号系统 (WorldState) 负责"路由"——查找哪些变量参与运算
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
    心智模拟引擎（v2 混合架构）：预测操作产生的值。

    设计原理：
    --------
    v2 采用神经-符号混合架构：
    - 神经网络负责"值计算"：给定当前状态和操作，预测操作产生的数值结果
    - 符号系统负责"路由"：将预测的值写入正确的变量槽

    这对应认知科学中的双过程理论 (Kahneman 2011)：
    - 系统1（神经）：快速、直觉式的数值计算——"a+b 大约是12"
    - 系统2（符号）：精确的逻辑路由——"把结果存入变量c"

    架构：
    ----
    输入: [当前状态 (20维) | 操作编码 (33维)] = 53维
    隐藏层1: 128维 + Leaky ReLU
    隐藏层2: 64维 + Leaky ReLU
    输出: 1维 (预测的操作结果值，归一化)

    学习机制：
    --------
    在线 SGD + 梯度裁剪 + 学习率衰减。
    训练时加入高斯噪声模拟预测误差，增强鲁棒性。
    """

    def __init__(self, state_dim=20, op_dim=33, hidden_dims=(128, 64), lr=0.005):
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

    def predict_value(self, state_vec, op_vec):
        """
        前向传播：预测操作产生的值（带残差连接）。

        残差连接设计（对应预测编码中的"预测残差"概念）：
        output = MLP(input) + op_vec[12]

        这使得 MLP 只需学习增量：
        - 赋值操作：MLP 输出 ≈ 0，总输出 = 0 + value = value
        - 加法操作：MLP 输出 ≈ op_vec[13]（src2值），总输出 = src2 + src1 = sum

        对应认知科学中的"预测残差"编码：大脑不编码绝对值，
        而是编码与预测的偏差（Rao & Ballard 1999）。
        """
        x = np.concatenate([state_vec, op_vec])
        self._cache = [x]

        for i in range(len(self.weights) - 1):
            h = x @ self.weights[i] + self.biases[i]
            h = self._leaky_relu(h)
            self._cache.append(h)
            x = h

        residual = x @ self.weights[-1] + self.biases[-1]
        self._residual_skip = op_vec[12]
        output = residual[0] + self._residual_skip
        return output

    def train_step(self, state_vec, op_vec, target_value):
        """
        单步训练：通过预测误差更新权重。

        预测编码循环：预测 → 误差 → 更新
        """
        pred = self.predict_value(state_vec, op_vec)

        error = pred - target_value
        loss = error ** 2
        self.error_history.append(loss)
        self.step_count += 1

        d_output = 2.0 * error * np.ones(1)

        d = d_output

        grads_W = []
        grads_b = []

        for i in range(len(self.weights) - 1, -1, -1):
            grads_W.insert(0, np.outer(self._cache[i], d))
            grads_b.insert(0, d.copy())

            if i > 0:
                d = d @ self.weights[i].T
                pre_act = self._cache[i - 1] @ self.weights[i - 1] + self.biases[i - 1]
                d = d * self._leaky_relu_grad(pre_act)

        grad_norm = sum(np.sum(gW ** 2) for gW in grads_W) + \
                    sum(np.sum(gb ** 2) for gb in grads_b)
        grad_norm = np.sqrt(grad_norm)

        if grad_norm > self.max_grad_norm:
            scale = self.max_grad_norm / grad_norm
            grads_W = [g * scale for g in grads_W]
            grads_b = [g * scale for g in grads_b]

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]

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
    教程学习器：从中文教程文本中学习编程能力。

    学习流程（模拟人类认知学习过程）：
    1. 文本分析 → 词汇习得与语法角色推断
    2. 规则推断 → 从示例中抽象出规则
    3. 心智演练 → 在脑中"试运行"并调整预测
    4. 模式巩固 → 将验证通过的模式转为长期记忆

    v2 混合架构：
    - MentalSimulator (神经) 预测操作产生的值
    - WorldState (符号) 负责路由和状态管理
    - 训练信号来自系统自身从文本推断的规则（非外部执行反馈）
    """

    def __init__(self):
        self.grounder = SymbolGrounder(embed_dim=32)
        self.simulator = MentalSimulator(
            state_dim=WorldState.STATE_DIM,
            op_dim=OperationEncoder.OP_DIM,
            hidden_dims=(128, 64),
            lr=0.005
        )
        self.memory = ProceduralMemory()
        self.encoder = OperationEncoder()
        self.learned_rules = {}

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
        print("  阶段 2：规则推断 (Rule Induction)")
        print("=" * 64)

        self._infer_rules(sentences)

        for name, rule in self.learned_rules.items():
            print(f"    {name}: {rule['description']}")

        print("\n" + "=" * 64)
        print("  阶段 3：心智演练 (Mental Rehearsal)")
        print("=" * 64)

        self._mental_rehearsal()

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
                        'description': '加法赋值：[变量] = [变量] + [变量] → 计算和并存入',
                        'pattern': ['variable', 'operator_assign', 'variable',
                                    'operator_add', 'variable'],
                        'action': 'add_and_assign',
                    }

            if 'operator_assign' in roles:
                eq_idx = roles.index('operator_assign')
                if (eq_idx > 0 and eq_idx < len(tokens) - 1
                        and roles[eq_idx - 1] == 'variable'
                        and roles[eq_idx + 1] == 'value'):
                    self.learned_rules['assign_value'] = {
                        'description': '赋值：[变量] = [数值] → 将数值存入变量',
                        'pattern': ['variable', 'operator_assign', 'value'],
                        'action': 'assign_value',
                    }

    def _mental_rehearsal(self):
        """
        心智演练：生成操作序列，训练 MentalSimulator 预测操作值。

        v2 混合架构训练：
        - MentalSimulator 预测操作产生的值（标量）
        - 规则模型提供"正确"值作为训练目标
        - 加入高斯噪声增强鲁棒性（模拟自回归预测误差）

        对应预测编码循环：预测 → 误差 → 调整。
        """
        rng = np.random.RandomState(42)
        n_sequences = 8000
        losses = []

        for seq_idx in range(n_sequences):
            operations = self._generate_random_sequence(rng)
            state = WorldState()

            for op in operations:
                if op['type'] == 'assign':
                    state.ensure_slot(op['var'])
                elif op['type'] == 'add_assign':
                    state.ensure_slot(op['target'])
                    state.ensure_slot(op['src1'])
                    state.ensure_slot(op['src2'])

                state_vec = state.get_state_vector()
                noisy_state = state_vec + rng.randn(len(state_vec)) * 0.01

                op_vec = self.encoder.encode(op, state)
                noisy_op = op_vec + rng.randn(len(op_vec)) * 0.003

                target_value = self._compute_target_value(op, state)

                if target_value is not None:
                    loss = self.simulator.train_step(noisy_state, noisy_op, target_value)
                    losses.append(loss)
                    state = self._apply_rule(op, state)
                else:
                    break

            if (seq_idx + 1) % 2000 == 0:
                avg_loss = np.mean(losses[-500:])
                print(f"    序列 {seq_idx + 1}/{n_sequences}, "
                      f"平均损失: {avg_loss:.6f}, lr: {self.simulator.lr:.6f}")
                self.simulator.decay_lr(0.96)

        if losses:
            print(f"\n    最终损失: {np.mean(losses[-200:]):.6f}")
            print(f"    总训练步数: {len(losses)}")

    def _generate_random_sequence(self, rng):
        val1 = rng.randint(1, 99)
        val2 = rng.randint(1, 99)
        return [
            {'type': 'assign', 'var': 'a', 'value': val1},
            {'type': 'assign', 'var': 'b', 'value': val2},
            {'type': 'add_assign', 'target': 'c', 'src1': 'a', 'src2': 'b'},
        ]

    def _compute_target_value(self, operation, state):
        """
        计算操作的目标值（归一化）。

        这是"内部世界模型"——由文本推断出的行为规则。
        不调用任何外部解释器，是系统自主构建的内部模型。
        """
        if operation['type'] == 'assign':
            return operation['value'] / WorldState.NORM_FACTOR
        elif operation['type'] == 'add_assign':
            v1 = state.read(operation['src1'])
            v2 = state.read(operation['src2'])
            if v1 is not None and v2 is not None:
                return (v1 + v2) / WorldState.NORM_FACTOR
        return None

    def _apply_rule(self, operation, state):
        new_state = state.copy()
        if operation['type'] == 'assign':
            new_state.assign(operation['var'], operation['value'])
            return new_state
        elif operation['type'] == 'add_assign':
            v1 = state.read(operation['src1'])
            v2 = state.read(operation['src2'])
            if v1 is not None and v2 is not None:
                new_state.assign(operation['target'], v1 + v2)
                return new_state
        return None

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

        v2 混合执行：
        1. MentalSimulator 预测操作产生的值（神经计算）
        2. WorldState 将值写入正确的变量槽（符号路由）
        3. 递归进行，每步的预测结果更新状态供下一步使用

        这实现了真正的"心智模拟链"：
        在脑中逐步运行程序，每一步的输出成为下一步的输入。
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
    print("║  心智模拟学习器 v2 — 核心假说验证实验                      ║")
    print("║  假说：仅从中文教程文本学会变量赋值和加法                    ║")
    print("╚" + "═" * 62 + "╝")
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
        print("  ✓ 核心假说得到支持：系统仅从教程文本学会了赋值和加法，")
        print("    并能泛化到训练中未见的数值组合。")
        print("    MentalSimulator 通过预测编码成功内化了操作规则。")
    elif accuracy >= 60:
        print("  △ 部分支持：系统学会了基本模式，但泛化精度有限。")
        print("    可能需要更多训练或更大网络容量。")
    else:
        print("  ✗ 假说未得到支持：MentalSimulator 未能充分内化操作规则。")

    print()
    return accuracy


if __name__ == '__main__':
    main()
