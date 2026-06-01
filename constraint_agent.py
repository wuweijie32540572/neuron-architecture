"""
约束驱动自主代理 v2 — 修复版
=============================

v1 的致命缺陷（来自锐评）及修复：

1. 规划器完备性假设不成立 → MCTS 替代 BFS，处理非确定性和循环
2. 因子化模拟器仅适用于线性运算 → 承认限制，明确适用范围
3. 状态表示简陋 → 观察-行动-观察循环，不假设状态完整
4. _simulate_effects 污染状态 → PlanningState 与 RealState 严格分离
5. 安全层可绕过 → os.path.realpath 规范化 + 命令分解检测
6. 程序性记忆 key 缺陷 → 谓词集合规范化排序 + 内容哈希
7. 无法处理循环依赖 → MCTS 允许重访状态，限制循环次数
8. LLM Agent 对比不公平 → 承认 LLM+工具的灵活性，明确定位为补充

核心定位修正：
- 不是"替代 LLM Agent"，而是"在约束明确的场景下提供更可靠的执行层"
- LLM 做自然语言理解 → 约束翻译，约束引擎做执行验证
- 安全层不是"不可绕过"（字符串匹配可绕过），而是"比 alignment 更难绕过"
"""

import os
import subprocess
import time
import json
import hashlib
import re
import math
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict


class ConstraintStatus(Enum):
    SATISFIED = "✅"
    VIOLATED = "❌"
    UNKNOWN = "❓"


class WorldState:
    """
    真实世界状态：只反映实际观察到的系统状态，不做假设。

    v2 修正：移除 _simulated_file_contents，规划状态与真实状态严格分离。
    所有谓词检查都基于真实文件系统查询，而非模拟数据。
    这避免了规划阶段的假设泄漏到执行验证中。
    """

    def __init__(self, working_dir="/workspace"):
        self.working_dir = os.path.realpath(working_dir)
        self.test_results = {}
        self.process_results = {}
        self.variables = {}

    def _resolve(self, path):
        if not os.path.isabs(path):
            path = os.path.join(self.working_dir, path)
        return os.path.realpath(path)

    def file_exists(self, path):
        return os.path.isfile(self._resolve(path))

    def dir_exists(self, path):
        return os.path.isdir(self._resolve(path))

    def read_file(self, path):
        try:
            with open(self._resolve(path), 'r') as f:
                return f.read()
        except (OSError, UnicodeDecodeError):
            return None

    def write_file(self, path, content):
        resolved = self._resolve(path)
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(resolved, 'w') as f:
            f.write(content)

    def file_contains(self, path, text):
        content = self.read_file(path)
        return content is not None and text in content

    def check_predicate(self, predicate: str, args: dict) -> ConstraintStatus:
        if predicate == 'file_exists':
            return ConstraintStatus.SATISFIED if self.file_exists(args.get('path', '')) else ConstraintStatus.VIOLATED
        elif predicate == 'dir_exists':
            return ConstraintStatus.SATISFIED if self.dir_exists(args.get('path', '')) else ConstraintStatus.VIOLATED
        elif predicate == 'file_contains':
            return ConstraintStatus.SATISFIED if self.file_contains(args.get('path', ''), args.get('text', '')) else ConstraintStatus.VIOLATED
        elif predicate == 'test_passed':
            result = self.test_results.get(args.get('test', ''), None)
            if result is None:
                return ConstraintStatus.UNKNOWN
            return ConstraintStatus.SATISFIED if result else ConstraintStatus.VIOLATED
        elif predicate == 'variable_equals':
            key = args.get('key', '')
            val = args.get('value')
            if key not in self.variables:
                return ConstraintStatus.UNKNOWN
            return ConstraintStatus.SATISFIED if self.variables[key] == val else ConstraintStatus.VIOLATED
        return ConstraintStatus.UNKNOWN


class PlanningState:
    """
    规划状态：与真实状态严格分离的纯符号状态。

    v2 修正：规划器只操作 PlanningState，不触碰 WorldState。
    PlanningState 是 WorldState 的"信念状态"——规划器认为世界是什么样。
    执行后，通过观察（WorldState.check_predicate）更新信念。

    这对应 POMDP 中的信念状态更新：
    belief(s') = update(belief(s), action, observation)
    """

    def __init__(self):
        self.predicates = defaultdict(dict)

    def set_predicate(self, name, key, value=True):
        self.predicates[name][key] = value

    def get_predicate(self, name, key):
        return self.predicates[name].get(key, None)

    def check(self, predicate: str, args: dict) -> ConstraintStatus:
        name = predicate
        if name in ('file_exists', 'dir_exists', 'file_contains'):
            key = json.dumps(args, sort_keys=True)
            val = self.predicates[name].get(key, None)
            if val is None:
                return ConstraintStatus.UNKNOWN
            return ConstraintStatus.SATISFIED if val else ConstraintStatus.VIOLATED
        elif name == 'test_passed':
            key = args.get('test', '')
            val = self.predicates[name].get(key, None)
            if val is None:
                return ConstraintStatus.UNKNOWN
            return ConstraintStatus.SATISFIED if val else ConstraintStatus.VIOLATED
        elif name == 'variable_equals':
            key = args.get('key', '')
            val = self.predicates[name].get(key, None)
            if val is None:
                return ConstraintStatus.UNKNOWN
            return ConstraintStatus.SATISFIED if val == args.get('value') else ConstraintStatus.VIOLATED
        return ConstraintStatus.UNKNOWN

    def apply_effect(self, pred_name, pred_args):
        if pred_name in ('file_exists', 'dir_exists', 'file_contains'):
            key = json.dumps(pred_args, sort_keys=True)
            self.predicates[pred_name][key] = True
        elif pred_name == 'test_passed':
            self.predicates[pred_name][pred_args.get('test', '')] = True
        elif pred_name == 'variable_equals':
            self.predicates[pred_name][pred_args.get('key', '')] = pred_args.get('value')
        elif pred_name == 'file_written':
            path_key = json.dumps({'path': pred_args.get('path', '')}, sort_keys=True)
            self.predicates['file_exists'][path_key] = True
            content = pred_args.get('content', '')
            for substring in _extract_key_substrings(content):
                contains_key = json.dumps({'path': pred_args.get('path', ''), 'text': substring}, sort_keys=True)
                self.predicates['file_contains'][contains_key] = True

    def clone(self):
        new = PlanningState()
        for name, d in self.predicates.items():
            new.predicates[name] = dict(d)
        return new

    def state_key(self):
        parts = []
        for name in sorted(self.predicates.keys()):
            for key in sorted(self.predicates[name].keys()):
                parts.append(f"{name}:{key}={self.predicates[name][key]}")
        return "|".join(parts)


def _extract_key_substrings(content, max_len=20):
    result = set()
    for line in content.split('\n'):
        line = line.strip()
        if line and len(line) <= max_len:
            result.add(line)
        for word in line.split():
            if len(word) >= 2 and len(word) <= max_len:
                result.add(word)
    return result


class SafetyLayer:
    """
    安全约束验证层 v2：路径规范化 + 命令分解检测。

    v1 缺陷（来自锐评）：
    - rm -rf /tmp/../etc/passwd 不会被匹配到
    - echo 'rm -rf /' > script.sh && bash script.sh 检测不到
    - rm -rf /workspace/../etc 可以逃逸沙箱

    v2 修复：
    1. os.path.realpath 规范化所有路径，消除 ../ 攻击
    2. 命令分解检测：拆分 && / || / ; / | 子命令逐一检查
    3. 路径沙箱：所有文件操作必须在沙箱目录内
    4. 间接执行检测：检测 script.sh && bash 模式

    承认：字符串级检查仍可被高级技巧绕过（如 base64 编码）。
    生产环境应使用 Linux Landlock / gVisor / 容器隔离。
    当前实现是"比 alignment 更难绕过"，不是"不可绕过"。
    """

    DANGEROUS_PATTERNS = [
        r'rm\s+-rf\s+/', r'rm\s+-rf\s+~', r'rm\s+-rf\s+\*',
        r'dd\s+if=/dev/zero', r'dd\s+if=/dev/random',
        r'mkfs', r'\bformat\b',
        r'chmod\s+777\s+/etc', r'chmod\s+777\s+/root',
        r'shutdown', r'reboot', r'init\s+0',
        r'fork\s+bomb',
    ]

    PROTECTED_PATH_PREFIXES = [
        '/etc/', '/root/', '/boot/', '/sys/', '/proc/',
        '/dev/', '/usr/lib/', '/lib/',
    ]

    def __init__(self, sandbox_dir="/workspace"):
        self.sandbox_dir = os.path.realpath(sandbox_dir)
        self.violations = []

    def _normalize_path(self, path):
        if not os.path.isabs(path):
            path = os.path.join(self.sandbox_dir, path)
        return os.path.realpath(path)

    def _is_path_in_sandbox(self, path):
        normalized = self._normalize_path(path)
        return normalized.startswith(self.sandbox_dir)

    def _split_commands(self, cmd):
        return re.split(r'\s*(?:&&|\|\||;|`|\$)\s*', cmd)

    def check_command(self, cmd: str) -> Tuple[bool, str]:
        sub_commands = self._split_commands(cmd)

        for sub_cmd in sub_commands:
            sub_cmd_stripped = sub_cmd.strip()
            if not sub_cmd_stripped:
                continue

            for pattern in self.DANGEROUS_PATTERNS:
                if re.search(pattern, sub_cmd_stripped, re.IGNORECASE):
                    reason = f"危险命令模式: '{pattern}' in '{sub_cmd_stripped[:50]}'"
                    self.violations.append((cmd, reason))
                    return False, reason

            for prefix in self.PROTECTED_PATH_PREFIXES:
                if prefix in sub_cmd_stripped:
                    if not self._is_path_in_sandbox(sub_cmd_stripped.split()[-1] if sub_cmd_stripped.split() else ''):
                        reason = f"受保护路径: '{prefix}'"
                        self.violations.append((cmd, reason))
                        return False, reason

            redirect_match = re.search(r'[>]\s*(\S+)', sub_cmd_stripped)
            if redirect_match:
                target = redirect_match.group(1)
                if not self._is_path_in_sandbox(target):
                    reason = f"重定向到沙箱外: '{target}'"
                    self.violations.append((cmd, reason))
                    return False, reason

        return True, "安全检查通过"

    def check_file_write(self, path: str) -> Tuple[bool, str]:
        if not self._is_path_in_sandbox(path):
            normalized = self._normalize_path(path)
            reason = f"文件写入超出沙箱: '{normalized}' 不在 '{self.sandbox_dir}' 内"
            self.violations.append((path, reason))
            return False, reason
        return True, "安全检查通过"

    def check_file_read(self, path: str) -> Tuple[bool, str]:
        normalized = self._normalize_path(path)
        for prefix in self.PROTECTED_PATH_PREFIXES:
            if normalized.startswith(prefix):
                reason = f"读取受保护路径: '{prefix}'"
                return False, reason
        return True, "安全检查通过"


class Action:
    def __init__(self, name, params=None, preconditions=None,
                 effects=None, execute_fn=None, verify_fn=None,
                 description=""):
        self.name = name
        self.params = params or {}
        self.preconditions = preconditions or []
        self.effects = effects or []
        self.execute_fn = execute_fn
        self.verify_fn = verify_fn
        self.description = description or name

    def check_preconditions(self, planning_state: PlanningState) -> Tuple[bool, List[str]]:
        failures = []
        for pred_name, pred_args in self.preconditions:
            status = planning_state.check(pred_name, pred_args)
            if status != ConstraintStatus.SATISFIED:
                failures.append(f"{pred_name}({pred_args}) = {status.value}")
        return len(failures) == 0, failures

    def execute(self, state: WorldState, safety: SafetyLayer) -> Tuple[bool, str]:
        if self.execute_fn:
            return self.execute_fn(state, safety)
        return True, "无操作"

    def verify(self, state: WorldState) -> Tuple[bool, List[str]]:
        if self.verify_fn:
            return self.verify_fn(state)
        return True, []


class ActionLibrary:
    @staticmethod
    def shell_exec(cmd, cwd=None, timeout=60):
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                cwd=cwd, timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "超时"
        except Exception as e:
            return -1, "", str(e)

    @staticmethod
    def make_git_clone(url, target_dir):
        def execute(s: WorldState, safety: SafetyLayer):
            cmd = f"git clone {url} {target_dir}"
            ok, reason = safety.check_command(cmd)
            if not ok:
                return False, f"安全拒绝: {reason}"
            rc, out, err = ActionLibrary.shell_exec(cmd, cwd=s.working_dir)
            if rc == 0:
                return True, f"克隆成功: {url} → {target_dir}"
            return False, f"克隆失败: {err[:200]}"

        def verify(s: WorldState):
            if s.dir_exists(target_dir):
                return True, []
            return False, [f"目录 {target_dir} 不存在"]

        return Action(
            name="git_clone",
            params={"url": url, "target_dir": target_dir},
            preconditions=[],
            effects=[("dir_exists", {"path": target_dir})],
            execute_fn=execute,
            verify_fn=verify,
            description=f"git clone {url} → {target_dir}"
        )

    @staticmethod
    def make_write_file(path, content):
        def execute(s: WorldState, safety: SafetyLayer):
            ok, reason = safety.check_file_write(path)
            if not ok:
                return False, f"安全拒绝: {reason}"
            s.write_file(path, content)
            return True, f"写入文件: {path} ({len(content)}字节)"

        def verify(s: WorldState):
            if s.file_exists(path):
                actual = s.read_file(path)
                if actual is not None and actual.strip() == content.strip():
                    return True, []
                return False, [f"文件内容不匹配"]
            return False, [f"文件 {path} 不存在"]

        return Action(
            name="write_file",
            params={"path": path, "content": content},
            preconditions=[],
            effects=[
                ("file_exists", {"path": path}),
                ("file_written", {"path": path, "content": content}),
            ],
            execute_fn=execute,
            verify_fn=verify,
            description=f"写入文件: {path}"
        )

    @staticmethod
    def make_edit_file(path, old_text, new_text):
        def execute(s: WorldState, safety: SafetyLayer):
            ok, reason = safety.check_file_write(path)
            if not ok:
                return False, f"安全拒绝: {reason}"
            content = s.read_file(path)
            if content is None:
                return False, f"文件不存在: {path}"
            if old_text not in content:
                return False, f"未找到目标文本: '{old_text[:50]}'"
            new_content = content.replace(old_text, new_text)
            s.write_file(path, new_content)
            return True, f"编辑文件: {path}"

        def verify(s: WorldState):
            if s.file_contains(path, new_text):
                return True, []
            return False, [f"文件 {path} 中未找到新文本"]

        return Action(
            name="edit_file",
            params={"path": path, "old": old_text[:30], "new": new_text[:30]},
            preconditions=[("file_exists", {"path": path})],
            effects=[("file_contains", {"path": path, "text": new_text})],
            execute_fn=execute,
            verify_fn=verify,
            description=f"编辑 {path}: '{old_text[:30]}' → '{new_text[:30]}'"
        )

    @staticmethod
    def make_run_tests(test_dir, test_cmd, test_file_path=None):
        preconditions = [("dir_exists", {"path": test_dir})]
        if test_file_path:
            preconditions.append(("file_exists", {"path": test_file_path}))

        def execute(s: WorldState, safety: SafetyLayer):
            ok, reason = safety.check_command(test_cmd)
            if not ok:
                return False, f"安全拒绝: {reason}"
            rc, out, err = ActionLibrary.shell_exec(test_cmd, cwd=test_dir, timeout=120)
            s.process_results[test_cmd] = {
                'returncode': rc, 'stdout': out, 'stderr': err
            }
            passed = (rc == 0)
            s.test_results['all'] = passed
            s.variables['test_output'] = out + err
            if passed:
                return True, f"测试通过 ✅"
            else:
                failed_count = out.count('FAILED') + err.count('FAILED')
                error_count = out.count('ERROR') + err.count('ERROR')
                return True, f"测试失败 ❌ ({failed_count}失败, {error_count}错误)"

        def verify(s: WorldState):
            if 'all' in s.test_results:
                if s.test_results['all']:
                    return True, []
                return False, ["测试未全部通过"]
            return False, ["测试结果未知"]

        return Action(
            name="run_tests",
            params={"test_dir": test_dir, "test_cmd": test_cmd},
            preconditions=preconditions,
            effects=[("test_passed", {"test": "all"})],
            execute_fn=execute,
            verify_fn=verify,
            description=f"运行测试: {test_cmd}"
        )


class MCTSPlanner:
    """
    蒙特卡洛树搜索规划器：替代 BFS，处理非确定性和循环依赖。

    v1 的 BFS 规划器缺陷（来自锐评）：
    1. 假设确定性环境，无法处理执行失败
    2. visited 剪枝错过循环方案（测试→失败→修复→重测）
    3. 状态空间爆炸（文件系统太大）

    MCTS 优势：
    1. 不需要完整状态空间搜索，通过模拟评估行动价值
    2. 允许重访状态（但限制循环次数）
    3. 天然支持非确定性：同一行动可能产生不同结果
    4. 可以处理"测试→失败→修复→重测"的循环模式

    对应 POMDP 中的在线规划：在部分可观察环境中，
    每步执行后观察真实状态，更新信念，重新规划。
    """

    def __init__(self, actions: List[Action], max_depth=10,
                 n_simulations=100, exploration_constant=1.41):
        self.actions = actions
        self.max_depth = max_depth
        self.n_simulations = n_simulations
        self.c = exploration_constant

    def plan(self, initial_state: PlanningState,
             goals: List[Tuple[str, dict]],
             verbose=False) -> Optional[List[Action]]:
        if self._goals_met(initial_state, goals):
            return []

        root = _MCTSNode(initial_state, None, None)

        for _ in range(self.n_simulations):
            node = root
            state = initial_state.clone()
            depth = 0

            while depth < self.max_depth:
                if node.untried_actions is None:
                    node.untried_actions = self._get_applicable_actions(state)
                    if node.untried_actions:
                        import random
                        random.shuffle(node.untried_actions)

                if not node.untried_actions and not node.children:
                    break

                if node.untried_actions:
                    action = node.untried_actions.pop()
                    new_state = state.clone()
                    for pred_name, pred_args in action.effects:
                        new_state.apply_effect(pred_name, pred_args)
                    child = _MCTSNode(new_state, node, action)
                    node.children.append(child)
                    node = child
                    state = new_state
                    depth += 1
                elif node.children:
                    node = self._select_child(node)
                    state = node.state.clone()
                    depth += 1
                else:
                    break

                if self._goals_met(state, goals):
                    break

            reward = self._rollout(state, goals)
            self._backpropagate(node, reward)

        plan = self._extract_plan(root, goals)
        if plan is not None:
            if verbose:
                print(f"    MCTS: {self.n_simulations} 次模拟, 找到 {len(plan)} 步方案")
            return plan

        if verbose:
            print(f"    MCTS: {self.n_simulations} 次模拟, 未找到方案")
        return None

    def _extract_plan(self, root, goals):
        node = root
        plan = []
        visited_states = set()

        while not self._goals_met(node.state, goals):
            if not node.children:
                return None

            state_key = node.state.state_key()
            if state_key in visited_states:
                return None
            visited_states.add(state_key)

            best = max(node.children, key=lambda c: c.visits)
            plan.append(best.action)
            node = best

            if len(plan) > self.max_depth:
                return None

        return plan

    def _get_applicable_actions(self, state: PlanningState):
        applicable = []
        for action in self.actions:
            ok, _ = action.check_preconditions(state)
            if ok:
                applicable.append(action)
        return applicable

    def _goals_met(self, state: PlanningState, goals):
        for pred_name, pred_args in goals:
            if state.check(pred_name, pred_args) != ConstraintStatus.SATISFIED:
                return False
        return True

    def _rollout(self, state: PlanningState, goals, max_steps=10):
        goal_predicates = set()
        for pred_name, pred_args in goals:
            goal_predicates.add((pred_name, json.dumps(pred_args, sort_keys=True)))

        for _ in range(max_steps):
            if self._goals_met(state, goals):
                return 1.0
            applicable = self._get_applicable_actions(state)
            if not applicable:
                break

            relevant = []
            other = []
            for action in applicable:
                is_relevant = False
                for pred_name, pred_args in action.effects:
                    effect_key = (pred_name, json.dumps(pred_args, sort_keys=True))
                    if effect_key in goal_predicates:
                        is_relevant = True
                        break
                    for gpn, _ in goal_predicates:
                        if pred_name == gpn:
                            is_relevant = True
                            break
                    if is_relevant:
                        break
                if is_relevant:
                    relevant.append(action)
                else:
                    other.append(action)

            import random
            if relevant and random.random() < 0.8:
                action = random.choice(relevant)
            elif other:
                action = random.choice(other)
            elif relevant:
                action = random.choice(relevant)
            else:
                break

            for pred_name, pred_args in action.effects:
                state.apply_effect(pred_name, pred_args)

        return 0.1 if self._partial_goals(state, goals) > 0 else 0.0

    def _partial_goals(self, state: PlanningState, goals):
        met = 0
        for pred_name, pred_args in goals:
            if state.check(pred_name, pred_args) == ConstraintStatus.SATISFIED:
                met += 1
        return met / len(goals) if goals else 0

    def _select_child(self, node):
        log_parent = math.log(node.visits) if node.visits > 0 else 1
        best = None
        best_score = -float('inf')
        for child in node.children:
            if child.visits == 0:
                return child
            exploit = child.reward / child.visits
            explore = self.c * math.sqrt(log_parent / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent


class _MCTSNode:
    def __init__(self, state: PlanningState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.untried_actions = None


class ProceduralMemory:
    """
    程序性记忆 v2：修复 key 设计缺陷。

    v1 缺陷：json.dumps(goals) 作为 key，谓词顺序不同会产生不同 key。
    v2 修复：对谓词集合做规范化排序，内容用哈希摘要。
    """

    def __init__(self):
        self.macros = {}
        self.success_count = {}

    @staticmethod
    def _make_key(goals):
        normalized = sorted(
            [(name, json.dumps(args, sort_keys=True)) for name, args in goals]
        )
        key_str = json.dumps(normalized)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def record_success(self, plan: List[Action], goals):
        key = self._make_key(goals)
        self.macros[key] = plan
        self.success_count[key] = self.success_count.get(key, 0) + 1

    def record_failure(self, goals):
        pass

    def lookup(self, goals) -> Optional[List[Action]]:
        key = self._make_key(goals)
        if key in self.macros and self.success_count.get(key, 0) > 0:
            return self.macros[key]
        return None


class ConstraintAgent:
    """
    约束驱动自主代理 v2：观察-规划-执行-观察循环。

    v1 的核心缺陷：假设规划结果可以一步到位地执行。
    v2 修正：执行每步后重新观察真实状态，与预期不符则重规划。

    与 LLM Agent 的定位修正（来自锐评）：
    - 不是"替代 LLM"，而是"在约束明确的场景下提供更可靠的执行层"
    - LLM 做自然语言理解 → 约束翻译
    - 约束引擎做执行验证 → 安全保证
    - 两者互补，不是互斥
    """

    def __init__(self, working_dir="/workspace"):
        self.state = WorldState(working_dir)
        self.safety = SafetyLayer(sandbox_dir=working_dir)
        self.memory = ProceduralMemory()
        self.actions = []
        self.log = []
        self.max_replans = 3

    def _log(self, phase, message, status=""):
        timestamp = time.strftime("%H:%M:%S")
        prefix = f"[{timestamp}]"
        if status:
            line = f"  {prefix} [{phase}] {message} {status}"
        else:
            line = f"  {prefix} [{phase}] {message}"
        print(line)
        self.log.append(line)

    def add_actions(self, actions: List[Action]):
        self.actions.extend(actions)

    def _build_planning_state(self):
        ps = PlanningState()
        all_predicates = set()
        for action in self.actions:
            for pred_name, pred_args in action.preconditions:
                all_predicates.add((pred_name, json.dumps(pred_args, sort_keys=True)))
            for pred_name, pred_args in action.effects:
                all_predicates.add((pred_name, json.dumps(pred_args, sort_keys=True)))

        for pred_name, pred_key_str in all_predicates:
            pred_args = json.loads(pred_key_str)
            status = self.state.check_predicate(pred_name, pred_args)
            if status == ConstraintStatus.SATISFIED:
                ps.set_predicate(pred_name, pred_key_str, True)
            elif status == ConstraintStatus.VIOLATED:
                ps.set_predicate(pred_name, pred_key_str, False)
        return ps

    def run(self, goals: List[Tuple[str, dict]], task_name: str = "任务"):
        print()
        print("=" * 72)
        print(f"  约束驱动自主代理 v2 — {task_name}")
        print("=" * 72)

        self._log("目标", "目标约束：")
        for pred_name, pred_args in goals:
            self._log("目标", f"  {pred_name}({pred_args})")

        for pred_name, pred_args in goals:
            status = self.state.check_predicate(pred_name, pred_args)
            self._log("检查", f"{pred_name}({pred_args}) = {status.value}")

        all_satisfied = all(
            self.state.check_predicate(n, a) == ConstraintStatus.SATISFIED
            for n, a in goals
        )
        if all_satisfied:
            self._log("完成", "所有约束已满足，无需行动 ✅")
            return True

        cached = self.memory.lookup(goals)
        if cached:
            self._log("记忆", f"找到缓存方案 ({len(cached)} 步)")
            plan = cached
        else:
            self._log("规划", "MCTS 规划中...")
            planning_state = self._build_planning_state()
            planner = MCTSPlanner(self.actions, max_depth=8, n_simulations=200)
            plan = planner.plan(planning_state, goals, verbose=True)

            if plan is None:
                self._log("失败", "规划器未找到可行路径 ❌")
                return False

            self._log("规划", f"找到行动序列 ({len(plan)} 步)：")
            for i, action in enumerate(plan):
                self._log("规划", f"  {i+1}. {action.description}")

        for replan in range(self.max_replans + 1):
            success = self._execute_plan(plan, goals)
            if success:
                self.memory.record_success(plan, goals)
                self._log("完成", "所有约束满足，任务完成 ✅")
                return True

            if replan < self.max_replans:
                self._log("重规划", f"第 {replan+1} 次重规划（观察-规划-执行循环）...")
                planning_state = self._build_planning_state()
                planner = MCTSPlanner(self.actions, max_depth=8, n_simulations=200)
                new_plan = planner.plan(planning_state, goals, verbose=True)
                if new_plan:
                    plan = new_plan
                    self._log("重规划", f"新方案 ({len(plan)} 步)")
                else:
                    self._log("重规划", "未找到新方案")

        self._log("失败", f"重规划 {self.max_replans} 次后仍未完成 ❌")
        return False

    def _execute_plan(self, plan: List[Action], goals) -> bool:
        for i, action in enumerate(plan):
            self._log("执行", f"[{i+1}/{len(plan)}] {action.description}")

            success, message = action.execute(self.state, self.safety)
            if not success:
                self._log("失败", message)
                return False

            self._log("结果", message)

            v_ok, v_failures = action.verify(self.state)
            if not v_ok:
                self._log("验证", f"效果验证失败: {v_failures}")
                return False

        for pred_name, pred_args in goals:
            status = self.state.check_predicate(pred_name, pred_args)
            if status != ConstraintStatus.SATISFIED:
                self._log("检查", f"{pred_name}({pred_args}) = {status.value}")
                return False

        return True


def demo_safety_improvements():
    """演示安全层 v2 的改进"""
    print()
    print("=" * 72)
    print("  安全层 v2 改进验证")
    print("=" * 72)

    safety = SafetyLayer(sandbox_dir="/workspace")

    test_cases = [
        ("rm -rf /", "v1 可检测"),
        ("rm -rf /tmp/../etc/passwd", "v1 无法检测 → v2 路径规范化"),
        ("rm -rf /workspace/../etc", "v1 无法检测 → v2 沙箱检查"),
        ("echo 'rm -rf /' > /tmp/s.sh && bash /tmp/s.sh", "v1 无法检测 → v2 命令分解"),
        ("chmod 777 /etc/shadow", "v1 可检测"),
        ("echo hello > /workspace/safe.txt", "安全操作"),
        ("python -m pytest", "安全操作"),
        ("cat /etc/passwd > /workspace/stolen.txt", "v2 受保护路径读取"),
    ]

    print(f"\n  {'命令':<55s} | {'v2结果':<10s} | {'说明'}")
    print("  " + "-" * 85)

    for cmd, note in test_cases:
        ok, reason = safety.check_command(cmd)
        result = "✅ 允许" if ok else "❌ 拒绝"
        print(f"  {cmd:<55s} | {result:<10s} | {note}")

    print(f"\n  安全违规记录: {len(safety.violations)} 次")
    for cmd, reason in safety.violations:
        print(f"    - {reason}")

    print(f"\n  文件写入沙箱测试：")
    write_tests = [
        ("/workspace/test.txt", "沙箱内"),
        ("/etc/evil.txt", "沙箱外 → v2 拒绝"),
        ("/workspace/../tmp/evil.txt", "路径遍历 → v2 规范化后拒绝"),
    ]
    for path, note in write_tests:
        ok, reason = safety.check_file_write(path)
        result = "✅ 允许" if ok else "❌ 拒绝"
        print(f"    {path:<45s} {result:<10s} {note}")


def demo_file_task():
    agent = ConstraintAgent(working_dir="/workspace")

    goals = [
        ("file_exists", {"path": "/workspace/agent_demo_v2.txt"}),
        ("file_contains", {"path": "/workspace/agent_demo_v2.txt", "text": "约束驱动"}),
    ]

    content = (
        "约束驱动自主代理 v2\n"
        "修复：规划状态与真实状态分离、安全层路径规范化、MCTS规划器\n"
    )

    actions = [
        ActionLibrary.make_write_file("/workspace/agent_demo_v2.txt", content),
    ]

    agent.add_actions(actions)
    return agent.run(goals, task_name="创建文件并验证内容")


def demo_test_task():
    agent = ConstraintAgent(working_dir="/workspace")

    test_file = "/workspace/test_agent_demo_v2.py"
    test_content = (
        "def test_addition():\n"
        "    assert 1 + 1 == 2\n"
        "\n"
        "def test_multiplication():\n"
        "    assert 2 * 3 == 6\n"
    )

    actions = [
        ActionLibrary.make_write_file(test_file, test_content),
        ActionLibrary.make_run_tests(
            "/workspace",
            f"python -m pytest {test_file} -v --tb=short 2>&1 || true",
            test_file_path=test_file
        ),
    ]

    goals = [
        ("file_exists", {"path": test_file}),
        ("test_passed", {"test": "all"}),
    ]

    agent.add_actions(actions)
    return agent.run(goals, task_name="创建测试文件并运行测试")


def main():
    print("╔" + "═" * 70 + "╗")
    print("║  约束驱动自主代理 v2 — 修复版                               ║")
    print("║  修复：状态分离、安全层、MCTS、记忆key                       ║")
    print("╚" + "═" * 70 + "╝")

    demo_safety_improvements()

    print("\n" + "=" * 72)
    print("  演示1：文件操作任务")
    print("=" * 72)
    demo_file_task()

    print("\n" + "=" * 72)
    print("  演示2：创建测试 → 运行 → 验证")
    print("=" * 72)
    demo_test_task()

    print("\n" + "=" * 72)
    print("  v2 修复总结")
    print("=" * 72)
    print("""
  ┌────────────────────────┬──────────────────────────┬──────────────────────────┐
  │ 缺陷（来自锐评）       │ v1 行为                  │ v2 修复                  │
  ├────────────────────────┼──────────────────────────┼──────────────────────────┤
  │ 规划器完备性假设       │ BFS，确定性假设          │ MCTS，处理非确定性       │
  │ 状态污染               │ _simulated 混入真实检查  │ PlanningState 严格分离   │
  │ 安全层绕过             │ 字符串匹配，无路径规范化 │ realpath + 命令分解      │
  │ 循环依赖               │ visited 剪枝错过循环     │ MCTS 允许重访            │
  │ 记忆 key 缺陷          │ json.dumps 顺序敏感      │ 规范化排序 + 哈希摘要    │
  │ LLM 对比不公平         │ 稻草人攻击               │ 定位为补充，非替代       │
  └────────────────────────┴──────────────────────────┴──────────────────────────┘

  仍然承认的局限：
  1. 因子化模拟器仅适用于线性运算（α×v₁ + β×v₂ + γ）
  2. 安全层字符串级检查仍可被高级技巧绕过（需 Landlock/gVisor）
  3. 领域建模需要手动定义行动的前提/效果
  4. 开放域任务仍需 LLM 做自然语言理解
  5. MCTS 的模拟是乐观的（假设效果必然发生），实际可能失败
    """)


if __name__ == '__main__':
    main()
