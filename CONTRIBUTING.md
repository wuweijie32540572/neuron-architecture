# Contributing to Neuron Architecture

感谢您对本项目的关注！

## 如何贡献

### 报告问题

如果您发现bug或有功能建议：
1. 检查是否已有相关issue
2. 如果没有，创建新issue，详细描述问题

### 提交代码

1. Fork本仓库
2. 创建特性分支：
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. 进行修改并测试：
   ```bash
   pytest tests/ -v
   ```
4. 提交更改：
   ```bash
   git commit -m "Add: your feature description"
   ```
5. 推送到分支：
   ```bash
   git push origin feature/your-feature-name
   ```
6. 创建Pull Request

### 代码规范

- 使用Python类型注解
- 添加docstring（Google风格）
- 遵循PEP 8
- 为新功能添加测试

### 文档规范

- 更新相关.md文件
- 添加数学公式时使用LaTeX
- 标注未实现功能为"Future Work"

## 开发环境

```bash
# 克隆仓库
git clone https://github.com/wuweijie32540572/neuron-architecture.git
cd neuron-architecture

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v --cov=src/neuron_arch

# 格式化代码
black src/ tests/
isort src/ tests/
```

## 项目结构

```
src/neuron_arch/     # 核心代码
tests/               # 测试
docs/                # 文档
examples/            # 示例
```

## 联系方式

aiwuweijie@foxmail.com

## 许可证

贡献的代码将采用MIT许可证。
