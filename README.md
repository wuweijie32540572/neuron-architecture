# Novel Neuron Architectures for Overcoming LLM Limitations

[![CI](https://github.com/wuweijie32540572/neuron-architecture/workflows/CI/badge.svg)](https://github.com/wuweijie32540572/neuron-architecture/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Neuroscience-inspired architectures to address core LLM limitations: catastrophic forgetting, local learning, event-driven computation, and symbol grounding.**

## Installation

```bash
pip install neuron-architecture
```

Or from source:

```bash
git clone https://github.com/wuweijie32540572/neuron-architecture.git
cd neuron-architecture
pip install -e .
```

## One-Click Reproduction

一键复现所有实验结果：

```bash
python run_all_experiments.py
```

输出保存到 `results/experiment_report.json`。

## Quick Start

```python
from neuron_arch import IntegratedSystem

# Create integrated system (197K parameters)
system = IntegratedSystem(n_sch=256, n_pc=256, n_hm=128)

# Train on data
for x, target in data:
    metrics = system.learn(x, target)
    print(f"Sparsity: {metrics.sparsity:.3f}")

# Consolidate memories
system.consolidate(n_replay=80)
```

## Architectures

| Architecture | Purpose | Key Innovation |
|-------------|---------|----------------|
| **SCH-Neuron** | Event-driven sparse computation | Adaptive threshold for target spike rate |
| **PC-Neuron** | Local predictive learning | Two-stage training (offline + online) |
| **HM-Neuron** | Continual learning | Risk-weighted replay with stability measure |
| **NG-Neuron** | Dynamic learning control | DA/5HT/ACh/NE neuromodulatory gating |
| **ESB-Neuron** | Symbol grounding | Orthogonal patterns + grounding matrix |

## Key Results

### Rigorous Comparison (Mean ± Std, 5 runs)

**8-Function Regression Task**:

| Method | Average MSE | Description |
|--------|-------------|-------------|
| Standard MLP (BP) | **0.0454 ± 0.0048** | Full backpropagation training |
| Pure Local Learning | 0.4120 ± 0.0067 | No BP pretraining, local rules only |
| Two-Stage Training | **0.1326 ± 0.0114** | BP pretrain + local finetune |

**Key Finding**: Two-stage training achieves **67.8% lower MSE** than pure local learning, while using BP only for initialization.

**Note on "98.1% improvement"**: This was compared to a random-initialized local learning baseline (MSE=6.64). The rigorous comparison above uses proper baselines.

### Sequential MNIST (Continual Learning)

| Method | Task1 Initial | Task1 Final | Forgetting |
|--------|---------------|-------------|------------|
| Vanilla MLP | 99.5% | 0.0% | 99.5% |
| Experience Replay | 99.5% | 95.4% | **4.0%** |
| HM-Neuron | 99.8% | 92.7% | **7.1%** |

### Other Results

| Experiment | Result |
|------------|--------|
| Symbol recognition confidence | **0.987** (vs 0.25 random) |
| DA explore/exploit | Correct tradeoff |
| Integrated system (197K params) | Validated |

## Documentation

- [Architecture Design](ARCHITECTURE.md) - Detailed design of each neuron type
- [Experiments](EXPERIMENTS.md) - Experimental results and analysis
- [Deep Analysis](docs/DEEP_ANALYSIS.py) - Multi-disciplinary analysis (physics, biology, math, electronics)

## Examples

See [examples/](examples/) directory:
- `quickstart.py` - Basic usage examples
- `demo.ipynb` - Jupyter notebook with visualizations (forgetting curves, spike rates)

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
neuron-architecture/
├── src/neuron_arch/     # Core implementations
│   ├── sch.py           # Spiking-Continuous Hybrid
│   ├── pc.py            # Predictive Coding
│   ├── hm.py            # Hippocampus-Cortex Memory
│   ├── ng.py            # Neuromodulator Gating
│   ├── esb.py           # Embodied-Symbolic Bridge
│   └── integrated.py    # Full system
├── tests/               # Unit tests
├── docs/                # Documentation
├── examples/            # Usage examples
└── .github/workflows/   # CI/CD
```

## Mathematical Foundations

### SCH-Neuron: Adaptive Threshold

```
τ_m · dV/dt = -V + R_m · I(t)
V_th(t) = V_th_base × (1 + α · (r - r_target))
```

### PC-Neuron: Free Energy Minimization

```
F = D_KL[q||p] - E[ln p(s|ψ)]
ΔW = -η · ∂F/∂W = η · ε · z^T
```

### HM-Neuron: Memory Stability

```
σ(m) = 1 / (1 + α·age + β/(access+1))
P(replay m) ∝ 1 - σ(m)
```

### NG-Neuron: Neuromodulatory Gating

```
gate = 0.4·DA + 0.2·5HT + 0.2·ACh + 0.2·NE
η_eff = η_base × gate × 2.0 / (1 + t·(1-DA))
```

## Limitations

1. **BP pretraining required**: Two-stage training uses BP for initialization (not purely local)
2. **HM insufficient for strong interference**: +194.8% forgetting on 8 orthogonal tasks
3. **Task similarity affects results**: Performance varies with task orthogonality
4. **Scale limited**: Only tested up to 197K parameters

## Related Work & References

This project builds on the following foundational work:

### Local Learning & Feedback Alignment
- **Lillicrap, T. P., et al. (2016)**. "Random synaptic feedback weights support error backpropagation for deep learning." *Nature Communications*. [DOI:10.1038/ncomms13276](https://doi.org/10.1038/ncomms13276)
  - Shows that random feedback weights can approximate backpropagation, supporting biological plausibility of local learning.

### Continual Learning
- **Kirkpatrick, J., et al. (2017)**. "Overcoming catastrophic forgetting in neural networks." *PNAS*. [DOI:10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114)
  - Introduces Elastic Weight Consolidation (EWC) for mitigating catastrophic forgetting.
- **McClelland, J. L., et al. (1995)**. "Why there are complementary learning systems in the hippocampus and neocortex." *Psychological Review*.
  - Complementary Learning Systems (CLS) theory underlying our HM-Neuron design.

### Predictive Coding
- **Friston, K. (2010)**. "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*. [DOI:10.1038/nrn2787](https://doi.org/10.1038/nrn2787)
  - Free energy principle underlying our PC-Neuron design.

### Neuromodulation
- **Schultz, W., et al. (1997)**. "A neural substrate of prediction and reward." *Science*. [DOI:10.1126/science.9358056](https://doi.org/10.1126/science.9358056)
  - Dopamine as reward prediction error, underlying our NG-Neuron DA dynamics.

### Spiking Neural Networks
- **Roy, K., et al. (2019)**. "Towards spike-based machine intelligence with neuromorphic computing." *Nature*. [DOI:10.1038/s41586-019-1677-2](https://doi.org/10.1038/s41586-019-1677-2)
  - Overview of neuromorphic computing and spike-based computation.

## Future Directions

- Elastic Weight Consolidation (EWC)
- Progressive Neural Networks (PGN)
- Larger scale validation (1M+ parameters)
- Real-world task evaluation

## Citation

```bibtex
@software{neuron_architecture_2024,
  title = {Novel Neuron Architectures for Overcoming LLM Limitations},
  author = {Neuron Architecture Research},
  year = {2024},
  url = {https://github.com/wuweijie32540572/neuron-architecture}
}
```

## License

MIT License - see [LICENSE](LICENSE)

## Contact

aiwuweijie@foxmail.com
