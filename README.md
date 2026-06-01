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

| Experiment | Result |
|------------|--------|
| Two-stage training vs pure local | **98.1% improvement** |
| HM forgetting mitigation | **143.9 pp reduction** |
| Symbol recognition confidence | **0.987** (vs 0.25 random) |
| DA explore/exploit | **Correct tradeoff** |
| Integrated system (197K params) | **Validated** |

## Documentation

- [Architecture Design](ARCHITECTURE.md) - Detailed design of each neuron type
- [Experiments](EXPERIMENTS.md) - Experimental results and analysis
- [Deep Analysis](docs/DEEP_ANALYSIS.py) - Multi-disciplinary analysis (physics, biology, math, electronics)

## Examples

See [examples/](examples/) directory:
- `quickstart.py` - Basic usage examples

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

1. **HM insufficient for strong interference**: +194.8% forgetting on 8 orthogonal tasks
2. **Task similarity affects results**: Performance varies with task orthogonality
3. **Scale limited**: Only tested up to 197K parameters

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
