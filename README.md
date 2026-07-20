# FitzHugh-Nagumo Operator Surrogate Modeling (PWFO & Recurrent Flow-Map)

This repository contains a high-performance trained amortized operator surrogate for FitzHugh–Nagumo (FHN) excitability dynamics. It solves the long-horizon non-divergence and non-collapse problems of traditional neural surrogates without relying on recursive simulation.

## Core Architectures

1. **Phase-Warped Floquet Operator (PWFO)**:
   * **Method**: Maps initial states and time-varying control currents to any query time in a single $O(1)$ forward pass. Phase accumulates as a prefix sum of instantaneous frequencies, the limit cycle is modeled via Fourier series, and transients decay via isostable modes.
   * **Key Property**: Bounded and eternal by construction. Does not suffer from error propagation over long rollouts.

2. **Recurrent Flow-Map Stepper**:
   * **Method**: A Markov residual stepper executing on a coarse grid. Designed for fast, high-accuracy tracking of high-frequency or highly oscillatory control currents.

3. **Hybrid Model**:
   * **Method**: Automatically routes predictions: the recurrent flow-map stepper is used for local accuracy under transient stimulation, while the PWFO is used for long-term asymptotic stability.

## Quick Start

Ensure you have Python 3.9+ installed, then run:

```bash
pip install -r requirements.txt
python operator_data.py
python pwfo_train.py
python flowmap_train.py
python pwfo_eval.py
python flowmap_eval.py
```

All detailed research findings, design decisions, and math derivations are available in **[REPORT.md](REPORT.md)**.

## Hodgkin–Huxley extension (Julia / CUDA) — `hh_julia/`

The project is carried over to **biophysical Hodgkin–Huxley** neurons in a native-Windows,
GPU-accelerated Julia package under **[`hh_julia/`](hh_julia/README.md)**. It adds GPU-batched
HH/stiff solvers (KernelAbstractions, CPU + CUDA), a multi-compartment cable with a line-source
**electrical image** forward model, the control-affine flow-map surrogate with its closed-form
inverse, and a **differentiable biophysical inverse** (channel-density recovery + neurostimulation
design), taking reference from Lotlikar et al. (2026), *Learning Biophysical Models of Large-Scale
Multineuronal Data to Enable Precise Neurostimulation*.

Julia is used because JAX has no native-Windows GPU backend (it needs WSL2); `CUDA.jl` targets the
GTX 1660 Ti directly with no WSL2. The Python/JAX code here is unchanged. See
[`hh_julia/README.md`](hh_julia/README.md) for setup, benchmarks, and validation.
