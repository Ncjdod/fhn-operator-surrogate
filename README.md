# Neuronal operator surrogates & control

Amortized operator surrogates for spiking-neuron dynamics that (1) match well-optimized numerical
solvers in a **multineuronal** setting and (2) are **invertible**, so they double as a real-time
control mechanism. The project began on FitzHugh–Nagumo (Python/JAX) and has been carried over to
biophysical **Hodgkin–Huxley** neurons in a native-Windows, GPU-accelerated Julia package.

## Repository layout

```
.
├── hh_julia/     ← ACTIVE: Hodgkin–Huxley surrogate + control, Julia/CUDA (native Windows, no WSL2)
├── python/       ← LEGACY: original FitzHugh–Nagumo operator surrogates, Python/JAX
└── README.md     ← you are here
```

Two self-contained tracks, one idea. Each has its own README and runs independently.

| | [`hh_julia/`](hh_julia/README.md) | [`python/`](python/README.md) |
|---|---|---|
| Language / accel | Julia + `CUDA.jl` (native-Windows GPU) | Python + JAX (GPU needs WSL2) |
| Neuron model | Hodgkin–Huxley (single, 7-D, RGC), multi-compartment cable | FitzHugh–Nagumo |
| Surrogate | control-affine flow-map (closed-form inverse) | PWFO + flow-map + control-affine |
| Extras | extracellular EI, differentiable biophysical inverse, neurostimulation design | long-horizon O(1) PWFO |
| Status | **active** | reference / baseline |

Reference for the HH track: **Lotlikar et al. (2026), *Learning Biophysical Models of Large-Scale
Multineuronal Data to Enable Precise Neurostimulation***.

## Quick start

**Hodgkin–Huxley / CUDA (recommended).** Native Windows + an NVIDIA GPU (e.g. GTX 1660 Ti), no WSL2:

```powershell
julia --project=hh_julia -e "using Pkg; Pkg.instantiate()"
julia --project=hh_julia hh_julia/test/runtests.jl      # 25 checks (CPU, no GPU needed)
julia --project=hh_julia hh_julia/scripts/demos.jl      # cable+EI, control, inverse demos

# GPU: add CUDA once, then everything scales onto the device with --gpu
julia --project=hh_julia -e "using Pkg; Pkg.add(\"CUDA\")"
julia --project=hh_julia hh_julia/scripts/train_surrogate.jl --model hh --steps 4000 --gpu
```

**Generate the comparison article (surrogate vs optimized solvers vs the article).**
One command trains + benchmarks + dumps results, the second renders the figures and fills the
report — see [`hh_julia/results/README.md`](hh_julia/results/README.md):

```powershell
julia --project=hh_julia hh_julia/scripts/run_comparison.jl --gpu   # train + measure -> results/data
python hh_julia/results/make_figures.py                            # figures + COMPARISON.md
```

**FitzHugh–Nagumo (legacy).** See [`python/README.md`](python/README.md).
