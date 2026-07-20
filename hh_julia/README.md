# HHSurrogate — Hodgkin–Huxley operator surrogate & control (Julia / CUDA)

A Julia port and **Hodgkin–Huxley** extension of the `fhn-operator-surrogate` project. It carries
the FitzHugh–Nagumo surrogate/control machinery over to biophysical HH neurons, adds the
**multi-compartment cable + extracellular electrical image** forward model, and makes the whole
system **invertible for control** — both a real-time closed-form current inverse and a
gradient-based biophysical parameter/stimulation inverse.

It takes reference from **Lotlikar et al. (2026), *Learning Biophysical Models of Large-Scale
Multineuronal Data to Enable Precise Neurostimulation*** (multi-compartment HH, differentiable
simulation, line-source EIs, and neurostimulation as the control application).

> The original Python/JAX code is **untouched** — this lives entirely under `hh_julia/`.

---

## Why Julia (and why move off the JAX/WSL2 stack)

The goal is to *match well-optimized numerical solvers in a multineuronal setting on your
Windows + GTX 1660 Ti box, without WSL2* — and to keep everything differentiable for control.

* **JAX's GPU backend has no native-Windows support** — it needs WSL2 or drops to CPU. Since you
  are moving off WSL2, the existing `jax[cuda12]` stack can't touch the GPU on native Windows.
* **Julia has first-class native-Windows CUDA** via `CUDA.jl` (no WSL2), compiles through LLVM
  (near-C, "compiler-level") and has best-in-class stiff ODE solvers — exactly the regime HH
  lives in.
* The entire compute core is written with **`KernelAbstractions.jl`**, so the *same* kernel
  source runs on the CPU (how the test-suite validates it) and on the GPU (`CuArray`) — the
  backend is chosen at runtime from the array type. One thread integrates one neuron.
* **No heavy AD dependency.** A small self-contained forward-mode dual (`src/ad.jl`) provides the
  Jacobians for the stiff solver and the gradients for biophysical inversion. It is `isbits`, so
  it is legal inside GPU kernels, and it means the package installs with **zero binary artifacts**
  (only `StaticArrays` + `KernelAbstractions`).

Your **GTX 1660 Ti** (Turing, `sm_75`, 6 GB) is fully supported by `CUDA.jl`. Use `Float32` (the
default here) — that card's FP64 is intentionally slow.

---

## Install & run (native Windows, no WSL2)

Install Julia (`winget install julia -s msstore`, or juliaup). Then, from the repo root:

```powershell
# CPU: run and validate everything (no GPU packages needed)
julia --project=hh_julia -e "using Pkg; Pkg.instantiate()"
julia --project=hh_julia hh_julia/test/runtests.jl        # 25 checks, all on CPU
julia --project=hh_julia hh_julia/scripts/demos.jl        # cable+EI, control, inference
julia --project=hh_julia hh_julia/bench/benchmarks.jl     # forward + control benchmarks

# GPU: add CUDA once, then pass --gpu to move the batched work onto the 1660 Ti
julia --project=hh_julia -e "using Pkg; Pkg.add(\"CUDA\")"
julia --project=hh_julia hh_julia/bench/benchmarks.jl --gpu
julia --project=hh_julia hh_julia/scripts/train_surrogate.jl --model hh --steps 4000 --gpu
```

The first run precompiles (Julia's one-time JIT/latency); subsequent runs are fast.

---

## What's inside

| File | Role |
|---|---|
| `src/models/models.jl` | Classic squid HH (`HHClassic`), 7-D fast-spiking `MultiChan`, and the article's Fohlmeister RGC channels (`RGCChannels`, Na/K/Ca/K-Ca/leak, Tables 4–5). All control-affine in the injected current. |
| `src/ad.jl` | Self-contained forward-mode dual number (Jacobians + parameter gradients, GPU-safe). |
| `src/solvers/solvers.jl` | Batched **RK4** and **Rosenbrock-W (ROS2)** stiff steppers as `KernelAbstractions` kernels (CPU + CUDA). |
| `src/models/cable.jl` | **Multi-compartment HH cable** (article Eq. 1/17) with an implicit-axial **IMEX** solver (unconditionally stable tridiagonal Thomas + exponential-Euler gates). |
| `src/inference/extracellular.jl` | **Line-source electrical image** (Eq. 4) + differentiable EI features (capacitive/Na/K peaks, duration, propagation delay). |
| `src/surrogate/*.jl` | Control-affine **flow-map surrogate** `x⁺ = F(x) + G(x)·u`, closed-form inverse, dense-MLP toolkit + Adam, and **curriculum BPTT** training — forward *and* backward are matrix algebra, so training runs on GPU with no AD package. |
| `src/control/inverse.jl` | Closed-form **controllers** (surrogate / one-shot linearization / Gauss-Newton) and a closed-loop driver that steers the true plant. |
| `src/inference/fit.jl` | **Gradient-based biophysical inverse**: recover channel densities from a trace; differentiable spike-probability relaxation for stimulus-threshold matching and neurostimulation design. |

## The two pillars, and how they map to the article

**1 — Match well-optimized numerical solvers in the multineuronal setting.**
The batched RK4 / Rosenbrock kernels *are* the optimized solver (one GPU thread per neuron); the
learned flow-map takes a single coarse step `D` that jumps over the stiff spike a stability-capped
explicit solver cannot. Benchmark (CPU, `HHClassic`, `D = 0.4 ms`, horizon 40 ms):

```
batch   fineRK4 ms   ROS2 ms   sur1 ms   surrogate speedup
 64          34.3      25.8     0.37       93×
1024        554.7     412.3     4.12      135×          (7-D multichannel: ~165×)
```

**2 — Invertible for control.** Because the current enters affinely, the steering current has a
closed form `u* = ⟨G, x_target − F⟩ / ⟨G, G⟩`. Steering the true HH plant to a reference:

```
controller                       tracking NRMSE   stiff solves/step
Gauss-Newton (exact plant)          1.0e-07              6
one-shot linearization              6.7e-03              1
surrogate (1 MLP forward)           amortized            0     ← no stiff solve in the loop
```

**Article forward + inverse.** A spike propagates along the multi-compartment cable at a
physiological **0.49 m/s** and produces the characteristic **three-phase EI** (capacitive +,
sodium −, potassium +) with propagation delays across the 7-electrode hex patch. The differentiable
inverse recovers known conductances exactly (`gNa 80→110`, `gK 45→30`, loss → 3e-12) and
`design_stimulus(p*)` returns the current achieving a target spike probability — the
neurostimulation control use.

## Validation

`hh_julia/test/runtests.jl` (25 checks, all CPU, no GPU/heavy deps):

* `HHClassic` and `MultiChan` vector fields + RK4 trajectories match an **independent NumPy
  oracle** to machine precision (`test/fixtures.jl`);
* the in-house AD Jacobian matches finite differences (`< 1e-5`);
* Rosenbrock shows clean 2nd-order subthreshold convergence;
* the surrogate's closed-form inverse residual is ⟂ to `G`, and BPTT gradients match finite
  differences (`< 1e-4`);
* Gauss-Newton closed-loop control tracks the true plant to `< 1e-4`;
* cable spike propagation + three-phase EI;
* conductance recovery + differentiable stimulus design.

## Honest notes

* **Spike resolution needs `dt ≲ 0.05 ms`** for *any* fixed-step solver — an HH spike upstroke is a
  ~0.1 ms event. Rosenbrock buys unconditional *stability* (and shines on the stiff-but-smooth
  axial diffusion of the cable, where it takes large steps), but resolving the spike itself still
  needs small steps — which is precisely why the learned coarse-step flow-map is worthwhile.
* `RGCChannels` is transcribed exactly from the article's Table 4/5 and is intended as the
  **per-compartment kinetics of the cable** (spikes initiated proximally and propagated). As an
  isolated point neuron its very fast 35 °C inactivation truncates the spike and it enters
  depolarization block under sustained drive — expected, not a bug. `HHClassic` is the robust
  single-cell workhorse used by the surrogate/control benchmarks.
* The deterministic stimulus threshold `P(I)` is near-step (spikes are all-or-none). The article
  widens it with an injected-current **noise model** and does Bayesian **SBI**; that stochastic
  extension is a natural next step on top of the differentiable forward model here.
