# Comparison article — how to generate it

Two commands produce `COMPARISON.md` and its five figures: a Julia pass that trains the surrogate
and measures it against the optimized solvers + the article's forward/inverse models, then a Python
pass that renders the figures and fills in the report.

## Run it (GPU, native Windows — no WSL2)

```powershell
# from the repo root, with CUDA added once:  julia --project=hh_julia -e 'using Pkg; Pkg.add("CUDA")'
julia --project=hh_julia hh_julia/scripts/run_comparison.jl --gpu     # ~minutes on a GTX 1660 Ti
python hh_julia/results/make_figures.py                               # needs numpy + matplotlib
```

Outputs (this directory):
`COMPARISON.md`, `fig1_forward.png`, `fig2_rollout.png`, `fig3_control.png`,
`fig4_cable_ei.png`, `fig5_inverse.png`.

Flags for `run_comparison.jl`:
- `--gpu` — move batched arrays to `CuArray` (the GTX 1660 Ti numbers). Omit for CPU.
- `--steps N` — training steps (default 4000; more → lower rollout NRMSE).
- `--quick` — tiny config for a fast pipeline smoke test.

## What it measures

| Section | Figure | Compares |
|---|---|---|
| Forward cost/accuracy | `fig1_forward` | surrogate 1 coarse step **vs** fine RK4 **vs** Rosenbrock-W, across batch size |
| Long-horizon rollout | `fig2_rollout` | surrogate V(t) vs true V(t) under held-out currents |
| Inverse for control | `fig3_control` | surrogate vs one-shot linearization vs Gauss-Newton on the true plant |
| Cable + electrical image | `fig4_cable_ei` | article forward model: propagation + 3-phase EI |
| Differentiable inverse | `fig5_inverse` | channel-density recovery + stimulus P(I) / threshold |

Intermediate data lands in `results/data/*.csv` (+ `meta.txt`); regenerate figures alone by
re-running `make_figures.py`.
