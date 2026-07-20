# FitzHugh–Nagumo operator surrogates (Python / JAX) — legacy track

The original research track: high-performance trained amortized operator surrogates for
**FitzHugh–Nagumo (FHN)** excitability dynamics, in JAX. It solves the long-horizon
non-divergence / non-collapse problems of neural surrogates without recursive simulation.

The **active** development has moved to the Hodgkin–Huxley + CUDA Julia package in
[`../hh_julia/`](../hh_julia/README.md); this track is kept intact as the reference/baseline that
the Julia port validates against.

> Note: JAX's GPU backend needs WSL2 on Windows. These scripts run CPU-only on native Windows —
> that is exactly why the HH work was ported to Julia/`CUDA.jl`.

## Files by role

| Group | Files |
|---|---|
| **Dynamics / models** | `dynamics.py` (FHN vector field), `hh_model.py` (classic HH), `multichan_model.py` (7-D fast-spiking), `fhn_theory.py` |
| **Datasets** | `operator_data.py` (FHN operator dataset), `neuron_data.py` (stiff-neuron coarse-grid ZOH) |
| **PWFO surrogate** | `pwfo_model.py`, `pwfo_train.py`, `pwfo_eval.py`, `pwfo_figures.py`, `pwfo_freq_table.py` — Phase-Warped Floquet Operator (non-recursive, O(1) at any query time) |
| **Flow-map surrogate** | `flowmap_model.py`, `flowmap_train.py`, `flowmap_eval.py`, `flowmap_fast.py`, `flowmap_fast_train.py`, `flowmap_speed_opt.py`, `flowmap_benchmark.py` — recurrent coarse-grid stepper |
| **Control-affine (invertible)** | `flowmap_affine.py`, `flowmap_affine_train.py` — `F(x)+G(x)u` with the closed-form current inverse |
| **Hybrid / benchmarks** | `hybrid_model.py`, `neuron_bench.py`, `neuron_crossover.py`, `simulation.py`, `results_figures.py` |
| **Tests** | `tests/` (pytest; `conftest.py` forces JAX onto CPU) |
| **Figures** | `plots/results/*.png` |

## Quick start

```bash
cd python
pip install -r requirements.txt
python operator_data.py
python pwfo_train.py && python flowmap_train.py
python pwfo_eval.py  && python flowmap_eval.py
pytest -q
```
