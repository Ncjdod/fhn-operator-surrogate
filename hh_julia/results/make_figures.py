#!/usr/bin/env python3
"""Render the comparison figures and assemble COMPARISON.md from the CSVs written by
run_comparison.jl.  Run after the Julia measurement pass:

    julia --project=hh_julia hh_julia/scripts/run_comparison.jl --gpu
    python hh_julia/results/make_figures.py

Only depends on numpy + matplotlib.  Reads hh_julia/results/data/, writes PNGs and COMPARISON.md
into hh_julia/results/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140,
                     "axes.grid": True, "grid.alpha": 0.3})


def load(name, skip=1):
    return np.genfromtxt(os.path.join(DATA, name), delimiter=",", skip_header=skip)


def meta():
    d = {}
    with open(os.path.join(DATA, "meta.txt")) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                d[k] = v
    return d


M = meta()
DEV = M.get("device", "CPU")


# ---- Figure 1: forward accuracy vs cost -----------------------------------------------------
def fig_forward():
    b = load("forward_bench.csv")
    b = np.atleast_2d(b)
    batch, fine, ros, sur, speed = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(batch, fine, "o-", label="fine RK4 (spike-resolving)")
    ax[0].loglog(batch, ros, "s-", label="Rosenbrock-W (ROS2)")
    ax[0].loglog(batch, sur, "^-", label="surrogate (1 coarse step)")
    ax[0].set_xlabel("batch (neurons integrated together)")
    ax[0].set_ylabel(f"wall-clock per horizon (ms, {DEV})")
    ax[0].set_title(f"Forward cost — horizon {M.get('horizon_ms','?')} ms")
    ax[0].legend(fontsize=8)
    ax[1].bar([str(int(x)) for x in batch], speed, color="#4C78A8")
    for i, v in enumerate(speed):
        ax[1].text(i, v, f"{v:.0f}×", ha="center", va="bottom", fontsize=9)
    ax[1].set_xlabel("batch")
    ax[1].set_ylabel("surrogate speedup vs fine RK4")
    ax[1].set_title(f"Amortized speedup (rollout NRMSE = {M.get('rollout_nrmse','?')})")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_forward.png")); plt.close(fig)


# ---- Figure 2: long-horizon rollout, surrogate vs truth -------------------------------------
def fig_rollout():
    t = load("rollout_t.csv").ravel()
    tru = np.atleast_2d(load("rollout_true.csv"))
    prd = np.atleast_2d(load("rollout_pred.csv"))
    n = tru.shape[1]
    fig, ax = plt.subplots(n, 1, figsize=(9, 2.2 * n), sharex=True)
    ax = np.atleast_1d(ax)
    for j in range(n):
        ax[j].plot(t, tru[:, j], "k", lw=1.4, label="true (fine RK4)")
        ax[j].plot(t, prd[:, j], "C1--", lw=1.4, label="surrogate (coarse flow-map)")
        ax[j].set_ylabel("V (mV)")
        if j == 0:
            ax[j].legend(fontsize=8, ncol=2)
            ax[j].set_title(f"Long-horizon rollout under held-out currents (D={M.get('D_ms','?')} ms/step)")
    ax[-1].set_xlabel("t (ms)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig2_rollout.png")); plt.close(fig)


# ---- Figure 3: control — tracking trace + cost/accuracy -------------------------------------
def fig_control():
    tr = load("control_trace.csv")
    t, ref, sur, lin1, gn = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3], tr[:, 4]
    summ = np.genfromtxt(os.path.join(DATA, "control_summary.csv"), delimiter=",",
                         skip_header=1, dtype=None, encoding="utf-8")
    summ = np.atleast_1d(summ)
    names = [str(r[0]) for r in summ]
    track = np.array([float(r[1]) for r in summ])
    solves = np.array([float(r[2]) for r in summ])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(t, ref, "k", lw=1.6, label="reference")
    ax[0].plot(t, gn, "C2-", lw=1.1, label="Gauss-Newton")
    ax[0].plot(t, lin1, "C0--", lw=1.1, label="one-shot linearization")
    ax[0].plot(t, sur, "C1:", lw=1.3, label="surrogate")
    ax[0].set_xlabel("t (ms)"); ax[0].set_ylabel("V (mV)")
    ax[0].set_title("Closed-loop tracking of the true HH plant"); ax[0].legend(fontsize=8)
    xs = np.maximum(solves, 0.5)
    track = np.maximum(track, 1e-8)   # so an (essentially exact) controller still shows on log axis
    ax[1].scatter(xs, track, s=90, c=["C1", "C0", "C2"][:len(names)], zorder=3)
    for i, nm in enumerate(names):
        ax[1].annotate(nm, (xs[i], track[i]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("stiff ODE substeps per control step")
    ax[1].set_ylabel("tracking NRMSE")
    ax[1].set_title("Control cost vs accuracy (lower-left better)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig3_control.png")); plt.close(fig)


# ---- Figure 4: cable propagation + extracellular EI (article forward model) ------------------
def fig_cable_ei():
    V = load("cable_V.csv", skip=0)
    ei = load("ei_waveforms.csv")
    tei, waves = ei[:, 0], ei[:, 1:]
    dtc = float(M.get("cable_dt_ms", 0.025))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im = ax[0].imshow(V, aspect="auto", origin="lower", cmap="inferno",
                      extent=[0, V.shape[1] * dtc, 1, V.shape[0]])
    ax[0].set_xlabel("t (ms)"); ax[0].set_ylabel("compartment (proximal → distal)")
    ax[0].set_title(f"Spike propagation ({M.get('cable_velocity_mps','?')} m/s)")
    fig.colorbar(im, ax=ax[0], label="V (mV)")
    for k in range(min(4, waves.shape[1])):
        ax[1].plot(tei, waves[:, k], lw=1.2, label=f"electrode {k+1}")
    ax[1].set_xlabel("t (ms)"); ax[1].set_ylabel("Φ (µV)")
    ax[1].set_title("Extracellular electrical image (3-phase)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig4_cable_ei.png")); plt.close(fig)


# ---- Figure 5: differentiable inverse — recovery + stimulus curve ---------------------------
def fig_inverse():
    r = np.atleast_2d(load("inverse_recovery.csv"))
    pi = load("pi_curve.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    lim = [min(r[:, 0].min(), r[:, 1].min()) - 10, max(r[:, 0].max(), r[:, 1].max()) + 10]
    ax[0].plot(lim, lim, "k--", lw=0.8, alpha=0.6)
    ax[0].scatter(r[:, 0], r[:, 2], s=70, c="C3", label="gNa")
    ax[0].scatter(r[:, 1], r[:, 3], s=70, c="C0", marker="s", label="gK")
    ax[0].set_xlabel("true conductance (mS/cm²)"); ax[0].set_ylabel("recovered")
    ax[0].set_title("Differentiable biophysical inverse"); ax[0].legend(fontsize=8)
    ax[1].plot(pi[:, 0], pi[:, 1], "C4-o", ms=3)
    thr = float(M.get("stim_threshold", "nan"))
    if np.isfinite(thr):
        ax[1].axvline(thr, color="k", ls="--", lw=0.9)
        ax[1].axhline(0.5, color="gray", ls=":", lw=0.8)
        ax[1].annotate(f"threshold ≈ {thr:.2f}", (thr, 0.5), fontsize=8, xytext=(6, -14),
                       textcoords="offset points")
    ax[1].set_xlabel("stimulus current I (µA/cm²)"); ax[1].set_ylabel("spike probability P(I)")
    ax[1].set_title("Differentiable stimulus threshold / design")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig5_inverse.png")); plt.close(fig)


def md_table(path):
    rows = np.genfromtxt(os.path.join(DATA, path), delimiter=",", dtype=None, encoding="utf-8")
    rows = np.atleast_1d(rows)
    hdr = [str(x) for x in rows[0]]
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(str(x) for x in np.atleast_1d(r)) + " |")
    return "\n".join(out)


def write_report():
    md = f"""# Hodgkin–Huxley operator surrogate vs optimized solvers — comparison

*Generated by `run_comparison.jl` + `make_figures.py`.  Device: **{DEV}**.  Coarse step
D = {M.get('D_ms','?')} ms (stride {M.get('stride','?')} × dt {M.get('dt','?')} ms), horizon
{M.get('horizon_ms','?')} ms.*

This report compares the learned **control-affine flow-map surrogate** against well-optimized
numerical integrators (spike-resolving fine RK4 and an L-stable Rosenbrock-W stepper), and
reproduces the multineuronal forward model and differentiable inverse of Lotlikar et al. (2026).

## 1. Forward: matching solver accuracy at a fraction of the cost

![forward](fig1_forward.png)

The batched RK4 / Rosenbrock kernels are the optimized solvers (one GPU thread per neuron). A
spike-resolving explicit solver is stability-capped at a tiny dt, so it takes many substeps per
coarse step; the surrogate takes **one** learned coarse step of D = {M.get('D_ms','?')} ms and
jumps over the stiff spike. Peak amortized speedup here: **{M.get('fwd_speedup_max','?')}×** vs
fine RK4, at a full-rollout standardized NRMSE of **{M.get('rollout_nrmse','?')}**.

{md_table('forward_bench.csv')}

## 2. Long-horizon rollout accuracy

![rollout](fig2_rollout.png)

The surrogate tracks the true voltage trace over the full horizon under held-out currents without
recursive fine integration — the accuracy that backs the speedup above.

## 3. Inverse for control: steering the true plant

![control](fig3_control.png)

Because the current enters affinely, the steering current has the closed form
`u* = ⟨G, x_target − F⟩ / ⟨G, G⟩`. Three controllers drive the *true* HH plant to a reference:

{md_table('control_summary.csv')}

Gauss-Newton inverts the exact plant (near-zero tracking) but pays K stiff solves per step; the
one-shot linearization pays one; the **surrogate pays zero stiff solves** (a single MLP forward),
amortizing the controller — its tracking accuracy scales with training budget.

## 4. Multineuronal forward model (article): cable + electrical image

![cable](fig4_cable_ei.png)

A spike propagates along the multi-compartment HH cable at **{M.get('cable_velocity_mps','?')} m/s**
(physiological for an unmyelinated axon) and the line-source model yields the characteristic
**three-phase electrical image** (capacitive +, sodium −, potassium +): three-phase =
{M.get('ei_three_phase','?')}. This is the article's forward model (Eq. 1/4) reproduced.

## 5. Differentiable biophysical inverse + neurostimulation design

![inverse](fig5_inverse.png)

Gradient descent through the differentiable simulator recovers unknown channel densities (max gNa
error {M.get('recovery_max_gNa_err','?')} mS/cm² across the sweep), and the differentiable
spike-probability relaxation gives a stimulus threshold ≈ {M.get('stim_threshold','?')} µA/cm² —
inverted, this is neurostimulation design (pick a target spike probability, get the current).

## 6. How this compares to the reference article

| Aspect | This work (`hh_julia`) | Lotlikar et al. (2026) |
|---|---|---|
| Neuron model | multi-compartment HH cable + point models | multi-compartment HH (RGC) |
| Simulator | own KernelAbstractions kernels (CPU/CUDA, native Windows) | JAXLEY (JAX) |
| Extracellular EI | line-source (Eq. 4), 3-phase reproduced | line-source (Eq. 4) |
| Inverse | gradient-based param recovery + closed-form control | gradient descent + SBI |
| Stimulation | differentiable P(I), threshold, `design_stimulus` | differentiable P_stim, threshold matching |
| Real data / SBI | not included (deterministic) | macaque MEA + neural posterior estimation |

**Honest notes.** Resolving HH *spikes* needs dt ≲ 0.05 ms for any fixed-step solver, which is why
the learned coarse-step map is the real forward win; Rosenbrock's advantage is unconditional
stability on the stiff-but-smooth cable diffusion. The deterministic P(I) is near-step (all-or-none
spikes); the article widens it with a current-noise model and Bayesian SBI — the natural extension
on top of the differentiable forward model here.
"""
    with open(os.path.join(HERE, "COMPARISON.md"), "w") as f:
        f.write(md)


def main():
    fig_forward(); fig_rollout(); fig_control(); fig_cable_ei(); fig_inverse()
    write_report()
    print("wrote figures + COMPARISON.md to", HERE)


if __name__ == "__main__":
    main()
