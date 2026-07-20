#!/usr/bin/env python3
"""Render the comparison figures and assemble COMPARISON.md from the CSVs written by
run_comparison.jl.  Run after the Julia measurement pass:

    julia --project=hh_julia hh_julia/scripts/run_comparison.jl --gpu
    python hh_julia/results/make_figures.py

Only depends on numpy + matplotlib.  Reads hh_julia/results/data/, writes PNGs and COMPARISON.md
into hh_julia/results/.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def _arg(name, default):
    """Tiny --name value parser (the script has no other CLI needs)."""
    return sys.argv[sys.argv.index(f"--{name}") + 1] if f"--{name}" in sys.argv[:-1] else default


# Primary run (the one the report is written from) and the optional second run to compare it
# against, e.g. --data data_gpu --compare data_cpu.
DATA = os.path.join(HERE, _arg("data", "data"))
CMP = _arg("compare", None)
CMP = os.path.join(HERE, CMP) if CMP else None
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140,
                     "axes.grid": True, "grid.alpha": 0.3})


def load(name, skip=1, root=None):
    return np.genfromtxt(os.path.join(root or DATA, name), delimiter=",", skip_header=skip)


def meta(root=None):
    d = {}
    with open(os.path.join(root or DATA, "meta.txt")) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                d[k] = v
    return d


M = meta()
DEV = M.get("device", "CPU")
MC = meta(CMP) if CMP and os.path.isdir(CMP) else None
DEVC = MC.get("device", "?") if MC else None


# ---- Figure 1: forward accuracy vs cost -----------------------------------------------------
def fig_forward():
    b = np.atleast_2d(load("forward_bench.csv"))
    batch, fine, ros, sur1, roll, speed = (b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(batch, fine, "o-", color="C3", label="fine RK4 (spike-resolving)")
    ax[0].loglog(batch, ros, "s-", color="C0", label="Rosenbrock-W (ROS2)")
    ax[0].loglog(batch, roll, "^-", color="C2", label="surrogate (same horizon)")
    ax[0].loglog(batch, sur1, "v:", color="C2", alpha=0.6, label="surrogate (1 coarse step)")
    ax[0].set_xlabel("batch (neurons integrated together)")
    ax[0].set_ylabel(f"wall-clock (ms, {DEV})")
    ax[0].set_title(f"Forward cost — horizon {M.get('horizon_ms','?')} ms "
                    f"({M.get('horizon_steps','?')} coarse steps)")
    ax[0].legend(fontsize=8)
    ax[1].bar([str(int(x)) for x in batch], speed,
              color=["#4C78A8" if v >= 1 else "#B4413C" for v in speed])
    ax[1].axhline(1.0, color="k", lw=0.9, ls="--")
    for i, v in enumerate(speed):
        ax[1].text(i, v, f"{v:.2f}×", ha="center", va="bottom", fontsize=9)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("batch")
    ax[1].set_ylabel("surrogate speedup vs fine RK4 (horizon-for-horizon)")
    ax[1].set_title(f"Like-for-like speedup (rollout NRMSE = {M.get('rollout_nrmse','?')})")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_forward.png")); plt.close(fig)


# ---- Figure 1b: accuracy vs cost, every method on one plane ---------------------------------
def fig_pareto():
    r = np.genfromtxt(os.path.join(DATA, "pareto.csv"), delimiter=",", skip_header=1,
                      dtype=None, encoding="utf-8")
    r = np.atleast_1d(r)
    meth = np.array([str(x[0]) for x in r])
    nsub = np.array([int(float(x[1])) for x in r])
    ms = np.array([float(x[2]) for x in r])
    err = np.array([float(x[3]) for x in r])
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    style = {"rk4": ("C3", "o", "RK4 (explicit)"),
             "ros2": ("C0", "s", "Rosenbrock-W (L-stable)"),
             "surrogate": ("C2", "*", "learned flow-map")}
    # A solver given too few substeps blows up (NaN). Those points are real results, so park them
    # on a "diverged" band above everything that converged instead of dropping them silently.
    fin = np.isfinite(err)
    ceil = max(err[fin].max() * 30, 10.0) if fin.any() else 10.0
    for m, (c, mk, lab) in style.items():
        k = meth == m
        if not k.any():
            continue
        o = np.argsort(ms[k])
        ax.plot(ms[k & fin][np.argsort(ms[k & fin])], err[k & fin][np.argsort(ms[k & fin])],
                mk + "-", color=c, ms=13 if m == "surrogate" else 7, lw=1.2, label=lab)
        ax.plot(ms[k & ~fin], np.full((~fin & k).sum(), ceil), "x", color=c, ms=8, mew=2)
        # Only label converged points; the diverged ones sit on top of each other in the band.
        for x, y, n, ok in zip(ms[k], err[k], nsub[k], fin[k]):
            if ok:
                ax.annotate(f"{n} substeps" if n else "0 substeps\n(1 MLP/step)", (x, y),
                            fontsize=7.5, xytext=(6, 5), textcoords="offset points", color=c)
    ax.set_xscale("log"); ax.set_yscale("log")
    if (~fin).any():
        ax.axhspan(ceil / 3, ceil * 3, color="0.5", alpha=0.12)
        # axes-fraction coords: an x in data coords would drag the tight bbox out to the label.
        ax.text(0.015, 0.955, "diverged (NaN)", transform=ax.transAxes, fontsize=8, color="0.35")
    ax.set_xlabel(f"wall-clock per {M.get('horizon_ms','?')} ms horizon, batch "
                  f"{M.get('pareto_batch','?')} (ms, {DEV})")
    ax.set_ylabel("standardized NRMSE vs converged reference")
    ax.set_title("Accuracy vs cost at a fixed coarse step "
                 f"D = {M.get('D_ms','?')} ms\n(down-left is better)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1b_pareto.png")); plt.close(fig)


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


def _cell(x):
    """Render a CSV cell for markdown: floats get 3 decimals (4 sig figs when tiny)."""
    s = str(x)
    try:
        v = float(s)
    except ValueError:
        return s
    if not np.isfinite(v):
        return "diverged" if np.isnan(v) else s
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    return f"{v:.3f}" if abs(v) >= 1e-3 else f"{v:.3g}"


def md_table(path, root=None):
    rows = np.genfromtxt(os.path.join(root or DATA, path), delimiter=",", dtype=None,
                         encoding="utf-8")
    rows = np.atleast_1d(rows)
    hdr = [str(x) for x in rows[0]]
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(_cell(x) for x in np.atleast_1d(r)) + " |")
    return "\n".join(out)


# ---- Figure 6 (optional): the same measurements on two devices ------------------------------
def fig_devices():
    """Wall-clock of every forward method on both devices, plus the training throughput."""
    a, b = np.atleast_2d(load("forward_bench.csv")), np.atleast_2d(load("forward_bench.csv", root=CMP))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # column 4 is the full-horizon surrogate rollout -- the only surrogate column comparable to the
    # solver columns, which also integrate the whole horizon.
    for arr, dev, ls in ((b, DEVC, "--"), (a, DEV, "-")):
        ax[0].loglog(arr[:, 0], arr[:, 1], "o" + ls, color="C3", label=f"fine RK4 ({dev})")
        ax[0].loglog(arr[:, 0], arr[:, 2], "s" + ls, color="C0", label=f"ROS2 ({dev})")
        ax[0].loglog(arr[:, 0], arr[:, 4], "^" + ls, color="C2", label=f"surrogate rollout ({dev})")
    ax[0].set_xlabel("batch (neurons integrated together)")
    ax[0].set_ylabel("wall-clock per horizon (ms)")
    ax[0].set_title(f"{DEV} (solid) vs {DEVC} (dashed): forward cost")
    ax[0].legend(fontsize=7, ncol=2)

    # throughput = neurons x coarse steps per second, the scale-free way to read the two devices
    K = float(M.get("horizon_ms", 0)) / max(float(M.get("D_ms", 1)), 1e-9)
    for arr, dev, c in ((b, DEVC, "#9C755F"), (a, DEV, "#4C78A8")):
        ax[1].loglog(arr[:, 0], arr[:, 0] * K / (arr[:, 1] / 1e3), "o-", color=c,
                     label=f"fine RK4 ({dev})")
        ax[1].loglog(arr[:, 0], arr[:, 0] / (arr[:, 3] / 1e3), "^--", color=c,
                     label=f"surrogate 1 step ({dev})")
    ax[1].set_xlabel("batch")
    ax[1].set_ylabel("neuron-steps / s")
    ax[1].set_title("Throughput (higher better)")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig6_devices.png")); plt.close(fig)


def device_section():
    if not MC:
        return ""
    ta, tb = float(M.get("train_seconds", "nan")), float(MC.get("train_seconds", "nan"))
    faster, slower = (DEVC, DEV) if tb < ta else (DEV, DEVC)
    ratio = max(ta, tb) / min(ta, tb)
    Cc, Cg = control_facts(CMP), control_facts()
    # cost of the cheapest dominating solver config on each device (nan if none dominates)
    bc, bg = pareto_facts(CMP)["best"], pareto_facts()["best"]
    bc_ms = bc[2] if bc else float("nan")
    bg_ms = bg[2] if bg else float("nan")
    return f"""
## 7. {DEV} vs {DEVC} — the same code on both backends

![devices](fig6_devices.png)

The compute core is written once with `KernelAbstractions` and the backend is chosen at runtime
from the array type, so these two runs execute the *same* kernel source on both devices. That
makes the split instructive.

**Training is faster on {faster}.** {M.get('train_steps','?')} BPTT steps took {ta:.0f} s on
{DEV} and {tb:.0f} s on {DEVC} — **{ratio:.1f}× in favour of {faster}**. The training step is a
128×128 MLP over a batch of 64 unrolled up to 120 steps: hundreds of tiny kernel launches per
step, almost no arithmetic per launch. That is the worst possible shape for a GPU and a fine
shape for a few CPU cores.

**The same effect sets the forward comparison.** On {DEVC} the surrogate rollout is
{1 / np.atleast_2d(load('forward_bench.csv', root=CMP))[:, 5].max():.1f}–\
{1 / np.atleast_2d(load('forward_bench.csv', root=CMP))[:, 5].min():.1f}× slower than fine RK4;
on {DEV} that gap widens to {1 / np.atleast_2d(load('forward_bench.csv'))[:, 5].max():.0f}–\
{1 / np.atleast_2d(load('forward_bench.csv'))[:, 5].min():.0f}×. Moving to the GPU makes the
launch-bound method *relatively worse*, not better — the fused solver kernel absorbs the extra
parallelism, the host-side rollout loop cannot.

**Where the GPU does win:** the fused, compute-heavy kernels. Gauss-Newton control costs
{Cc['gn6'][2]:.3f} ms on {DEVC} versus {Cg['gn6'][2]:.3f} ms on {DEV}
({Cc['gn6'][2] / Cg['gn6'][2]:.1f}×), and the cheapest solver configuration that beats the
surrogate at batch {M.get('pareto_batch','?')} drops from {bc_ms:.0f} ms on {DEVC} to
{bg_ms:.1f} ms on {DEV} ({bc_ms / bg_ms:.0f}×). One thread per neuron, all the work inside the
kernel — that is the shape this card rewards.

{DEVC} forward benchmark, for reference:

{md_table('forward_bench.csv', root=CMP)}
"""


def pareto_facts(root=None):
    """Pull the surrogate point and the best solver point that dominates it (faster AND better)."""
    r = np.atleast_1d(np.genfromtxt(os.path.join(root or DATA, "pareto.csv"), delimiter=",",
                                    skip_header=1, dtype=None, encoding="utf-8"))
    rows = [(str(x[0]), int(float(x[1])), float(x[2]), float(x[3])) for x in r]
    sur = next(x for x in rows if x[0] == "surrogate")
    dom = [x for x in rows if x[0] != "surrogate" and np.isfinite(x[3])
           and x[2] < sur[2] and x[3] < sur[3]]
    diverged = [x for x in rows if not np.isfinite(x[3])]
    best = min(dom, key=lambda x: x[2]) if dom else None
    return {"sur_ms": sur[2], "sur_err": sur[3], "best": best, "n_dom": len(dom),
            "diverged": diverged}


def control_facts(root=None):
    r = np.atleast_1d(np.genfromtxt(os.path.join(root or DATA, "control_summary.csv"),
                                    delimiter=",", skip_header=1, dtype=None, encoding="utf-8"))
    return {str(x[0]): (float(x[1]), int(float(x[2])), float(x[3])) for x in r}


def write_report():
    P = pareto_facts()
    C = control_facts()
    fb = np.atleast_2d(load("forward_bench.csv"))
    sp_lo, sp_hi = fb[:, 5].min(), fb[:, 5].max()
    if P["best"]:
        bm, bn, bt, be = P["best"]
        verdict = (f"**{bm.upper()} with {bn} substeps is {P['sur_ms'] / bt:.1f}× faster *and* "
                   f"{P['sur_err'] / be:.0f}× more accurate** than the learned flow-map "
                   f"({bt:.1f} ms / {be:.2g} vs {P['sur_ms']:.1f} ms / {P['sur_err']:.3f}).")
    else:
        verdict = ("No classical configuration dominates the surrogate on both axes — the learned "
                   "step is on the Pareto front.")
    ndiv = len(P["diverged"])
    md = f"""# Hodgkin–Huxley operator surrogate vs optimized solvers — comparison

*Generated by `run_comparison.jl` + `make_figures.py`.  Device: **{DEV}**.  Coarse step
D = {M.get('D_ms','?')} ms (stride {M.get('stride','?')} × dt {M.get('dt','?')} ms), horizon
{M.get('horizon_ms','?')} ms.*

This report compares the learned **control-affine flow-map surrogate** against well-optimized
numerical integrators (spike-resolving fine RK4 and an L-stable Rosenbrock-W stepper), and
reproduces the multineuronal forward model and differentiable inverse of Lotlikar et al. (2026).

## 1. Forward cost: the learned step vs the optimized solvers

![forward](fig1_forward.png)

The batched RK4 / Rosenbrock kernels are the optimized solvers (one GPU thread per neuron). A
spike-resolving explicit solver is stability-capped at a tiny dt, so it takes many substeps per
coarse step; the surrogate takes **one** learned coarse step of D = {M.get('D_ms','?')} ms and
jumps over the stiff spike.

Two surrogate columns are reported, and the distinction is the whole story. `sur1step_ms` is a
single coarse step — that is the per-step cost the method is really about. `surRollout_ms` is the
same {M.get('horizon_steps','?')}-step horizon the solver columns integrate. **Only the second one
is comparable to them**, so the speedup column is computed from it; dividing a full solver rollout
by a single surrogate step would mostly be measuring the horizon length
({M.get('horizon_steps','?')}×).

Measured like-for-like, the surrogate is **not faster**: the speedup ranges from
{sp_lo:.2f}× to {sp_hi:.2f}×, i.e. it is {1 / sp_hi:.0f}–{1 / sp_lo:.0f}× *slower* than fine RK4
over the same horizon, at a full-rollout standardized NRMSE of {M.get('rollout_nrmse','?')}.
The per-step figure is genuine — one coarse step costs {fb[:, 3].min():.2f}–{fb[:, 3].max():.2f} ms
— but {M.get('horizon_steps','?')} of them, each launching a handful of small kernels from a
host-side loop, cost far more than one fused solver kernel that loops internally on the device.

{md_table('forward_bench.csv')}

## 1b. The same comparison with the accuracy held honest

![pareto](fig1b_pareto.png)

The table above times each method at *its own* accuracy, which flatters the surrogate. So spend
the same wall clock on the classical solvers instead: hold the coarse step D fixed, give RK4 and
Rosenbrock progressively fewer substeps per step, and measure error against a converged reference
(4× the substeps used to build the training data). Every method then lands on one
accuracy-vs-cost plane, and the surrogate — zero substeps, one MLP forward per coarse step — is
just another point on it.

{md_table('pareto.csv')}

**Verdict at batch {M.get('pareto_batch','?')}.** {verdict} {P['n_dom']} classical configurations
dominate it outright. What the classical solvers *cannot* do is take the coarse step at all:
{ndiv} of the configurations tried blow up to NaN, because an explicit method at
D = {M.get('D_ms','?')} ms is far past its stability limit on an HH spike. That — unconditional
stability at a coarse step, and a closed-form inverse — is the surrogate's actual contribution
here, not throughput.

## 2. Long-horizon rollout accuracy

![rollout](fig2_rollout.png)

The surrogate tracks the true voltage trace over the full horizon under held-out currents without
recursive fine integration. It follows the subthreshold envelope and fires roughly in the right
places, which is what an NRMSE of {M.get('rollout_nrmse','?')} looks like — useful as a coarse
predictor, not as a replacement for an accurate integrator.

## 3. Inverse for control: steering the true plant

![control](fig3_control.png)

Because the current enters affinely, the steering current has the closed form
`u* = ⟨G, x_target − F⟩ / ⟨G, G⟩`. Three controllers drive the *true* HH plant to a reference:

{md_table('control_summary.csv')}

Gauss-Newton inverts the exact plant (tracking NRMSE {C['gn6'][0]:.2g}) but pays
{C['gn6'][1]} stiff substeps per control step; the one-shot linearization pays {C['lin1'][1]}
and still tracks to {C['lin1'][0]:.2g}. The surrogate pays **zero** stiff solves — but on this
hardware that does not translate into either accuracy or speed: it tracks to
{C['surrogate'][0]:.2g} (three to four orders of magnitude worse) at {C['surrogate'][2]:.3f} ms
versus {C['gn6'][2]:.3f} ms for Gauss-Newton. Both controller kernels are single fused launches,
so the model-based ones are simply better here on every axis. The surrogate controller's appeal is
structural — no ODE solve in the loop, one closed-form expression — and it would need either a
much better-trained flow-map or a per-step cost regime where the stiff solve genuinely dominates
before that structure pays off.

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

Note the article reports **no runtime, hardware or batch-size figures at all**, and uses no learned
surrogate — full biophysical simulation runs inside both its gradient fitting and its SBI. So
nothing in §1–§1b is a comparison *against* the article; the article is the source of the forward
model and the inverse formulation reproduced in §4–§5. Its 90.6% is stimulation-response
prediction accuracy on macaque MEA recordings, which this repository does not attempt.
{device_section()}
## 8. What this run actually establishes

1. **The learned flow-map is not a speedup on this hardware.** Measured horizon-for-horizon it is
   {1 / sp_hi:.0f}–{1 / sp_lo:.0f}× slower than fine RK4 on {DEV}. Earlier numbers in this
   repository compared *one* surrogate coarse step against a *full* solver rollout; that ratio was
   mostly the horizon length ({M.get('horizon_steps','?')}) rather than a property of the method.
2. **The bottleneck is kernel launches, not arithmetic.** The rollout is a host-side loop issuing a
   handful of small kernels per coarse step; the solvers issue one kernel that loops on the device.
   That is why the gap *widens* on the GPU and why training is faster on the CPU. Fusing the
   rollout into a single kernel (or capturing it as a CUDA graph) is the fix, and it is a code
   change rather than a modelling one.
3. **The genuine advantages survive and are structural, not statistical.** The surrogate takes a
   coarse step at which explicit solvers diverge outright ({ndiv} configurations here returned
   NaN), and it inverts in closed form. Neither of those is a throughput claim.
4. **Accuracy is the binding constraint.** At NRMSE {P['sur_err']:.3f} the flow-map is roughly
   three orders of magnitude coarser than RK4 at four substeps. Until that closes, the speed
   question is secondary.

**Honest notes.** Resolving HH *spikes* needs dt ≲ 0.05 ms for any fixed-step solver. Rosenbrock's
advantage is unconditional stability on the stiff-but-smooth axial diffusion of the cable, not on
the spike itself. The deterministic P(I) is near-step (all-or-none spikes); the article widens it
with a current-noise model and Bayesian SBI — the natural extension on top of the differentiable
forward model here. All timings are single-run minimum-of-8 on one machine
(GTX 1660 Ti Max-Q, sm_75, native Windows, no WSL2) and no error bars are reported.
"""
    with open(os.path.join(HERE, "COMPARISON.md"), "w", encoding="utf-8") as f:
        f.write(md)


def main():
    fig_forward(); fig_pareto(); fig_rollout(); fig_control(); fig_cable_ei(); fig_inverse()
    if MC:
        fig_devices()
    write_report()
    print("wrote figures + COMPARISON.md to", HERE)


if __name__ == "__main__":
    main()
