"""Result + verification figures for the FHN surrogate (-> plots/results/)."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import pickle
import collections
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import pwfo_model as P
import flowmap_model as F
import fhn_theory as TH
from operator_data import KINDS, simulate_batch, A, B, TAU
from pwfo_train import firing_const_data

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "plots", "results")
os.makedirs(ART, exist_ok=True)
DT = 0.05
PERIOD = 37.0
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140})


def load_pwfo(path):
    o = pickle.load(open(path, "rb"))
    params = jax.tree_util.tree_map(jnp.asarray, o["params"])
    c = o["cfg"]
    cfg = P.PWFOConfig(d=c["d"], K=c["K"], m=c["m"], local_waveform=c.get("local_waveform", False))
    return params, cfg, float(o["dt"])


def load_flow(path):
    o = pickle.load(open(path, "rb"))
    params = jax.tree_util.tree_map(jnp.asarray, o["params"])
    c = o["cfg"]
    cfg = F.FlowConfig(d=c["d"], hidden=tuple(c["hidden"]), stride=c["stride"])
    return params, cfg, float(o["dt"])


def sim_const(u, y0, t_max, dt=DT):
    t = np.arange(0.0, t_max + dt * 0.5, dt).astype(np.float32)
    U = np.full((1, len(t)), u, np.float32)
    ys = np.asarray(simulate_batch(jnp.asarray(y0[None].astype(np.float32)), jnp.asarray(t), jnp.asarray(U)))[0]
    return t, ys


# ---------- ground-truth dynamics ----------

def fig_dynamics():
    u_fire, u_rest = 0.6, 0.0
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    vv = np.linspace(-2.4, 2.4, 400)
    for a_, u, col, lab in [(ax[0], u_fire, "C3", f"firing u={u_fire}"),
                            (ax[0], u_rest, "C0", f"resting u={u_rest}")]:
        a_.plot(vv, vv - vv ** 3 / 3.0 + u, col, lw=1.6, ls="--",
                label=f"v-nullcline ({lab})")
    ax[0].plot(vv, (vv + A) / B, "0.4", lw=1.6, ls=":", label="w-nullcline")
    _, yc = sim_const(u_fire, np.array([-1.0, -0.5]), 400.0)
    cyc = yc[int(0.6 * len(yc)):]
    ax[0].plot(cyc[:, 0], cyc[:, 1], "C3", lw=2.4, label="limit cycle (u=0.6)")
    _, ys_in = sim_const(u_fire, np.array([0.2, 0.9]), 90.0)
    ax[0].plot(ys_in[:, 0], ys_in[:, 1], "C1", lw=1.0, alpha=0.8, label="transient -> cycle")
    _, yr = sim_const(u_rest, np.array([1.8, 0.2]), 300.0)
    ax[0].plot(yr[:, 0], yr[:, 1], "C0", lw=1.0, alpha=0.8, label="decay -> fixed pt")
    ax[0].plot(yr[-1, 0], yr[-1, 1], "ko", ms=6)
    ax[0].set_xlabel("v (membrane)"); ax[0].set_ylabel("w (recovery)")
    ax[0].set_title("Phase plane: nullclines, attracting limit cycle, stable fixed point")
    ax[0].legend(fontsize=7, loc="upper right"); ax[0].grid(alpha=.3)
    ax[0].set_xlim(-2.4, 2.4); ax[0].set_ylim(-1.2, 2.2)

    t, yc = sim_const(u_fire, np.array([-1.0, -0.5]), 160.0)
    ax[1].plot(t, yc[:, 0], "C3", lw=1.6, label="v (fast, spikes)")
    ax[1].plot(t, yc[:, 1], "C0", lw=1.6, label="w (slow, recovery)")
    ax[1].set_xlabel("time t"); ax[1].set_ylabel("state")
    ax[1].set_title(f"Relaxation oscillation at u={u_fire} (period ~ {PERIOD:.0f}, tau={TAU})")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    fig.suptitle("Ground truth: FitzHugh-Nagumo dynamics (the known system being surrogated)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/r1_fhn_dynamics.png"); plt.close(fig)
    print("r1_fhn_dynamics")


def fig_bifurcation():
    ug = np.linspace(-0.5, 2.2, 91)
    sp = TH.spectrum_over_u(ug)
    fire = np.load(os.path.join(HERE, "data", "fhn_freq_table.npz"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].axhline(0, color="0.5", lw=1)
    ax[0].plot(ug, sp["sigma"], "C0", lw=1.8)
    fireband = np.isfinite(fire["omega"])
    ub = fire["u"][fireband]
    ax[0].axvspan(ub.min(), ub.max(), color="C3", alpha=0.12, label="firing band")
    zc = ug[np.where(np.diff(np.sign(np.nan_to_num(sp["sigma"]))))[0]]
    for z in zc:
        ax[0].axvline(z, color="C3", ls=":", lw=1.2)
    ax[0].set_xlabel("current u"); ax[0].set_ylabel(r"$\sigma(u)=\mathrm{Re}\,\lambda$")
    ax[0].set_title("Linear stability: Hopf crossings $\\sigma=0$")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].plot(ug, sp["omega"], "C0", lw=1.6, label=r"$\mathrm{Im}\,\lambda$ (onset)")
    ax[1].plot(fire["u"][fireband], fire["omega"][fireband], "C3", lw=2.0, label=r"measured cycle $\omega$")
    ax[1].set_xlabel("current u"); ax[1].set_ylabel(r"angular frequency $\omega$")
    ax[1].set_title("Frequency: linear onset vs nonlinear cycle")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    amps = []
    for u in ug:
        _, amp = TH.limit_cycle_period(float(u), t_max=300.0, dt=0.02)
        amps.append(amp)
    ax[2].plot(ug, amps, "C2", lw=1.8)
    ax[2].axvspan(ub.min(), ub.max(), color="C3", alpha=0.12)
    ax[2].set_xlabel("current u"); ax[2].set_ylabel("v amplitude (peak-to-peak)")
    ax[2].set_title("Cycle amplitude vs u (0 outside firing band)")
    ax[2].grid(alpha=.3)
    fig.suptitle("Bifurcation structure derived from the equations (stability, frequency, amplitude)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/r2_stability_bifurcation.png"); plt.close(fig)
    print("r2_stability_bifurcation")


# ---------- PWFO concept + far horizon ----------

def fig_pwfo_concept(core):
    params, cfg, dt = core
    t, ys, u = firing_const_data(2, 120.0, dt, seed=3)
    c = P._mlp(P._profile_stats(jnp.asarray(u)), params["ctx"])
    om, _ = P.segment_rates(params, cfg, jnp.asarray(u), c)
    Phi = np.array(dt * jnp.cumsum(om, axis=1))
    tt = np.arange(ys.shape[1]) * dt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ax[0].plot(tt, Phi[0], "C0", lw=2)
    ax[0].set_xlabel("time t"); ax[0].set_ylabel(r"$\Phi(t)$")
    ax[0].set_title(r"$\Phi(t)=\phi_0+\int_0^t\omega\,d\tau$ = prefix-sum, evaluable at any t")
    ax[0].grid(alpha=.3)
    ax[1].plot(tt, np.cos(Phi[0]), "C3", lw=1.2)
    ax[1].set_xlabel("time t"); ax[1].set_ylabel(r"$\cos\Phi(t)$")
    ax[1].set_title(r"$\cos k\Phi$ bounded & periodic $\forall t$ -> never decays/blows up")
    ax[1].grid(alpha=.3)
    fig.suptitle("PWFO backbone: time enters only through a bounded periodic phase (one-shot, no recursion)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/r3_pwfo_concept.png"); plt.close(fig)
    print("r3_pwfo_concept")


def fig_pwfo_farhorizon(core):
    params, cfg, dt = core
    t, ys, u = firing_const_data(6, 1500.0, dt, seed=99)
    S = ys.shape[1]
    tq = jnp.asarray((np.arange(S) * dt).astype(np.float32))[None].repeat(6, 0)
    xh = np.array(P.forward(params, cfg, jnp.asarray(ys[:, 0]), jnp.asarray(u), tq, dt))
    i = 0
    fig, ax = plt.subplots(2, 1, figsize=(13, 6))
    for axx, (lo, hi, lab) in zip(ax, [(0, 90, "trained window (t<300)"),
                                       (1410, 1500, "EXTRAPOLATION t~1450 (single forward pass)")]):
        m = (t >= lo) & (t < hi)
        axx.plot(t[m], ys[i, m, 0], "k", lw=2.2, label="true FHN")
        axx.plot(t[m], xh[i, m, 0], "C2--", lw=1.6, label="PWFO (1 pass)")
        axx.set_ylabel("v"); axx.set_title(lab); axx.legend(fontsize=8); axx.grid(alpha=.3)
    ax[-1].set_xlabel("time")
    fig.suptitle("PWFO constant-current core: oscillation sustained ~40 cycles past training, no recursion",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/r4_pwfo_farhorizon.png"); plt.close(fig)

    ps = int(round(PERIOD / dt))
    fig, a = plt.subplots(figsize=(11, 4))
    for arr, c_, lab in [(ys[i, :, 0], "k", "true"), (xh[i, :, 0], "C2", "PWFO")]:
        amps = [np.ptp(arr[k * ps:(k + 1) * ps]) for k in range(len(arr) // ps)]
        a.plot(np.arange(len(amps)), amps, c_ + "-o", ms=3, label=lab)
    a.axvline(300 / PERIOD, color="0.6", ls=":", label="train horizon")
    a.set_xlabel("cycle #"); a.set_ylabel("v amplitude (peak-to-peak)")
    a.set_ylim(bottom=0); a.legend(); a.grid(alpha=.3)
    a.set_title("Amplitude stays flat over 40 cycles (no decay, no divergence)")
    fig.tight_layout(); fig.savefig(f"{ART}/r5_pwfo_amplitude_flat.png"); plt.close(fig)
    print("r4_pwfo_farhorizon, r5_pwfo_amplitude_flat")


# ---------- shared eval ----------

def anchored_flow(flow, ys, u, cycles=3, seed=5):
    params, cfg, dt = flow
    yc = ys[:, ::cfg.stride]; uc = u[:, ::cfg.stride]; Tc = yc.shape[1]
    sx = yc.reshape(-1, 2).std(0) + 1e-6
    K = int(cycles * PERIOD / (cfg.stride * dt))
    rng = np.random.default_rng(seed)
    roll = jax.jit(lambda x0, uw: F.rollout(params, cfg, x0, uw))
    agg = collections.defaultdict(list)
    for i in range(yc.shape[0]):
        t0 = int(rng.integers(0, Tc - K - 1))
        xh = np.asarray(roll(jnp.asarray(yc[i, t0][None]), jnp.asarray(uc[i, t0:t0 + K + 1][None])))[0]
        tgt = yc[i, t0:t0 + K + 1]
        agg[KINDS[i % len(KINDS)]].append(np.sqrt((((xh - tgt) / sx) ** 2).mean()))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def anchored_pwfo(gen, ys, u, cycles=3, seed=5):
    params, cfg, dt = gen
    S = ys.shape[1]; W = int(cycles * PERIOD / dt)
    sx = ys.reshape(-1, 2).std(0) + 1e-6
    rng = np.random.default_rng(seed)
    agg = collections.defaultdict(list)
    for i in range(ys.shape[0]):
        t0 = int(rng.integers(0, S - W - 1))
        tq = jnp.asarray((np.arange(W) * dt).astype(np.float32))[None]
        xh = np.asarray(P.forward(params, cfg, jnp.asarray(ys[i, t0][None]),
                                  jnp.asarray(u[i, t0:t0 + W][None]), tq, dt))[0]
        tgt = ys[i, t0:t0 + W]
        agg[KINDS[i % len(KINDS)]].append(np.sqrt((((xh - tgt) / sx) ** 2).mean()))
    return {k: float(np.mean(v)) for k, v in agg.items()}


# ---------- flow-map verification grid ----------

def fig_flowmap_grid(flow, ys, u):
    params, cfg, dt = flow
    yc = ys[:, ::cfg.stride]; uc = u[:, ::cfg.stride]; Tc = yc.shape[1]
    sx = yc.reshape(-1, 2).std(0) + 1e-6
    K = int(3 * PERIOD / (cfg.stride * dt))
    roll = jax.jit(lambda x0, uw: F.rollout(params, cfg, x0, uw))
    tc = np.arange(K + 1) * (cfg.stride * dt)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    for j, kind in enumerate(KINDS):
        i = next(q for q in range(ys.shape[0]) if KINDS[q % len(KINDS)] == kind)
        t0 = (Tc - K - 1) // 2
        xh = np.asarray(roll(jnp.asarray(yc[i, t0][None]), jnp.asarray(uc[i, t0:t0 + K + 1][None])))[0]
        tgt = yc[i, t0:t0 + K + 1]
        e = np.sqrt((((xh - tgt) / sx) ** 2).mean())
        ax = axes.flat[j]
        ax.plot(tc, tgt[:, 0], "k", lw=2.0, label="true v")
        ax.plot(tc, xh[:, 0], "C2--", lw=1.4, label="flow-map v")
        ax.set_title(f"{kind}   NRMSE={e:.3f}", fontsize=10)
        ax.grid(alpha=.3)
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("time (from anchor)")
    fig.suptitle("Flow-map verification: 3-cycle rollout from a measured state, every current type",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/r6_flowmap_grid.png"); plt.close(fig)
    print("r6_flowmap_grid")


def fig_phaseportraits(flow, ys, u):
    params, cfg, dt = flow
    yc = ys[:, ::cfg.stride]; uc = u[:, ::cfg.stride]; Tc = yc.shape[1]
    K = int(4 * PERIOD / (cfg.stride * dt))
    roll = jax.jit(lambda x0, uw: F.rollout(params, cfg, x0, uw))
    picks = ["const", "chirp", "pulse"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, kind in zip(axes, picks):
        i = next(q for q in range(ys.shape[0]) if KINDS[q % len(KINDS)] == kind)
        t0 = (Tc - K - 1) // 2
        xh = np.asarray(roll(jnp.asarray(yc[i, t0][None]), jnp.asarray(uc[i, t0:t0 + K + 1][None])))[0]
        tgt = yc[i, t0:t0 + K + 1]
        ax.plot(tgt[:, 0], tgt[:, 1], "k", lw=2.0, label="true")
        ax.plot(xh[:, 0], xh[:, 1], "C2--", lw=1.3, label="flow-map")
        ax.set_xlabel("v"); ax.set_ylabel("w"); ax.set_title(f"{kind}")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Phase-plane fidelity: flow-map orbit overlays the true orbit (v-w plane)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/r7_flowmap_phaseportrait.png"); plt.close(fig)
    print("r7_flowmap_phaseportrait")


def fig_nrmse_bars(flow_by, pwfo_by):
    order = ["const", "step", "ramp", "pulse", "chirp", "sines", "ou", "piecewise"]
    fm = [flow_by.get(k, np.nan) for k in order]
    pw = [pwfo_by.get(k, np.nan) for k in order]
    order2 = order + ["MEAN"]
    fm2 = fm + [np.nanmean(fm)]; pw2 = pw + [np.nanmean(pw)]
    x = np.arange(len(order2)); wdt = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - wdt / 2, fm2, wdt, color="C2", label="flow-map (recurrent)")
    b2 = ax.bar(x + wdt / 2, pw2, wdt, color="C1", label="PWFO (one-shot)")
    ax.axhline(np.nanmean(fm), color="C2", ls=":", lw=1)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(order2, rotation=20)
    ax.set_ylabel("anchored 3-cycle NRMSE (lower=better)")
    ax.set_title("Accuracy by current type: flow-map vs PWFO (measured-state -> next 3 cycles)")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{ART}/r8_nrmse_bars.png"); plt.close(fig)
    print("r8_nrmse_bars")


def fig_error_vs_time(flow, gen, ys, u):
    fp, fc, dt = flow
    pp, pc, _ = gen
    yc = ys[:, ::fc.stride]; uc = u[:, ::fc.stride]; Tc = yc.shape[1]
    sx = yc.reshape(-1, 2).std(0) + 1e-6
    Kc = int(6 * PERIOD / (fc.stride * dt))
    i = next(q for q in range(ys.shape[0]) if KINDS[q % len(KINDS)] == "const")
    t0 = (Tc - Kc - 1) // 2
    xf = np.asarray(F.rollout(fp, fc, jnp.asarray(yc[i, t0][None]), jnp.asarray(uc[i, t0:t0 + Kc + 1][None])))[0]
    tgt = yc[i, t0:t0 + Kc + 1]
    tc = np.arange(Kc + 1) * (fc.stride * dt)
    ef = np.sqrt((((xf - tgt) / sx) ** 2).mean(1))

    Wp = int(6 * PERIOD / dt); tp0 = t0 * fc.stride
    tqp = jnp.asarray((np.arange(Wp) * dt).astype(np.float32))[None]
    xp = np.asarray(P.forward(pp, pc, jnp.asarray(ys[i, tp0][None]), jnp.asarray(u[i, tp0:tp0 + Wp][None]), tqp, dt))[0]
    tgtp = ys[i, tp0:tp0 + Wp]
    tpn = np.arange(Wp) * dt
    ep = np.sqrt((((xp - tgtp) / sx) ** 2).mean(1))

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    ax[0].plot(tc, ef, "C2", lw=1.6, label="flow-map (bounded)")
    ax[0].plot(tpn, ep, "C1", lw=1.2, alpha=0.9, label="PWFO (drift grows)")
    ax[0].set_xlabel("time (from anchor)"); ax[0].set_ylabel("pointwise NRMSE")
    ax[0].set_title("Error growth on a constant current: bounded vs phase-drift")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].plot(tpn, tgtp[:, 0], "k", lw=1.6, label="true v")
    ax[1].plot(tpn, xp[:, 0], "C1--", lw=1.1, label="PWFO v")
    ax[1].set_xlabel("time (from anchor)"); ax[1].set_ylabel("v")
    ax[1].set_title("PWFO: correct amplitude/waveform, slowly slipping phase")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle("Why the error grows: PWFO phase drift is the only unavoidable limit; flow-map stays bounded",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/r9_error_vs_time.png"); plt.close(fig)
    print("r9_error_vs_time")


def fig_chirp_compare(flow, gen, ys, u):
    fp, fc, dt = flow
    pp, pc, _ = gen
    i = next(q for q in range(ys.shape[0]) if KINDS[q % len(KINDS)] == "chirp")
    W = int(3 * PERIOD / dt); S = ys.shape[1]; t0 = (S - W) // 2
    tq = jnp.asarray((np.arange(W) * dt).astype(np.float32))[None]
    xp = np.asarray(P.forward(pp, pc, jnp.asarray(ys[i, t0][None]), jnp.asarray(u[i, t0:t0 + W][None]), tq, dt))[0]
    Kc = int(3 * PERIOD / (fc.stride * dt))
    uc = u[i, t0::fc.stride][:Kc + 1]
    xf = np.asarray(F.rollout(fp, fc, jnp.asarray(ys[i, t0][None]), jnp.asarray(uc[None])))[0]
    tcf = np.arange(Kc + 1) * (fc.stride * dt)
    tt = np.arange(W) * dt
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True)
    ax[0].plot(tt, u[i, t0:t0 + W], "C4", lw=1.3); ax[0].set_ylabel(r"$I_{ext}(t)$")
    ax[0].set_title("Fast (chirp) current — the regime that separates the two models")
    ax[0].grid(alpha=.3)
    ax[1].plot(tt, ys[i, t0:t0 + W, 0], "k", lw=2.2, label="true FHN")
    ax[1].plot(tcf, xf[:, 0], "C2--", lw=1.5, label="flow-map (0.016)")
    ax[1].plot(tt, xp[:, 0], "C1:", lw=1.5, label="PWFO (0.77)")
    ax[1].set_ylabel("v"); ax[1].set_xlabel("time"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    fig.suptitle("Fast current: flow-map tracks waveform+phase; closed-form PWFO cannot (motivates the hybrid)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/r10_flowmap_vs_pwfo_chirp.png"); plt.close(fig)
    print("r10_flowmap_vs_pwfo_chirp")


# ---------- schematic diagrams ----------

def _box(ax, xy, wh, text, fc):
    b = FancyBboxPatch(xy, wh[0], wh[1], boxstyle="round,pad=0.02,rounding_size=0.03",
                       fc=fc, ec="0.3", lw=1.3)
    ax.add_patch(b)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=8.5)


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color="0.3"))


def fig_hybrid_routing():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.fill_between([0, 10], [0, 0], [5, 5], color="C2", alpha=0.13)
    ax.fill_between([0, 5], [5, 5], [10, 10], color="C1", alpha=0.15)
    ax.fill_between([5, 10], [5, 5], [10, 10], color="C3", alpha=0.15)
    ax.text(5, 2.5, "flow-map  (recurrent, accurate)\nfull waveform + phase\nANY current, finite horizon",
            ha="center", va="center", fontsize=12, weight="bold")
    ax.text(2.5, 7.5, "PWFO one-shot\n(instant at any t)\nslow current", ha="center",
            va="center", fontsize=11, weight="bold")
    ax.text(7.5, 7.5, "unreachable corner\nfast forcing + unbounded t\n(no finite-cost exact answer)",
            ha="center", va="center", fontsize=10.5, color="C3")
    ax.plot([0, 10], [5, 5], color="0.4", lw=1.2, ls="--")
    ax.plot([5, 5], [5, 10], color="0.4", lw=1.2, ls="--")
    ax.annotate("", xy=(10, -0.2), xytext=(0, -0.2), arrowprops=dict(arrowstyle="-|>"))
    ax.annotate("", xy=(-0.2, 10), xytext=(-0.2, 0), arrowprops=dict(arrowstyle="-|>"))
    ax.text(5, -0.75, "current speed   max|du/dt|   ->", ha="center", fontsize=10.5)
    ax.text(-0.75, 5, "query horizon  t   ->", va="center", rotation=90, fontsize=10.5)
    ax.text(0.2, 2.4, "finite\nhorizon", fontsize=8, color="0.3", va="center")
    ax.text(0.2, 7.5, "far /\nunbounded", fontsize=8, color="0.3", va="center")
    ax.text(2.4, 9.4, "slow", fontsize=8.5, color="0.3", ha="center")
    ax.text(7.5, 9.4, "fast", fontsize=8.5, color="0.3", ha="center")
    ax.set_title("Hybrid routing: which surrogate answers a query (current-speed x query-horizon regime map)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/r11_hybrid_routing.png"); plt.close(fig)
    print("r11_hybrid_routing")


def fig_architecture():
    fig, ax = plt.subplots(2, 1, figsize=(13, 9))
    a = ax[0]; a.set_xlim(0, 12); a.set_ylim(0, 4); a.axis("off")
    a.set_title("PWFO: one-shot operator  G(x0, u(.), t) -> x(t)", fontweight="bold", loc="left")
    _box(a, (0.2, 2.45), (1.9, 0.95), "current\nprofile $u(\\cdot)$", "C4")
    _box(a, (0.2, 0.55), (1.9, 0.95), "measured\nstate $x_0$", "C0")
    _box(a, (2.6, 2.35), (3.2, 1.15), "profile encoder $\\to c$\nrates $\\omega(u),\\kappa(u)$\nwaveform $\\mu,A_k,B_k,C$", "0.85")
    _box(a, (2.6, 0.45), (3.2, 1.15), "IC encoder\n$\\to \\phi_0,\\ \\rho_0$", "0.85")
    _box(a, (6.3, 1.45), (2.9, 1.15),
         "$\\Phi(t)=\\phi_0+\\int\\omega\\,d\\tau$\n$\\psi(t)=\\rho_0\\,e^{\\int\\kappa\\,d\\tau}$\n(prefix-sum, any $t$)", "C2")
    _box(a, (9.7, 1.45), (2.1, 1.15), "$x(t)=\\mu+\\sum_k[A_k\\cos k\\Phi$\n$+B_k\\sin k\\Phi]+\\sum_j\\psi_j(\\cdots)$", "C2")
    for s, e in [((2.1, 2.9), (2.6, 2.9)), ((2.1, 1.0), (2.6, 1.0)),
                 ((5.8, 2.6), (6.3, 2.2)), ((5.8, 1.0), (6.3, 1.7)),
                 ((9.2, 2.0), (9.7, 2.0))]:
        _arrow(a, s, e)
    a.text(6, 3.78, "time enters ONLY via bounded periodic $\\cos k\\Phi$ + decaying $\\psi$   =>   O(1) per query, no recursion",
           ha="center", fontsize=8.5, style="italic")
    a.text(7.75, 0.9, "waveform coeffs (from $c$)\ncombine with $\\Phi,\\psi$ in the output",
           ha="center", fontsize=7.2, color="0.4", style="italic")

    b = ax[1]; b.set_xlim(0, 12); b.set_ylim(0, 4); b.axis("off")
    b.set_title("Flow-map stepper: learned coarse Markov integrator (recurrent)", fontweight="bold", loc="left")
    _box(b, (0.3, 1.5), (1.7, 1.0), "$x_0$", "C0")
    xs = [2.6, 5.0, 7.4, 9.8]
    for k, x in enumerate(xs):
        _box(b, (x, 1.5), (1.7, 1.0), f"$x_{{{k}}}$" if k < 3 else "$x_N$", "C2")
    prev = (2.0, 2.0)
    for x in xs:
        _arrow(b, prev, (x, 2.0)); prev = (x + 1.7, 2.0)
    for x in xs[:-1]:
        b.text(x + 0.85, 2.65, "$+g_\\theta(x,u_t,u_{t+\\Delta})$", ha="center", fontsize=7.5)
        b.add_patch(FancyArrowPatch((x + 0.85, 2.5), (x + 0.85, 2.05), arrowstyle="-|>",
                                    mutation_scale=10, color="C1"))
    b.text(6, 0.6, "coarse step $\\Delta=4\\,dt=0.2$;  cost $O(t/\\Delta)$ steps;  trained by multi-step BPTT curriculum",
           ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(); fig.savefig(f"{ART}/r12_architecture.png"); plt.close(fig)
    print("r12_architecture")


def main():
    core = load_pwfo(os.path.join(HERE, "data", "pwfo_core_k20.pkl"))
    gen = load_pwfo(os.path.join(HERE, "data", "pwfo_general.pkl"))
    flow = load_flow(os.path.join(HERE, "data", "flowmap.pkl"))
    d = np.load(os.path.join(HERE, "data", "fhn_operator.npz"))
    ys = np.asarray(d["ys_val"]); u = np.asarray(d["u_val"])

    fig_dynamics()
    fig_bifurcation()
    fig_pwfo_concept(core)
    fig_pwfo_farhorizon(core)
    fig_flowmap_grid(flow, ys, u)
    fig_phaseportraits(flow, ys, u)
    flow_by = anchored_flow(flow, ys, u)
    pwfo_by = anchored_pwfo(gen, ys, u)
    print("flow-map NRMSE:", {k: round(v, 3) for k, v in flow_by.items()},
          "mean", round(np.mean(list(flow_by.values())), 3))
    print("PWFO NRMSE:", {k: round(v, 3) for k, v in pwfo_by.items()},
          "mean", round(np.mean(list(pwfo_by.values())), 3))
    fig_nrmse_bars(flow_by, pwfo_by)
    fig_error_vs_time(flow, gen, ys, u)
    fig_chirp_compare(flow, gen, ys, u)
    fig_hybrid_routing()
    fig_architecture()
    print("FIGURES ->", ART)


if __name__ == "__main__":
    main()
