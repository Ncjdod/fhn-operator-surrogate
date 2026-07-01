"""Benchmark: learned flow-map vs standard RK4 integration of FHN, across time scales."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flowmap_model as F
from operator_data import simulate_batch, random_profile, KINDS

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "plots", "results")
os.makedirs(ART, exist_ok=True)
DT = 0.05
PERIOD = 37.0
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140})


def load_flow(path):
    o = pickle.load(open(path, "rb"))
    params = jax.tree_util.tree_map(jnp.asarray, o["params"])
    c = o["cfg"]
    cfg = F.FlowConfig(d=c["d"], hidden=tuple(c["hidden"]), stride=c["stride"])
    return params, cfg, float(o["dt"])


def limit_cycle_state(u=0.7, t_warm=400.0):
    """One point on the attracting limit cycle at constant current u."""
    t = np.arange(0.0, t_warm + DT * 0.5, DT).astype(np.float32)
    U = np.full((1, len(t)), u, np.float32)
    ys = np.asarray(simulate_batch(jnp.asarray([[-1.0, -0.5]], np.float32), jnp.asarray(t), jnp.asarray(U)))[0]
    return ys[-1]


def make_currents(B, t_fine, seed):
    rng = np.random.default_rng(seed)
    return np.stack([random_profile(rng, t_fine, KINDS[i % len(KINDS)]) for i in range(B)], 0).astype(np.float32)


def timed(fn, *a, reps=10):
    fn(*a).block_until_ready()
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*a).block_until_ready()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_one(T, B, params, cfg, x0, seed):
    stride = cfg.stride
    Sf = int(round(T / DT))
    Sf -= Sf % stride
    t_fine = (np.arange(Sf + 1) * DT).astype(np.float32)
    U_fine = make_currents(B, t_fine, seed)
    t_coarse = t_fine[::stride]
    U_coarse = U_fine[:, ::stride]
    x0b = jnp.asarray(np.repeat(x0[None], B, 0))
    tf = jnp.asarray(t_fine); tc = jnp.asarray(t_coarse)
    Uf = jnp.asarray(U_fine); Uc = jnp.asarray(U_coarse)
    n_coarse = U_coarse.shape[1] - 1

    rk4_fine = jax.jit(lambda y0, U: simulate_batch(y0, tf, U))
    rk4_coarse = jax.jit(lambda y0, U: simulate_batch(y0, tc, U))
    flow = jax.jit(lambda y0, U: F.rollout(params, cfg, y0, U))

    reps = 12 if n_coarse < 5000 else 5
    t_fine_s = timed(rk4_fine, x0b, Uf, reps=reps)
    t_coarse_s = timed(rk4_coarse, x0b, Uc, reps=reps)
    t_flow_s = timed(flow, x0b, Uc, reps=reps)

    yf = np.asarray(rk4_fine(x0b, Uf))[:, ::stride]
    yc = np.asarray(rk4_coarse(x0b, Uc))
    yh = np.asarray(flow(x0b, Uc))
    sx = yf.reshape(-1, 2).std(0) + 1e-6
    nrmse_flow = float(np.sqrt((((yh - yf) / sx) ** 2).mean()))
    nrmse_coarse = float(np.sqrt((((yc - yf) / sx) ** 2).mean()))

    return {"T": T, "cycles": T / PERIOD, "n_fine": Sf, "n_coarse": n_coarse,
            "t_fine": t_fine_s, "t_coarse": t_coarse_s, "t_flow": t_flow_s,
            "speedup_vs_fine": t_fine_s / t_flow_s,
            "speedup_vs_coarse": t_coarse_s / t_flow_s,
            "nrmse_flow": nrmse_flow, "nrmse_coarse": nrmse_coarse}


def figures(rows, dev, B):
    T = np.array([r["T"] for r in rows])
    tf = np.array([r["t_fine"] for r in rows]) * 1e3
    tc = np.array([r["t_coarse"] for r in rows]) * 1e3
    th = np.array([r["t_flow"] for r in rows]) * 1e3
    sp = np.array([r["speedup_vs_fine"] for r in rows])
    ef = np.array([r["nrmse_flow"] for r in rows])
    ec = np.array([r["nrmse_coarse"] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.loglog(T, tf, "C3-o", lw=1.8, label=r"RK4 fine ($dt=0.05$, reference accuracy)")
    ax.loglog(T, tc, "C0-s", lw=1.6, label=r"RK4 coarse ($\Delta=0.2$)")
    ax.loglog(T, th, "C2-^", lw=1.8, label=r"flow-map ($\Delta=0.2$)")
    ax.set_xlabel("integration horizon  T  (time units)")
    ax.set_ylabel("wall-clock per batch (ms)")
    ax.set_title(f"Wall-clock vs time scale  ({dev}, batch={B})")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / PERIOD, lambda x: x * PERIOD))
    sec.set_xlabel("cycles")
    fig.tight_layout(); fig.savefig(f"{ART}/bench_walltime.png"); plt.close(fig)
    print("bench_walltime")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].semilogx(T, sp, "C2-o", lw=1.9)
    ax[0].axhline(1.0, color="0.5", ls="--", lw=1, label="parity (=numerical integration)")
    ax[0].set_xlabel("integration horizon  T  (time units)")
    ax[0].set_ylabel(r"speedup  $t_{\mathrm{RK4\,fine}}/t_{\mathrm{flow\text{-}map}}$")
    ax[0].set_title("Flow-map speedup over reference RK4")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)
    for x, y in zip(T, sp):
        ax[0].annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0, 6),
                       ha="center", fontsize=7)

    ax[1].loglog(T, ef, "C2-^", lw=1.8, label=r"flow-map ($\Delta=0.2$)")
    ax[1].loglog(T, ec, "C0-s", lw=1.6, label=r"RK4 coarse ($\Delta=0.2$)")
    ax[1].set_xlabel("integration horizon  T  (time units)")
    ax[1].set_ylabel("NRMSE vs RK4-fine reference")
    ax[1].set_title(r"Accuracy at the coarse step $\Delta=0.2$")
    ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)
    fig.suptitle("How much faster is the flow-map than numerical integration, and at what accuracy",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/bench_speedup_accuracy.png"); plt.close(fig)
    print("bench_speedup_accuracy")


def main():
    p = argparse.ArgumentParser(description="Flow-map vs RK4 speed benchmark")
    p.add_argument("--model", default=os.path.join(HERE, "data", "flowmap.pkl"))
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--horizons", type=float, nargs="+",
                   default=[10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0])
    args = p.parse_args()

    params, cfg, _ = load_flow(args.model)
    dev = str(jax.devices()[0].platform).upper()
    print(f"device={dev}  batch={args.batch}  stride={cfg.stride} (Delta={cfg.stride*DT})")
    x0 = limit_cycle_state()

    rows = [bench_one(T, args.batch, params, cfg, x0, seed=100 + i)
            for i, T in enumerate(args.horizons)]

    print(f"\n{'T':>7} {'cycles':>7} {'n_fine':>8} {'n_coarse':>9} "
          f"{'RK4fine':>9} {'RK4coar':>9} {'flowmap':>9} {'speedup':>8} {'e_flow':>7} {'e_coar':>7}")
    for r in rows:
        print(f"{r['T']:7.0f} {r['cycles']:7.1f} {r['n_fine']:8d} {r['n_coarse']:9d} "
              f"{r['t_fine']*1e3:8.2f}m {r['t_coarse']*1e3:8.2f}m {r['t_flow']*1e3:8.2f}m "
              f"{r['speedup_vs_fine']:7.2f}x {r['nrmse_flow']:7.3f} {r['nrmse_coarse']:7.3f}")
    print(f"\nmedian speedup vs RK4-fine: {np.median([r['speedup_vs_fine'] for r in rows]):.2f}x")
    figures(rows, dev, args.batch)
    print("FIGURES ->", ART)


if __name__ == "__main__":
    main()
