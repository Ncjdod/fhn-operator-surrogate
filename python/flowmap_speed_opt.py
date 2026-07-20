"""Optimized-inference benchmark: large-stride flow-maps vs RK4, exact times + accuracy."""
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

import flowmap_model as F0
import flowmap_fast as FF
from operator_data import simulate_batch, random_profile, KINDS

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "plots", "results")
os.makedirs(ART, exist_ok=True)
DT = 0.05
PERIOD = 37.0
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140})


def load_old(path):
    o = pickle.load(open(path, "rb")); c = o["cfg"]
    return jax.tree_util.tree_map(jnp.asarray, o["params"]), F0.FlowConfig(d=c["d"], hidden=tuple(c["hidden"]), stride=c["stride"])


def load_fast(path):
    o = pickle.load(open(path, "rb")); c = o["cfg"]
    return jax.tree_util.tree_map(jnp.asarray, o["params"]), FF.FastConfig(d=c["d"], hidden=tuple(c["hidden"]), stride=c["stride"], n_samp=c["n_samp"])


def lc_state(u=0.7, t_warm=400.0):
    t = np.arange(0.0, t_warm + DT * 0.5, DT).astype(np.float32)
    U = np.full((1, len(t)), u, np.float32)
    return np.asarray(simulate_batch(jnp.asarray([[-1.0, -0.5]], np.float32), jnp.asarray(t), jnp.asarray(U)))[0, -1]


def timed(fn, *a, reps=12):
    fn(*a).block_until_ready()
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fn(*a).block_until_ready(); best = min(best, time.perf_counter() - t0)
    return best


def bench(T, B, old, fasts, x0, seed):
    Sf = int(round(T / DT)); Sf -= Sf % 16
    t_fine = (np.arange(Sf + 1) * DT).astype(np.float32)
    U = np.stack([random_profile(np.random.default_rng(seed + i), t_fine, KINDS[i % len(KINDS)]) for i in range(B)], 0).astype(np.float32)
    x0b = jnp.asarray(np.repeat(x0[None], B, 0))
    tf = jnp.asarray(t_fine); Uf = jnp.asarray(U)
    rk4_fine = jax.jit(lambda y, u: simulate_batch(y, tf, u))
    t_ref = timed(rk4_fine, x0b, Uf)
    yf = np.asarray(rk4_fine(x0b, Uf))
    sx = yf.reshape(-1, 2).std(0) + 1e-6
    row = {"T": T, "n_fine": Sf, "t_fine_ms": t_ref * 1e3, "methods": {}}

    for name, stride in [("coarseRK4_0.2", 4), ("coarseRK4_0.4", 8), ("coarseRK4_0.8", 16)]:
        tc = jnp.asarray(t_fine[::stride]); Uc = jnp.asarray(U[:, ::stride])
        rk = jax.jit(lambda y, u: simulate_batch(y, tc, u))
        t = timed(rk, x0b, Uc)
        yc = np.asarray(rk(x0b, Uc)); ref = yf[:, ::stride]
        e = float(np.sqrt((((yc - ref) / sx) ** 2).mean()))
        row["methods"][name] = {"ms": t * 1e3, "speedup": t_ref / t, "nrmse": e, "steps": Uc.shape[1] - 1}

    p0, c0 = old
    uc0 = jnp.asarray(U[:, ::c0.stride])
    roll0 = jax.jit(lambda y, u: F0.rollout(p0, c0, y, u))
    t = timed(roll0, x0b, uc0)
    yh = np.asarray(roll0(x0b, uc0)); ref = yf[:, ::c0.stride]
    e = float(np.sqrt((((yh - ref) / sx) ** 2).mean()))
    row["methods"][f"flowmap_0.2"] = {"ms": t * 1e3, "speedup": t_ref / t, "nrmse": e, "steps": uc0.shape[1] - 1}

    for tag, (pf, cf) in fasts.items():
        us = jnp.asarray(FF.build_samples(U, cf.stride, cf.n_samp))
        roll = jax.jit(lambda y, u: FF.rollout(pf, cf, y, u))
        t = timed(roll, x0b, us)
        yh = np.asarray(roll(x0b, us)); ref = yf[:, ::cf.stride][:, :us.shape[1] + 1]
        m = min(yh.shape[1], ref.shape[1])
        e = float(np.sqrt((((yh[:, :m] - ref[:, :m]) / sx) ** 2).mean()))
        row["methods"][tag] = {"ms": t * 1e3, "speedup": t_ref / t, "nrmse": e, "steps": us.shape[1]}
    return row


def figure(rows):
    Ts = [r["T"] for r in rows]
    names = ["flowmap_0.2", "flowmap_0.4", "flowmap_0.8"]
    cols = {"flowmap_0.2": "C0", "flowmap_0.4": "C1", "flowmap_0.8": "C2"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for nm in names:
        sp = [r["methods"].get(nm, {}).get("speedup", np.nan) for r in rows]
        ax[0].plot(Ts, sp, "-o", color=cols[nm], lw=1.8, label=nm.replace("flowmap_", "flow-map Δ="))
    ax[0].axhline(1, color="0.5", ls="--", lw=1, label="fine RK4 parity")
    ax[0].axhline(2, color="C3", ls=":", lw=1.2, label="2x target")
    ax[0].set_xscale("log"); ax[0].set_xlabel("horizon T (time units)")
    ax[0].set_ylabel("speedup vs fine RK4"); ax[0].set_title("Larger stride → more steps removed → more speedup")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)

    r = rows[len(rows) // 2]
    ds = [0.2, 0.4, 0.8]
    fm = [r["methods"][f"flowmap_{d}"]["nrmse"] for d in ds]
    rk = [r["methods"][f"coarseRK4_{d}"]["nrmse"] for d in ds]
    ax[1].plot(ds, fm, "C2-^", lw=1.9, label="flow-map (learned)")
    ax[1].plot(ds, rk, "C0-s", lw=1.6, label="coarse RK4 (classical)")
    ax[1].set_xlabel("coarse step Δ"); ax[1].set_ylabel(f"NRMSE vs fine RK4 (T={r['T']:.0f})")
    ax[1].set_title("Accuracy at each step: learned map holds where RK4 breaks")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.suptitle("Optimized flow-map inference: larger-stride distillation pushes speedup past 2x",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/bench_stride_speedup.png"); plt.close(fig)
    print("bench_stride_speedup")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--horizons", type=float, nargs="+", default=[30.0, 100.0, 300.0, 1000.0])
    args = p.parse_args()
    dev = str(jax.devices()[0].platform).upper()
    old = load_old(os.path.join(HERE, "data", "flowmap.pkl"))
    fasts = {}
    for st, tag in [(8, "flowmap_0.4"), (16, "flowmap_0.8")]:
        pth = os.path.join(HERE, "data", f"flowmap_fast_s{st}.pkl")
        if os.path.exists(pth):
            fasts[tag] = load_fast(pth)
    print(f"device={dev} batch={args.batch}  models: flowmap_0.2 + {list(fasts)}")
    x0 = lc_state()
    rows = [bench(T, args.batch, old, fasts, x0, seed=200 + i) for i, T in enumerate(args.horizons)]
    for r in rows:
        print(f"\nT={r['T']:.0f}  fine RK4 {r['t_fine_ms']:.2f} ms ({r['n_fine']} steps)")
        for nm, m in r["methods"].items():
            print(f"   {nm:16s} {m['ms']:8.2f} ms  {m['speedup']:6.2f}x  NRMSE {m['nrmse']:.3f}  ({m['steps']} steps)")
    figure(rows)
    print("FIGURE ->", ART)


if __name__ == "__main__":
    main()
