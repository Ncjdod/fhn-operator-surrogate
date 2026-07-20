"""Train the control-affine flow-map (FHN or HH) via multi-step BPTT curriculum.

Loss = peak-weighted standardized state MSE over a K-step rollout + a G-conditioning
keep-alive that keeps <G,G> above a floor so the closed-form inverse stays well-posed.
The affine-in-u structure plus ZOH currents that vary across nearby states identify the
drift F and the control sensitivity G without any separate supervision.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

import flowmap_affine as A


def train(yc, uc, cfg, g_true=None, steps=7000, batch=64, lr=2e-3, seed=0, stages=None, verbose=True):
    yc = jnp.asarray(yc); uc = jnp.asarray(uc)
    N, Tcp1, d = yc.shape
    Tc = uc.shape[1]
    mu = np.asarray(yc.reshape(-1, d).mean(0))
    sd = np.asarray(yc.reshape(-1, d).std(0)) + 1e-6
    sdj = jnp.asarray(sd)
    has_g = g_true is not None
    gt = jnp.asarray(g_true) if has_g else None

    params = A.init_affine(cfg, jax.random.PRNGKey(seed), x_mu=mu, x_sd=sd)
    sched = optax.cosine_decay_schedule(lr, steps, alpha=1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-5))
    st = opt.init(params)
    gfloor2 = (cfg.g_floor * 1.5) ** 2

    def make_step(K):
        K = int(min(K, Tc))

        def loss_fn(p, idx, t0):
            offu = t0[:, None] + jnp.arange(K)[None, :]
            offy = t0[:, None] + jnp.arange(K + 1)[None, :]
            u_win = jnp.take_along_axis(uc[idx], offu, axis=1)
            y_win = jnp.take_along_axis(yc[idx], offy[:, :, None].repeat(d, 2), axis=1)
            xh = A.rollout(p, cfg, y_win[:, 0], u_win)
            w = 1.0 + 3.0 * jnp.abs((y_win - y_win.mean(1, keepdims=True)) / sdj)
            fit = jnp.mean(w * ((xh - y_win) / sdj) ** 2)
            # G on the visited (stepped) states
            _, G = jax.vmap(lambda xx: A.FG(p, cfg, xx))(y_win[:, :-1].reshape(-1, d))
            gg = jnp.sum(G * G, axis=-1)
            cond = jnp.mean(jax.nn.relu(gfloor2 - gg))
            lg = 0.0
            if has_g:
                g_win = jnp.take_along_axis(gt[idx], offu[:, :, None].repeat(d, 2), axis=1)
                lg = jnp.mean(((G - g_win.reshape(-1, d)) / sdj) ** 2)
            return fit + 0.1 * cond + 1.0 * lg

        @jax.jit
        def stp(p, st, key):
            ki, kt = jax.random.split(key)
            idx = jax.random.randint(ki, (batch,), 0, N)
            t0 = jax.random.randint(kt, (batch,), 0, max(1, Tc - K))
            l, g = jax.value_and_grad(loss_fn)(p, idx, t0)
            upd, st = opt.update(g, st, p)
            return optax.apply_updates(p, upd), st, l
        return stp

    stages = stages or [(1, 0.15), (4, 0.15), (16, 0.20), (64, 0.25), (128, 0.25)]
    key = jax.random.PRNGKey(seed + 1)
    t0 = time.time(); i = 0
    for K, frac in stages:
        stp = make_step(K)
        for _ in range(int(frac * steps)):
            key, k = jax.random.split(key)
            params, st, l = stp(params, st, k)
            if verbose and i % max(1, steps // 25) == 0:
                print(f"step {i:5d} K={int(min(K,Tc)):3d} | loss {float(l):.4f} ({time.time()-t0:.0f}s)", flush=True)
            i += 1
    return params, {"mu": mu, "sd": sd}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Train control-affine flow-map")
    p.add_argument("--data", default=os.path.join(here, "data", "hh_operator.npz"))
    p.add_argument("--out", default=os.path.join(here, "data", "affine_hh.pkl"))
    p.add_argument("--steps", type=int, default=7000)
    p.add_argument("--g-floor", type=float, default=None)
    args = p.parse_args()

    dd = np.load(args.data)
    yc, uc = dd["ys_train"], dd["u_train"]
    g_true = dd["g_train"] if "g_train" in dd.files else None
    d = yc.shape[-1]
    D = float(dd["D"]) if "D" in dd else float(dd["dt"]) * int(dd["stride"])
    g_floor = args.g_floor if args.g_floor is not None else max(0.05, 0.3 * D)
    cfg = A.AffineConfig(d=d, hidden=(128, 128), stride=int(dd["stride"]), g_floor=g_floor)
    print(f"train affine d={d} D={D} ms  g_floor={g_floor:.3f}  data {yc.shape}  L_G={'on' if g_true is not None else 'off'}")
    params, extra = train(yc, uc, cfg, g_true=g_true, steps=args.steps)
    neuron = str(dd["neuron"]) if "neuron" in dd.files else "hh_model"
    with open(args.out, "wb") as f:
        pickle.dump({"params": jax.tree_util.tree_map(np.array, params),
                     "cfg": {"d": d, "hidden": cfg.hidden, "stride": cfg.stride,
                             "g_floor": g_floor, "v_chan": cfg.v_chan},
                     "D": D, "dt": float(dd["dt"]), "neuron": neuron, **extra}, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
