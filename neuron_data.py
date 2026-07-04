"""Stiff neuron dataset on a COARSE grid with zero-order-hold currents (model-agnostic).

Works for any control-affine neuron module exposing f, simulate_batch, random_init,
U_LO/U_HI, FIRE_BAND, D_STATE (hh_model, multichan_model, ...). Truth is fine-RK4;
states subsampled to the coarse grid D=stride*dt; currents piecewise-constant per coarse
step (ZOH). Also stores the finite-difference control sensitivity g_true=dphi_D/du.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import importlib
import numpy as np
import jax
import jax.numpy as jnp


def coarse_current(rng, Tc, kind, M):
    lo, hi = M.FIRE_BAND
    if kind == "const":
        return np.full(Tc, rng.uniform(lo, hi))
    if kind == "step":
        a, b = rng.uniform(M.U_LO, lo + 2), rng.uniform(lo, hi)
        k = rng.integers(int(Tc * 0.15), int(Tc * 0.6)); u = np.full(Tc, a); u[k:] = b; return u
    if kind == "ramp":
        return np.linspace(rng.uniform(M.U_LO, lo), rng.uniform(lo, hi), Tc)
    if kind == "pulse":
        u = np.full(Tc, rng.uniform(M.U_LO, lo)); w = max(1, int(rng.uniform(2, 8)))
        per = rng.integers(w + 3, w + 20); h = rng.uniform(lo, hi)
        for s in range(0, Tc, per):
            u[s:s + w] = h
        return u
    if kind == "ou":
        theta, mu, sig = 0.05, rng.uniform(lo, hi), rng.uniform(2.0, 6.0)
        u = np.empty(Tc); u[0] = mu; z = rng.standard_normal(Tc)
        for i in range(1, Tc):
            u[i] = u[i - 1] + theta * (mu - u[i - 1]) + sig * z[i]
        return u
    u = np.empty(Tc); i = 0
    while i < Tc:
        seg = int(rng.uniform(3, 20)); u[i:i + seg] = rng.uniform(M.U_LO, hi); i += seg
    return u


KINDS = ["const", "step", "ramp", "pulse", "ou", "piecewise"]


def phi_D_factory(M, dt, nsub):
    def phi(x, u):
        def body(y, _):
            k1 = M.f(y, u); k2 = M.f(y + 0.5 * dt * k1, u)
            k3 = M.f(y + 0.5 * dt * k2, u); k4 = M.f(y + dt * k3, u)
            return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None
        y, _ = jax.lax.scan(body, x, None, length=nsub)
        return y
    return phi


def generate(M, n, Tc, stride, dt, seed):
    rng = np.random.default_rng(seed)
    Uc = np.stack([coarse_current(rng, Tc, KINDS[i % len(KINDS)], M) for i in range(n)], 0)
    Uc = np.clip(Uc, M.U_LO, M.U_HI).astype(np.float32)
    Uf = np.repeat(Uc, stride, axis=1)
    Uf = np.concatenate([Uf, Uf[:, -1:]], axis=1).astype(np.float32)
    t_fine = (np.arange(Uf.shape[1]) * dt).astype(np.float32)
    y0 = M.random_init(rng, n)
    ys = np.asarray(M.simulate_batch(jnp.asarray(y0), jnp.asarray(t_fine), jnp.asarray(Uf)))
    yc = ys[:, ::stride]
    phi = phi_D_factory(M, dt, stride)
    xk = jnp.asarray(yc[:, :-1]); uk = jnp.asarray(Uc); dc = 0.5
    fp = jax.vmap(jax.vmap(lambda x, u: phi(x, u + dc)))(xk, uk)
    fm = jax.vmap(jax.vmap(lambda x, u: phi(x, u - dc)))(xk, uk)
    g_true = np.asarray((fp - fm) / (2.0 * dc)).astype(np.float32)
    return yc.astype(np.float32), Uc, g_true


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Generate stiff-neuron coarse-grid ZOH dataset")
    p.add_argument("--neuron", default="hh_model")
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--tc", type=int, default=250)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    M = importlib.import_module(args.neuron)
    D = args.stride * args.dt
    out = args.out or os.path.join(here, "data", f"{args.neuron.split('_')[0]}_operator.npz")
    print(f"[{args.neuron}] D={D} ms (stride {args.stride}, dt {args.dt}), Tc={args.tc} -> T={args.tc*D:.0f} ms, d={M.D_STATE}")
    ytr, utr, gtr = generate(M, args.n_train, args.tc, args.stride, args.dt, 101)
    yva, uva, gva = generate(M, args.n_val, args.tc, args.stride, args.dt, 202)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, ys_train=ytr, u_train=utr, g_train=gtr, ys_val=yva, u_val=uva, g_val=gva,
             dt=np.float32(args.dt), stride=np.int32(args.stride), D=np.float32(D), neuron=args.neuron)
    print(f"  train {ytr.shape}  frac depol(V>0)={float((ytr[...,0]>0).mean()):.3f}  saved {out}")


if __name__ == "__main__":
    main()
