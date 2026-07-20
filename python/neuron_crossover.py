"""Crossover map (model-agnostic): where does 1 MLP forward+inverse beat 2 stiff solves?

Sweeps the control-step wall-clock ratio over (nsub = substeps per coarse step) x (batch),
pure timing, for a given neuron model. Costlier / stiffer neuron -> surrogate wins a wider
region. rk4_gn(K) row optional to show the iterative baseline's cost when linearize-once fails.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import importlib
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flowmap_affine as A

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "plots", "results")
plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
D = 0.4
delta = 0.5


def phi_factory(M, nsub):
    dt = D / nsub
    def phi(x, u):
        def b(y, _):
            k1 = M.f(y, u); k2 = M.f(y + 0.5 * dt * k1, u)
            k3 = M.f(y + 0.5 * dt * k2, u); k4 = M.f(y + dt * k3, u)
            return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None
        y, _ = jax.lax.scan(b, x, None, length=nsub)
        return y
    return phi


def T(fn, *a, r=25):
    fn(*a).block_until_ready(); b = np.inf
    for _ in range(r):
        t0 = time.perf_counter(); fn(*a).block_until_ready(); b = min(b, time.perf_counter() - t0)
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--neuron", default="hh_model")
    args = p.parse_args()
    M = importlib.import_module(args.neuron)
    d = M.D_STATE
    cfg = A.AffineConfig(d=d, hidden=(128, 128), stride=1, g_floor=0.1)
    P = A.init_affine(cfg, jax.random.PRNGKey(0), x_mu=np.zeros(d, np.float32), x_sd=np.ones(d, np.float32))

    def sur(x, tgt):
        F, G = A.FG(P, cfg, x)
        return jnp.clip(jnp.sum(G * (tgt - F), -1) / (jnp.sum(G * G, -1) + 1e-8), M.U_LO, M.U_HI)
    sur_j = jax.jit(jax.vmap(sur))

    def lin1(M, phi):
        def fn(x, tgt):
            f0 = phi(x, jnp.zeros(())); g = (phi(x, delta) - f0) / delta
            return jnp.clip(jnp.sum(g * (tgt - f0)) / (jnp.sum(g * g) + 1e-8), M.U_LO, M.U_HI)
        return fn

    subs = [2, 4, 8, 16, 32, 64]; batches = [1, 16, 128, 1024, 8192]
    speed = np.zeros((len(subs), len(batches)))
    print(f"[{args.neuron}] device={jax.devices()[0].platform.upper()}  speedup = rk4_lin1 / surrogate")
    print("nsub\\batch  " + "  ".join(f"{b:>8d}" for b in batches))
    for i, ns in enumerate(subs):
        lin_j = jax.jit(jax.vmap(lin1(M, phi_factory(M, ns))))
        row = []
        for k, B in enumerate(batches):
            rng = np.random.default_rng(B + ns)
            x = jnp.asarray(M.random_init(rng, B))
            tgt = x + jnp.asarray(rng.uniform(-5, 5, (B, d)).astype(np.float32))
            speed[i, k] = T(lin_j, x, tgt) / T(sur_j, x, tgt); row.append(speed[i, k])
        print(f"{ns:4d}       " + "  ".join(f"{v:8.2f}" for v in row))

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    im = ax.imshow(np.log2(speed), aspect="auto", cmap="RdBu_r", vmin=-np.log2(8), vmax=np.log2(8), origin="lower")
    ax.set_xticks(range(len(batches))); ax.set_xticklabels(batches)
    ax.set_yticks(range(len(subs))); ax.set_yticklabels(subs)
    ax.set_xlabel("batch (neurons stepped together)")
    ax.set_ylabel("stiff substeps per coarse step")
    for i in range(len(subs)):
        for k in range(len(batches)):
            v = speed[i, k]
            ax.text(k, i, f"{v:.1f}x", ha="center", va="center",
                    color="white" if abs(np.log2(v)) > 1.2 else "black", fontsize=9)
    cb = fig.colorbar(im, ax=ax, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb.set_ticklabels(["1/8", "1/4", "1/2", "1", "2x", "4x", "8x"]); cb.set_label("surrogate speedup")
    ax.set_title(f"{args.neuron} (d={d}): where invertible surrogate wins (red)\n"
                 "1 MLP forward+inverse vs 2 stiff phi_D solves per control step", fontsize=10)
    fig.tight_layout(); out = f"{ART}/{args.neuron.split('_')[0]}_crossover.png"
    fig.savefig(out); plt.close(fig)
    print(f"\nFIGURE -> {out}")


if __name__ == "__main__":
    main()
