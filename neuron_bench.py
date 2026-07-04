"""Fair stiff-neuron benchmark (model-agnostic): forward stability/cost + 3-way control.

Honest accounting per the design review:
- FORWARD: surrogate one big coarse step vs fine RK4 (stiffness-forced) vs coarse RK4 at
  the same D (diverges). Report substep depth, wall-clock, accuracy.
- CONTROL: steer the TRUE plant to a reference; three controllers, all warm-started, all
  saturated, same task:
    surrogate  = FG(x) once + closed-form u*=<G,tgt-F>/<G,G>        [1 MLP forward]
    rk4_lin1   = phi_D^RK4(x,0) + one sensitivity + IDENTICAL closed form [2 stiff solves]
    rk4_gn(K)  = Gauss-Newton on the true model                     [K stiff solves]
  Primary metric = true-model vector-field evals / sequential substeps per control step;
  wall-clock + batched throughput secondary.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import importlib
import pickle
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
os.makedirs(ART, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140})


def load_affine(path):
    o = pickle.load(open(path, "rb")); c = o["cfg"]
    cfg = A.AffineConfig(d=c["d"], hidden=tuple(c["hidden"]), stride=c["stride"],
                         g_floor=c["g_floor"], v_chan=c["v_chan"])
    params = jax.tree_util.tree_map(jnp.asarray, o["params"])
    return params, cfg, float(o["D"]), float(o["dt"]), o.get("neuron", "hh_model")


def phi_D_factory(M, D, dt_sub):
    nsub = int(round(D / dt_sub))
    def phi(x, u):
        def body(y, _):
            k1 = M.f(y, u); k2 = M.f(y + 0.5 * dt_sub * k1, u)
            k3 = M.f(y + 0.5 * dt_sub * k2, u); k4 = M.f(y + dt_sub * k3, u)
            return y + dt_sub / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None
        y, _ = jax.lax.scan(body, x, None, length=nsub)
        return y
    return phi, nsub


def timed(fn, *a, reps=15):
    fn(*a).block_until_ready(); best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fn(*a).block_until_ready(); best = min(best, time.perf_counter() - t0)
    return best


def forward_bench(M, params, cfg, D, dt_data, batch=256, n_coarse=200, dt_stable=0.05):
    lo, hi = M.FIRE_BAND
    print(f"\n== FORWARD ==  D={D:.3f} ms  batch={batch}  horizon={n_coarse*D:.0f} ms")
    rng = np.random.default_rng(0)
    Uc = np.clip(rng.uniform(lo, hi, (batch, n_coarse)), M.U_LO, M.U_HI).astype(np.float32)
    x0 = M.random_init(rng, batch)
    x0j = jnp.asarray(x0); Ucj = jnp.asarray(Uc)
    sub_fine = int(round(D / dt_data))
    phi_fine, _ = phi_D_factory(M, D, dt_data)
    def rollf(x, U):
        def b(x, u): xn = phi_fine(x, u); return xn, xn
        _, xs = jax.lax.scan(b, x, U); return xs
    rollf_j = jax.jit(jax.vmap(rollf, in_axes=(0, 0)))
    y_true = np.asarray(rollf_j(x0j, Ucj)); sx = y_true.reshape(-1, cfg.d).std(0) + 1e-6

    res = {}
    roll_sur = jax.jit(lambda x, U: A.rollout(params, cfg, x, U))
    t_sur = timed(roll_sur, x0j, Ucj); y_sur = np.asarray(roll_sur(x0j, Ucj))[:, 1:]
    res["surrogate"] = {"ms": t_sur * 1e3, "nrmse": float(np.sqrt((((y_sur - y_true) / sx) ** 2).mean())),
                        "subs": 1, "seq": n_coarse}
    t_fine = timed(rollf_j, x0j, Ucj)
    res["rk4_fine"] = {"ms": t_fine * 1e3, "nrmse": 0.0, "subs": sub_fine, "seq": n_coarse * sub_fine}
    for dt_c, tag in [(dt_stable, f"rk4@{dt_stable}"), (D, "rk4@D")]:
        nsub = max(1, int(round(D / dt_c))); phi_c, _ = phi_D_factory(M, D, D / nsub)
        def rollc(x, U):
            def b(x, u): xn = phi_c(x, u); return xn, xn
            _, xs = jax.lax.scan(b, x, U); return xs
        rollc_j = jax.jit(jax.vmap(rollc, in_axes=(0, 0)))
        yc = np.asarray(rollc_j(x0j, Ucj))
        blew = (not np.isfinite(yc).all()) or np.nanmax(np.abs(yc[..., 0])) > 1e3
        ec = np.inf if blew else float(np.sqrt((((yc - y_true) / sx) ** 2).mean()))
        res[tag] = {"ms": timed(rollc_j, x0j, Ucj) * 1e3, "nrmse": ec, "subs": nsub,
                    "seq": n_coarse * nsub, "diverged": bool(blew)}
    print(f"  {'method':12s} {'ms':>8s} {'nrmse':>9s} {'subs/D':>7s}  speedup(vs fine)")
    for k, v in res.items():
        nr = "DIVERGED" if v.get("diverged") else f"{v['nrmse']:.3f}"
        print(f"  {k:12s} {v['ms']:8.2f} {nr:>9s} {v['subs']:7d}  {res['rk4_fine']['ms']/v['ms']:.2f}x")
    return res, y_true, y_sur


def control_bench(M, params, cfg, D, dt_data, batch=128, n_steps=200, n_gn=6, dt_stable=0.05):
    lo, hi = M.FIRE_BAND
    print(f"\n== CONTROL ==  D={D:.3f} ms  batch={batch}  steps={n_steps}")
    phi_plant, sub_plant = phi_D_factory(M, D, dt_data)
    phi_ctrl, sub_ctrl = phi_D_factory(M, D, dt_stable)
    delta = 0.5; vc = cfg.v_chan
    rng = np.random.default_rng(7)
    I_ref = np.clip(rng.uniform(lo, hi, (batch, n_steps)), M.U_LO, M.U_HI).astype(np.float32)
    x0 = M.random_init(rng, batch)
    phi_plant_v = jax.jit(jax.vmap(phi_plant))
    def ref(x, U):
        def b(x, u): xn = phi_plant(x, u); return xn, xn
        _, xs = jax.lax.scan(b, x, U); return xs
    x_ref = np.asarray(jax.jit(jax.vmap(ref))(jnp.asarray(x0), jnp.asarray(I_ref)))
    sx = x_ref.reshape(-1, cfg.d).std(0) + 1e-6

    def u_sur(x, tgt, uw):
        F, G = A.FG(params, cfg, x)
        return jnp.clip(jnp.sum(G * (tgt - F), -1) / (jnp.sum(G * G, -1) + 1e-8), M.U_LO, M.U_HI)
    def u_lin1(x, tgt, uw):
        f0 = phi_ctrl(x, jnp.zeros(())); g = (phi_ctrl(x, delta) - f0) / delta
        return jnp.clip(jnp.sum(g * (tgt - f0)) / (jnp.sum(g * g) + 1e-8), M.U_LO, M.U_HI)
    def u_gn(x, tgt, uw):
        def body(u, _):
            f0 = phi_ctrl(x, u); g = (phi_ctrl(x, u + delta) - f0) / delta
            du = jnp.sum(g * (tgt - f0)) / (jnp.sum(g * g) + 1e-8)
            return jnp.clip(u + du, M.U_LO, M.U_HI), None
        u, _ = jax.lax.scan(body, uw, None, length=n_gn); return u
    fns = {"surrogate": (jax.jit(jax.vmap(u_sur)), 0),
           "rk4_lin1": (jax.jit(jax.vmap(u_lin1)), 2 * sub_ctrl),
           f"rk4_gn(K={n_gn})": (jax.jit(jax.vmap(u_gn)), 2 * n_gn * sub_ctrl)}

    def loop(fn):
        x = jnp.asarray(x0); uprev = jnp.zeros(batch); traj = [np.asarray(x)]; us = []
        for k in range(n_steps):
            u = fn(x, jnp.asarray(x_ref[:, k]), uprev); x = phi_plant_v(x, u); uprev = u
            traj.append(np.asarray(x)); us.append(np.asarray(u))
        return np.stack(traj, 1), np.stack(us, 1)

    out = {}
    for name, (fn, seq) in fns.items():
        tr, us = loop(fn)
        track = float(np.sqrt((((tr[:, 1:] - x_ref) / sx) ** 2).mean()))
        x0j = jnp.asarray(x0); t0 = jnp.asarray(x_ref[:, 0]); u0 = jnp.zeros(batch)
        best = _bt(fn, x0j, t0, u0)
        out[name] = {"track": track, "vf_evals": seq, "seq": seq,
                     "us_per_neuron": best / batch * 1e6, "traj": tr, "us": us}
    print(f"  {'controller':14s} {'track':>7s} {'vf/step':>8s} {'seq':>5s} {'us/neuron':>10s}")
    for k, v in out.items():
        print(f"  {k:14s} {v['track']:7.3f} {v['vf_evals']:8d} {v['seq']:5d} {v['us_per_neuron']:10.3f}")
    return out, x_ref


def _bt(fn, *a, reps=12):
    fn(*a).block_until_ready(); b = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fn(*a).block_until_ready(); b = min(b, time.perf_counter() - t0)
    return b


def figures(M, fwd, ctl, y_true, y_sur, D, neuron):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    tt = np.arange(y_true.shape[1]) * D
    ax[0].plot(tt, y_true[0, :, 0], "k", lw=1.3, label="true (fine RK4)")
    ax[0].plot(tt, y_sur[0, :, 0], "C1--", lw=1.3, label="affine surrogate (1 step/D)")
    ax[0].set_title(f"{neuron}: forward V(t), 1 big step D={D:.2f} ms")
    ax[0].set_xlabel("t (ms)"); ax[0].set_ylabel("V (mV)"); ax[0].legend(fontsize=8)
    names = list(fwd); seq = [fwd[k]["seq"] for k in names]
    cols = ["C1" if "surro" in k else ("C3" if fwd[k].get("diverged") else "C0") for k in names]
    ax[1].bar(range(len(names)), seq, color=cols); ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax[1].set_ylabel("sequential vf-evals (horizon)"); ax[1].set_title("Sequential depth")
    cn = list(ctl); seqd = [max(ctl[k]["seq"], 0.5) for k in cn]; trk = [ctl[k]["track"] for k in cn]
    ax[2].scatter(seqd, trk, c=["C1" if "surro" in k else "C0" for k in cn], s=90, zorder=3)
    for k in cn:
        ax[2].annotate(k, (max(ctl[k]["seq"], 0.5), ctl[k]["track"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax[2].set_xscale("log"); ax[2].set_xlabel("stiff vf-evals per control step"); ax[2].set_ylabel("tracking NRMSE")
    ax[2].set_title("Control: cost vs accuracy"); ax[2].grid(alpha=.3)
    fig.suptitle(f"{neuron}: control-affine surrogate vs fair classical integration", fontweight="bold")
    fig.tight_layout(); out = f"{ART}/{neuron.split('_')[0]}_bench.png"; fig.savefig(out); plt.close(fig)
    print(f"\nFIGURE -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(HERE, "data", "affine_hh.pkl"))
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    params, cfg, D, dt_data, neuron = load_affine(args.model)
    M = importlib.import_module(neuron)
    print(f"device={jax.devices()[0].platform.upper()}  neuron={neuron} d={cfg.d} D={D:.3f} ms g_floor={cfg.g_floor:.3f}")
    fwd, y_true, y_sur = forward_bench(M, params, cfg, D, dt_data, batch=args.batch)
    ctl, x_ref = control_bench(M, params, cfg, D, dt_data, batch=min(128, args.batch))
    figures(M, fwd, ctl, y_true, y_sur, D, neuron)


if __name__ == "__main__":
    main()
