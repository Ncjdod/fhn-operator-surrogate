"""Large-stride flow-map: coarser step with interior forcing samples for fewer scan steps."""
import numpy as np
import jax
import jax.numpy as jnp


def _init_mlp(sizes, key, scale_last=0.1):
    ks = jax.random.split(key, len(sizes) - 1)
    ps = []
    for i in range(len(sizes) - 1):
        a, b = sizes[i], sizes[i + 1]
        lim = np.sqrt(6.0 / (a + b)) * (scale_last if i == len(sizes) - 2 else 1.0)
        ps.append({"w": jax.random.uniform(ks[i], (a, b), minval=-lim, maxval=lim),
                   "b": jnp.zeros((b,))})
    return ps


def _mlp(x, ps):
    a = x
    for layer in ps[:-1]:
        a = jax.nn.swish(a @ layer["w"] + layer["b"])
    return a @ ps[-1]["w"] + ps[-1]["b"]


class FastConfig:
    """Config for the large-stride flow-map (stride, interior forcing samples)."""

    def __init__(self, d=2, hidden=(96, 96), stride=16, n_samp=5):
        self.d = d
        self.hidden = tuple(hidden)
        self.stride = stride
        self.n_samp = n_samp

    def __repr__(self):
        return f"FastConfig(d={self.d}, stride={self.stride}, n_samp={self.n_samp})"


def init_fast(cfg, key):
    """Init the residual stepper MLP [x, u_samples] -> Δx."""
    return {"step": _init_mlp([cfg.d + cfg.n_samp, *cfg.hidden, cfg.d], key)}


def step(params, cfg, x, usamp):
    """One coarse step x -> x_next given n_samp forcing samples across the interval."""
    return x + _mlp(jnp.concatenate([x, usamp], axis=-1), params["step"])


def rollout(params, cfg, x0, usamps):
    """Roll over coarse steps; usamps (B, Tc, n_samp). Returns (B, Tc+1, d)."""
    @jax.checkpoint
    def body(x, us):
        xn = step(params, cfg, x, us)
        return xn, xn

    _, xs = jax.lax.scan(body, x0, jnp.moveaxis(usamps, 1, 0))
    xs = jnp.moveaxis(xs, 0, 1)
    return jnp.concatenate([x0[:, None, :], xs], axis=1)


def build_samples(u_full, stride, n_samp):
    """Sample the native current at n_samp evenly-spaced points inside each coarse step."""
    u_full = np.asarray(u_full)
    S = u_full.shape[1]
    Tc = (S - 1) // stride
    offs = np.rint(np.linspace(0, stride, n_samp)).astype(int)
    starts = (np.arange(Tc) * stride)[:, None]
    idx = np.clip(starts + offs[None, :], 0, S - 1)
    return u_full[:, idx]


def coarse_states(ys, stride):
    """Subsample native states to the coarse grid (B, Tc+1, d)."""
    return np.asarray(ys)[:, ::stride]
