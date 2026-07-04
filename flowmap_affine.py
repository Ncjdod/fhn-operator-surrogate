"""Control-affine flow-map: x_{t+D} = F(x) + G(x)*u, closed-form invertible in u.

The injected current enters FHN/HH affinely and only in the voltage channel, so a
coarse flow-map that is affine in a zero-order-hold current is structurally exact in
its u-dependence and inverts in one pass:
    u* = <G(x), x_target - F(x)> / <G(x),G(x)>          (least-squares steering current)
No optimizer loop -> live intervention. F,G are unconstrained MLPs sharing a trunk;
only u enters affinely, so forward accuracy is never traded for invertibility. Inputs
are per-channel standardized (V and gates live on very different scales); G is floored
on the voltage channel so <G,G> is bounded away from zero and the inverse never blows up.
"""
import numpy as np
import jax
import jax.numpy as jnp


def _init_mlp(sizes, key, scale_last=1.0):
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


class AffineConfig:
    def __init__(self, d=2, hidden=(128, 128), stride=8, g_floor=0.1, v_chan=0):
        self.d = d
        self.hidden = tuple(hidden)
        self.stride = stride
        self.g_floor = g_floor       # absolute floor on the voltage-channel gain
        self.v_chan = v_chan         # index of the voltage channel (0 for FHN & HH)


def init_affine(cfg, key, x_mu=None, x_sd=None):
    kt, kf, kg = jax.random.split(key, 3)
    trunk = _init_mlp([cfg.d, *cfg.hidden], kt)
    fh = _init_mlp([cfg.hidden[-1], cfg.d], kf, scale_last=0.1)
    gh = _init_mlp([cfg.hidden[-1], cfg.d], kg, scale_last=0.05)
    mu = jnp.zeros(cfg.d) if x_mu is None else jnp.asarray(x_mu, jnp.float32)
    sd = jnp.ones(cfg.d) if x_sd is None else jnp.asarray(x_sd, jnp.float32)
    return {"trunk": trunk, "F": fh, "G": gh, "mu": mu, "sd": sd}


def _trunk(x, params):
    a = (x - params["mu"]) / params["sd"]
    for layer in params["trunk"]:
        a = jax.nn.swish(a @ layer["w"] + layer["b"])
    return a


def FG(params, cfg, x):
    """Drift F(x) (next state at u=0) and control sensitivity G(x)=d(next)/du."""
    h = _trunk(x, params)
    sd = params["sd"]
    F = x + sd * _mlp(h, params["F"])
    g = sd * _mlp(h, params["G"])
    floor = jnp.zeros((cfg.d,)).at[cfg.v_chan].set(cfg.g_floor)
    return F, g + floor


def step(params, cfg, x, u):
    F, G = FG(params, cfg, x)
    return F + G * u[..., None]


def invert(params, cfg, x, x_target, clip_lo=None, clip_hi=None):
    """Closed-form least-squares steering current to drive x -> x_target in one step.

    Returns (u*, x_next, r_reach) where r_reach is the unreachable residual (scalar u
    can only reach the rank-1 span of G in the d-dim target space).
    """
    F, G = FG(params, cfg, x)
    diff = x_target - F
    num = jnp.sum(G * diff, axis=-1)
    den = jnp.sum(G * G, axis=-1) + 1e-8
    u = num / den
    if clip_lo is not None:
        u = jnp.clip(u, clip_lo, clip_hi)
    x_next = F + G * u[..., None]
    r_reach = jnp.linalg.norm(x_target - x_next, axis=-1)
    return u, x_next, r_reach


def rollout(params, cfg, x0, u_coarse):
    """Simulate over a zero-order-hold coarse current u_coarse (B, Tc) -> (B, Tc+1, d)."""
    @jax.checkpoint
    def body(x, u):
        xn = step(params, cfg, x, u)
        return xn, xn

    _, xs = jax.lax.scan(body, x0, jnp.moveaxis(u_coarse, 1, 0))
    xs = jnp.moveaxis(xs, 0, 1)
    return jnp.concatenate([x0[:, None, :], xs], axis=1)
