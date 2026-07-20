"""Hodgkin-Huxley (1952 squid axon), control-affine in injected current.

State x=(V,m,h,n). Injected current u=I_ext enters ONLY dV/dt, affinely:
  C dV/dt = u - g_Na m^3 h (V-E_Na) - g_K n^4 (V-E_K) - g_L (V-E_L)
so the exact input matrix is B=[1/C,0,0,0] -> the same closed-form inverse as FHN
but on a STIFF system where explicit RK4 is stability-capped at a tiny dt.
"""
import numpy as np
import jax
import jax.numpy as jnp

C_M = 1.0
G_NA, G_K, G_L = 120.0, 36.0, 0.3
E_NA, E_K, E_L = 50.0, -77.0, -54.387
U_LO, U_HI = -5.0, 60.0
V0 = -65.0
D_STATE = 4
FIRE_BAND = (2.0, 35.0)


def _safe_exprel(x):
    """x/(exp(x)-1) with the removable singularity at x=0 handled (-> -1 shift)."""
    small = jnp.abs(x) < 1e-4
    return jnp.where(small, 1.0 - x / 2.0, x / jnp.expm1(x))


def _alpha_m(V):
    return _safe_exprel(-(V + 40.0) / 10.0)


def _beta_m(V):
    return 4.0 * jnp.exp(-(V + 65.0) / 18.0)


def _alpha_h(V):
    return 0.07 * jnp.exp(-(V + 65.0) / 20.0)


def _beta_h(V):
    return 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))


def _alpha_n(V):
    return 0.1 * _safe_exprel(-(V + 55.0) / 10.0)


def _beta_n(V):
    return 0.125 * jnp.exp(-(V + 65.0) / 80.0)


def f(x, u):
    """HH vector field; u=I_ext (uA/cm^2) enters dV/dt affinely."""
    V, m, h, n = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    i_na = G_NA * m ** 3 * h * (V - E_NA)
    i_k = G_K * n ** 4 * (V - E_K)
    i_l = G_L * (V - E_L)
    dV = (u - i_na - i_k - i_l) / C_M
    dm = _alpha_m(V) * (1.0 - m) - _beta_m(V) * m
    dh = _alpha_h(V) * (1.0 - h) - _beta_h(V) * h
    dn = _alpha_n(V) * (1.0 - n) - _beta_n(V) * n
    return jnp.stack([dV, dm, dh, dn], axis=-1)


def gate_inf(V):
    """Steady-state gating (m,h,n)_inf(V) for building rest / initial states."""
    m = _alpha_m(V) / (_alpha_m(V) + _beta_m(V))
    h = _alpha_h(V) / (_alpha_h(V) + _beta_h(V))
    n = _alpha_n(V) / (_alpha_n(V) + _beta_n(V))
    return jnp.stack([m, h, n], axis=-1)


def rest_state(V=V0):
    g = np.asarray(gate_inf(jnp.asarray(float(V))))
    return np.array([V, g[0], g[1], g[2]], np.float32)


def random_init(rng, n):
    V = rng.uniform(-75.0, -50.0, n).astype(np.float32)
    g = np.asarray(gate_inf(jnp.asarray(V)))
    jit = rng.uniform(-0.05, 0.05, (n, 3)).astype(np.float32)
    return np.concatenate([V[:, None], np.clip(g + jit, 0.0, 1.0)], axis=1).astype(np.float32)


def _rk4_profile(y0, t_grid, u_grid):
    dt = t_grid[1] - t_grid[0]
    u_t, u_next = u_grid[:-1], u_grid[1:]
    u_mid = 0.5 * (u_t + u_next)

    def step(y, inp):
        ut, um, un = inp
        k1 = f(y, ut)
        k2 = f(y + 0.5 * dt * k1, um)
        k3 = f(y + 0.5 * dt * k2, um)
        k4 = f(y + dt * k3, un)
        y2 = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y2, y2

    _, ys = jax.lax.scan(step, y0, (u_t, u_mid, u_next))
    return jnp.concatenate([y0[None, :], ys], axis=0)


simulate_batch = jax.jit(jax.vmap(_rk4_profile, in_axes=(0, None, 0)))
