"""Multi-channel conductance-based neuron (7-D), control-affine in injected current.

Textbook HH (Na, Kd, leak) + I_M (slow non-inactivating K, spike-frequency adaptation)
+ I_A (transient A-type K, fast activation) with a kinetic speed-up factor PHI so the
cell is fast-spiking and genuinely stiff. State x=(V,m,h,n,p,a,b). Injected current u
enters ONLY dV/dt, so the exact input matrix is B=[1/C,0,...,0] and the same closed-form
inverse applies -- but now each vector-field eval is expensive (many exponentials) and the
max stable explicit step is small, so one classical phi_D solve costs a lot.
"""
import numpy as np
import jax
import jax.numpy as jnp

C_M = 1.0
G_NA, G_K, G_L = 120.0, 36.0, 0.3
G_M, G_A = 1.0, 20.0
E_NA, E_K, E_L = 50.0, -77.0, -54.387
PHI = 1.5                      # kinetic speed-up -> stiff, biologically realistic ~55-110 Hz
U_LO, U_HI = -5.0, 40.0
V0 = -70.0
D_STATE = 7
FIRE_BAND = (18.0, 32.0)


def _safe_exprel(x):
    small = jnp.abs(x) < 1e-4
    return jnp.where(small, 1.0 - x / 2.0, x / jnp.expm1(x))


def _am(V): return _safe_exprel(-(V + 40.0) / 10.0)
def _bm(V): return 4.0 * jnp.exp(-(V + 65.0) / 18.0)
def _ah(V): return 0.07 * jnp.exp(-(V + 65.0) / 20.0)
def _bh(V): return 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))
def _an(V): return 0.1 * _safe_exprel(-(V + 55.0) / 10.0)
def _bn(V): return 0.125 * jnp.exp(-(V + 65.0) / 80.0)


def _p_inf(V): return 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))
def _p_tau(V): return 100.0 / (3.3 * jnp.exp((V + 35.0) / 20.0) + jnp.exp(-(V + 35.0) / 20.0))
def _a_inf(V): return 1.0 / (1.0 + jnp.exp(-(V + 50.0) / 20.0))
def _a_tau(V): return 0.5 + 1.5 / (1.0 + jnp.exp((V + 40.0) / 10.0))
def _b_inf(V): return 1.0 / (1.0 + jnp.exp((V + 80.0) / 6.0))
def _b_tau(V): return 8.0 + 12.0 / (1.0 + jnp.exp((V + 55.0) / 10.0))


def f(x, u):
    V, m, h, n, p, a, b = (x[..., i] for i in range(7))
    i_na = G_NA * m ** 3 * h * (V - E_NA)
    i_k = G_K * n ** 4 * (V - E_K)
    i_l = G_L * (V - E_L)
    i_m = G_M * p * (V - E_K)
    i_a = G_A * a ** 3 * b * (V - E_K)
    dV = (u - i_na - i_k - i_l - i_m - i_a) / C_M
    dm = PHI * (_am(V) * (1.0 - m) - _bm(V) * m)
    dh = PHI * (_ah(V) * (1.0 - h) - _bh(V) * h)
    dn = PHI * (_an(V) * (1.0 - n) - _bn(V) * n)
    dp = (_p_inf(V) - p) / _p_tau(V)
    da = PHI * (_a_inf(V) - a) / _a_tau(V)
    db = (_b_inf(V) - b) / _b_tau(V)
    return jnp.stack([dV, dm, dh, dn, dp, da, db], axis=-1)


def gate_inf(V):
    m = _am(V) / (_am(V) + _bm(V))
    h = _ah(V) / (_ah(V) + _bh(V))
    n = _an(V) / (_an(V) + _bn(V))
    return jnp.stack([m, h, n, _p_inf(V), _a_inf(V), _b_inf(V)], axis=-1)


def rest_state(V=V0):
    g = np.asarray(gate_inf(jnp.asarray(float(V))))
    return np.concatenate([[V], g]).astype(np.float32)


def random_init(rng, n):
    V = rng.uniform(-75.0, -55.0, n).astype(np.float32)
    g = np.asarray(gate_inf(jnp.asarray(V)))
    jit = rng.uniform(-0.03, 0.03, (n, 6)).astype(np.float32)
    return np.concatenate([V[:, None], np.clip(g + jit, 0.0, 1.0)], axis=1).astype(np.float32)


def _rk4_profile(y0, t_grid, u_grid):
    dt = t_grid[1] - t_grid[0]
    u_t, u_next = u_grid[:-1], u_grid[1:]
    u_mid = 0.5 * (u_t + u_next)

    def step(y, inp):
        ut, um, un = inp
        k1 = f(y, ut); k2 = f(y + 0.5 * dt * k1, um)
        k3 = f(y + 0.5 * dt * k2, um); k4 = f(y + dt * k3, un)
        return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    _, ys = jax.lax.scan(step, y0, (u_t, u_mid, u_next))
    return jnp.concatenate([y0[None, :], ys], axis=0)


simulate_batch = jax.jit(jax.vmap(_rk4_profile, in_axes=(0, None, 0)))
