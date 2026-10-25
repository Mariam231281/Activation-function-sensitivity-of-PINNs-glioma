"""Proliferation-invasion (Fisher-KPP) glioma model: definitions, reference solver
and Crank-Nicolson baseline (Phase 2).

Dimensional model on a 1-D line of tissue of length L = 5 cm through grey and white matter:
    c_t = d/dx( D(x) c_x ) + rho c (1 - c) - k(t) c,   c_x = 0 at both ends,
with c the tumour cell density normalised by the carrying capacity.

Non-dimensionalisation: x~ = x/L, t~ = rho0 t, D~ = D/(rho0 L^2), rho~ = rho/rho0, k~ = k/rho0.
The PINN uses s = t~/T~ in [0, 1] as its time input.
"""
import numpy as np
from scipy.linalg import solve_banded

# --- literature values (Swanson et al. 2000, as tabulated by Martin-Landrove 2017) ---
D_GREY = 0.0013      # cm^2/day
D_WHITE = 0.0065     # cm^2/day
RHO0 = 0.012         # 1/day
L = 5.0              # cm, length of the tissue line
T_DAYS = 180.0       # simulated period (days): diagnosis to 6-month follow-up
TT = RHO0 * T_DAYS   # non-dimensional final time (= 2.16)
DG = D_GREY / (RHO0 * L ** 2)   # 4.333e-3
DW = D_WHITE / (RHO0 * L ** 2)  # 2.167e-2
X_IFACE = 0.5        # grey | white interface (x~)
EPS = 0.02           # interface smoothing half-width (x~), i.e. 1 mm
X0, SIG0, A0 = 0.3, 0.1, 0.8    # initial Gaussian seed: centre, width (5 mm), peak density


def c0(x):
    """Gaussian seed plus its mirror images in x = 0 and x = 1, so that c0'(0) = c0'(1) = 0
    and the initial condition is compatible with the no-flux boundary conditions."""
    g = lambda z: np.exp(-z ** 2 / (2 * SIG0 ** 2))
    return A0 * (g(x - X0) + g(x + X0) + g(x - (2 - X0)))


def D_of_x(x, dg=DG, dw=DW, hetero=True):
    if not hetero:
        return np.full_like(np.asarray(x, dtype=float), dw)
    return dg + (dw - dg) * 0.5 * (1 + np.tanh((x - X_IFACE) / EPS))


def dD_of_x(x, dg=DG, dw=DW, hetero=True):
    if not hetero:
        return np.zeros_like(np.asarray(x, dtype=float))
    return (dw - dg) * 0.5 / EPS / np.cosh((x - X_IFACE) / EPS) ** 2


# --- therapy schedule following Stupp et al. (2005): treatment starts 30 days after the
#     first scan; 6 weeks of chemoradiotherapy, a 4-week break, then 6 adjuvant TMZ cycles
#     of 5 days every 28 days. Box functions are smoothed with 1-day tanh ramps. ---
K_CRT = 2.0      # kill rate during chemoradiotherapy, in units of rho0
K_TMZ = 3.0      # kill rate during adjuvant TMZ days, in units of rho0
import os
RAMP = float(os.environ.get("PI_RAMP", 3.0))  # days (3-day ramps; 1-day ramps used as a stress test)
T_START = 30.0
PULSES = [(T_START, T_START + 42.0, K_CRT)] + [(T_START + 70.0 + 28.0 * i, T_START + 75.0 + 28.0 * i, K_TMZ)
                                              for i in range(6) if T_START + 70.0 + 28.0 * i < T_DAYS]


def _box(td, a, b):
    return 0.5 * (np.tanh((td - a) / RAMP) - np.tanh((td - b) / RAMP))


def k_of_t(t_nd):
    """Kill rate in units of rho0 as a function of non-dimensional time t~ (numpy)."""
    td = np.asarray(t_nd) / RHO0
    return sum(k * _box(td, a, b) for a, b, k in PULSES)


PROBLEMS = {
    "pi_homog":   dict(hetero=False, therapy=False, title="Homogeneous white matter"),
    "pi_het":     dict(hetero=True,  therapy=False, title="Grey/white heterogeneity"),
    "pi_therapy": dict(hetero=True,  therapy=True,  title="Heterogeneity + chemoradiotherapy"),
}


def crank_nicolson(problem, N, Nt, dg=DG, dw=DW, rho=1.0, store_every=None, newton_tol=1e-12):
    """Vertex-centred finite volumes + Crank-Nicolson in time, Newton on the reaction term.
    Returns x, stored times (non-dimensional t~), and solution array (times x N)."""
    cfg = PROBLEMS[problem]
    x = np.linspace(0.0, 1.0, N)
    h = x[1] - x[0]
    xf = 0.5 * (x[1:] + x[:-1])
    Df = D_of_x(xf, dg, dw, cfg["hetero"])
    # control-volume widths (half cells at boundaries)
    w = np.full(N, h); w[0] = w[-1] = h / 2
    # diffusion operator A (tridiagonal), zero flux at both ends
    lower = np.zeros(N); diag = np.zeros(N); upper = np.zeros(N)
    flux = Df / h
    diag[:-1] -= flux; upper[1:] += flux      # flux between i and i+1 acting on i
    diag[1:] -= flux; lower[:-1] += flux
    diag /= w; upper[1:] /= w[:-1]; lower[:-1] /= w[1:]
    # banded storage: row0 = superdiag (shifted), row1 = diag, row2 = subdiag
    def Aprod(c):
        out = diag * c
        out[:-1] += upper[1:] * c[1:]
        out[1:] += lower[:-1] * c[:-1]
        return out
    dt = TT / Nt
    kfun = (lambda t: k_of_t(t)) if cfg["therapy"] else (lambda t: 0.0)
    f = lambda c, t: Aprod(c) + rho * c * (1 - c) - kfun(t) * c
    c = c0(x)
    store_every = store_every or Nt
    ts, sol = [0.0], [c.copy()]
    t = 0.0
    for n in range(Nt):
        rhs = c + 0.5 * dt * f(c, t)
        t1 = t + dt
        cn = c.copy()
        for _ in range(20):
            G = cn - 0.5 * dt * f(cn, t1) - rhs
            # Jacobian J = I - dt/2 (A + diag(rho(1-2c) - k))
            ab = np.zeros((3, N))
            ab[0, 1:] = -0.5 * dt * upper[1:]
            ab[1] = 1 - 0.5 * dt * (diag + rho * (1 - 2 * cn) - kfun(t1))
            ab[2, :-1] = -0.5 * dt * lower[:-1]
            dc = solve_banded((1, 1), ab, -G)
            cn += dc
            if np.max(np.abs(dc)) < newton_tol:
                break
        c, t = cn, t1
        if (n + 1) % store_every == 0:
            ts.append(t); sol.append(c.copy())
    return x, np.array(ts), np.array(sol)


def reference(problem, N=4001, Nt=40000):
    """Fine-grid CN solution stored on the 201 x 201 evaluation grid."""
    x, ts, sol = crank_nicolson(problem, N, Nt, store_every=Nt // 200)
    stride = (N - 1) // 200
    return x[::stride], ts / TT, sol[:, ::stride]
