"""PINN for the proliferation-invasion glioma model (Phase 2): forward and inverse problems.

Forward:  c_s = T~ [ D(x) c_xx + D'(x) c_x + rho c (1 - c) - k(s) c ],  c_x(0,s) = c_x(1,s) = 0,
          c(x, 0) = c0(x), (x, s) in [0, 1]^2, with s = t~/T~.
Inverse:  same PDE (heterogeneous, no therapy) with unknown D_grey, D_white and rho, recovered
          from sparse noisy observations of c at two follow-up scan times.

Usage examples
  python pinn_pi.py --problem pi_het --act tanh --seed 1
  python pinn_pi.py --problem inverse --act swish --seed 1 --noise 0.02
  python pinn_pi.py --problem pi_het --act tanh --seed 1 --ic soft
  python pinn_pi.py --problem pi_het --act tanh --seed 1 --colloc sobol
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import qmc
import pi_model as m

torch.set_default_dtype(torch.float64)


class AdaptiveTanh(torch.nn.Module):
    """Layer-wise locally adaptive tanh(n * a * z) (Jagtap et al. 2020), n = 10, a0 = 1/n."""
    def __init__(self, n=10.0):
        super().__init__()
        self.n = n
        self.a = torch.nn.Parameter(torch.tensor(1.0 / n))

    def forward(self, z):
        return torch.tanh(self.n * self.a * z)


FIXED = {
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "swish": lambda z: z * torch.sigmoid(z),
    "sine": torch.sin,
    "gelu": F.gelu,
    "softplus": F.softplus,
    "relu": torch.relu,
}
ACT_NAMES = list(FIXED) + ["atanh"]


class MLP(torch.nn.Module):
    def __init__(self, act, width=48, depth=4):
        super().__init__()
        sizes = [2] + [width] * depth + [1]
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        if act == "atanh":
            self.acts = torch.nn.ModuleList(AdaptiveTanh() for _ in range(depth))
        else:
            self.acts = [FIXED[act]] * depth

    def forward(self, x, s):
        h = torch.cat([2 * x - 1, 2 * s - 1], dim=1)   # inputs rescaled to [-1, 1]
        for layer, a in zip(self.layers[:-1], self.acts):
            h = a(layer(h))
        return self.layers[-1](h)


def t_D(x, dg, dw, hetero):
    if not hetero:
        return dw * torch.ones_like(x), torch.zeros_like(x)
    th = torch.tanh((x - m.X_IFACE) / m.EPS)
    return dg + (dw - dg) * 0.5 * (1 + th), (dw - dg) * 0.5 / m.EPS * (1 - th ** 2)


def t_k(s):
    td = s * m.T_DAYS
    out = torch.zeros_like(s)
    for a, b, k in m.PULSES:
        out = out + k * 0.5 * (torch.tanh((td - a) / m.RAMP) - torch.tanh((td - b) / m.RAMP))
    return out


def t_c0(x):
    g = lambda z: torch.exp(-z ** 2 / (2 * m.SIG0 ** 2))
    return m.A0 * (g(x - m.X0) + g(x + m.X0) + g(x - (2 - m.X0)))


def collocation(kind, n, seed):
    if kind == "lhs":
        return qmc.LatinHypercube(d=2, seed=seed).random(n)
    if kind == "sobol":
        return qmc.Sobol(d=2, scramble=True, seed=seed).random(n)
    if kind == "random":
        return np.random.default_rng(seed).random((n, 2))
    raise ValueError(kind)


# inverse-problem set-up
SCAN_DAYS = (90.0, 180.0)    # two follow-up scans after the diagnostic scan at day 0
N_OBS = 50                   # sensor points per scan
INIT_FACTORS = dict(dg=2.0, dw=0.5, rho=0.5)  # initial guess = truth x factor


def run(problem, act, seed, out_dir, noise=0.0, ic="hard", colloc="lhs", n_r=2048,
        epochs=4000, lbfgs_iters=2000, lam_data=1.0, causal=0.0, n_bins=32):
    inverse = problem == "inverse"
    cfg = m.PROBLEMS["pi_het" if inverse else problem]
    torch.manual_seed(seed); np.random.seed(seed)
    net = MLP(act)

    # parameters (trainable only in the inverse problem)
    if inverse:
        log_dg = torch.nn.Parameter(torch.tensor(math.log(m.DG * INIT_FACTORS["dg"])))
        log_dw = torch.nn.Parameter(torch.tensor(math.log(m.DW * INIT_FACTORS["dw"])))
        log_rho = torch.nn.Parameter(torch.tensor(math.log(1.0 * INIT_FACTORS["rho"])))
        extra = [log_dg, log_dw, log_rho]
        params = lambda: (torch.exp(log_dg), torch.exp(log_dw), torch.exp(log_rho))
    else:
        extra = []
        params = lambda: (torch.tensor(m.DG), torch.tensor(m.DW), torch.tensor(1.0))
    all_params = list(net.parameters()) + extra

    def c_hat(x, s):
        if ic == "hard":
            return t_c0(x) + s * net(x, s)
        return net(x, s)

    pts = collocation(colloc, n_r, seed)
    xr = torch.tensor(pts[:, :1], requires_grad=True)
    sr = torch.tensor(pts[:, 1:], requires_grad=True)
    bins = torch.clamp((sr.detach().ravel() * n_bins).long(), max=n_bins - 1)
    use_causal = [causal > 0]
    Dr_fixed = None
    kr = t_k(sr.detach()) if cfg["therapy"] else torch.zeros_like(sr)
    sb = torch.linspace(0, 1, 200).reshape(-1, 1)
    xb0 = torch.zeros_like(sb, requires_grad=True)
    xb1 = torch.ones_like(sb, requires_grad=True)
    xi = torch.linspace(0, 1, 200).reshape(-1, 1)
    si = torch.zeros_like(xi)
    ci = t_c0(xi)

    # observations for the inverse problem (from the fine reference solution)
    if inverse:
        ref = np.load(os.path.join(REF_DIR, "ref_pi_het.npz"))
        rng = np.random.default_rng(1000 + seed)
        xo, so, co = [], [], []
        for day in SCAN_DAYS:
            j = int(round(day / m.T_DAYS * 200))
            xs = np.sort(rng.random(N_OBS))
            cs = np.interp(xs, ref["x"], ref["c"][j]) + noise * rng.standard_normal(N_OBS)
            xo.append(xs); so.append(np.full(N_OBS, ref["s"][j])); co.append(cs)
        xo = torch.tensor(np.concatenate(xo)).reshape(-1, 1)
        so = torch.tensor(np.concatenate(so)).reshape(-1, 1)
        co = torch.tensor(np.concatenate(co)).reshape(-1, 1)

    def loss_fn():
        dg, dw, rho = params()
        c = c_hat(xr, sr)
        cs, cx = torch.autograd.grad(c.sum(), (sr, xr), create_graph=True)
        cxx = torch.autograd.grad(cx.sum(), xr, create_graph=True)[0]
        D, dD = t_D(xr, dg, dw, cfg["hetero"])
        res = cs / m.TT - (D * cxx + dD * cx + rho * c * (1 - c) - kr * c)
        if use_causal[0]:
            # causal weighting (Wang, Sankaran & Perdikaris 2024): bin residuals in time and
            # down-weight later bins until earlier ones are resolved
            Lb = torch.zeros(n_bins).index_add(0, bins, res.ravel() ** 2) / torch.bincount(bins, minlength=n_bins).clamp(min=1)
            w = torch.exp(-causal * torch.cumsum(torch.cat([Lb.new_zeros(1), Lb[:-1]]), 0)).detach()
            lr = (w * Lb).mean()
        else:
            lr = (res ** 2).mean()
        g0 = torch.autograd.grad(c_hat(xb0, sb).sum(), xb0, create_graph=True)[0]
        g1 = torch.autograd.grad(c_hat(xb1, sb).sum(), xb1, create_graph=True)[0]
        lb = (g0 ** 2).mean() + (g1 ** 2).mean()
        L = lr + lb
        li = torch.tensor(0.0)
        if ic == "soft":
            li = ((c_hat(xi, si) - ci) ** 2).mean()
            L = L + li
        ld = torch.tensor(0.0)
        if inverse:
            ld = ((c_hat(xo, so) - co) ** 2).mean()
            L = L + lam_data * ld
        return L, lr, lb, li, ld

    ref = np.load(os.path.join(REF_DIR, f"ref_{'pi_het' if inverse else problem}.npz"))
    X, S = np.meshgrid(ref["x"], ref["s"])
    Xt = torch.tensor(X.reshape(-1, 1)); St = torch.tensor(S.reshape(-1, 1))
    U = ref["c"]

    def evaluate():
        with torch.no_grad():
            P = c_hat(Xt, St).numpy().reshape(X.shape)
        if not np.all(np.isfinite(P)):
            return None, None
        E = U - P
        out = dict(global_rel_l2=float(np.linalg.norm(E) / np.linalg.norm(U)),
                   global_linf=float(np.abs(E).max()),
                   rel_l2_vs_t=[float(np.linalg.norm(E[i]) / max(np.linalg.norm(U[i]), 1e-30))
                                for i in range(U.shape[0])])
        if inverse:
            dg, dw, rho = [float(v) for v in params()]
            out["params"] = dict(dg=dg, dw=dw, rho=rho)
            out["param_rel_err"] = dict(dg=abs(dg / m.DG - 1), dw=abs(dw / m.DW - 1), rho=abs(rho - 1))
        return out, P

    hist = []
    def log(it, stage, L, lr, lb, li, ld):
        h = dict(it=it, stage=stage, loss=L.item(), res=lr.item(), bc=lb.item(), ic=li.item(), data=ld.item())
        if inverse:
            h.update({k: float(v.detach()) for k, v in zip(["dg", "dw", "rho"], params())})
        hist.append(h)

    opt = torch.optim.Adam(all_params, lr=1e-3)
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        L, *parts = loss_fn()
        L.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            log(ep, "adam", L, *parts)
    t_adam = time.time() - t0
    m_adam, P_adam = evaluate()
    state = {k: v.clone() for k, v in net.state_dict().items()}
    extra_state = [p.detach().clone() for p in extra]

    use_causal[0] = False   # L-BFGS refines the plain (unweighted) loss
    lb_opt = torch.optim.LBFGS(all_params, lr=1.0, max_iter=lbfgs_iters, history_size=50,
                               tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
    count = [0]
    def closure():
        lb_opt.zero_grad()
        L, *parts = loss_fn()
        L.backward()
        if count[0] % 10 == 0:
            log(epochs + count[0], "lbfgs", L, *parts)
        count[0] += 1
        return L
    t1 = time.time()
    ok = True
    try:
        lb_opt.step(closure)
    except Exception:
        ok = False
    m_final, P_final = evaluate()
    if m_final is None or not ok:
        net.load_state_dict(state)
        with torch.no_grad():
            for p, v in zip(extra, extra_state):
                p.copy_(v)
        m_final, P_final = evaluate()
        ok = False
    t_lbfgs = time.time() - t1

    tag = f"{problem}_{act}_s{seed}" + (f"_n{noise:g}" if inverse else "") + \
          ("" if ic == "hard" else f"_ic{ic}") + ("" if colloc == "lhs" else f"_{colloc}")
    rec = dict(problem=problem, act=act, seed=seed, noise=noise, ic=ic, colloc=colloc, n_r=n_r,
               causal=causal, epochs=epochs, lbfgs_iters=lbfgs_iters,
               n_params=sum(p.numel() for p in net.parameters()), adam=m_adam, final=m_final,
               lbfgs_ok=ok, lbfgs_evals=count[0], time_adam=t_adam, time_lbfgs=t_lbfgs,
               final_loss=hist[-1]["loss"], history=hist)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rec, open(os.path.join(out_dir, tag + ".json"), "w"))
    np.savez_compressed(os.path.join(out_dir, tag + "_field.npz"), final=P_final.astype(np.float32))
    return rec


REF_DIR = os.environ.get("PI_REF_DIR", "../results_p2")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(m.PROBLEMS) + ["inverse"])
    ap.add_argument("--act", required=True, choices=ACT_NAMES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--ic", default="hard", choices=["hard", "soft"])
    ap.add_argument("--colloc", default="lhs", choices=["lhs", "sobol", "random"])
    ap.add_argument("--n_r", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lbfgs", type=int, default=2000)
    ap.add_argument("--causal", type=float, default=0.0)
    ap.add_argument("--out", default="../results_p2")
    a = ap.parse_args()
    torch.set_num_threads(1)
    r = run(a.problem, a.act, a.seed, a.out, a.noise, a.ic, a.colloc, a.n_r, a.epochs, a.lbfgs, causal=a.causal)
    msg = f"{a.problem} {a.act} s{a.seed} n{a.noise:g} {a.ic} {a.colloc}: adam {r['adam']['global_rel_l2']:.3e} final {r['final']['global_rel_l2']:.3e} t={r['time_adam']:.0f}+{r['time_lbfgs']:.0f}s"
    if a.problem == "inverse":
        msg += " params " + " ".join(f"{k}={v:.3f}" for k, v in r["final"]["param_rel_err"].items())
    print(msg, flush=True)
