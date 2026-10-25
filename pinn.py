"""PINN solver for three 1-D reaction-diffusion test problems derived from the
Burgess/Wein-Koplow glioma model (revised protocol).

    u_t = 1/2 u_xx + F(x, t, u),   (x, t) in [0, 1]^2,
    u(x, 0) = u0(x),  Dirichlet data at x = 0, 1 from the exact solution.

Changes from the original protocol:
  * Example 3 uses the kill term F = -t^2 u, consistent with the exact solution
    exp(2x + 2t - t^3/3) (the original +t^2 u growth term was inconsistent).
  * Errors are evaluated over the full domain on a 201 x 201 grid.
  * Adam (4000 epochs) is followed by an L-BFGS refinement stage; metrics are
    recorded after both stages.
  * Loss histories are saved.

Usage: python pinn.py --example 1 --act tanh --seed 1234 --out results/
"""
import argparse, json, math, os, time
import numpy as np
import torch
from scipy.stats import qmc

torch.set_default_dtype(torch.float64)

EXAMPLES = {
    1: dict(name="linear",
            F=lambda x, t, u: 0.5 * u,
            u0=lambda x: torch.exp(x),
            exact=lambda x, t: np.exp(x + t)),
    2: dict(name="nonlinear",
            F=lambda x, t, u: torch.exp(-u) + 0.5 * torch.exp(-2.0 * u),
            u0=lambda x: torch.log(x + 2.0),
            exact=lambda x, t: np.log(x + t + 2.0)),
    3: dict(name="time-dependent kill term",
            F=lambda x, t, u: -(t ** 2) * u,
            u0=lambda x: torch.exp(2.0 * x),
            exact=lambda x, t: np.exp(2.0 * x + 2.0 * t - t ** 3 / 3.0)),
}

ACTS = {
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "swish": lambda z: z * torch.sigmoid(z),
    "relu": torch.relu,
}


class MLP(torch.nn.Module):
    def __init__(self, act, width=48, depth=4):
        super().__init__()
        sizes = [2] + [width] * depth + [1]
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        self.act = act

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        return self.layers[-1](h)


def run(example, act_name, seed, out_dir, epochs=4000, lbfgs_iters=1000):
    ex = EXAMPLES[example]
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = MLP(ACTS[act_name])
    n_params = sum(p.numel() for p in net.parameters())

    def u_hat(x, t):  # hard initial-condition ansatz
        return ex["u0"](x) + t * net(x, t)

    # collocation: Latin Hypercube interior, uniform boundary times
    pts = qmc.LatinHypercube(d=2, seed=seed).random(2000)
    xr = torch.tensor(pts[:, :1], requires_grad=True)
    tr = torch.tensor(pts[:, 1:], requires_grad=True)
    tb = torch.linspace(0, 1, 200).reshape(-1, 1)
    x0, x1 = torch.zeros_like(tb), torch.ones_like(tb)
    g0 = torch.tensor(ex["exact"](0.0, tb.numpy()))
    g1 = torch.tensor(ex["exact"](1.0, tb.numpy()))

    def loss_fn():
        u = u_hat(xr, tr)
        ut, ux = torch.autograd.grad(u.sum(), (tr, xr), create_graph=True)
        uxx = torch.autograd.grad(ux.sum(), xr, create_graph=True)[0]
        res = ut - 0.5 * uxx - ex["F"](xr, tr, u)
        lr_ = (res ** 2).mean()
        lb_ = ((u_hat(x0, tb) - g0) ** 2 + (u_hat(x1, tb) - g1) ** 2).mean()
        return lr_ + lb_, lr_, lb_

    # evaluation grid
    xs = np.linspace(0, 1, 201)
    ts = np.linspace(0, 1, 201)
    X, T = np.meshgrid(xs, ts)  # rows = time
    Xt = torch.tensor(X.reshape(-1, 1))
    Tt = torch.tensor(T.reshape(-1, 1))
    U = ex["exact"](X, T)
    early = [round(0.01 * k, 2) for k in range(1, 11)]
    probe = [0.1, 0.25, 0.5, 0.75, 1.0]

    def evaluate():
        with torch.no_grad():
            P = u_hat(Xt, Tt).numpy().reshape(X.shape)
        E = U - P
        if not np.all(np.isfinite(P)):
            return None, None
        out = dict(global_rel_l2=float(np.linalg.norm(E) / np.linalg.norm(U)),
                   global_linf=float(np.abs(E).max()),
                   global_mae=float(np.abs(E).mean()))
        def at(tv):
            xx = xs
            with torch.no_grad():
                p = u_hat(torch.tensor(xx.reshape(-1, 1)),
                          torch.full((len(xx), 1), tv)).numpy().ravel()
            e = ex["exact"](xx, tv) - p
            return dict(rel_l2=float(np.linalg.norm(e) / np.linalg.norm(ex["exact"](xx, tv))),
                        linf=float(np.abs(e).max()),
                        rms=float(np.sqrt((e ** 2).mean())))
        out["early_mean_rel_l2"] = float(np.mean([at(tv)["rel_l2"] for tv in early]))
        out["early_max_linf"] = float(np.max([at(tv)["linf"] for tv in early]))
        out["probe"] = {str(tv): at(tv) for tv in probe}
        out["rel_l2_vs_t"] = [float(np.linalg.norm(E[i]) / np.linalg.norm(U[i])) for i in range(len(ts))]
        return out, P

    hist = []
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        L, lr_, lb_ = loss_fn()
        L.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            hist.append(dict(it=ep, stage="adam", loss=L.item(), res=lr_.item(), bc=lb_.item()))
    t_adam = time.time() - t0
    m_adam, P_adam = evaluate()
    state = {k: v.clone() for k, v in net.state_dict().items()}

    # L-BFGS refinement
    lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=lbfgs_iters, history_size=50,
                           tolerance_grad=1e-12, tolerance_change=1e-15,
                           line_search_fn="strong_wolfe")
    count = [0]

    def closure():
        lb.zero_grad()
        L, lr_, lb_ = loss_fn()
        L.backward()
        if count[0] % 10 == 0:
            hist.append(dict(it=epochs + count[0], stage="lbfgs", loss=L.item(),
                             res=lr_.item(), bc=lb_.item()))
        count[0] += 1
        return L

    t1 = time.time()
    lbfgs_ok = True
    try:
        lb.step(closure)
    except Exception as e:  # numerical failure -> keep Adam weights
        lbfgs_ok = False
    m_final, P_final = evaluate()
    if m_final is None or not lbfgs_ok:
        net.load_state_dict(state)
        m_final, P_final = evaluate()
        lbfgs_ok = False
    t_lbfgs = time.time() - t1

    rec = dict(example=example, act=act_name, seed=seed, n_params=n_params,
               adam=m_adam, final=m_final, lbfgs_ok=lbfgs_ok, lbfgs_evals=count[0],
               time_adam=t_adam, time_lbfgs=t_lbfgs, final_loss=hist[-1]["loss"],
               history=hist)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"ex{example}_{act_name}_s{seed}"
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump(rec, f)
    np.savez_compressed(os.path.join(out_dir, tag + "_field.npz"),
                        x=xs, t=ts, exact=U, adam=P_adam, final=P_final)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", type=int, required=True)
    ap.add_argument("--act", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lbfgs", type=int, default=1000)
    a = ap.parse_args()
    torch.set_num_threads(1)
    r = run(a.example, a.act, a.seed, a.out, a.epochs, a.lbfgs)
    print(f"ex{a.example} {a.act} s{a.seed}: adam relL2={r['adam']['global_rel_l2']:.3e} "
          f"final relL2={r['final']['global_rel_l2']:.3e} t={r['time_adam']:.0f}+{r['time_lbfgs']:.0f}s")
