"""Aggregate Phase 2 runs (PI model) into LaTeX tables, figures and a JSON summary."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pi_model as m

RES = sys.argv[1] if len(sys.argv) > 1 else "../results_p2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../manuscript/figures"
os.makedirs(OUT, exist_ok=True)

ACTS = ["tanh", "atanh", "sine", "swish", "gelu", "softplus", "sigmoid"]
LABEL = {"tanh": "Tanh", "atanh": "Adaptive Tanh", "sine": "Sine", "swish": "Swish",
         "gelu": "GELU", "softplus": "Softplus", "sigmoid": "Sigmoid"}
# validated categorical palette (dataviz reference), fixed order
COL = {"tanh": "#2a78d6", "swish": "#eb6834", "sigmoid": "#1baf7a", "sine": "#e87ba4",
       "atanh": "#0d366b", "gelu": "#eda100", "softplus": "#008300"}
MRK = {"tanh": "o", "atanh": "v", "sine": "P", "swish": "s", "gelu": "X", "softplus": "*", "sigmoid": "^"}
PROBS = ["pi_homog", "pi_het", "pi_therapy"]
PNAME = {"pi_homog": "Homogeneous", "pi_het": "Grey/white", "pi_therapy": "Grey/white + therapy"}
INV_ACTS = ["tanh", "atanh", "sine", "swish"]
NOISES = [0.0, 0.01, 0.05]

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#888888", "legend.frameon": False})


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, pattern)))
            if not f.endswith("_field.npz")]


def pm(a, d=2):
    a = np.asarray(a, float)
    if len(a) == 1:
        return sci(a[0], d)
    mu, sd = a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)
    mant, e = f"{mu:.{d}e}".split("e"); e = int(e)
    return rf"$({mant}\pm{sd / 10 ** e:.{d}f})\times10^{{{e}}}$"


def sci(v, d=2):
    if not np.isfinite(v):
        return "--"
    mant, e = f"{v:.{d}e}".split("e")
    return rf"${mant}\times10^{{{int(e)}}}$"


def pct(a):
    a = 100 * np.asarray(a, float)
    if len(a) == 1:
        return rf"${a[0]:.1f}$"
    return rf"${a.mean():.1f}\pm{(a.std(ddof=1) if len(a) > 1 else 0):.1f}$"


S = {}
# ---------------- forward problems ----------------
fwd = {}
for p in PROBS:
    for a in ACTS:
        rs = [r for r in load(f"{p}_{a}_s[0-9].json") if r["ic"] == "hard" and r["colloc"] == "lhs"]
        if rs:
            fwd[(p, a)] = sorted(rs, key=lambda r: r["seed"])

rows = []
best = {}
for p in PROBS:
    avail = [a for a in ACTS if (p, a) in fwd]
    if avail:
        best[p] = min(avail, key=lambda a: np.mean([r["final"]["global_rel_l2"] for r in fwd[(p, a)]]))
for a in ACTS:
    cells = []
    for p in PROBS:
        if (p, a) not in fwd:
            cells.append("--"); continue
        v = [r["final"]["global_rel_l2"] for r in fwd[(p, a)]]
        c = pm(v)
        if best.get(p) == a:
            c += r"$^{\ast}$"
        cells.append(c)
        S[f"{p}_{a}"] = dict(final_mean=float(np.mean(v)), final_sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                             seeds=v, n=len(v),
                             adam_mean=float(np.mean([r["adam"]["global_rel_l2"] for r in fwd[(p, a)]])),
                             linf_mean=float(np.mean([r["final"]["global_linf"] for r in fwd[(p, a)]])),
                             time=float(np.mean([r["time_adam"] + r["time_lbfgs"] for r in fwd[(p, a)]])),
                             lbfgs_ok=int(sum(r["lbfgs_ok"] for r in fwd[(p, a)])))
    t = np.mean([S[f"{p}_{a}"]["time"] for p in PROBS if f"{p}_{a}" in S]) if any(f"{p}_{a}" in S for p in PROBS) else float("nan")
    rows.append(f"{LABEL[a]} & " + " & ".join(cells) + f" & {t:.0f} \\\\")
open(os.path.join(OUT, "table_p2_forward.tex"), "w").write("\n".join(rows))
S["best_forward"] = best
# per-seed wins and pairwise ratio best vs second
for p in PROBS:
    avail = [a for a in ACTS if (p, a) in fwd and len(fwd[(p, a)]) == 3]
    if len(avail) >= 2:
        M = np.array([[r["final"]["global_rel_l2"] for r in fwd[(p, a)]] for a in avail])
        order = np.argsort(M.mean(1))
        wins = {}
        for j in M.argmin(0):
            wins[avail[j]] = wins.get(avail[j], 0) + 1
        S[f"{p}_wins"] = wins
        S[f"{p}_ranking"] = [avail[i] for i in order]
        S[f"{p}_best_vs_second"] = dict(best=avail[order[0]], second=avail[order[1]],
                                        ratio=float(M[order[1]].mean() / M[order[0]].mean()),
                                        seeds_lower=int((M[order[0]] < M[order[1]]).sum()))
        S[f"{p}_spread"] = float(M.mean(1).max() / M.mean(1).min())

# ---------------- CN baseline vs PINN: error vs time ----------------
cn = json.load(open(os.path.join(RES, "cn_baseline.json")))
S["cn"] = cn
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, p in zip(axs, PROBS):
    rows_cn = [r for r in cn if r["problem"] == p]
    ax.plot([r["time"] for r in rows_cn], [r["rel_l2"] for r in rows_cn], "-o", color="#555555", ms=3.5,
            lw=1.4, label="Crank–Nicolson")
    for r in rows_cn:
        if r["N"] in (26, 201, 801):
            ax.annotate(f"N={r['N']}", (r["time"], r["rel_l2"]), textcoords="offset points", xytext=(4, 3),
                        fontsize=6.5, color="#555555")
    for a in ACTS:
        if (p, a) in fwd:
            rs = fwd[(p, a)]
            ax.scatter([r["time_adam"] + r["time_lbfgs"] for r in rs], [r["final"]["global_rel_l2"] for r in rs],
                       color=COL[a], marker=MRK[a], s=22, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Wall-clock time (s)")
    ax.set_title(PNAME[p], fontsize=8.5)
axs[0].set_ylabel(r"Global relative $L^2$ error")
h = [plt.Line2D([], [], color="#555555", marker="o", lw=1.4, ms=3.5, label="Crank–Nicolson")] + \
    [plt.Line2D([], [], color=COL[a], marker=MRK[a], lw=0, ms=5, label=LABEL[a]) for a in ACTS]
fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(os.path.join(OUT, "fig_p2_cost.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig_p2_cost.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- forward solution profiles (best activation, median seed) ----------------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
S["profile_seed"] = {}
for ax, p in zip(axs, PROBS):
    if p not in best:
        continue
    a = {"pi_homog": best["pi_homog"], "pi_het": "tanh", "pi_therapy": "tanh"}[p]
    rs = fwd[(p, a)]
    g = [r["final"]["global_rel_l2"] for r in rs]
    seed = rs[int(np.argsort(g)[len(g) // 2])]["seed"] if p != "pi_therapy" else rs[int(np.argmin(g))]["seed"]
    S["profile_seed"][p] = [a, int(seed)]
    ref = np.load(os.path.join(RES, f"ref_{p}.npz"))
    P = np.load(os.path.join(RES, f"{p}_{a}_s{seed}_field.npz"))["final"]
    shades = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    for k, (j, sh) in enumerate(zip([0, 50, 100, 150, 200], shades)):
        day = ref["s"][j] * m.T_DAYS
        ax.plot(ref["x"] * m.L, ref["c"][j], color=sh, lw=2.0, label=f"day {day:.0f}")
        ax.plot(ref["x"] * m.L, P[j], color="#eb6834", lw=1.0, ls="--")
        if p == "pi_therapy" and j in (100, 200):
            worst = rs[int(np.argmax(g))]["seed"]
            Pw = np.load(os.path.join(RES, f"{p}_{a}_s{worst}_field.npz"))["final"]
            ax.plot(ref["x"] * m.L, Pw[j], color="#e87ba4", lw=1.2, ls=":")
            S["profile_fail_seed"] = int(worst)
    if p != "pi_homog":
        ax.axvline(m.X_IFACE * m.L, color="#888888", lw=0.8, ls=":")
        ax.text(m.X_IFACE * m.L - 0.1, 0.93, "grey", ha="right", fontsize=7, color="#555555")
        ax.text(m.X_IFACE * m.L + 0.1, 0.93, "white", ha="left", fontsize=7, color="#555555")
    ax.set_xlabel("$x$ (cm)"); ax.set_title(f"{PNAME[p]} ({LABEL[a]}, seed {seed})", fontsize=8.5)
    ax.set_ylim(0, 1.0)
axs[0].set_ylabel("Tumour cell density $c$")
h = [plt.Line2D([], [], color=c, lw=2, label=f"day {d}") for c, d in zip(["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"], [0, 45, 90, 135, 180])] + \
    [plt.Line2D([], [], color="#eb6834", lw=1.0, ls="--", label="PINN"),
     plt.Line2D([], [], color="#e87ba4", lw=1.2, ls=":", label="failed PINN (P3, days 90, 180)")]
fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12), fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(OUT, "fig_p2_profiles.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig_p2_profiles.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# therapy schedule panel + error vs time for therapy problem
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
sd = np.linspace(0, m.T_DAYS, 2000)
axs[0].plot(sd, m.k_of_t(sd * m.RHO0), color="#2a78d6", lw=1.6, label="3-day ramps (main)")
import importlib, os as _os
_os.environ["PI_RAMP"] = "1"; m1 = importlib.reload(m)
axs[0].plot(sd, m1.k_of_t(sd * m1.RHO0), color="#eb6834", lw=1.0, ls="--", label="1-day ramps (stress test)")
_os.environ["PI_RAMP"] = "3"; m = importlib.reload(m1)
axs[0].set_xlabel("Days since first scan"); axs[0].set_ylabel(r"Kill rate $k(t)/\rho$")
axs[0].legend(fontsize=7, loc="upper right")
for a in ACTS:
    if ("pi_therapy", a) in fwd:
        for r in fwd[("pi_therapy", a)]:
            E = np.array(r["final"]["rel_l2_vs_t"])[1:]
            axs[1].plot(np.linspace(0, m.T_DAYS, 201)[1:], E, color=COL[a], lw=0.9, alpha=0.85)
axs[1].axhline(0.02, color="#555555", lw=0.8, ls="--")
axs[1].text(178, 0.022, "2%", ha="right", va="bottom", fontsize=7, color="#555555")
axs[1].set_yscale("log"); axs[1].set_xlabel("Days since first scan"); axs[1].set_ylabel(r"Relative $L^2$ error in $x$")
for a_, b_, k_ in m.PULSES:
    axs[1].axvspan(a_, b_, color="#999999", alpha=0.12, lw=0)
axs[1].legend(handles=[plt.Line2D([], [], color=COL[a], lw=1.2, label=LABEL[a]) for a in ACTS], fontsize=6.5, ncol=2, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_p2_therapy.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig_p2_therapy.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- per-seed strip plot ----------------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, p in zip(axs, PROBS):
    for k, a in enumerate(ACTS):
        if (p, a) not in fwd:
            continue
        v = [r["final"]["global_rel_l2"] for r in fwd[(p, a)]]
        ax.hlines(np.median(v), k - 0.3, k + 0.3, color=COL[a], lw=2.0)
        ax.scatter(k + np.array([-0.12, 0, 0.12])[:len(v)], v, color=COL[a], marker=MRK[a], s=22,
                   edgecolor="white", linewidth=0.5, zorder=3)
    if p == "pi_therapy":
        ax.axhline(0.02, color="#555555", lw=0.8, ls="--")
    ax.set_xticks(range(len(ACTS)), [LABEL[a].replace("Adaptive Tanh", "Adapt. Tanh") for a in ACTS], rotation=45, ha="right", fontsize=7.5)
    ax.set_yscale("log"); ax.set_title(PNAME[p], fontsize=8.5); ax.grid(axis="x", visible=False)
axs[0].set_ylabel(r"Global relative $L^2$ error")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_p2_strip.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig_p2_strip.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- inverse problem ----------------
inv = {}
for a in INV_ACTS:
    for n in NOISES:
        rs = load(f"inverse_{a}_s[0-9]_n{n:g}.json")
        if rs:
            inv[(a, n)] = sorted(rs, key=lambda r: r["seed"])
rows = []
for a in INV_ACTS:
    for i, n in enumerate(NOISES):
        if (a, n) not in inv:
            continue
        rs = inv[(a, n)]
        pe = {k: [r["final"]["param_rel_err"][k] for r in rs] for k in ["dg", "dw", "rho"]}
        fe = [r["final"]["global_rel_l2"] for r in rs]
        lead = rf"\multirow{{3}}{{*}}{{{LABEL[a]}}}" if i == 0 else ""
        rows.append(f"{lead} & {100 * n:g}\\% & {pct(pe['dg'])} & {pct(pe['dw'])} & {pct(pe['rho'])} & {pm(fe)} \\\\")
        S[f"inv_{a}_{n:g}"] = dict(dg=pe["dg"], dw=pe["dw"], rho=pe["rho"], field=fe,
                                   dg_mean=float(np.mean(pe["dg"])), dw_mean=float(np.mean(pe["dw"])),
                                   rho_mean=float(np.mean(pe["rho"])), field_mean=float(np.mean(fe)),
                                   adam_param_err={k: float(np.mean([r["adam"]["param_rel_err"][k] for r in rs])) for k in ["dg", "dw", "rho"]},
                                   time=float(np.mean([r["time_adam"] + r["time_lbfgs"] for r in rs])))
    if a != INV_ACTS[-1]:
        rows.append(r"\midrule")
open(os.path.join(OUT, "table_p2_inverse.tex"), "w").write("\n".join(rows))

# inverse figure: parameter trajectories (noise 1%, all inverse acts, seed 1) + error vs noise
if inv:
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5))
    truth = dict(dg=m.DG, dw=m.DW, rho=1.0)
    names = dict(dg=r"$D_g/D_g^{\mathrm{true}}$", dw=r"$D_w/D_w^{\mathrm{true}}$", rho=r"$\rho/\rho^{\mathrm{true}}$")
    for ax, k in zip(axs[:3], ["dg", "dw", "rho"]):
        for a in INV_ACTS:
            if (a, 0.01) in inv:
                for r in inv[(a, 0.01)]:
                    h = r["history"]
                    ax.plot([x["it"] for x in h], [x[k] / truth[k] for x in h], color=COL[a], lw=0.8, alpha=0.8)
        ax.axhline(1.0, color="#555555", lw=0.8, ls=":")
        ax.axvline(4000, color="#888888", lw=0.6, ls="--")
        ax.set_xlabel("Iteration"); ax.set_title(names[k], fontsize=8.5)
    ax = axs[3]
    for a in INV_ACTS:
        y = [100 * np.mean([max(r["final"]["param_rel_err"].values()) for r in inv[(a, n)]]) if (a, n) in inv else np.nan for n in NOISES]
        ax.plot([100 * n for n in NOISES], y, "-", marker=MRK[a], color=COL[a], ms=4, lw=1.2, label=LABEL[a])
    ax.set_yscale("log"); ax.set_xlabel("Noise level (%)"); ax.set_title("Worst parameter error (%)", fontsize=8.5)
    ax.set_yticks([3, 10, 30, 100]); ax.set_yticklabels(["3", "10", "30", "100"]); ax.minorticks_off()
    ax.set_xticks([0, 1, 5])
    h = [plt.Line2D([], [], color=COL[a], marker=MRK[a], lw=1.2, ms=4, label=LABEL[a]) for a in INV_ACTS]
    fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "fig_p2_inverse.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig_p2_inverse.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------- ablations ----------------
abl = {}
abl["hard IC, LHS (main)"] = fwd.get(("pi_het", "tanh"), [])
abl["soft IC, LHS"] = load("pi_het_tanh_s[0-9]_icsoft.json")
abl["hard IC, Sobol"] = load("pi_het_tanh_s[0-9]_sobol.json")
abl["hard IC, uniform random"] = load("pi_het_tanh_s[0-9]_random.json")
rows = []
for k, rs in abl.items():
    if not rs:
        continue
    v = [r["final"]["global_rel_l2"] for r in rs]
    va = [r["adam"]["global_rel_l2"] for r in rs]
    rows.append(f"{k} & {pm(va)} & {pm(v)} \\\\")
    S[f"abl_{k}"] = dict(final=v, adam=va, final_mean=float(np.mean(v)), adam_mean=float(np.mean(va)))
open(os.path.join(OUT, "table_p2_ablation.tex"), "w").write("\n".join(rows))

# ---------------- stress test (1-day ramps) ----------------
rows = []
for a in ACTS:
    f = os.path.join(RES, "ramp1", f"pi_therapy_{a}_s1.json")
    if os.path.exists(f):
        r = json.load(open(f))
        S[f"ramp1_{a}"] = dict(final=r["final"]["global_rel_l2"], adam=r["adam"]["global_rel_l2"])
        main = S.get(f"pi_therapy_{a}", {}).get("seeds", [np.nan])[0]
        rows.append(f"{LABEL[a]} & {sci(r['final']['global_rel_l2'])} & {sci(r['adam']['global_rel_l2'])} & {sci(main)} \\\\")
open(os.path.join(OUT, "table_p2_stress.tex"), "w").write("\n".join(rows))

json.dump(S, open(os.path.join(OUT, "summary_p2.json"), "w"), indent=1)
print("forward cfgs", len(fwd), "inverse cfgs", len(inv), "ablation", {k: len(v) for k, v in abl.items()})
