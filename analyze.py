"""Aggregate the 60 runs into tables (LaTeX) and figures (PDF) for the manuscript."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "../results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../manuscript/figures"
os.makedirs(OUT, exist_ok=True)

ACTS = ["tanh", "swish", "sigmoid", "relu"]
LABEL = {"tanh": "Tanh", "swish": "Swish", "sigmoid": "Sigmoid", "relu": "ReLU"}
COL = {"tanh": "#2a78d6", "swish": "#eb6834", "sigmoid": "#1baf7a", "relu": "#eda100"}
MRK = {"tanh": "o", "swish": "s", "sigmoid": "^", "relu": "D"}
EXS = [1, 2, 3]
EXNAME = {1: "Example 1 (constant growth)", 2: "Example 2 (nonlinear source)", 3: "Example 3 (therapy kill term)"}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#888888", "legend.frameon": False})

runs = {}
for f in glob.glob(os.path.join(RES, "ex*_s*.json")):
    r = json.load(open(f))
    runs.setdefault((r["example"], r["act"]), []).append(r)
for k in runs:
    runs[k].sort(key=lambda r: r["seed"])

def arr(ex, act, fn):
    return np.array([fn(r) for r in runs[(ex, act)]])

def ms(a):
    return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0

def sci(v, d=2):
    m, e = f"{v:.{d}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"

def pm(a):
    m, s = ms(a)
    m_, e = f"{m:.2e}".split("e"); e = int(e)
    return rf"$({m_}\pm{s / 10**e:.2f})\times10^{{{e}}}$"

summary = {}
# ---------- Table: main results ----------
rows = []
for ex in EXS:
    best_final = min(ACTS, key=lambda a: np.mean(arr(ex, a, lambda r: r["final"]["global_rel_l2"])))
    best_adam = min(ACTS, key=lambda a: np.mean(arr(ex, a, lambda r: r["adam"]["global_rel_l2"])))
    for i, a in enumerate(ACTS):
        g_adam = arr(ex, a, lambda r: r["adam"]["global_rel_l2"])
        g_fin = arr(ex, a, lambda r: r["final"]["global_rel_l2"])
        linf = arr(ex, a, lambda r: r["final"]["global_linf"])
        t_tot = arr(ex, a, lambda r: r["time_adam"] + r["time_lbfgs"])
        ok = sum(r["lbfgs_ok"] for r in runs[(ex, a)])
        ca = pm(g_adam); cf = pm(g_fin)
        if a == best_adam: ca = ca + r"$^{\ast}$"
        if a == best_final: cf = cf + r"$^{\ast}$"
        lead = rf"\multirow{{4}}{{*}}{{{ex}}}" if i == 0 else ""
        rows.append(f"{lead} & {LABEL[a]} & {ca} & {cf} & {pm(linf)} & {np.mean(t_tot):.0f} \\\\")
        summary[f"ex{ex}_{a}"] = dict(
            adam_mean=ms(g_adam)[0], adam_sd=ms(g_adam)[1], final_mean=ms(g_fin)[0], final_sd=ms(g_fin)[1],
            linf_mean=ms(linf)[0], linf_sd=ms(linf)[1], time=float(np.mean(t_tot)),
            time_adam=float(np.mean(arr(ex, a, lambda r: r["time_adam"]))),
            lbfgs_ok=int(ok), n=len(runs[(ex, a)]),
            early_adam_mean=ms(arr(ex, a, lambda r: r["adam"]["early_mean_rel_l2"]))[0],
            early_adam_sd=ms(arr(ex, a, lambda r: r["adam"]["early_mean_rel_l2"]))[1],
            early_final_mean=ms(arr(ex, a, lambda r: r["final"]["early_mean_rel_l2"]))[0],
            early_adam_linf=ms(arr(ex, a, lambda r: r["adam"]["early_max_linf"]))[0],
            adam_linf_mean=ms(arr(ex, a, lambda r: r["adam"]["global_linf"]))[0],
            probe={t: ms(arr(ex, a, lambda r, t=t: r["final"]["probe"][t]["rel_l2"]))
                   for t in ["0.1", "0.25", "0.5", "0.75", "1.0"]},
            probe_linf={t: ms(arr(ex, a, lambda r, t=t: r["final"]["probe"][t]["linf"]))
                        for t in ["0.1", "0.25", "0.5", "0.75", "1.0"]},
            seeds_final=g_fin.tolist(), seeds_adam=g_adam.tolist())
    if ex != EXS[-1]:
        rows.append(r"\midrule")
    summary[f"ex{ex}_best_final"] = best_final
    summary[f"ex{ex}_best_adam"] = best_adam
    # rank wins per seed
    for stage in ["adam", "final"]:
        M = np.array([arr(ex, a, lambda r, s=stage: r[s]["global_rel_l2"]) for a in ACTS])
        wins = {ACTS[i]: int(c) for i, c in zip(*np.unique(M.argmin(0), return_counts=True))}
        summary[f"ex{ex}_wins_{stage}"] = wins
        order = np.argsort(M.mean(1))
        b, s2 = order[0], order[1]
        summary[f"ex{ex}_pair_{stage}"] = dict(best=ACTS[b], second=ACTS[s2],
            seeds_best_lower=int((M[b] < M[s2]).sum()), ratio=float(M[s2].mean() / M[b].mean()))
        # full ranking per seed
        summary[f"ex{ex}_ranking_{stage}"] = [ACTS[i] for i in order]

with open(os.path.join(OUT, "table_main.tex"), "w") as f:
    f.write("\n".join(rows))

# ---------- Table: errors at probe times (final, all activations) ----------
rows = []
for ex in EXS:
    for i, a in enumerate(ACTS):
        lead = rf"\multirow{{4}}{{*}}{{{ex}}}" if i == 0 else ""
        cells = []
        for t in ["0.1", "0.25", "0.5", "0.75", "1.0"]:
            m, s = summary[f"ex{ex}_{a}"]["probe"][t]
            cells.append(f"${sci(m, 1)}$")
        rows.append(f"{lead} & {LABEL[a]} & " + " & ".join(cells) + r" \\")
    if ex != EXS[-1]:
        rows.append(r"\midrule")
with open(os.path.join(OUT, "table_probe.tex"), "w") as f:
    f.write("\n".join(rows))

# ---------- Table: early window vs full domain (Adam stage, original protocol) ----------
rows = []
for ex in EXS:
    for i, a in enumerate(ACTS):
        s = summary[f"ex{ex}_{a}"]
        lead = rf"\multirow{{4}}{{*}}{{{ex}}}" if i == 0 else ""
        rows.append(f"{lead} & {LABEL[a]} & ${sci(s['early_adam_mean'])}$ & ${sci(s['adam_mean'])}$ & "
                    f"{s['adam_mean'] / s['early_adam_mean']:.1f} & ${sci(s['early_adam_linf'])}$ & ${sci(s['adam_linf_mean'])}$ \\\\")
    if ex != EXS[-1]:
        rows.append(r"\midrule")
with open(os.path.join(OUT, "table_window.tex"), "w") as f:
    f.write("\n".join(rows))

json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

# ---------- Figure 1: global error, Adam vs Adam+L-BFGS ----------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
for ax, ex in zip(axs, EXS):
    xpos = np.arange(len(ACTS))
    for j, (stage, off, alpha) in enumerate([("adam", -0.18, 0.45), ("final", 0.18, 1.0)]):
        for k, a in enumerate(ACTS):
            v = arr(ex, a, lambda r, s=stage: r[s]["global_rel_l2"])
            ax.bar(k + off, np.mean(v), width=0.32, color=COL[a], alpha=alpha,
                   edgecolor="white", linewidth=0.8, zorder=2)
            ax.scatter(np.full(len(v), k + off) + np.linspace(-0.07, 0.07, len(v)), v,
                       s=7, color="#333333", zorder=3, linewidths=0)
    ax.set_yscale("log")
    ax.set_xticks(xpos, [LABEL[a] for a in ACTS], fontsize=7.5)
    ax.set_title(EXNAME[ex], fontsize=8.5)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel(r"Global relative $L^2$ error")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="#777777", alpha=0.45, label="Adam (4,000 epochs)"),
                    Patch(color="#777777", label="Adam + L-BFGS")],
           loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "fig_global_error.pdf")); fig.savefig(os.path.join(OUT, "fig_global_error.png"), dpi=150)
plt.close(fig)

# ---------- Figure 2: relative L2 error vs time (final) ----------
ts = np.linspace(0, 1, 201)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, ex in zip(axs, EXS):
    for a in ACTS:
        E = np.array([r["final"]["rel_l2_vs_t"] for r in runs[(ex, a)]])[:, 1:]
        ax.plot(ts[1:], np.median(E, 0), color=COL[a], lw=1.6, label=LABEL[a])
        ax.fill_between(ts[1:], E.min(0), E.max(0), color=COL[a], alpha=0.15, linewidth=0)
        ax.plot(ts[1::25], np.median(E, 0)[::25], MRK[a], color=COL[a], ms=3.5, mec="white", mew=0.5)
    ax.axvspan(0.01, 0.10, color="#999999", alpha=0.12, linewidth=0)
    ax.set_yscale("log"); ax.set_xlabel("$t$"); ax.set_title(EXNAME[ex], fontsize=8.5)
axs[0].set_ylabel(r"Relative $L^2$ error in $x$")
h = [plt.Line2D([], [], color=COL[a], marker=MRK[a], lw=1.6, ms=4, label=LABEL[a]) for a in ACTS]
fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "fig_error_time.pdf")); fig.savefig(os.path.join(OUT, "fig_error_time.png"), dpi=150)
plt.close(fig)

# ---------- Figure 3: loss histories (median over seeds) ----------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, ex in zip(axs, EXS):
    for a in ACTS:
        H = runs[(ex, a)]
        ad = np.array([[h["loss"] for h in r["history"] if h["stage"] == "adam"] for r in H])
        it = [h["it"] for h in H[0]["history"] if h["stage"] == "adam"]
        ax.plot(it, np.median(ad, 0), color=COL[a], lw=1.2, label=LABEL[a])
        # L-BFGS: ragged lengths -> plot each seed thinly
        for r in H:
            lb = [(h["it"], h["loss"]) for h in r["history"] if h["stage"] == "lbfgs"]
            if lb:
                ax.plot(*zip(*lb), color=COL[a], lw=0.6, alpha=0.6)
    ax.axvline(4000, color="#888888", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("Iteration"); ax.set_title(EXNAME[ex], fontsize=8.5)
axs[0].set_ylabel("Training loss")
h = [plt.Line2D([], [], color=COL[a], lw=1.6, label=LABEL[a]) for a in ACTS]
fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "fig_loss.pdf")); fig.savefig(os.path.join(OUT, "fig_loss.png"), dpi=150)
plt.close(fig)

# ---------- Figure 4: full-domain fields, best activation, median seed ----------
fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.6))
for row, ex in enumerate(EXS):
    a = summary[f"ex{ex}_best_final"]
    g = arr(ex, a, lambda r: r["final"]["global_rel_l2"])
    seed = runs[(ex, a)][int(np.argsort(g)[len(g) // 2])]["seed"]
    d = np.load(os.path.join(RES, f"ex{ex}_{a}_s{seed}_field.npz"))
    ext = [0, 1, 0, 1]
    vmin, vmax = d["exact"].min(), d["exact"].max()
    err = np.abs(d["exact"] - d["final"])
    ex_p = int(np.floor(np.log10(err.max())))
    for col, (Z, title, cmap) in enumerate([(d["final"], f"PINN ({LABEL[a]})", "viridis"),
                                             (d["exact"], "Exact", "viridis"),
                                             (err / 10.0**ex_p, "|error| (" + r"$\times10^{" + str(ex_p) + "}$)", "magma")]):
        ax = axs[row, col]
        kw = dict(vmin=vmin, vmax=vmax) if col < 2 else {}
        im = ax.imshow(Z, origin="lower", extent=ext, aspect="auto", cmap=cmap, **kw)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(f"Ex. {ex}: {title}", fontsize=8.5)
        ax.set_xlabel("$x$"); ax.set_ylabel("$t$")
    summary[f"ex{ex}_field_seed"] = int(seed)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_fields.pdf")); fig.savefig(os.path.join(OUT, "fig_fields.png"), dpi=150)
plt.close(fig)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print("ok", {k: len(v) for k, v in runs.items()})
