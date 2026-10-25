"""Crank-Nicolson baseline: error versus wall-clock time on the three forward PI problems."""
import json, time, sys
import numpy as np
import pi_model as m

out = sys.argv[1] if len(sys.argv) > 1 else "../results_p2/cn_baseline.json"
rows = []
for p in ["pi_homog", "pi_het", "pi_therapy"]:
    ref = np.load(f"../results_p2/ref_{p}.npz")
    U = ref["c"]
    for N in [26, 51, 101, 201, 401, 801]:
        Nt = int(np.ceil(10 * (N - 1) / 200) * 200)
        times = []
        for rep in range(3):
            t0 = time.perf_counter()
            x, ts, sol = m.crank_nicolson(p, N, Nt, store_every=Nt // 200)
            times.append(time.perf_counter() - t0)
        P = np.array([np.interp(ref["x"], x, row) for row in sol])
        E = U - P
        rows.append(dict(problem=p, N=N, Nt=Nt, time=min(times),
                         rel_l2=float(np.linalg.norm(E) / np.linalg.norm(U)),
                         linf=float(np.abs(E).max())))
        print(rows[-1])
json.dump(rows, open(out, "w"), indent=1)
