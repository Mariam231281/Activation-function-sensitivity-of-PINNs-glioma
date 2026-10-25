# PINN activation-function study for glioma reaction–diffusion models

Code for *Activation-function sensitivity of physics-informed neural networks for forward and inverse glioma proliferation–invasion models* (Raza & Sultana).

Repository: https://github.com/Mariam231281/Activation-function-sensitivity-of-PINNs-glioma

## Contents

| File | Purpose |
|---|---|
| `pinn.py` | Trains one PINN (example × activation × seed): Adam 4,000 epochs, then L-BFGS; evaluates on a 201×201 grid; writes `<tag>.json` (metrics, loss history) and `<tag>_field.npz` (solution fields). |
| `run_all.sh` | Runs all 60 configurations (3 examples × 4 activations × 5 seeds), two at a time. Skips runs already completed. |
| `verify_exact.py` | Symbolic check (SymPy) that each exact solution satisfies its PDE. |
| `analyze.py` | Aggregates results into the LaTeX tables and PDF figures used in the manuscript. |

## Requirements

Python ≥ 3.10, PyTorch ≥ 2.0, NumPy, SciPy, Matplotlib, SymPy. All runs use float64 on a single CPU thread.

## Setup

The scripts read and write results in folders next to the code folder:

```bash
pip install torch numpy scipy matplotlib sympy
mkdir -p ../results ../results_p2/ramp1
```

## Reproduce

```bash
python verify_exact.py
./run_all.sh                       # ~1.5–2 h on two CPU cores
python analyze.py ../results ../manuscript/figures
```

Single run: `python pinn.py --example 3 --act swish --seed 1234 --out ../results`

## Problems

| | Source F(x,t,u) | u0(x) | Exact u(x,t) |
|---|---|---|---|
| Example 1 | u/2 | e^x | e^(x+t) |
| Example 2 | e^(−u) + e^(−2u)/2 | ln(x+2) | ln(x+t+2) |
| Example 3 | −t² u (therapy kill term) | e^(2x) | exp(2x + 2t − t³/3) |

All with u_t = u_xx/2 + F on [0,1]², Dirichlet data from the exact solution.

## Part II: proliferation–invasion model

| File | Purpose |
|---|---|
| `pi_model.py` | Model definition (literature D, ρ; grey/white interface; mirrored Gaussian seed; Stupp-timed therapy schedule), Crank–Nicolson finite-volume solver and fine-grid reference solutions. |
| `pinn_pi.py` | PINN for the forward problems P1–P3 (`--problem pi_homog | pi_het | pi_therapy`), seven activations (`tanh, atanh, sine, swish, gelu, softplus, sigmoid`), ablations (`--ic soft`, `--colloc sobol|random`). Inputs scaled to [−1,1], 2,048 collocation points, Adam 4,000 + L-BFGS 2,000. `--problem inverse --noise σ` solves P4: recovers D_grey, D_white and ρ from two simulated scans (days 90 and 180, 50 points each). |
| `cn_baseline.py` | Crank–Nicolson error versus wall-clock time on 26–801 nodes. |
| `jobs_remaining.sh`, `run_next.sh`, `run_phase2.sh`, `run_inverse.sh` | Job lists and runners for the Part II forward and inverse sweeps (resumable; skip completed runs). |
| `analyze_p2.py` | Part II tables and figures. |

Reproduce Part II:

```bash
python -c "import numpy as np, pi_model as m; [np.savez_compressed(f'../results_p2/ref_{p}.npz', **dict(zip(['x','s','c'], m.reference(p)))) for p in m.PROBLEMS]"
python cn_baseline.py
./run_phase2.sh            # forward runs, ablations, stress test (or repeatedly: ./run_next.sh 2)
./run_inverse.sh           # inverse problem P4
python analyze_p2.py ../results_p2 ../manuscript/figures
```
The 1-day-ramp stress test uses `PI_RAMP=1` with its own reference in `results_p2/ramp1/`, created with:

```bash
PI_RAMP=1 python -c "import numpy as np, pi_model as m; x,s,c=m.reference('pi_therapy'); np.savez_compressed('../results_p2/ramp1/ref_pi_therapy.npz', x=x, s=s, c=c)"
```
