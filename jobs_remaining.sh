#!/bin/bash
# Prints the pending Phase 2 jobs (one command per line), skipping completed runs.
R=../results_p2
{
  for p in pi_therapy pi_homog; do for a in tanh swish sine atanh sigmoid gelu softplus; do
    echo "$R/${p}_${a}_s3.json|python3 pinn_pi.py --problem $p --act $a --seed 3"; done; done
  for s in 1 2 3; do
    echo "$R/pi_het_tanh_s${s}_icsoft.json|python3 pinn_pi.py --problem pi_het --act tanh --seed $s --ic soft"
    echo "$R/pi_het_tanh_s${s}_sobol.json|python3 pinn_pi.py --problem pi_het --act tanh --seed $s --colloc sobol"
    echo "$R/pi_het_tanh_s${s}_random.json|python3 pinn_pi.py --problem pi_het --act tanh --seed $s --colloc random"
  done
  for a in tanh swish sine atanh sigmoid gelu softplus; do
    echo "$R/ramp1/pi_therapy_${a}_s1.json|PI_RAMP=1 PI_REF_DIR=$R/ramp1 python3 pinn_pi.py --problem pi_therapy --act $a --seed 1 --out $R/ramp1"; done
} | while IFS='|' read -r f cmd; do [ -f "$f" ] || echo "$cmd"; done
