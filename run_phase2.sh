#!/bin/bash
# Phase 2 sweep: forward PI problems, ablations and stress test (2 jobs at a time).
cd "$(dirname "$0")"
R=../results_p2
jobs() {
  for s in 1 2 3; do for p in pi_het pi_therapy pi_homog; do for a in tanh swish sine atanh sigmoid gelu softplus; do
    echo "$R/${p}_${a}_s${s}.json|python3 pinn_pi.py --problem $p --act $a --seed $s"; done; done; done
  for s in 1 2 3; do
    echo "$R/pi_het_tanh_s${s}_icsoft.json|python3 pinn_pi.py --problem pi_het --act tanh --seed $s --ic soft"
    echo "$R/pi_het_tanh_s${s}_sobol.json|python3 pinn_pi.py --problem pi_het --act tanh --seed $s --colloc sobol"
    echo "$R/pi_het_tanh_s${s}_random.json|python3 pinn_pi.py --problem pi_het --act tanh --seed $s --colloc random"
  done
  for a in tanh swish sine atanh sigmoid gelu softplus; do
    echo "$R/ramp1/pi_therapy_${a}_s1.json|PI_RAMP=1 PI_REF_DIR=$R/ramp1 python3 pinn_pi.py --problem pi_therapy --act $a --seed 1 --out $R/ramp1"; done
}
jobs | while IFS='|' read -r f cmd; do [ -f "$f" ] || echo "$cmd"; done \
  | xargs -P 2 -I{} sh -c '{} 2>/dev/null' >> $R/run.log
echo DONE >> $R/run.log
