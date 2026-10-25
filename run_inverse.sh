#!/bin/bash
# Inverse-problem sweep (P4): 4 activations x noise {1%, 5%} x 3 seeds, plus noise-free x 1 seed.
cd "$(dirname "$0")"
R=../results_p2
{
  for s in 1 2 3; do for n in 0.01 0.05; do for a in tanh atanh sine swish; do
    echo "$R/inverse_${a}_s${s}_n${n}.json|python3 pinn_pi.py --problem inverse --act $a --seed $s --noise $n"; done; done; done
  for a in tanh atanh sine swish; do
    echo "$R/inverse_${a}_s1_n0.json|python3 pinn_pi.py --problem inverse --act $a --seed 1 --noise 0"; done
} | while IFS='|' read -r f cmd; do [ -f "$f" ] || echo "$cmd"; done \
  | xargs -P 2 -I{} sh -c '{} 2>/dev/null' >> $R/run_inverse.log
echo DONE >> $R/run_inverse.log
