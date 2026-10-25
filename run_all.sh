#!/bin/bash
# Runs all 60 configurations (3 examples x 4 activations x 5 seeds), 2 at a time.
cd "$(dirname "$0")"
for s in 1234 1235 1236 1237 1238; do for e in 1 2 3; do for a in tanh swish sigmoid relu; do
  [ -f ../results/ex${e}_${a}_s${s}.json ] || echo "$e $a $s"
done; done; done | xargs -P 2 -n 3 sh -c 'python3 pinn.py --example $0 --act $1 --seed $2 --out ../results' >> ../results/run.log 2>&1
echo DONE >> ../results/run.log
