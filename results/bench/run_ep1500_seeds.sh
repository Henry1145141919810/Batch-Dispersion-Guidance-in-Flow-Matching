#!/bin/bash
cd "/c/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
for s in 1 2; do
  .venv/Scripts/python.exe -u proj1/scripts/benchmark_base.py --ckpt betty_pull/fm_last.pt --n 10000 --steps 100 --solver euler --seed $s --out results/bench/fmlast_ep1500_nfe100_euler_s$s.json 2>&1 | grep -v Warning
done
.venv/Scripts/python.exe -u proj1/scripts/benchmark_base.py --rescore results/bench/fmlast_ep1500_nfe100_euler_s1_samples.pt results/bench/fmlast_ep1500_nfe100_euler_s2_samples.pt --out results/bench/rescored 2>&1 | grep -v Warning
echo "### EP1500 SEEDS DONE"
