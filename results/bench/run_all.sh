#!/bin/bash
# Full benchmark sweep. Sequential so the GPU is never shared.
cd "/c/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
PY=.venv/Scripts/python.exe
B=proj1/scripts/benchmark_base.py
CK=betty_pull/fm.pt
O=results/bench
run() { $PY -u $B "$@" 2>&1 | grep -v Warning; }

echo "### main: NFE 100 euler, 3 seeds x 10k (our operating point)"
for s in 0 1 2; do run --ckpt $CK --n 10000 --steps 100 --solver euler --seed $s --out $O/fm_nfe100_euler_s$s.json; done

echo "### heun NFE 100 (200 evaluations), 1 seed x 10k"
run --ckpt $CK --n 10000 --steps 100 --solver heun --seed 0 --out $O/fm_nfe100_heun_s0.json

echo "### NFE 250 / 500, 1 seed x 10k"
run --ckpt $CK --n 10000 --steps 250 --solver euler --seed 0 --out $O/fm_nfe250_euler_s0.json
run --ckpt $CK --n 10000 --steps 500 --solver euler --seed 0 --out $O/fm_nfe500_euler_s0.json

echo "### NFE 1000 (EDM protocol), 1 seed x 5k"
run --ckpt $CK --n 5000 --steps 1000 --solver euler --seed 0 --out $O/fm_nfe1000_euler_s0.json

echo "### epoch curve: every snapshot, 2k samples, NFE 100, fresh seed 7"
for e in 0100 0200 0300 0400 0500 0600 0700 0800 0900 1000 1100 1200 1300 1400 1500; do
  run --ckpt betty_pull/fm_ep$e.pt --n 2000 --steps 100 --solver euler --seed 7 --out $O/ep${e}_nfe100_s7.json
done

echo "### epoch 1500 (fm_last EMA) vs selected 1300, 10k, NFE 100, seed 0 -- the selection question"
run --ckpt betty_pull/fm_last.pt --n 10000 --steps 100 --solver euler --seed 0 --out $O/fmlast_ep1500_nfe100_euler_s0.json

echo "### raw (non-EMA) weights, 5k, NFE 100 -- is EMA helping?"
run --ckpt $CK --n 5000 --steps 100 --solver euler --seed 0 --raw --out $O/fm_raw_nfe100_euler_s0.json

echo "### ALL DONE $(date -Is)"
