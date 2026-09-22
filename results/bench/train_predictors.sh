#!/bin/bash
cd "/c/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
PY=.venv/Scripts/python.exe
for prop in mu alpha gap; do
  for split in train_a train_b; do
    $PY -u proj1/scripts/train_predictor.py --split $split --arch ridge --prop $prop --fresh 2>&1 | grep -E "done:|ridge fitted"
  done
done
for prop in mu alpha gap; do
  for split in train_a train_b; do
    $PY -u proj1/scripts/train_predictor.py --split $split --arch transformer --prop $prop \
        --epochs 120 --hidden 128 --layers 4 --heads 8 --fresh 2>&1 | grep -E "done:"
  done
done
echo "### PREDICTORS DONE"
