#!/bin/bash
cd "/mnt/c/Users/user/Documents/projects/sonic-manifold-flow" || exit 1
mkdir -p artifacts/sonic_noun_target_probe
setsid /home/user/gtr_venv/bin/python -m manifoldflow.experiments.sonic_noun_target_probe \
  --mode retrieval --steps 10000 --eval-every 100 --eval-episodes 60 \
  --recency-weight 0.3 --recency-decay 0.7 \
  --held-out-eval-episodes 100 \
  --output artifacts/sonic_noun_target_probe/digit_frozen_generalization.json \
  < /dev/null > /tmp/digit_frozen_generalization.log 2>&1 &
disown
sleep 2
echo "started pid $!"
