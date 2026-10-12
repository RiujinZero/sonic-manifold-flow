#!/bin/bash
cd "/mnt/c/Users/user/Documents/projects/sonic-manifold-flow" || exit 1
mkdir -p artifacts/sonic_noun_target_probe
setsid /home/user/gtr_venv/bin/python -m manifoldflow.experiments.sonic_noun_target_probe \
  --mode decode --steps 6000 --eval-every 100 --eval-episodes 60 \
  --held-out-eval-episodes 100 \
  --output artifacts/sonic_noun_target_probe/decode_frozen_generalization.json \
  < /dev/null > /tmp/noun_decode_frozen_generalization.log 2>&1 &
disown
sleep 2
echo "started pid $!"
