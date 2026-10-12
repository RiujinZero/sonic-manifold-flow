#!/bin/bash
cd "/mnt/c/Users/user/Documents/projects/sonic-manifold-flow" || exit 1
mkdir -p artifacts/sonic_noun_target_probe
setsid /home/user/gtr_venv/bin/python -m manifoldflow.experiments.sonic_noun_target_probe \
  --mode decode --steps 4000 --eval-every 100 --eval-episodes 60 \
  --output artifacts/sonic_noun_target_probe/decode_frozen.json \
  < /dev/null > /tmp/noun_target_decode_frozen.log 2>&1 &
disown
sleep 2
echo "started pid $!"
