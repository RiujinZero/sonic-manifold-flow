#!/bin/bash
cd "/mnt/c/Users/user/Documents/projects/sonic-manifold-flow" || exit 1
mkdir -p artifacts/sonic_reverse_query_probe
setsid /home/user/gtr_venv/bin/python -m manifoldflow.experiments.sonic_reverse_query_probe \
  --mode decode --steps 4000 --eval-every 100 --eval-episodes 60 \
  --unfreeze-sonic-at-step 2000 \
  --output artifacts/sonic_reverse_query_probe/decode_staged_unfreeze.json \
  < /dev/null > /tmp/reverse_decode_staged.log 2>&1 &
disown
sleep 2
echo "started pid $!"
