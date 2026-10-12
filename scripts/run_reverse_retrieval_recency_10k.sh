#!/bin/bash
cd "/mnt/c/Users/user/Documents/projects/sonic-manifold-flow" || exit 1
mkdir -p artifacts/sonic_reverse_query_probe
setsid /home/user/gtr_venv/bin/python -m manifoldflow.experiments.sonic_reverse_query_probe \
  --mode retrieval --steps 10000 --eval-every 100 --eval-episodes 60 \
  --recency-weight 0.3 --recency-decay 0.7 \
  --output artifacts/sonic_reverse_query_probe/retrieval_recency_10k.json \
  < /dev/null > /tmp/reverse_retrieval_recency_10k.log 2>&1 &
disown
sleep 2
echo "started pid $!"
