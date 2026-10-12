# Vendored from GTR-sonic-ssm

A wholesale copy of `gator/` and `gator_linear/` from
the `GTR-sonic-ssm` repository (2026-08-28),
not a hand-picked subset. This is deliberately a bulk drop rather than the
surgical, file-at-a-time vendoring `src/manifoldflow/sonic_ssm.py` and
`src/gator_linear/` used earlier - the request behind this copy was "pull
the hard-won Sonic optimizations over" without spending time dissecting
which exact lines are load-bearing.

Not wired into `src/manifoldflow` or `pyproject.toml`'s packaging - this
is reference material for later training-infrastructure work, not an
active dependency of anything in this repo yet.

## What's actually in here

- `gator/models/sonic_ssm.py` - the same model `src/manifoldflow/sonic_ssm.py`
  was vendored from (byte-identical at copy time); this bulk copy is not a
  replacement for that more carefully provenance-tracked file.
- `gator/sonic_ssm_train.py`, `gator/smollm3_train.py`,
  `gator/smollm3_recipe.py` - Sonic/SmolLM3-specific training entry
  points, narrower and more directly relevant than the top-level
  `gator_language_train.py` CLI script (which was *not* copied here -
  importing the full trainer is on hold; see the project's session notes).
- `gator_linear/_hgrn_compiled.py`, `gator_linear/_hgrn_fixed_decay.py` -
  the real Triton HGRN kernel source (`src/gator_linear/_hgrn_compiled.py`
  only has the same file with a guarded import - functionally identical,
  this copy just sits alongside the rest of `gator_linear/` instead of
  standing alone).
- The rest of `gator/` (`cli.py`, `data/`, `tasks/`, `optimizers/`,
  `continual/`, `memory/`, CUDA-graph helpers, etc.) - largely
  Gator/GTR-ARW-specific, not inspected yet for Sonic relevance.

## Triton/`fla` status

Neither is installed in this machine's WSL (Ubuntu) environment - a
search turned up only a `torchinductor` cache directory, no real package.
The actual GPU-optimized training environment (the one reportedly
reaching ~57k tokens/sec on a 5080 with `sonic_ssm.py`'s
`scan_backend="hgrn"` path) is not this machine's WSL install; where it
actually lives has not been located yet. Until it is, `sonic_ssm.py`'s
default `scan_backend="doubling"` (pure PyTorch, no Triton) is the only
runnable path here, on CPU, matching every probe run in this project so
far.
