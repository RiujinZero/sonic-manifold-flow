"""Minimal vendored subset of GTR-sonic-ssm's `gator_linear` package.

Only `_hgrn_compiled` is vendored here - `sonic_ssm.SonicSSMBlock` imports
`compiled_chunk_hgrn` from it unconditionally at module load time, even
though that function is only ever called when `scan_backend="hgrn"` is
explicitly selected (never the default, and unavailable without Triton/
`fla` on this machine - see `_hgrn_compiled.py`'s own guarded import).
The upstream package's other submodules (`coalition`, `core`, `prefix`,
`evotoken` - a different, unrelated linear-attention architecture in the
same source repo) are not part of Sonic and were not copied.
"""

from __future__ import annotations
