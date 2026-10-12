"""Muon: an optimizer for a network's own hidden 2D weight matrices.

Implemented directly from the published algorithm (Keller Jordan et al.,
"Muon: An optimizer for the hidden layers of neural networks", and the
`modded-nanogpt` speedrun lineage that popularized it) rather than vendored
from an external file - see each function's docstring for the specific
formulation followed.

Muon replaces Adam's per-parameter second-moment normalization with an
explicit orthogonalization of the momentum-averaged gradient, via a fast
Newton-Schulz iteration that approximates the matrix sign function (the
zeroth power of the singular value decomposition, hence "zeropower"). It
is defined only for 2D parameters - embeddings, output/bias vectors, and
normalization gains are out of scope by construction (a lookup table or a
1D vector has no meaningful orthogonalization) and should stay on Adam/
AdamW, which this module does not replace.
"""

from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximates `G`'s zeroth matrix power (i.e. `U @ V.T` from its SVD
    `G = U @ S @ V.T`) via `steps` iterations of a quintic Newton-Schulz
    polynomial, following Jordan et al.'s tuned coefficients. Converges to
    an orthogonal matrix (in the sense that `X @ X.T` approaches the
    identity on the matrix's smaller dimension) without ever computing an
    SVD - Newton-Schulz needs only matrix multiplies, which is what makes
    this cheap enough to run every optimizer step on every hidden matrix.

    Operates in bf16 for speed; the whole point of the polynomial's odd
    coefficients (rather than the textbook `1.5, -0.5` Newton-Schulz
    iteration for matrix inverse square roots) is that it stays numerically
    well-behaved at bf16 precision over a handful of steps.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """One 2D-only optimizer group per Jordan et al.: momentum (optionally
    Nesterov-accelerated) accumulated in fp32, then orthogonalized via
    `zeropower_via_newtonschulz5` before being applied as the update.

    The `max(1, rows/cols) ** 0.5` scale factor matches the update's RMS
    to roughly what a well-tuned per-element optimizer (Adam) would
    produce on a non-square matrix - Newton-Schulz's output is orthogonal
    regardless of aspect ratio, but a wide-and-short update needs scaling
    up relative to a tall-and-narrow one to move the parameter the "same
    distance" in aggregate. Decoupled weight decay (applied to the
    parameter directly, not folded into the gradient) matches AdamW's own
    convention, for consistency with the AdamW groups this optimizer runs
    alongside in this project's cross-optimizer setup.

    Silently skips any parameter with `requires_grad=False` or no `.grad`
    - the same convention `torch.optim.Adam`/`AdamW` already follow, which
    this project's staged Sonic-unfreeze design (Sonic's own params join
    the optimizer from step 1 but stay frozen until an accuracy gate
    fires) depends on."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be positive")
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay
        )
        params = list(params)
        for p in params:
            if p.ndim != 2:
                raise ValueError(
                    f"Muon only supports 2D parameters (got shape {tuple(p.shape)}) - "
                    "route embeddings/biases/norms/scalars to an Adam(W) group instead"
                )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad.float())
                update = grad.float().add(buf, alpha=momentum) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update.to(p.dtype), alpha=-lr * scale)

        return loss
