import pytest
import torch

from manifoldflow.sonic_memory_port import SonicMemoryPort, SonicMemoryPortConfig


def _port(**overrides: object) -> SonicMemoryPort:
    defaults: dict[str, object] = {"d_model": 8, "memory_dim": 4, "rank": 2}
    defaults.update(overrides)
    return SonicMemoryPort(SonicMemoryPortConfig(**defaults))


def test_output_shape_matches_hidden() -> None:
    port = _port()
    hidden = torch.randn(2, 5, 8)
    memory_vector = torch.randn(2, 4)

    output = port(hidden, memory_vector)

    assert output.shape == hidden.shape


def test_starts_as_a_near_no_op() -> None:
    # Zero-initialized up-projection + very negative gate bias means the
    # port should start close to the identity, so grafting it onto
    # ordinary language modeling doesn't destabilize it before the gate
    # has learned anything.
    port = _port()
    hidden = torch.randn(1, 3, 8)
    memory_vector = torch.randn(1, 4)

    output = port(hidden, memory_vector)

    assert torch.allclose(output, hidden, atol=1e-6)


def test_gradients_reach_all_trainable_parameters() -> None:
    # A wide batch/rank, not a single random draw: `up`'s input passes
    # through a ReLU, and a single unlucky draw where every pre-activation
    # lands negative would zero `up.weight.grad` regardless of the gate -
    # not a bug in the module, but a real trap for this test specifically
    # (hit non-deterministically when this file ran inside the full suite
    # rather than alone, because the earlier tests' torch.randn calls left
    # global RNG state at a different point). A wide batch makes an
    # all-negative draw astronomically unlikely instead of guessing a seed.
    port = _port(d_model=8, memory_dim=4, rank=8)
    hidden = torch.randn(32, 3, 8, requires_grad=True)
    memory_vector = torch.randn(32, 4)

    output = port(hidden, memory_vector)
    output.sum().backward()

    assert port.down.weight.grad is not None
    assert port.up.weight.grad is not None
    assert port.gate.weight.grad is not None
    assert port.gate.bias.grad is not None
    # up starts at zero, so its gradient is the one place a genuinely
    # zero gradient would silently prevent the port from ever leaving
    # the no-op state - confirm it is nonzero.
    assert port.up.weight.grad.abs().sum().item() > 0


def test_memory_vector_influences_output_once_up_is_nonzero() -> None:
    # Wide batch/rank, not two single draws: `down`'s output passes through
    # a ReLU, and a single unlucky pair where both memory_a and memory_b
    # land on the same all-negative side (zeroing `injected` for both,
    # making the outputs bitwise identical) is a real, observed flake at
    # rank=2 - the same trap as test_gradients_reach_all_trainable_parameters
    # above, worked around the same way.
    port = _port(d_model=8, memory_dim=4, rank=8)
    with torch.no_grad():
        port.up.weight.fill_(1.0)
        port.gate.bias.fill_(4.0)  # force the gate open
    hidden = torch.randn(32, 2, 8)
    memory_a = torch.randn(32, 4)
    memory_b = torch.randn(32, 4)

    output_a = port(hidden, memory_a)
    output_b = port(hidden, memory_b)

    assert not torch.allclose(output_a, output_b)


def test_rejects_wrong_hidden_shape() -> None:
    port = _port()
    with pytest.raises(ValueError):
        port(torch.randn(1, 8), torch.randn(1, 4))


def test_rejects_mismatched_memory_vector_batch() -> None:
    port = _port()
    with pytest.raises(ValueError):
        port(torch.randn(2, 3, 8), torch.randn(1, 4))


def test_rejects_wrong_memory_dim() -> None:
    port = _port()
    with pytest.raises(ValueError):
        port(torch.randn(1, 3, 8), torch.randn(1, 5))


def test_config_rejects_non_positive_dims() -> None:
    with pytest.raises(ValueError):
        SonicMemoryPortConfig(d_model=0, memory_dim=4, rank=2)
    with pytest.raises(ValueError):
        SonicMemoryPortConfig(d_model=8, memory_dim=0, rank=2)
    with pytest.raises(ValueError):
        SonicMemoryPortConfig(d_model=8, memory_dim=4, rank=0)
