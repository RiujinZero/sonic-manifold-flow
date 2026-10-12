import pytest
import torch

from manifoldflow.flow import analytic_topk_flow, basin_energy, positive_rate


def test_positive_rate_is_bounded() -> None:
    raw = torch.tensor([-1e6, -2.0, 0.0, 2.0, 1e6])
    rate = positive_rate(raw, rate_min=0.01, rate_max=3.0)
    assert torch.all(rate >= 0.01)
    assert torch.all(rate <= 3.0)
    assert torch.isfinite(rate).all()


def test_single_basin_matches_closed_form() -> None:
    z0 = torch.tensor([[3.0, -1.0]])
    mu = torch.tensor([[[1.0, 2.0]]])
    raw_rate = torch.zeros_like(mu)
    bias = torch.zeros(1, 1)

    zt, _, weight = analytic_topk_flow(z0, mu, raw_rate, bias, t=0.7)
    rate = positive_rate(raw_rate[:, 0])
    expected = mu[:, 0] + (z0 - mu[:, 0]) * torch.exp(-rate * 0.7)

    torch.testing.assert_close(zt, expected)
    torch.testing.assert_close(weight, torch.ones_like(weight))


def test_zero_time_is_identity_and_weights_normalize() -> None:
    generator = torch.Generator().manual_seed(9)
    z0 = torch.randn(3, 5, generator=generator)
    mu = torch.randn(3, 4, 5, generator=generator)
    raw_rate = torch.randn(3, 4, 5, generator=generator)
    bias = torch.randn(3, 4, generator=generator)

    zt, _, weight = analytic_topk_flow(z0, mu, raw_rate, bias, t=0.0)

    torch.testing.assert_close(zt, z0)
    torch.testing.assert_close(weight.sum(-1), torch.ones(3))


def test_per_sample_flow_times_broadcast_over_latent_dimension() -> None:
    z0 = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    mu = torch.zeros(2, 1, 2)
    raw_rate = torch.zeros_like(mu)
    bias = torch.zeros(2, 1)

    zt, _, _ = analytic_topk_flow(z0, mu, raw_rate, bias, t=torch.tensor([0.0, 1.0]))

    torch.testing.assert_close(zt[0], z0[0])
    assert torch.all(zt[1] < z0[1])


def test_flow_has_finite_active_row_gradients() -> None:
    generator = torch.Generator().manual_seed(12)
    z0 = torch.randn(2, 3, generator=generator, requires_grad=True)
    mu = torch.randn(2, 2, 3, generator=generator, requires_grad=True)
    raw_rate = torch.randn(2, 2, 3, generator=generator, requires_grad=True)
    bias = torch.randn(2, 2, generator=generator, requires_grad=True)

    zt, energy, weight = analytic_topk_flow(z0, mu, raw_rate, bias, t=0.4)
    (zt.square().mean() + energy.mean() + weight.square().mean()).backward()

    for value in (z0, mu, raw_rate, bias):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_bias_changes_routing_energy_not_rate() -> None:
    z0 = torch.zeros(1, 2)
    mu = torch.ones(1, 2, 2)
    raw_rate = torch.zeros_like(mu)
    unbiased = basin_energy(z0, mu, raw_rate, torch.zeros(1, 2))
    biased = basin_energy(z0, mu, raw_rate, torch.tensor([[2.0, 0.0]]))

    torch.testing.assert_close(biased[0, 0], unbiased[0, 0] - 2.0)
    torch.testing.assert_close(biased[0, 1], unbiased[0, 1])


@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_temperature_must_be_positive(temperature: float) -> None:
    z0 = torch.zeros(1, 2)
    mu = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match="temperature"):
        analytic_topk_flow(
            z0, mu, torch.zeros_like(mu), torch.zeros(1, 1), temperature=temperature
        )
