from types import SimpleNamespace

import torch
from torch import nn

from manifoldflow.gemma_world import GemmaWorldModel, LatentMemoryPort


class _FakeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.last_call = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, **kwargs):
        self.last_call = kwargs
        return SimpleNamespace(loss=kwargs["inputs_embeds"].sum())

    def generate(self, **kwargs):
        self.last_call = kwargs
        return torch.tensor([[1, 2, 3]])


def test_memory_tokens_are_inserted_after_bos_and_labels_are_ignored() -> None:
    base = _FakeCausalLM()
    wrapper = GemmaWorldModel(
        base,
        LatentMemoryPort(state_dimension=4, hidden_dimension=8, bottleneck_dimension=3),
    )
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    labels = ids.clone()

    prepared = wrapper.prepare_inputs(ids, mask, torch.ones((1, 2, 4)), labels=labels)

    assert prepared.inputs_embeds.shape == (1, 5, 8)
    assert prepared.attention_mask.tolist() == [[1, 1, 1, 1, 1]]
    assert prepared.labels.tolist() == [[1, -100, -100, 2, 3]]
    assert torch.equal(prepared.inputs_embeds[:, 0], base.embedding(ids)[:, 0])


def test_memory_mask_can_disable_padded_records() -> None:
    wrapper = GemmaWorldModel(
        _FakeCausalLM(),
        LatentMemoryPort(state_dimension=4, hidden_dimension=8, bottleneck_dimension=3),
    )
    prepared = wrapper.prepare_inputs(
        torch.tensor([[1, 2]]),
        torch.ones((1, 2), dtype=torch.long),
        torch.zeros((1, 3, 4)),
        memory_mask=torch.tensor([[1, 0, 0]]),
    )

    assert prepared.attention_mask.tolist() == [[1, 1, 0, 0, 1]]


def test_identity_keys_are_fused_into_memory_tokens() -> None:
    port = LatentMemoryPort(
        state_dimension=4,
        key_dimension=3,
        hidden_dimension=8,
        bottleneck_dimension=3,
    )
    values = torch.zeros((1, 1, 4))

    first = port(values, identity_keys=torch.tensor([[[1.0, 0.0, 0.0]]]))
    second = port(values, identity_keys=torch.tensor([[[0.0, 1.0, 0.0]]]))

    assert not torch.equal(first, second)


def test_forward_and_generation_use_inputs_embeds() -> None:
    base = _FakeCausalLM()
    wrapper = GemmaWorldModel(
        base,
        LatentMemoryPort(state_dimension=4, hidden_dimension=8, bottleneck_dimension=3),
    )
    common = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "memory_values": torch.zeros((1, 1, 4)),
    }

    wrapper(**common)
    assert "inputs_embeds" in base.last_call
    assert wrapper.generate(**common).tolist() == [[1, 2, 3]]
