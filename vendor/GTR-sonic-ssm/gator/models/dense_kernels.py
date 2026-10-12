"""Dense grouped linear kernels for Arrow cell populations.

The semantic reference uses compact ``einsum`` expressions.  The grouped
backend reshapes the population into batched matrix multiplications so CUDA
libraries and ``torch.compile`` see regular GEMMs instead of many implicit
cell-local contractions. Optional zero padding aligns reduction/output
dimensions without adding trainable parameters.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


DenseKernelBackend = Literal["einsum", "grouped_bmm"]
DenseNodeBackend = Literal["einsum", "folded_bmm"]


def _validate_padding(pad_to_multiple: int) -> None:
    if (
        isinstance(pad_to_multiple, bool)
        or not isinstance(pad_to_multiple, int)
        or pad_to_multiple <= 0
    ):
        raise ValueError("pad_to_multiple must be a positive integer")


def _padded_size(size: int, multiple: int) -> int:
    return ((size + multiple - 1) // multiple) * multiple


def _pad_linear_operands(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    pad_to_multiple: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    """Zero-pad the final input/output dimensions of linear operands."""

    _validate_padding(pad_to_multiple)
    input_features = inputs.shape[-1]
    output_features = weight.shape[-2]
    padded_input = _padded_size(input_features, pad_to_multiple)
    padded_output = _padded_size(output_features, pad_to_multiple)
    input_padding = padded_input - input_features
    output_padding = padded_output - output_features
    if input_padding:
        inputs = F.pad(inputs, (0, input_padding))
    if input_padding or output_padding:
        weight = F.pad(
            weight,
            (0, input_padding, 0, output_padding),
        )
    if bias is not None and output_padding:
        bias = F.pad(bias, (0, output_padding))
    return inputs, weight, bias, output_features


def cell_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    backend: DenseKernelBackend = "grouped_bmm",
    pad_to_multiple: int = 1,
) -> torch.Tensor:
    """Apply independent cell weights to ``[B,C,T,I]`` inputs.

    ``weight`` and ``bias`` have shapes ``[C,O,I]`` and ``[C,O]``. The result
    has shape ``[B,C,T,O]``.
    """

    if inputs.ndim != 4:
        raise ValueError("cell inputs must have [batch,cells,tokens,input] shape")
    if weight.ndim != 3 or (bias is not None and bias.ndim != 2):
        raise ValueError("cell weight/bias must have [cells,output,input] shapes")
    batch, cells, tokens, input_features = inputs.shape
    if weight.shape[0] != cells or weight.shape[2] != input_features:
        raise ValueError("cell weight must match input cells/features")
    if bias is not None and bias.shape != weight.shape[:2]:
        raise ValueError("cell bias must match weight cells/output")
    if backend == "einsum":
        output = torch.einsum("bcti,coi->bcto", inputs, weight)
        return output if bias is None else output + bias[None, :, None]
    if backend != "grouped_bmm":
        raise ValueError("dense kernel backend must be 'einsum' or 'grouped_bmm'")

    inputs, weight, bias, output_features = _pad_linear_operands(
        inputs,
        weight,
        bias,
        pad_to_multiple=pad_to_multiple,
    )
    grouped_inputs = inputs.permute(1, 0, 2, 3).reshape(
        cells, batch * tokens, inputs.shape[-1]
    )
    grouped_output = torch.bmm(
        grouped_inputs,
        weight.transpose(1, 2),
    )
    if bias is not None:
        grouped_output = grouped_output + bias[:, None]
    return grouped_output[..., :output_features].reshape(
        cells, batch, tokens, output_features
    ).permute(1, 0, 2, 3)


def cell_node_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    backend: DenseKernelBackend = "grouped_bmm",
    pad_to_multiple: int = 1,
) -> torch.Tensor:
    """Apply independent cell/node weights to ``[B,C,T,N,I]`` inputs."""

    if inputs.ndim != 5:
        raise ValueError(
            "cell-node inputs must have [batch,cells,tokens,nodes,input] shape"
        )
    if weight.ndim != 4 or bias.ndim != 3:
        raise ValueError(
            "cell-node weight/bias must have [cells,nodes,output,input] shapes"
        )
    batch, cells, tokens, nodes, input_features = inputs.shape
    if (
        weight.shape[0] != cells
        or weight.shape[1] != nodes
        or weight.shape[3] != input_features
    ):
        raise ValueError("cell-node weight must match input cells/nodes/features")
    if bias.shape != weight.shape[:3]:
        raise ValueError("cell-node bias must match weight cells/nodes/output")
    if backend == "einsum":
        return (
            torch.einsum("bctni,cnoi->bctno", inputs, weight)
            + bias[None, :, None]
        )
    if backend != "grouped_bmm":
        raise ValueError("dense kernel backend must be 'einsum' or 'grouped_bmm'")

    inputs, weight, bias, output_features = _pad_linear_operands(
        inputs,
        weight,
        bias,
        pad_to_multiple=pad_to_multiple,
    )
    grouped_inputs = inputs.permute(1, 3, 0, 2, 4).reshape(
        cells * nodes,
        batch * tokens,
        inputs.shape[-1],
    )
    grouped_weight = weight.reshape(
        cells * nodes, weight.shape[-2], weight.shape[-1]
    )
    grouped_bias = bias.reshape(cells * nodes, bias.shape[-1])
    grouped_output = torch.bmm(
        grouped_inputs,
        grouped_weight.transpose(1, 2),
    )
    grouped_output = grouped_output + grouped_bias[:, None]
    return grouped_output[..., :output_features].reshape(
        cells, nodes, batch, tokens, output_features
    ).permute(2, 0, 3, 1, 4)


def padded_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    pad_to_multiple: int = 1,
) -> torch.Tensor:
    """Apply one shared linear transform with parameter-free tile padding."""

    if inputs.ndim < 2 or weight.ndim != 2:
        raise ValueError("linear inputs/weight must have [...,input] and [output,input]")
    if inputs.shape[-1] != weight.shape[-1]:
        raise ValueError("linear input and weight features must match")
    if bias is not None and bias.shape != weight.shape[:1]:
        raise ValueError("linear bias must match weight output")
    inputs, weight, bias, output_features = _pad_linear_operands(
        inputs,
        weight,
        bias,
        pad_to_multiple=pad_to_multiple,
    )
    return F.linear(inputs, weight, bias)[..., :output_features]


def fold_cell_node_parameters(
    node_embedding: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold N affine node transforms into one wide transform per cell.

    For node ``n`` the original affine input is ``base + node_embedding[n]``.
    Linearity permits:

    ``W[n] @ base + (W[n] @ node_embedding[n] + bias[n])``.

    The first term becomes one ``D -> N*O`` grouped GEMM while the second term
    is a small relay-invariant offset.
    """

    if node_embedding.ndim != 3 or weight.ndim != 4 or bias.ndim != 3:
        raise ValueError(
            "node embedding/weight/bias must have [C,N,I], [C,N,O,I], "
            "and [C,N,O] shapes"
        )
    cells, nodes, input_features = node_embedding.shape
    if weight.shape[:2] != (cells, nodes) or weight.shape[-1] != input_features:
        raise ValueError("node weight must match embedding cells/nodes/features")
    if bias.shape != weight.shape[:3]:
        raise ValueError("node bias must match weight cells/nodes/output")
    output_features = weight.shape[-2]
    folded_weight = weight.reshape(
        cells, nodes * output_features, input_features
    )
    offset = (
        torch.einsum("cni,cnoi->cno", node_embedding, weight)
        + bias
    )
    return folded_weight, offset


def folded_cell_node_linear(
    base: torch.Tensor,
    node_embedding: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    backend: DenseNodeBackend = "folded_bmm",
    pad_to_multiple: int = 1,
) -> torch.Tensor:
    """Evaluate node-affine outputs without materializing repeated node input."""

    if base.ndim != 4:
        raise ValueError("base must have [batch,cells,tokens,input] shape")
    if backend == "einsum":
        return cell_node_linear(
            base.unsqueeze(3) + node_embedding[None, :, None],
            weight,
            bias,
            backend="einsum",
        )
    if backend != "folded_bmm":
        raise ValueError("dense node backend must be 'einsum' or 'folded_bmm'")
    folded_weight, offset = fold_cell_node_parameters(
        node_embedding, weight, bias
    )
    projected = cell_linear(
        base,
        folded_weight,
        backend="grouped_bmm",
        pad_to_multiple=pad_to_multiple,
    )
    return projected.view(
        base.shape[0],
        base.shape[1],
        base.shape[2],
        weight.shape[1],
        weight.shape[2],
    ) + offset[None, :, None]
